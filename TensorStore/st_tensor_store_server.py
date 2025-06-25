# Single Threaded Tensor Store Server

import argparse
import glob
import json
from typing import List, Optional, Union
import huggingface_hub.constants
from huggingface_hub import HfFileSystem, snapshot_download
import tempfile
import os
import hashlib
import filelock
import fnmatch
from pathlib import Path
from tqdm.auto import tqdm
import time

from safetensors import safe_open
from multiprocessing.managers import BaseManager, DictProxy
import torch.multiprocessing as mp
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] [Server] - %(message)s')

TENSOR_DICT = {}

MANAGER_HOST = '127.0.0.1'
MANAGER_PORT = 50001
MANAGER_AUTHKEY = b'param_store'

DTYPE = torch.float16

# PP Parallelism 을 위해서
START_LAYER_ID = -1
END_LAYER_ID = -1 # 마지막 포함하지 않음
TOTAL_LAYER_NUM = -1
# TP Parallelism 을 위해서
TENSOR_PARALLEL_SIZE = -1
TENSOR_PARALLEL_RANK = -1
LOCAL_RANK = -1
DEVICE="cuda"
# 열 단위로 쪼개는 경우 1번째 차원 (0부터 세면 0번째 차원) 을 쪼개야 한다 -> output dim
# 행 단위로 쪼개는 경우 2번째 차원 (0부터 세면 1번째 차원) 을 쪼개야 한다 -> input dim
# vLLM 에서는 input_dim 이 1, output_dim 이 0이다.
COLUMN_WAY = 0
ROW_WAY = 1
DIV_COLUMN_WISE_LIST = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
DIV_ROW_WISE_LIST = ["o_proj", "down_proj"]
VOCAB_PADDING_SIZE = 64

def get_tensor_dict():
    return TENSOR_DICT

class TensorManager(BaseManager):
    pass

# 매니저에 공유 데이터 접근 함수 등록 (get_tensors)
# get_tensor 이름 함수 등록하고 DictProxy 사용
TensorManager.register('get_tensor_dict', callable=get_tensor_dict, proxytype=DictProxy)

### VLLM 의 weight_utils.py 로부터 가져온 소스
class DisabledTqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)

temp_dir = tempfile.gettempdir()

def get_lock(model_name_or_path: Union[str, Path],
             cache_dir: Optional[str] = None):
    lock_dir = cache_dir or temp_dir
    model_name_or_path = str(model_name_or_path)
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name),
                             mode=0o666)
    return lock

def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: Optional[str],
    allow_patterns: List[str],
    revision: Optional[str] = None,
    ignore_patterns: Optional[Union[str, List[str]]] = None,
) -> str:
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE
    if not local_only:
        # Before we download we look at that is available:
        fs = HfFileSystem()
        file_list = fs.ls(model_name_or_path, detail=False, revision=revision)

        # depending on what is available we download different things
        for pattern in allow_patterns:
            matching = fnmatch.filter(file_list, pattern)
            if len(matching) > 0:
                allow_patterns = [pattern]
                break

    logging.info("Using model weights format %s", allow_patterns)
    allow_patterns.append("config.json")
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        start_time = time.perf_counter()
        hf_folder = snapshot_download(
            model_name_or_path,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
            tqdm_class=DisabledTqdm,
            revision=revision,
            local_files_only=local_only,
        )
        time_taken = time.perf_counter() - start_time
        if time_taken > 0.5:
            logging.info("Time spent downloading weights for %s: %.6f seconds",
                        model_name_or_path, time_taken)
    return hf_folder
### VLLM 의 weight_utils.py 로부터 가져온 소스 끝

# vocabulary embedding 텐서는 output dim 을 쪼갠다. 즉, 0번째 차원을 쪼갠다.
# LoRA 도입하면 바로터짐
def get_range_vocabulary_embedding_tensor(vocab_size: int) -> torch.Tensor:
    padding_size = VOCAB_PADDING_SIZE

    vocab_size_padded = ((vocab_size + padding_size - 1) // padding_size) * padding_size
    assert vocab_size_padded % TENSOR_PARALLEL_SIZE == 0
    per_shard_vocab_size = vocab_size_padded // TENSOR_PARALLEL_SIZE
    padded_vocab_idx_start = TENSOR_PARALLEL_RANK * per_shard_vocab_size
    padded_vocab_idx_end = padded_vocab_idx_start + per_shard_vocab_size
    
    # remove padding
    vocab_start_idx = min(padded_vocab_idx_start, vocab_size)
    vocab_end_idx = min(padded_vocab_idx_end, vocab_size)
    
    return vocab_start_idx, vocab_end_idx, per_shard_vocab_size

def get_tensor_idx_range(dim: int) -> torch.Tensor:
    assert dim % TENSOR_PARALLEL_SIZE == 0
    per_shard_dim_size = dim // TENSOR_PARALLEL_SIZE
    shard_idx_start = TENSOR_PARALLEL_RANK * per_shard_dim_size
    shard_idx_end = shard_idx_start + per_shard_dim_size
    return shard_idx_start, shard_idx_end, per_shard_dim_size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--start-layer-id", type=int, default=0)
    parser.add_argument("--end-layer-id", type=int, default=-1)
    return parser.parse_args()

def set_global_variables(args: argparse.Namespace, config_dict: dict):
    global TENSOR_PARALLEL_SIZE, LOCAL_RANK, DTYPE, START_LAYER_ID, END_LAYER_ID, TOTAL_LAYER_NUM, DEVICE, TENSOR_PARALLEL_RANK
    TENSOR_PARALLEL_SIZE = args.tensor_parallel_size
    LOCAL_RANK = args.local_rank
    TENSOR_PARALLEL_RANK = args.local_rank
    DEVICE = f"cuda:{LOCAL_RANK}"
    TOTAL_LAYER_NUM = config_dict["num_hidden_layers"]
    START_LAYER_ID = args.start_layer_id
    END_LAYER_ID = args.end_layer_id
    if END_LAYER_ID == -1:
        END_LAYER_ID = config_dict["num_hidden_layers"]

    if args.dtype == "float16":
        DTYPE = torch.float16
    elif args.dtype == "float32":
        DTYPE = torch.float32
    elif args.dtype == "bfloat16":
        DTYPE = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

def main():
    args = parse_args()
    model_name = args.model_name
    logging.info(f"args: {args}")

    try:
        mp.set_start_method('fork', force=True)
        logging.info("Set multiprocessing start method to 'fork'.")
    except RuntimeError:
        logging.warning("Could not set start method to 'fork'. Using default.")

    logging.info("Loading model directly in the main process...")

    # 모델 다운로딩
    download_start = time.perf_counter()
    allow_patterns = ["*.safetensors", "*.bin"]
    hf_folder = download_weights_from_hf(model_name, None, allow_patterns)
    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if len(hf_weights_files) > 0:
            break
    download_end = time.perf_counter()
    logging.info(f"Model Download time: {download_end - download_start} seconds")

    # config 파일 보기
    config_path = os.path.join(hf_folder, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    tie_word_embeddings = config_dict["tie_word_embeddings"]
    logging.info(f"model config: {config_dict}")
    
    set_global_variables(args, config_dict)

    load_start = time.perf_counter()
    # 모델 로딩 해서 텐서에 저장
    try:
        for st_file in tqdm(hf_weights_files, desc="Loading safetensors checkpoint shards", disable=True):
            with safe_open(st_file, framework="pt", device=DEVICE) as f:
                for tensor_name in f.keys():
                    if tensor_name.startswith("model.layers"):
                        layer_idx = int(tensor_name.split('.')[2])
                        if not (START_LAYER_ID <= layer_idx < END_LAYER_ID):
                            continue
                    elif tensor_name.startswith("model.embed_tokens"):
                        if not (START_LAYER_ID <= 0):
                            if not tie_word_embeddings:
                                continue
                            # tie_word_embeddings 가 True 일 때 마지막 스테이지는 embed_tokens 를 가져야 함
                            if not (TOTAL_LAYER_NUM <= END_LAYER_ID):
                                continue
                    elif tensor_name.startswith("model.norm"):
                        # 마지막 스테이지만 가질 것
                        if not TOTAL_LAYER_NUM <= END_LAYER_ID:
                            continue
                    elif tensor_name.startswith("lm_head"):
                        # lm_head 는 마지막 스테이지가 갖는다.
                        if not (TOTAL_LAYER_NUM <= END_LAYER_ID):
                            continue
                    
                    tensor_slice = f.get_slice(tensor_name)

                    if tensor_name.split('.')[-2] == "embed_tokens":
                        vocab_size, hidden_size = tensor_slice.get_shape()
                        start_idx, end_idx, per_shard_dim_size = get_range_vocabulary_embedding_tensor(vocab_size)
                        if end_idx - start_idx > per_shard_dim_size:
                            raise ValueError(f"vocab_end_idx - vocab_start_idx > per_shard_vocab_size 임. 있을 수 없는 일")
                        elif end_idx - start_idx < per_shard_dim_size:
                            tensor = torch.zeros(per_shard_dim_size, hidden_size, dtype=DTYPE)
                            tensor[:end_idx - start_idx, :] = tensor_slice[start_idx:end_idx, :].to(dtype=DTYPE)
                        else:
                            tensor = tensor_slice[start_idx:end_idx, :].to(dtype=DTYPE)
                    elif tensor_name.split('.')[-2] in DIV_COLUMN_WISE_LIST:
                        output_dim, input_dim = tensor_slice.get_shape()
                        start_idx, end_idx, per_shard_dim_size = get_tensor_idx_range(output_dim)
                        tensor = tensor_slice[start_idx:end_idx, :].to(dtype=DTYPE)
                    elif tensor_name.split('.')[-2] in DIV_ROW_WISE_LIST:
                        output_dim, input_dim = tensor_slice.get_shape()
                        start_idx, end_idx, per_shard_dim_size = get_tensor_idx_range(input_dim)
                        tensor = tensor_slice[:, start_idx:end_idx].to(dtype=DTYPE)
                    else:
                        tensor = tensor_slice[:].to(dtype=DTYPE)
                    
                    assert tensor is not None
                    assert tensor.dtype == DTYPE

                    TENSOR_DICT[tensor_name] = tensor
                    logging.info(f"Loaded {tensor_name} / shape: {tensor.shape} / dtype: {tensor.dtype}")
                    del tensor
    except Exception as e:
        logging.error(f"Model Loading 중 Error 발생")
        raise e
    load_end = time.perf_counter()
    logging.info(f"Model Loading time: {load_end - load_start} seconds")
    
    if not TENSOR_DICT:
        raise ValueError("텐서 로딩 실패 : TENSOR_DICT 가 비어있음")
    
    manager = TensorManager(address=(MANAGER_HOST, MANAGER_PORT), authkey=MANAGER_AUTHKEY)

    try:
        logging.info(f"TensorManager 서버 시작 {MANAGER_HOST}:{MANAGER_PORT}...")
        server = manager.get_server()
        logging.info("Manager server running. Waiting for client connections...")
        logging.info("Press Ctrl+C to stop.")
        server.serve_forever() # 블로킹 호출

    except OSError as bind_e:
        logging.error(f"Failed to bind manager server: {bind_e}. Port {MANAGER_PORT} might be in use.")
    except KeyboardInterrupt:
        logging.info("Received Ctrl+C. Shutting down manager server...")
    except Exception as e:
        logging.exception(f"An unexpected error occurred in manager server loop: {e}")
    finally:
        logging.info("Server shutting down.")

if __name__ == "__main__":
    main()