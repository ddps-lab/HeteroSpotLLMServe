# Multi-threaded Tensor Store Server

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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from safetensors import safe_open
from multiprocessing.managers import BaseManager, DictProxy
import torch.multiprocessing as mp
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] [Server] - %(message)s')

TENSOR_DICT = {}
TENSOR_DICT_LOCK = threading.Lock()

MANAGER_HOST = '127.0.0.1'
MANAGER_PORT = 50001
MANAGER_AUTHKEY = b'param_store'

DTYPE = torch.float16

NUM_FILE_WORKERS = 4
NUM_TENSOR_WORKERS = 16

# PP Parallelism 을 위해서
START_LAYER_ID = -1
END_LAYER_ID = -1 # 마지막 포함하지 않음
TOTAL_LAYER_NUM = -1
# TP Parallelism 을 위해서
TENSOR_PARALLEL_SIZE = -1
TENSOR_PARALLEL_RANK = -1
LOCAL_RANK = -1
DEVICE="cuda"
# 로딩 방식 설정
USE_CPU_LOADING = False  # False: 직접 GPU 로딩, True: CPU 로딩 후 GPU 전송
# 열 단위로 쪼개는 경우 1번째 차원 (0부터 세면 0번째 차원) 을 쪼개야 한다 -> output dim
# 행 단위로 쪼개는 경우 2번째 차원 (0부터 세면 1번째 차원) 을 쪼개야 한다 -> input dim
# vLLM 에서는 input_dim 이 1, output_dim 이 0이다.
INPUT_DIM = 0
OUTPUT_DIM = 1
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
    parser.add_argument("--use-cpu-loading", action="store_true", 
                        help="Load tensors to CPU first, then transfer to GPU (safer for multi-threading)")
    return parser.parse_args()

def set_global_variables(args: argparse.Namespace, config_dict: dict):
    global TENSOR_PARALLEL_SIZE, LOCAL_RANK, DTYPE, START_LAYER_ID, END_LAYER_ID, TOTAL_LAYER_NUM, DEVICE, TENSOR_PARALLEL_RANK, USE_CPU_LOADING
    TENSOR_PARALLEL_SIZE = args.tensor_parallel_size
    LOCAL_RANK = args.local_rank
    TENSOR_PARALLEL_RANK = args.local_rank
    DEVICE = f"cuda:{LOCAL_RANK}"
    START_LAYER_ID = args.start_layer_id
    END_LAYER_ID = args.end_layer_id
    TOTAL_LAYER_NUM = config_dict["num_hidden_layers"]
    
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

    USE_CPU_LOADING = args.use_cpu_loading

def process_tensor(tensor_name: str, tensor_slice, tie_word_embeddings: bool):
    """개별 텐서를 처리하는 함수 (IO intensive한 get_slice 이후의 처리)"""
    try:
        if tensor_name.split('.')[-2] == "embed_tokens":
            vocab_size, hidden_size = tensor_slice.get_shape()
            start_idx, end_idx, per_shard_dim_size = get_range_vocabulary_embedding_tensor(vocab_size)
            if end_idx - start_idx > per_shard_dim_size:
                raise ValueError(f"vocab_end_idx - vocab_start_idx > per_shard_vocab_size 임. 있을 수 없는 일")
            elif end_idx - start_idx < per_shard_dim_size:
                tensor = torch.zeros(per_shard_dim_size, hidden_size, dtype=DTYPE, device=DEVICE if not USE_CPU_LOADING else "cpu")
                if USE_CPU_LOADING:
                    tensor[:end_idx - start_idx, :] = tensor_slice[start_idx:end_idx, :].to(dtype=DTYPE)
                    tensor = tensor.to(device=DEVICE)
                else:
                    tensor[:end_idx - start_idx, :] = tensor_slice[start_idx:end_idx, :].to(dtype=DTYPE)
            else:
                if USE_CPU_LOADING:
                    tensor = tensor_slice[start_idx:end_idx, :].to(dtype=DTYPE).to(device=DEVICE)
                else:
                    tensor = tensor_slice[start_idx:end_idx, :].to(dtype=DTYPE)
        elif tensor_name.split('.')[-2] in DIV_COLUMN_WISE_LIST:
            output_dim, input_dim = tensor_slice.get_shape()
            start_idx, end_idx, per_shard_dim_size = get_tensor_idx_range(output_dim)
            if USE_CPU_LOADING:
                tensor = tensor_slice[start_idx:end_idx, :].to(dtype=DTYPE).to(device=DEVICE)
            else:
                tensor = tensor_slice[start_idx:end_idx, :].to(dtype=DTYPE)
        elif tensor_name.split('.')[-2] in DIV_ROW_WISE_LIST:
            output_dim, input_dim = tensor_slice.get_shape()
            start_idx, end_idx, per_shard_dim_size = get_tensor_idx_range(input_dim)
            if USE_CPU_LOADING:
                tensor = tensor_slice[:, start_idx:end_idx].to(dtype=DTYPE).to(device=DEVICE)
            else:
                tensor = tensor_slice[:, start_idx:end_idx].to(dtype=DTYPE)
        else:
            if USE_CPU_LOADING:
                tensor = tensor_slice[:].to(dtype=DTYPE).to(device=DEVICE)
            else:
                tensor = tensor_slice[:].to(dtype=DTYPE)
        
        assert tensor is not None
        assert tensor.dtype == DTYPE
        assert tensor.device.type == "cuda"

        # Thread-safe하게 TENSOR_DICT에 추가
        with TENSOR_DICT_LOCK:
            TENSOR_DICT[tensor_name] = tensor
            
        logging.info(f"Loaded {tensor_name} / shape: {tensor.shape} / dtype: {tensor.dtype} / device: {tensor.device}")
        del tensor
        return True
        
    except Exception as e:
        logging.error(f"Error processing tensor {tensor_name}: {e}")
        raise e

def load_tensors_from_file(st_file: str, tie_word_embeddings: bool):
    """각 파일에서 텐서들을 병렬로 로딩하는 함수"""
    logging.info(f"Starting to load file: {st_file} (CPU loading: {USE_CPU_LOADING})")
    
    # CUDA 컨텍스트 설정 (직접 GPU 로딩 시 필요)
    if not USE_CPU_LOADING:
        torch.cuda.set_device(LOCAL_RANK)
    
    # 로딩 방식에 따라 device 설정
    load_device = "cpu" if USE_CPU_LOADING else DEVICE
    
    with safe_open(st_file, framework="pt", device=load_device) as f:
        # 먼저 필요한 텐서들을 필터링
        valid_tensors = []
        invalid_tensors = []
        for tensor_name in f.keys():
            should_load = True
            
            if tensor_name.startswith("model.layers"):
                layer_idx = int(tensor_name.split('.')[2])
                if not (START_LAYER_ID <= layer_idx < END_LAYER_ID):
                    should_load = False
            elif tensor_name.startswith("model.embed_tokens"):
                if not (START_LAYER_ID <= 0):
                    if not tie_word_embeddings:
                        should_load = False
                    # tie_word_embeddings 가 True 일 때 마지막 스테이지는 embed_tokens 를 가져야 함
                    elif not (TOTAL_LAYER_NUM <= END_LAYER_ID):
                        should_load = False
            elif tensor_name.startswith("model.norm"):
                # 마지막 스테이지만 가질 것
                if not (TOTAL_LAYER_NUM <= END_LAYER_ID):
                    should_load = False
            elif tensor_name.startswith("lm_head"):
                # lm_head 는 마지막 스테이지가 갖는다.
                if not (TOTAL_LAYER_NUM <= END_LAYER_ID):
                    should_load = False
            
            if should_load:
                valid_tensors.append(tensor_name)
            else:
                invalid_tensors.append(tensor_name)
        
        logging.info(f"File {st_file}: Found {len(valid_tensors)} valid tensors to load / {len(invalid_tensors)} invalid tensors")
        
        # 텐서별로 병렬 처리 (IO intensive한 get_slice 부분)
        max_tensor_workers = min(NUM_TENSOR_WORKERS, len(valid_tensors))  # 텐서별 워커 수 제한
        
        if not max_tensor_workers == 0:
            with ThreadPoolExecutor(max_workers=max_tensor_workers) as tensor_executor:
                tensor_futures = []
                
                for tensor_name in valid_tensors:
                    # get_slice는 IO intensive하므로 이것도 병렬로 처리
                    future = tensor_executor.submit(lambda tn=tensor_name: (tn, f.get_slice(tn)))
                    tensor_futures.append(future)
                
                # 완료된 순서대로 텐서 처리
                for future in as_completed(tensor_futures):
                    tensor_name, tensor_slice = future.result()
                    # 실제 텐서 처리는 메인 스레드에서 (또는 별도 처리)
                    process_tensor(tensor_name, tensor_slice, tie_word_embeddings)
    
    logging.info(f"Finished loading file: {st_file}")


def fuse_tensor(layer_idx: int):
    # bias 는 나중에 처리하자

    # 1. q_proj, k_proj, v_proj 를 가져온다.
    q_proj_tensor_name = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
    k_proj_tensor_name = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
    v_proj_tensor_name = f"model.layers.{layer_idx}.self_attn.v_proj.weight"

    q_proj_tensor = TENSOR_DICT[q_proj_tensor_name]
    k_proj_tensor = TENSOR_DICT[k_proj_tensor_name]
    v_proj_tensor = TENSOR_DICT[v_proj_tensor_name]

    # q_proj, k_proj, v_proj 를 합친다.
    # input dim : 1, 이 dimension 은 그대로여야 함
    # Input Dimension 을 그대로 두고 이어붙인다.
    qkv_proj_tensor = torch.cat((q_proj_tensor, k_proj_tensor, v_proj_tensor), dim=INPUT_DIM)
    qkv_proj_tensor_name = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"

    # qkv_proj_tensor 를 저장한다.
    TENSOR_DICT[qkv_proj_tensor_name] = qkv_proj_tensor

    # 기존 q_proj, k_proj, v_proj 는 삭제한다.
    del TENSOR_DICT[q_proj_tensor_name]
    del TENSOR_DICT[k_proj_tensor_name]
    del TENSOR_DICT[v_proj_tensor_name]

    # 2. gate_proj, up_proj 를 가져온다.
    gate_proj_tensor_name = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
    up_proj_tensor_name = f"model.layers.{layer_idx}.mlp.up_proj.weight"

    gate_proj_tensor = TENSOR_DICT[gate_proj_tensor_name]
    up_proj_tensor = TENSOR_DICT[up_proj_tensor_name]

    # gate_proj, up_proj 를 합친다.
    # input dim : 1, 이 dimension 은 그대로여야 함
    gate_up_proj_tensor = torch.cat((gate_proj_tensor, up_proj_tensor), dim=INPUT_DIM)
    gate_up_proj_tensor_name = f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"

    # gate_up_proj_tensor 를 저장한다.
    TENSOR_DICT[gate_up_proj_tensor_name] = gate_up_proj_tensor

    # 기존 gate_proj, up_proj 는 삭제한다.
    del TENSOR_DICT[gate_proj_tensor_name]
    del TENSOR_DICT[up_proj_tensor_name]

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
    
    logging.info(f"Loading strategy: {'CPU -> GPU transfer' if USE_CPU_LOADING else 'Direct GPU loading'}")

    load_start = time.perf_counter()
    # 모델 로딩 - 두 단계 병렬화: 파일별 + 텐서별
    try:
        # 파일별 병렬 처리
        max_file_workers = min(NUM_FILE_WORKERS, len(hf_weights_files))  # GPU 메모리 고려하여 파일 워커 수 제한
        logging.info(f"Using {max_file_workers} file workers for {len(hf_weights_files)} files")
        
        if not max_file_workers == 0:
            with ThreadPoolExecutor(max_workers=max_file_workers) as file_executor:
                file_futures = []
                
                for st_file in hf_weights_files:
                    future = file_executor.submit(load_tensors_from_file, st_file, tie_word_embeddings)
                    file_futures.append(future)
                
                # 모든 파일 로딩 완료 대기
                for future in as_completed(file_futures):
                    future.result()  # 예외 발생 시 여기서 catch
                
    except Exception as e:
        logging.error(f"Model Loading 중 Error 발생: {e}")
        raise e

    # Fusing 하기 전 memory usage check
    logging.info(f"Before fusing, memory usage: {torch.cuda.memory_summary(device=DEVICE)}")
    # 여기에 이제 tensor 를 fuse 해야 함.
    # 1 : q_proj, k_proj, v_proj 를 input_dim (1) 을 그대로 두고
    # qkv_proj 로 합쳐야 함.
    # 2 : gate_proj, up_proj 를 input_dim (1) 을 그대로 두고
    # gate_up_proj 로 합쳐야 함.
    # 이는 layer 별로 이루어져야 함.
    logging.info(f"Start fusing tensors for layers {START_LAYER_ID} to {END_LAYER_ID}")
    for layer_idx in range(START_LAYER_ID, END_LAYER_ID):
        fuse_tensor(layer_idx)
    logging.info(f"Finished fusing tensors for layers {START_LAYER_ID} to {END_LAYER_ID}")
    # Fusing 후 memory usage check
    logging.info("Emptying CUDA cache after fusing...")
    torch.cuda.empty_cache()
    logging.info(f"After emptying cache, memory usage: {torch.cuda.memory_summary(device=DEVICE)}")

    # 최종적인 tensor들 print 하기
    for tensor_name in TENSOR_DICT:
        logging.info(f"tensor_name: {tensor_name} / shape: {TENSOR_DICT[tensor_name].shape} / dtype: {TENSOR_DICT[tensor_name].dtype} / device: {TENSOR_DICT[tensor_name].device}")
        
    load_end = time.perf_counter()
    logging.info(f"Model Loading time: {load_end - load_start} seconds")
    
    if not TENSOR_DICT:
        raise ValueError("텐서 로딩 실패 : TENSOR_DICT 가 비어있음")
    
    logging.info(f"Successfully loaded {len(TENSOR_DICT)} tensors")
    
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
