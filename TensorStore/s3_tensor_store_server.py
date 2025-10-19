# S3 Tensor Store Server - Downloads from S3 and serves tensors

import argparse
import json
from typing import List, Optional, Union
import os
import hashlib
import filelock
import tempfile
from pathlib import Path
from tqdm.auto import tqdm
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import io

import torch
from multiprocessing.managers import BaseManager, DictProxy
import torch.multiprocessing as mp
import logging
import socket
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from botocore.config import Config

# Add parent directory to path for protocols import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocols import TensorStoreRequest, TensorStoreResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] [S3Server] - %(message)s')

TENSOR_DICT = {}
TENSOR_DICT_LOCK = threading.Lock()

MANAGER_HOST = '127.0.0.1'
MANAGER_PORT = 50001
MANAGER_AUTHKEY = b'param_store'

STATUS_HOST = '0.0.0.0'

DTYPE = None  # Will be set based on loaded tensors

NUM_TENSOR_WORKERS = -1

# PP Parallelism variables
START_LAYER_ID = -1
END_LAYER_ID = -1
TOTAL_LAYER_NUM = -1
# TP Parallelism variables
TENSOR_PARALLEL_SIZE = -1
TENSOR_PARALLEL_RANK = -1
LOCAL_RANK = -1
DEVICE = "cuda"

# Tensor partitioning constants (from mt_tensor_store_server.py)
INPUT_DIM = 0
OUTPUT_DIM = 1
DIV_COLUMN_WISE_LIST = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
DIV_ROW_WISE_LIST = ["o_proj", "down_proj"]
VOCAB_PADDING_SIZE = 64

TENSOR_INITIALIZE_COMPLETE = False
SHUTDOWN_EVENT = threading.Event()

# KV cache related global variables
BLOCK_SIZE = -1 # BLOCK_SIZE defines how many tokens are contained in one block.
GPU_MEMORY_UTILIZATION = -1.0
SWAP_SPACE_BYTES = -1
CACHE_DTYPE = None  # Will be set based on model dtype
PIPELINE_PARALLEL_SIZE = -1
PIPELINE_PARALLEL_RANK = -1
MAX_MODEL_LEN = -1

def get_tensor_dict():
    return TENSOR_DICT

class TensorManager(BaseManager):
    pass

TensorManager.register('get_tensor_dict', callable=get_tensor_dict, proxytype=DictProxy)

### Utility functions from mt_tensor_store_server.py
class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def should_load_tensor(tensor_name: str, tie_word_embeddings: bool) -> bool:
    """Check if tensor should be loaded based on layer partitioning logic (same as mt_tensor_store_server.py)"""
    should_load = True
    
    if tensor_name.startswith("model.layers"):
        layer_idx = int(tensor_name.split('.')[2])
        if not (START_LAYER_ID <= layer_idx < END_LAYER_ID):
            should_load = False
    elif tensor_name.startswith("model.embed_tokens"):
        if not (START_LAYER_ID <= 0):
            if not tie_word_embeddings:
                should_load = False
            elif not (TOTAL_LAYER_NUM <= END_LAYER_ID):
                should_load = False
    elif tensor_name.startswith("model.norm"):
        if not (TOTAL_LAYER_NUM <= END_LAYER_ID):
            should_load = False
    elif tensor_name.startswith("lm_head"):
        if not (TOTAL_LAYER_NUM <= END_LAYER_ID):
            should_load = False
    
    return should_load

def list_tensor_files_from_s3_with_boto3(s3_client, bucket_name: str, base_s3_path: str) -> List[str]:
    """Get list of all tensor file names available in S3 for this TP configuration"""
    tensor_names = []

    try:
        # Set up prefix for listing with TP-specific path: TP{tp_size}/shard{tp_rank}/
        if base_s3_path:
            prefix = f"{base_s3_path}/TP{TENSOR_PARALLEL_SIZE}/shard{TENSOR_PARALLEL_RANK}/"
        else:
            prefix = f"TP{TENSOR_PARALLEL_SIZE}/shard{TENSOR_PARALLEL_RANK}/"

        logging.info(f"Listing tensors from S3 with prefix: {prefix}")

        # List objects with pagination
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']

                    if key.endswith('.bin') and not key.endswith('config.json'):
                        # Extract tensor name (remove prefix and .bin extension)
                        if key.startswith(prefix):
                            tensor_name = key[len(prefix):][:-4]  # Remove .bin
                            tensor_names.append(tensor_name)

        logging.info(f"Found {len(tensor_names)} tensor files in S3 for TP{TENSOR_PARALLEL_SIZE}/shard{TENSOR_PARALLEL_RANK}")

    except ClientError as e:
        logging.error(f"Error listing tensors from S3: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error listing tensors from S3: {e}")
        raise

    return tensor_names

def filter_required_tensors(all_tensor_names: List[str], tie_word_embeddings: bool) -> List[str]:
    """Filter tensor names to only include those that need to be loaded"""
    required_tensors = []
    
    for tensor_name in all_tensor_names:
        if should_load_tensor(tensor_name, tie_word_embeddings):
            required_tensors.append(tensor_name)
    
    logging.info(f"Filtered to {len(required_tensors)} required tensors out of {len(all_tensor_names)} total")
    return required_tensors

def log_required_tensors_summary(required_tensor_names: List[str]):
    """Log required tensors in a formatted, readable way"""
    logging.info("=" * 80)
    logging.info(f"[Required Tensors Summary] Total: {len(required_tensor_names)} tensors")
    logging.info("=" * 80)
    
    # Group tensors by type
    embed_tensors = [t for t in required_tensor_names if "embed_tokens" in t]
    norm_tensors = [t for t in required_tensor_names if "norm" in t and "layer" not in t]
    lm_head_tensors = [t for t in required_tensor_names if "lm_head" in t]
    layer_tensors = [t for t in required_tensor_names if "model.layers." in t]
    
    # Group layer tensors by layer number
    layer_dict = {}
    for tensor in layer_tensors:
        if "model.layers." in tensor:
            layer_num = int(tensor.split('.')[2])
            if layer_num not in layer_dict:
                layer_dict[layer_num] = []
            layer_dict[layer_num].append(tensor)
    
    # Log embedding tensors
    if embed_tensors:
        logging.info(f"[Embedding Tensors] ({len(embed_tensors)} tensors)")
        for tensor in sorted(embed_tensors):
            logging.info(f"  - {tensor}")
    
    # Log layer tensors
    if layer_dict:
        logging.info(f"[Layer Tensors] ({len(layer_tensors)} tensors across {len(layer_dict)} layers)")
        for layer_num in sorted(layer_dict.keys()):
            logging.info(f"  Layer {layer_num}: {len(layer_dict[layer_num])} tensors")
            for tensor in sorted(layer_dict[layer_num]):
                tensor_type = tensor.split('.')[-2]
                logging.info(f"    - {tensor_type}")
    
    # Log normalization tensors
    if norm_tensors:
        logging.info(f"[Normalization Tensors] ({len(norm_tensors)} tensors)")
        for tensor in sorted(norm_tensors):
            logging.info(f"  - {tensor}")
    
    # Log LM head tensors
    if lm_head_tensors:
        logging.info(f"[LM Head Tensors] ({len(lm_head_tensors)} tensors)")
        for tensor in sorted(lm_head_tensors):
            logging.info(f"  - {tensor}")
    
    # Log other tensors
    other_tensors = [t for t in required_tensor_names 
                     if t not in embed_tensors + norm_tensors + lm_head_tensors + layer_tensors]
    if other_tensors:
        logging.info(f"[Other Tensors] ({len(other_tensors)} tensors)")
        for tensor in sorted(other_tensors):
            logging.info(f"  - {tensor}")
    
    logging.info("=" * 80)

def load_tensor_from_s3_direct(s3_client: boto3.client, bucket_name: str, s3_key: str) -> torch.Tensor:
    """Load a tensor directly from S3 to CPU memory without saving to disk"""
    try:
        # Get object from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        
        # Read object body into memory
        tensor_bytes = response['Body'].read()
        content_length = response['ContentLength']
        
        logging.info(f"Loading {s3_key} from S3 ({content_length:,} bytes) directly to memory")
        
        # Load tensor directly from bytes
        buffer = io.BytesIO(tensor_bytes)
        tensor = torch.load(buffer, map_location="cpu")
        
        # Clean up
        buffer.close()
        del tensor_bytes
        
        return tensor
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ['NoSuchKey', '404']:
            logging.error(f"Tensor file not found in S3: {s3_key}")
            raise
        else:
            logging.error(f"Failed to load {s3_key} from S3: {e}")
            raise
    except Exception as e:
        logging.error(f"Unexpected error loading {s3_key}: {e}")
        raise

def load_tensor_to_gpu(tensor_name: str, partitioned_tensor: torch.Tensor):
    """Load a pre-partitioned tensor to GPU (no processing needed)"""
    global DTYPE

    try:
        # Set global DTYPE from first tensor if not set yet
        if DTYPE is None:
            DTYPE = partitioned_tensor.dtype
            logging.info(f"Set global DTYPE to {DTYPE} from tensor")

        # Move tensor to GPU (already partitioned for this TP rank)
        tensor = partitioned_tensor.to(device=DEVICE)

        assert tensor is not None
        assert tensor.device.type == "cuda"

        # Thread-safe add to TENSOR_DICT
        with TENSOR_DICT_LOCK:
            TENSOR_DICT[tensor_name] = tensor

        logging.info(f"Loaded {tensor_name} to GPU / shape: {tensor.shape} / dtype: {tensor.dtype} / device: {tensor.device}")

        # Clean up CPU tensor
        del partitioned_tensor
        del tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return True

    except Exception as e:
        logging.error(f"Error loading tensor {tensor_name} to GPU: {e}")
        raise e

def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int) -> tuple:
    """Get KV cache shape for flash attention backend"""
    # Shape: (2, num_blocks, block_size, num_kv_heads, head_size)
    # 2 is for key and value caches
    return (2, num_blocks, block_size, num_kv_heads, head_size)

def get_cache_block_size_bytes(config_dict: dict) -> int:
    """Calculate the size of a cache block in bytes"""
    global BLOCK_SIZE, CACHE_DTYPE
    
    # Get model parameters from config
    num_attention_layers = END_LAYER_ID - START_LAYER_ID
    num_kv_heads = config_dict.get("num_key_value_heads", config_dict["num_attention_heads"])
    head_size = config_dict["hidden_size"] // config_dict["num_attention_heads"]
    
    # Adjust num_kv_heads for tensor parallelism
    if TENSOR_PARALLEL_SIZE > 1:
        num_kv_heads = num_kv_heads // TENSOR_PARALLEL_SIZE
    
    # Calculate entries per block
    key_cache_entry = num_kv_heads * head_size
    value_cache_entry = key_cache_entry  # Same size for key and value
    
    # Total size for all layers
    total = num_attention_layers * BLOCK_SIZE * (key_cache_entry + value_cache_entry)
    
    # Get dtype size
    dtype_size = CACHE_DTYPE.itemsize if CACHE_DTYPE else 2  # Default to 2 bytes (float16)
    
    return dtype_size * total

def get_model_forward_memory(config_dict: dict) -> int:
    """Get the memory required for model forward pass"""
    num_attention_layers = END_LAYER_ID - START_LAYER_ID
    intermediate_size = config_dict["intermediate_size"]

    # Intermediate Tensor Size 가 가장 큰 것은 FFN 에서이다.
    # FFN Network 를 진행할 때 가장 큰 부분은 up&gate projection 이다.
    global CACHE_DTYPE, MAX_MODEL_LEN
    dtype_size = CACHE_DTYPE.itemsize if CACHE_DTYPE else 2  # Default to 2 bytes (float16)
    max_model_len = MAX_MODEL_LEN
    intermediate_total_dim = max_model_len * intermediate_size * 2 # 2 는 up&gate projection 이기 때문
    intermediate_total_size = intermediate_total_dim * dtype_size
    return intermediate_total_size

def determine_num_available_blocks(config_dict: dict) -> tuple[int, int]:
    """Profile memory and determine number of available KV cache blocks"""
    global CACHE_DTYPE, DTYPE
    
    # Set cache dtype if auto
    if CACHE_DTYPE is None:
        CACHE_DTYPE = DTYPE
    
    # Profile memory after model loading
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_memory_after, total_memory = torch.cuda.mem_get_info(LOCAL_RANK)
    using_memory = total_memory - free_memory_after
    model_forward_memory = get_model_forward_memory(config_dict)
    
    # Calculate available memory for KV cache
    available_kv_cache_memory = int(total_memory * GPU_MEMORY_UTILIZATION - using_memory - model_forward_memory)

    assert available_kv_cache_memory > 0, f"Insufficient memory to allocate space for KV cache." \
        f" Total memory: {total_memory / (1024**3):.2f} GiB, " \
        f"Free memory after model loading: {free_memory_after / (1024**3):.2f} GiB, " \
        f"Memory used by the model: {using_memory / (1024**3):.2f} GiB, " \
        f"Memory reserved for forward pass: {model_forward_memory / (1024**3):.2f} GiB, " \
        f"Available KV cache memory: {available_kv_cache_memory / (1024**3):.2f} GiB"
    
    # Calculate cache block size
    cache_block_size = get_cache_block_size_bytes(config_dict)
    
    # Calculate number of blocks
    num_gpu_blocks = max(0, available_kv_cache_memory // cache_block_size)
    num_cpu_blocks = max(0, SWAP_SPACE_BYTES // cache_block_size)

    assert num_gpu_blocks > 0, f"!!!num_gpu_blocks < 0!!!"
    
    logging.info(f"[Memory Profiling Complete]")
    logging.info(f"  - Total GPU memory: {total_memory / (1024**3):.2f} GiB")
    logging.info(f"  - Memory used by model weights: {using_memory / (1024**3):.2f} GiB")
    logging.info(f"  - Memory reserved for forward pass: {model_forward_memory / (1024**3):.2f} GiB")
    logging.info(f"  - Memory available for KV cache: {available_kv_cache_memory / (1024**3):.2f} GiB")
    logging.info(f"[KV Cache Block Calculation]")
    logging.info(f"  - Cache block size: {cache_block_size} bytes ({cache_block_size / 1024:.2f} KB)")
    logging.info(f"  - Number of GPU blocks: {num_gpu_blocks}")
    logging.info(f"  - Number of CPU blocks: {num_cpu_blocks}")
    
    return num_gpu_blocks, num_cpu_blocks

def allocate_kv_cache(config_dict: dict, num_gpu_blocks: int) -> dict:
    """Allocate KV cache tensors for all virtual engines"""
    # Get model parameters
    num_attention_layers = END_LAYER_ID - START_LAYER_ID
    num_kv_heads = config_dict.get("num_key_value_heads", config_dict["num_attention_heads"])
    head_size = config_dict["hidden_size"] // config_dict["num_attention_heads"]
    
    # Adjust for tensor parallelism
    if TENSOR_PARALLEL_SIZE > 1:
        num_kv_heads = num_kv_heads // TENSOR_PARALLEL_SIZE

    # virtual engine 에 block 들이 나누어 들어간다.
    num_gpu_blocks = num_gpu_blocks // PIPELINE_PARALLEL_SIZE
    
    # Get KV cache shape
    kv_cache_shape = get_kv_cache_shape(num_gpu_blocks, BLOCK_SIZE, num_kv_heads, head_size)
    
    # Allocate KV cache for each virtual engine
    for ve in range(PIPELINE_PARALLEL_SIZE):
        # Allocate GPU cache for each layer
        for layer_idx in range(num_attention_layers):
            # Create KV cache tensor for this layer
            layer_kv_cache = torch.zeros(kv_cache_shape, dtype=CACHE_DTYPE, device=DEVICE)
            
            # Store in TENSOR_DICT with appropriate key
            cache_key = f"kv_cache.ve_{ve}.layer_{START_LAYER_ID + layer_idx}"
            with TENSOR_DICT_LOCK:
                TENSOR_DICT[cache_key] = layer_kv_cache
        
        # Store metadata
        metadata_key = f"kv_cache_metadata.ve_{ve}"
        metadata = {
            "num_gpu_blocks": num_gpu_blocks,
            "block_size": BLOCK_SIZE,
            "num_layers": num_attention_layers,
            "start_layer_id": START_LAYER_ID,
            "end_layer_id": END_LAYER_ID,
            "shape": kv_cache_shape
        }
        
        with TENSOR_DICT_LOCK:
            TENSOR_DICT[metadata_key] = metadata
    
    cache_block_size = get_cache_block_size_bytes(config_dict)
    return {
        "num_gpu_blocks": num_gpu_blocks,
        "total_gpu_cache_size_gb": (num_gpu_blocks * cache_block_size) / (1024**3),
    }

def load_and_process_tensor(s3_client, bucket_name: str, base_s3_path: str, tensor_name: str):
    """Load a pre-partitioned tensor from S3 and load it directly to GPU"""
    # Construct S3 key with TP-specific path: TP{tp_size}/shard{tp_rank}/{tensor_name}.bin
    if base_s3_path:
        s3_key = f"{base_s3_path}/TP{TENSOR_PARALLEL_SIZE}/shard{TENSOR_PARALLEL_RANK}/{tensor_name}.bin"
    else:
        s3_key = f"TP{TENSOR_PARALLEL_SIZE}/shard{TENSOR_PARALLEL_RANK}/{tensor_name}.bin"

    # Load pre-partitioned tensor directly from S3 to CPU memory
    partitioned_tensor = load_tensor_from_s3_direct(s3_client, bucket_name, s3_key)

    # Load tensor to GPU (no processing needed - already partitioned)
    load_tensor_to_gpu(tensor_name, partitioned_tensor)

def load_required_tensors(s3_client, bucket_name: str, base_s3_path: str, required_tensor_names: List[str]):
    """Load required tensors directly from S3"""
    if not required_tensor_names:
        logging.warning("No tensors to load")
        return

    logging.info(f"Loading {len(required_tensor_names)} pre-partitioned tensors directly from S3")

    # Use threading for parallel loading
    max_tensor_workers = min(NUM_TENSOR_WORKERS, len(required_tensor_names))
    logging.info(f"Using {max_tensor_workers} workers for tensor loading")

    with ThreadPoolExecutor(max_workers=max_tensor_workers) as executor:
        futures = []

        for tensor_name in required_tensor_names:
            # Submit loading task
            future = executor.submit(
                load_and_process_tensor,
                s3_client, bucket_name, base_s3_path, tensor_name
            )
            futures.append(future)

        # Wait for all tasks to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in tensor loading: {e}")

# Function removed - tensors are now processed directly after download

def fuse_tensor(layer_idx: int):
    """Fuse tensors (same logic as mt_tensor_store_server.py)"""
    # QKV fusion
    q_proj_tensor_name = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
    k_proj_tensor_name = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
    v_proj_tensor_name = f"model.layers.{layer_idx}.self_attn.v_proj.weight"

    if all(name in TENSOR_DICT for name in [q_proj_tensor_name, k_proj_tensor_name, v_proj_tensor_name]):
        q_proj_tensor = TENSOR_DICT[q_proj_tensor_name]
        k_proj_tensor = TENSOR_DICT[k_proj_tensor_name]
        v_proj_tensor = TENSOR_DICT[v_proj_tensor_name]

        qkv_proj_tensor = torch.cat((q_proj_tensor, k_proj_tensor, v_proj_tensor), dim=INPUT_DIM)
        qkv_proj_tensor_name = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"

        TENSOR_DICT[qkv_proj_tensor_name] = qkv_proj_tensor

        del TENSOR_DICT[q_proj_tensor_name]
        del TENSOR_DICT[k_proj_tensor_name]
        del TENSOR_DICT[v_proj_tensor_name]

    # Gate-Up fusion
    gate_proj_tensor_name = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
    up_proj_tensor_name = f"model.layers.{layer_idx}.mlp.up_proj.weight"

    if all(name in TENSOR_DICT for name in [gate_proj_tensor_name, up_proj_tensor_name]):
        gate_proj_tensor = TENSOR_DICT[gate_proj_tensor_name]
        up_proj_tensor = TENSOR_DICT[up_proj_tensor_name]

        gate_up_proj_tensor = torch.cat((gate_proj_tensor, up_proj_tensor), dim=INPUT_DIM)
        gate_up_proj_tensor_name = f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"

        TENSOR_DICT[gate_up_proj_tensor_name] = gate_up_proj_tensor

        del TENSOR_DICT[gate_proj_tensor_name]
        del TENSOR_DICT[up_proj_tensor_name]

def _status_server(host: str, port: int):
    """TCP server that handles status checks and shutdown commands using protocol enum."""
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(1.0)  # 1 second timeout for accept()
        srv.bind((host, port))
        srv.listen()
        logging.info(f"Status server is listening on {host}:{port}")
        
        while not SHUTDOWN_EVENT.is_set():
            try:
                conn, addr = srv.accept()
                with conn:
                    conn.settimeout(5.0)  # 5 second timeout for recv
                    try:
                        # Receive single byte command
                        data = conn.recv(1)
                        
                        if not data:
                            # Empty connection - legacy status check
                            msg = TensorStoreResponse.READY.value if TENSOR_INITIALIZE_COMPLETE else TensorStoreResponse.NOT_READY.value
                            conn.sendall(msg)
                        elif data == TensorStoreRequest.STATUS_CHECK.value:
                            # Explicit status check
                            msg = TensorStoreResponse.READY.value if TENSOR_INITIALIZE_COMPLETE else TensorStoreResponse.NOT_READY.value
                            conn.sendall(msg)
                        elif data == TensorStoreRequest.SHUTDOWN.value:
                            # Shutdown command
                            conn.sendall(TensorStoreResponse.OK.value)
                            logging.info("Received shutdown command via status server")
                            SHUTDOWN_EVENT.set()  # Signal the main thread AFTER sending response
                            break  # Exit the loop immediately
                        else:
                            # Unknown command
                            conn.sendall(TensorStoreResponse.ERROR.value)
                    except socket.timeout:
                        pass
                    except Exception as e:
                        logging.debug(f"Error handling connection: {e}")
            except socket.timeout:
                # This is expected due to the timeout we set
                continue
            except Exception as e:
                if not SHUTDOWN_EVENT.is_set():
                    logging.error(f"Error in status server: {e}")
                    
        logging.info("Status server shutting down")

def parse_args():
    parser = argparse.ArgumentParser(description="S3 Tensor Store Server")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--s3-path", type=str, required=True,
                        help="S3 path where pre-partitioned tensors are stored (e.g., s3://bucket-name/path/to/models)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--start-layer-id", type=int, default=0)
    parser.add_argument("--end-layer-id", type=int, default=-1)
    parser.add_argument("--status-host", type=str, default=STATUS_HOST,
                        help="Host interface for readiness TCP server")
    parser.add_argument("--status-port", type=int, required=True,
                        help="Port for readiness TCP server")
    parser.add_argument("--aws-profile", type=str, default=None,
                        help="AWS profile to use for S3 access")
    
    # KV cache related arguments
    parser.add_argument("--block-size", type=int, default=16, choices=[8, 16, 32],
                        help="Token block size for KV cache (default: 16)")
    parser.add_argument("--gpu-num-blocks", type=int, default=None,
                        help="Number of GPU blocks for KV cache (default: None, calculated based on memory)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="Fraction of GPU memory to use for KV cache (default: 0.9)")
    parser.add_argument("--swap-space", type=float, default=4.0,
                        help="Size of CPU swap space per GPU in GiB (default: 4.0)")
    parser.add_argument("--cache-dtype", type=str, default="auto",
                        help="Data type for KV cache storage (default: auto, uses model dtype)")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                        help="Pipeline parallel size for virtual engine allocation")
    parser.add_argument("--pipeline-parallel-rank", type=int, default=0,
                        help="Pipeline parallel rank (default: 0)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Maximum model sequence length (default: from model config)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel workers for tensor loading (default: 8)")
    
    return parser.parse_args()

def set_global_variables(args: argparse.Namespace, config_dict: dict):
    global TENSOR_PARALLEL_SIZE, LOCAL_RANK, START_LAYER_ID, END_LAYER_ID, TOTAL_LAYER_NUM, DEVICE, TENSOR_PARALLEL_RANK, NUM_TENSOR_WORKERS
    global BLOCK_SIZE, GPU_MEMORY_UTILIZATION, SWAP_SPACE_BYTES, PIPELINE_PARALLEL_SIZE, PIPELINE_PARALLEL_RANK, MAX_MODEL_LEN
    
    TENSOR_PARALLEL_SIZE = args.tensor_parallel_size
    LOCAL_RANK = args.local_rank
    TENSOR_PARALLEL_RANK = args.tensor_parallel_rank
    DEVICE = f"cuda:{LOCAL_RANK}"
    START_LAYER_ID = args.start_layer_id
    END_LAYER_ID = args.end_layer_id
    TOTAL_LAYER_NUM = config_dict["num_hidden_layers"]
    NUM_TENSOR_WORKERS = args.num_workers
    
    if END_LAYER_ID == -1:
        END_LAYER_ID = config_dict["num_hidden_layers"]
    
    # Set KV cache related globals
    BLOCK_SIZE = args.block_size
    GPU_MEMORY_UTILIZATION = args.gpu_memory_utilization
    SWAP_SPACE_BYTES = int(args.swap_space * 1024 * 1024 * 1024)  # Convert GiB to bytes
    PIPELINE_PARALLEL_SIZE = args.pipeline_parallel_size
    PIPELINE_PARALLEL_RANK = args.pipeline_parallel_rank
    
    # Set max model length - use from args if provided, otherwise from config
    if args.max_model_len is not None:
        MAX_MODEL_LEN = args.max_model_len
    else:
        MAX_MODEL_LEN = config_dict.get("max_position_embeddings", 4096)  # Default to 4096 if not in config
    
    # Log all global variables in a formatted way
    logging.info("=" * 80)
    logging.info("[Global Variables Initialization Summary]")
    logging.info("=" * 80)
    
    logging.info("[Model Configuration]")
    logging.info(f"  - Model Name: {args.model_name}")
    logging.info(f"  - Total Layers: {TOTAL_LAYER_NUM}")
    logging.info(f"  - Max Model Length: {MAX_MODEL_LEN}")
    
    logging.info("[Layer Assignment]")
    logging.info(f"  - Start Layer ID: {START_LAYER_ID}")
    logging.info(f"  - End Layer ID: {END_LAYER_ID}")
    logging.info(f"  - Assigned Layers: {END_LAYER_ID - START_LAYER_ID} layers [{START_LAYER_ID}, {END_LAYER_ID})")
    
    logging.info("[Parallelism Configuration]")
    logging.info(f"  - Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
    logging.info(f"  - Tensor Parallel Rank: {TENSOR_PARALLEL_RANK}")
    logging.info(f"  - Pipeline Parallel Size: {PIPELINE_PARALLEL_SIZE}")
    logging.info(f"  - Pipeline Parallel Rank: {PIPELINE_PARALLEL_RANK}")
    
    logging.info("[Device Configuration]")
    logging.info(f"  - Local Rank: {LOCAL_RANK}")
    logging.info(f"  - CUDA Device: {DEVICE}")
    
    logging.info("[KV Cache Configuration]")
    logging.info(f"  - Block Size: {BLOCK_SIZE}")
    logging.info(f"  - GPU Memory Utilization: {GPU_MEMORY_UTILIZATION * 100:.1f}%")
    logging.info(f"  - Swap Space: {args.swap_space:.2f} GiB ({SWAP_SPACE_BYTES:,} bytes)")
    logging.info(f"  - Cache Dtype: {args.cache_dtype}")
    
    logging.info("=" * 80)

def main():
    args = parse_args()

    # Set CUDA device for this process BEFORE any CUDA operations
    # This prevents all processes from defaulting to cuda:0
    torch.cuda.set_device(args.local_rank)
    logging.info(f"Set CUDA device to cuda:{args.local_rank} for this process")

    # Start status server
    threading.Thread(target=_status_server,
                     args=(args.status_host, args.status_port),
                     daemon=True).start()

    logging.info(f"args: {args}")
    
    # Parse S3 path
    if not args.s3_path.startswith("s3://"):
        raise ValueError("S3 path must start with s3://")
    
    # Remove s3:// prefix and split into parts
    s3_path_without_prefix = args.s3_path[5:]
    s3_parts = s3_path_without_prefix.split('/', 1)
    
    bucket_name = s3_parts[0]
    if len(s3_parts) > 1:
        # There's a path after bucket name
        base_s3_path = s3_parts[1].rstrip('/')
    else:
        # Only bucket name provided
        base_s3_path = ""
    
    logging.info(f"S3 bucket: {bucket_name}")
    logging.info(f"S3 base path: '{base_s3_path}'")
    
    # Initialize S3 client with connection pool configuration
    try:
        # Configure boto3 with much larger connection pool
        config = Config(
            max_pool_connections=256,  # Large pool to handle many concurrent downloads
            retries={'max_attempts': 3}
        )
        
        if args.aws_profile:
            session = boto3.Session(profile_name=args.aws_profile)
            s3_client = session.client('s3', config=config)
        else:
            s3_client = boto3.client('s3', config=config)
        
        # Test S3 access
        s3_client.head_bucket(Bucket=bucket_name)
        logging.info(f"Successfully connected to S3 bucket: {bucket_name}")
        
    except NoCredentialsError:
        logging.error("AWS credentials not found. Please configure your credentials.")
        return
    except ClientError as e:
        logging.error(f"Error accessing S3 bucket: {e}")
        return
    except Exception as e:
        logging.error(f"Error initializing S3 client: {e}")
        return
    
    try:
        mp.set_start_method('spawn', force=True)
        logging.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        logging.warning("Could not set start method to 'spawn'. Using default.")
    
    # Step 1: Load config.json from S3 to get model configuration
    # Config is stored at the model root directory (not in TP-specific folders)
    logging.info("Step 2: Loading config.json from S3...")

    try:
        if base_s3_path:
            config_s3_key = f"{base_s3_path}/config.json"
        else:
            config_s3_key = "config.json"

        logging.info(f"Loading config from S3 key: {config_s3_key}")

        # Load config directly from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=config_s3_key)
        config_content = response['Body'].read()
        config_dict = json.loads(config_content)

        tie_word_embeddings = config_dict["tie_word_embeddings"]
        logging.info(f"Loaded model config from S3")
        logging.debug(f"Config: {config_dict}")
    except ClientError as e:
        logging.error(f"Failed to load config.json from S3: {e}")
        return
    except Exception as e:
        logging.error(f"Failed to parse config: {e}")
        return
    
    set_global_variables(args, config_dict)

    logging.info("Loading strategy: Direct S3 -> CPU -> GPU transfer (pre-partitioned tensors)")
    logging.info(f"Loading pre-partitioned tensors for TP{TENSOR_PARALLEL_SIZE}/shard{TENSOR_PARALLEL_RANK}")

    # Step 2: List all tensor files from S3 for this TP configuration
    logging.info("Step 1: Listing pre-partitioned tensor files from S3...")
    all_tensor_names = list_tensor_files_from_s3_with_boto3(s3_client, bucket_name, base_s3_path)
    
    if not all_tensor_names:
        logging.error("No tensor files found in S3")
        return
    
    # Step 3: Filter to only required tensors
    logging.info("Step 3: Filtering to required tensors...")
    required_tensor_names = filter_required_tensors(all_tensor_names, tie_word_embeddings)
    
    if not required_tensor_names:
        logging.error("No required tensors found after filtering")
        return
    
    # Log the required tensors in a formatted way
    log_required_tensors_summary(required_tensor_names)

    # Step 4: Load pre-partitioned tensors directly from S3
    logging.info("Step 4: Loading pre-partitioned tensors from S3...")
    load_start = time.perf_counter()

    try:
        load_required_tensors(s3_client, bucket_name, base_s3_path, required_tensor_names)
        
        load_end = time.perf_counter()
        logging.info(f"Loading and processing completed in {load_end - load_start:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Model Loading Error: {e}")
        raise e
    
    logging.info(f"Final DTYPE determined from tensors: {DTYPE}")
    
    # Fuse tensors
    logging.info(f"[Tensor Fusion Phase] Starting fusion for layers {START_LAYER_ID} to {END_LAYER_ID}")
    
    for layer_idx in range(START_LAYER_ID, END_LAYER_ID):
        fuse_tensor(layer_idx)
    
    logging.info(f"[Tensor Fusion Complete] Fused tensors for layers {START_LAYER_ID} to {END_LAYER_ID}")
    logging.info("[Memory Cleanup] Emptying CUDA cache after fusion...")
    torch.cuda.empty_cache()
    logging.debug(f"After emptying cache, memory usage: {torch.cuda.memory_summary(device=DEVICE)}")
    
    # Print final tensor list
    for tensor_name in TENSOR_DICT:
        logging.info(f"tensor_name: {tensor_name} / shape: {TENSOR_DICT[tensor_name].shape} / dtype: {TENSOR_DICT[tensor_name].dtype} / device: {TENSOR_DICT[tensor_name].device}")
    
    load_end = time.perf_counter()
    logging.info(f"[Model Loading Complete] Total loading time: {load_end - load_start:.2f} seconds")
    
    if not TENSOR_DICT:
        raise ValueError("Tensor loading failed: TENSOR_DICT is empty")
    
    logging.info(f"Successfully loaded {len(TENSOR_DICT)} tensors")
    
    # Allocate KV cache after model loading
    logging.info("[KV Cache Allocation Phase Started]")
    num_gpu_blocks, num_cpu_blocks = determine_num_available_blocks(config_dict)
    if args.gpu_num_blocks is not None:
        logging.info(f"[GPU Blocks Override] Overriding calculated num_gpu_blocks ({num_gpu_blocks}) with provided value: {args.gpu_num_blocks}")
        num_gpu_blocks = args.gpu_num_blocks
        logging.info(f"[GPU Blocks Override] Using {num_gpu_blocks} GPU blocks as specified")
    kv_cache_info = allocate_kv_cache(config_dict, num_gpu_blocks)
    logging.info("[KV Cache Allocation Complete]")
    logging.info(f"  - Total GPU cache size: {kv_cache_info['total_gpu_cache_size_gb']:.2f} GiB")
    
    # Start TensorManager server
    manager_port = MANAGER_PORT + LOCAL_RANK
    manager = TensorManager(address=(MANAGER_HOST, manager_port), authkey=MANAGER_AUTHKEY)
    
    try:
        logging.info(f"TensorManager server starting {MANAGER_HOST}:{manager_port}...")
        server = manager.get_server()
        logging.info("Manager server running. Waiting for client connections...")
        logging.info("Press Ctrl+C to stop.")
        
        global TENSOR_INITIALIZE_COMPLETE
        TENSOR_INITIALIZE_COMPLETE = True
        logging.info("[All Initialization Complete] Model weights loaded and KV cache allocated. Status server returning 'READY'.")
        
        # Start serve_forever in a separate thread so we can check shutdown flag
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        # Main thread waits for shutdown signal (efficient - no CPU usage while waiting)
        SHUTDOWN_EVENT.wait()
            
        logging.info("Shutdown signal received, stopping manager server...")
        
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