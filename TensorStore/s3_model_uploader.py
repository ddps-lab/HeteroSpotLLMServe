# S3 Model Uploader - Downloads from HuggingFace and uploads to S3

import argparse
import os
import json
import glob
import time
import logging
from typing import List, Optional, Union
from pathlib import Path
import tempfile
import hashlib
import filelock
import fnmatch
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import gc

import torch
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from safetensors import safe_open
import huggingface_hub.constants
from huggingface_hub import HfFileSystem, snapshot_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] [Uploader] - %(message)s')

# Global variables for thread safety
UPLOAD_LOCK = threading.Lock()
UPLOADED_COUNT = 0
TOTAL_COUNT = 0

### Utility functions from mt_tensor_store_server.py
class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)

temp_dir = tempfile.gettempdir()

def get_lock(model_name_or_path: Union[str, Path], cache_dir: Optional[str] = None):
    lock_dir = cache_dir or temp_dir
    model_name_or_path = str(model_name_or_path)
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    lock_file_name = hash_name + model_name + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
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
        fs = HfFileSystem()
        file_list = fs.ls(model_name_or_path, detail=False, revision=revision)
        
        for pattern in allow_patterns:
            matching = fnmatch.filter(file_list, pattern)
            if len(matching) > 0:
                allow_patterns = [pattern]
                break
    
    logging.info("Using model weights format %s", allow_patterns)
    allow_patterns.append("config.json")
    
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
            logging.info("Time spent downloading weights for %s: %.6f seconds", model_name_or_path, time_taken)
    return hf_folder


def save_tensor_to_file(tensor_name: str, tensor_data: torch.Tensor, output_dir: str, target_dtype: Optional[torch.dtype] = None) -> str:
    """Save a single tensor to a binary file"""
    # Create directory structure if needed
    file_path = os.path.join(output_dir, f"{tensor_name}.bin")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert dtype if specified, otherwise use original
    if target_dtype is not None:
        tensor_data = tensor_data.to(dtype=target_dtype)
    
    # Save tensor to file
    torch.save(tensor_data, file_path)
    
    # Explicitly free memory
    del tensor_data
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    return file_path


def upload_file_to_s3(file_path: str, s3_client, bucket_name: str, s3_key: str):
    """Upload a single file to S3"""
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        logging.info(f"Successfully uploaded {s3_key} to s3://{bucket_name}")
        
        # Delete local file after successful upload
        os.remove(file_path)
        
        global UPLOADED_COUNT
        with UPLOAD_LOCK:
            UPLOADED_COUNT += 1
            logging.info(f"Progress: {UPLOADED_COUNT}/{TOTAL_COUNT} tensors uploaded")
            
    except ClientError as e:
        logging.error(f"Failed to upload {file_path} to S3: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error uploading {file_path}: {e}")
        raise


def process_and_upload_tensor(args, s3_client, bucket_name: str, base_s3_path: str, 
                            tensor_name: str, tensor_slice, output_dir: str, target_dtype: Optional[torch.dtype] = None):
    """Process a single tensor: save to file and upload to S3"""
    try:
        # Get tensor data (preserve original dtype from safetensor)
        tensor_data = tensor_slice[:].cpu()
        
        # Save tensor to local file
        local_file_path = save_tensor_to_file(tensor_name, tensor_data, output_dir, target_dtype)
        
        # Use original tensor name for S3 key
        if base_s3_path:
            s3_key = f"{base_s3_path}/{args.model_name}/{tensor_name}.bin"
        else:
            s3_key = f"{args.model_name}/{tensor_name}.bin"
        
        # Upload to S3
        upload_file_to_s3(local_file_path, s3_client, bucket_name, s3_key)
        
    except Exception as e:
        logging.error(f"Error processing tensor {tensor_name}: {e}")
        raise


def process_safetensor_file(args, s3_client, bucket_name: str, base_s3_path: str, 
                          st_file: str, output_dir: str, executor: ThreadPoolExecutor, target_dtype: Optional[torch.dtype] = None):
    """Process a single safetensor file"""
    logging.info(f"Processing file: {st_file}")
    
    futures = []
    
    with safe_open(st_file, framework="pt", device="cpu") as f:
        tensor_names = list(f.keys())
        
        global TOTAL_COUNT
        TOTAL_COUNT += len(tensor_names)
        
        logging.info(f"Found {len(tensor_names)} tensors in {st_file}")
        
        for tensor_name in tensor_names:
            # Submit tensor processing to thread pool
            future = executor.submit(
                process_and_upload_tensor,
                args, s3_client, bucket_name, base_s3_path,
                tensor_name, f.get_slice(tensor_name), output_dir, target_dtype
            )
            futures.append(future)
    
    # Wait for all tensors from this file to complete
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            logging.error(f"Error in tensor processing: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download model from HuggingFace and upload to S3")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--s3-path", type=str, required=True, 
                        help="S3 path (e.g., s3://bucket-name/path/to/models)")
    parser.add_argument("--num-workers", type=int, default=8, 
                        help="Number of worker threads for parallel processing (default: 8, max recommended: 16)")
    parser.add_argument("--aws-profile", type=str, default=None,
                        help="AWS profile to use for S3 access")
    parser.add_argument("--temp-dir", type=str, default=None,
                        help="Temporary directory for storing tensor files")
    parser.add_argument("--dtype", type=str, default=None,
                        help="Target dtype for tensors (float16, float32, bfloat16). If not specified, uses original dtype")
    return parser.parse_args()

def get_dtype_from_string(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    """Convert string to torch.dtype"""
    if dtype_str is None:
        return None
    
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported types: {list(dtype_map.keys())}")
    
    return dtype_map[dtype_str]


def main():
    args = parse_args()
    
    # Convert dtype string to torch.dtype
    target_dtype = get_dtype_from_string(args.dtype)
    if target_dtype is not None:
        logging.info(f"Target dtype set to: {target_dtype}")
    else:
        logging.info("Using original tensor dtypes")
    
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
    
    # Initialize S3 client with increased connection pool
    try:
        from botocore.config import Config
        
        # Configure boto3 with larger connection pool
        config = Config(
            max_pool_connections=args.num_workers + 5,  # Add buffer for safety
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
    
    # Download model from HuggingFace
    logging.info(f"Downloading model {args.model_name} from HuggingFace...")
    download_start = time.perf_counter()
    
    allow_patterns = ["*.safetensors", "*.bin"]
    hf_folder = download_weights_from_hf(args.model_name, None, allow_patterns)
    
    # Also upload config.json
    config_path = os.path.join(hf_folder, "config.json")
    if os.path.exists(config_path):
        if base_s3_path:
            config_s3_key = f"{base_s3_path}/{args.model_name}/config.json"
        else:
            config_s3_key = f"{args.model_name}/config.json"
        try:
            s3_client.upload_file(config_path, bucket_name, config_s3_key)
            logging.info(f"Uploaded config.json to s3://{bucket_name}/{config_s3_key}")
        except Exception as e:
            logging.error(f"Failed to upload config.json: {e}")
    
    # Find weight files
    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if len(hf_weights_files) > 0:
            break
    
    download_end = time.perf_counter()
    logging.info(f"Model download completed in {download_end - download_start:.2f} seconds")
    logging.info(f"Found {len(hf_weights_files)} weight files")
    
    # Create temporary directory for tensor files
    output_dir = args.temp_dir or tempfile.mkdtemp(prefix="tensor_upload_")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Using temporary directory: {output_dir}")
    
    # Process and upload tensors
    upload_start = time.perf_counter()
    
    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for st_file in hf_weights_files:
                process_safetensor_file(args, s3_client, bucket_name, base_s3_path, 
                                      st_file, output_dir, executor, target_dtype)
        
        upload_end = time.perf_counter()
        logging.info(f"Upload completed in {upload_end - upload_start:.2f} seconds")
        logging.info(f"Successfully uploaded {UPLOADED_COUNT} tensors to S3")
        
        # Clean up temporary directory if it's empty
        if not os.listdir(output_dir):
            os.rmdir(output_dir)
            logging.info("Cleaned up temporary directory")
            
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise
    finally:
        # Clean up any remaining files
        remaining_files = glob.glob(os.path.join(output_dir, "*.bin"))
        if remaining_files:
            logging.warning(f"Found {len(remaining_files)} files not uploaded. Cleaning up...")
            for f in remaining_files:
                os.remove(f)


if __name__ == "__main__":
    main()