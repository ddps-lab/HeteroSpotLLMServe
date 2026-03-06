# SPDX-License-Identifier: Apache-2.0
"""CacheEngine class for managing the KV cache."""
from typing import List
import time
import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)

logger = init_logger(__name__)

# Used to retrieve tensor information from the tensor server store
from multiprocessing.managers import BaseManager, DictProxy

class TensorManager(BaseManager):
    pass

TENSOR_DICT = {}

TENSOR_SERVER_HOST = '127.0.0.1'
TENSOR_SERVER_PORT = 50001
TENSOR_SERVER_AUTHKEY = b'param_store'

MANAGER_INSTANCE = None

TensorManager.register('get_tensor_dict', proxytype=DictProxy)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        ve: int,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            # Pipeline parallelism requires overlapping multiple requests.
            # vLLM refers to this as a virtual engine.
            # The number of blocks per virtual engine is computed as follows.
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free,
                                             use_mla=model_config.use_mla)

        # Initialize the cache.
        # self.gpu_cache = self._allocate_kv_cache(
        #     self.num_gpu_blocks, self.device_config.device_type)
        self.gpu_cache = self.get_gpu_cache_from_manager(ve)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def get_gpu_cache_from_manager(self, ve: int):
        if self.parallel_config.local_rank == -1:
            raise ValueError("local_rank is not set")
        tensor_server_port = TENSOR_SERVER_PORT + self.parallel_config.local_rank
        MANAGER_INSTANCE = TensorManager(address=(TENSOR_SERVER_HOST, tensor_server_port), authkey=TENSOR_SERVER_AUTHKEY)
        if MANAGER_INSTANCE is None:
            raise ValueError("Failed to create TensorManager instance")
        max_retries = 3 # max 9s (3s * 3 retries)
        wait_time = 3
        for attempt in range(max_retries):
            try:
                MANAGER_INSTANCE.connect()
                logger.info("Connected to TensorManager server for CacheEngine.")
                break
            except ConnectionRefusedError:
                logger.info(f"Connection refused (Attempt {attempt + 1}/{max_retries}). Server might not be ready. Retrying in {wait_time}s...")
                if attempt == max_retries - 1:
                    raise ValueError("Max connection attempts reached. Exiting.")
                logger.info(f"Sleep {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Error connecting to manager")
                raise e
        logger.info("TensorManager server connected successfully.")

        try:
            logger.info("Accessing Tensor Dict via Manager for CacheEngine.")
            TENSOR_DICT = MANAGER_INSTANCE.get_tensor_dict()
            if not TENSOR_DICT:
                raise ValueError("Tensor Dictionary is empty")
        except Exception as e:
            logger.error(f"Error accessing Tensor Dict via Manager for CacheEngine: {e}")
            raise e

        kv_cache: List[torch.Tensor] = []
        try:
            start_layer_idx = self.parallel_config.start_layer_idx
            for layer_idx in range(self.num_attention_layers):
                cache_key = f"kv_cache.ve_{ve}.layer_{start_layer_idx + layer_idx}"
                layer_kv_cache = TENSOR_DICT[cache_key]
                kv_cache.append(layer_kv_cache)
        except Exception as e:
            logger.error(f"Error accessing Tensor Dict via Manager for CacheEngine: {e}")
            raise e

        return kv_cache


    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []

        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            layer_kv_cache = torch.zeros(kv_cache_shape,
                                         dtype=self.dtype,
                                         pin_memory=pin_memory,
                                         device=device)

            # view back to (TOTAL_PAGES, PAGE_SIZE, entry_shape...) for cases
            # when entry_shape is higher than 1D
            kv_cache.append(layer_kv_cache)
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        key_cache_entry = num_heads * head_size

        # For MLA there is no value cache, since the latent vector
        # is joint keys and values.
        value_cache_entry = key_cache_entry if not model_config.use_mla else 0
        total = num_attention_layers * cache_config.block_size * \
            (key_cache_entry + value_cache_entry)

        dtype_size = get_dtype_size(dtype)
        return dtype_size * total
