# SPDX-License-Identifier: Apache-2.0

# Backported Qwen3 support for vLLM 0.8.1
# Based on llama.py's Tensor Store pattern with Qwen3-specific QK-Norm.
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
"""Inference-only Qwen3 model compatible with HuggingFace weights."""
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union

import torch
from torch import nn
from multiprocessing.managers import BaseManager, DictProxy

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import is_first_stage, is_last_stage
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (PPMissingLayer, extract_layer_index,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

logger = init_logger(__name__)

# Tensor Store connection constants (same as llama.py)
TENSOR_SERVER_HOST = '127.0.0.1'
TENSOR_SERVER_PORT = 50001
TENSOR_SERVER_AUTHKEY = b'param_store'

class TensorManager(BaseManager):
    pass

TENSOR_DICT = {}
MANAGER_INSTANCE = None

TensorManager.register('get_tensor_dict', proxytype=DictProxy)


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj_tensor_name = f"{prefix}.gate_up_proj.weight"
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            weight_tensor=TENSOR_DICT[self.gate_up_proj_tensor_name]
        )
        self.down_proj_tensor_name = f"{prefix}.down_proj.weight"
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            weight_tensor=TENSOR_DICT[self.down_proj_tensor_name]
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class Qwen3Attention(nn.Module):
    """Qwen3 attention with QK-Norm and no QKV bias."""

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int = 128,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 1000000,
                 rms_norm_eps: float = 1e-6,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[Tuple] = None,
                 prefix: str = "",
                 attn_type: str = AttentionType.DECODER) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # Qwen3 uses explicit head_dim from config (not hidden_size // num_heads)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj_tensor_name = f"{prefix}.qkv_proj.weight"
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            weight_tensor=TENSOR_DICT[self.qkv_proj_tensor_name]
        )

        self.o_proj_tensor_name = f"{prefix}.o_proj.weight"
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            weight_tensor=TENSOR_DICT[self.o_proj_tensor_name]
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=attn_type)

        # QK-Norm: Qwen3's key difference from Qwen2/Llama
        self.q_norm_tensor_name = f"{prefix}.q_norm.weight"
        self.k_norm_tensor_name = f"{prefix}.k_norm.weight"
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps,
                              weight_tensor=TENSOR_DICT[self.q_norm_tensor_name])
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps,
                              weight_tensor=TENSOR_DICT[self.k_norm_tensor_name])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Apply QK-Norm per head
        q = q.view(*q.shape[:-1], -1, self.head_dim)
        q = self.q_norm(q)
        q = q.view(*q.shape[:-2], -1)

        k = k.view(*k.shape[:-1], -1, self.head_dim)
        k = self.k_norm(k)
        k = k.view(*k.shape[:-2], -1)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", self.hidden_size // config.num_attention_heads),
            max_position=config.max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm_tensor_name = f"{prefix}.input_layernorm.weight"
        self.post_attention_layernorm_tensor_name = f"{prefix}.post_attention_layernorm.weight"
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=rms_norm_eps,
                                       weight_tensor=TENSOR_DICT[self.input_layernorm_tensor_name])
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=rms_norm_eps,
                                                weight_tensor=TENSOR_DICT[self.post_attention_layernorm_tensor_name])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Qwen3Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        self.embed_tokens_tensor_name = f"{prefix}.embed_tokens.weight"
        self.norm_tensor_name = f"{prefix}.norm.weight"

        if is_first_stage(get_pp_group().rank) or (config.tie_word_embeddings
                                                    and is_last_stage(get_pp_group().rank)):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
                weight_tensor=TENSOR_DICT[self.embed_tokens_tensor_name]
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3DecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        if is_last_stage(get_pp_group().rank):
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
                               weight_tensor=TENSOR_DICT[self.norm_tensor_name])
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # if get_pp_group().is_first_rank:
        if is_first_stage(get_pp_group().rank):
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        # if not get_pp_group().is_last_rank:
        if not is_last_stage(get_pp_group().rank):
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        ### Begin loading tensors from the tensor store server
        global TENSOR_DICT
        global MANAGER_INSTANCE
        if vllm_config.parallel_config.local_rank == -1:
            raise ValueError("local_rank is not set")
        tensor_server_port = TENSOR_SERVER_PORT + vllm_config.parallel_config.local_rank
        MANAGER_INSTANCE = TensorManager(
            address=(TENSOR_SERVER_HOST, tensor_server_port),
            authkey=TENSOR_SERVER_AUTHKEY
        )
        if MANAGER_INSTANCE is None:
            raise ValueError("Failed to create TensorManager instance")
        max_retries = 120
        wait_time = 5
        for attempt in range(max_retries):
            try:
                MANAGER_INSTANCE.connect()
                logger.info("Connected to TensorManager server.")
                break
            except ConnectionRefusedError:
                logger.info(f"Connection refused (Attempt {attempt + 1}/{max_retries}). "
                           f"Server might not be ready. Retrying in {wait_time}s...")
                if attempt == max_retries - 1:
                    raise ValueError("Max connection attempts reached. Exiting.")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Error connecting to manager")
                raise e
        logger.info("TensorManager server connected successfully.")

        try:
            logger.info("Accessing Tensor Dict via Manager")
            TENSOR_DICT = MANAGER_INSTANCE.get_tensor_dict()
            if not TENSOR_DICT:
                raise ValueError("Tensor Dictionary is empty")
        except Exception as e:
            logger.error(f"Error accessing Tensor Dict via Manager: {e}")
            raise e
        ### End loading tensors from the tensor store server

        self.model = Qwen3Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if is_last_stage(get_pp_group().rank):
            if config.tie_word_embeddings:
                # When tying weights, create ParallelLMHead without weight_tensor
                # (it will share weights with embed_tokens via tie_weights)
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)
            else:
                lm_head_tensor_name = "lm_head.weight"
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                    weight_tensor=TENSOR_DICT[lm_head_tensor_name]
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        # Weights are loaded via Tensor Store, not through this method.
        # This is kept for interface compatibility.
        loaded_params: Set[str] = set()
        return loaded_params
