import numpy as np
import torch
from typing import List
from estimator_utils import *
from transformers import AutoConfig
from hardware_specs import GPU_SPEC, INTERCONNECT_SPEC, INSTANCE_SPEC


def get_latency(
    start_layer: int,
    end_layer: int,
    gpu_type: str,
    tp_size: int,
    p2p_bandwidth: float,
    avg_input_len: int,
    avg_output_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    total_num_layers: int,
    intermediate_dim: int,
    vocab_size: int,
    dtype: torch.dtype = torch.float16
) -> float:
    """[start_layer, end_layer) 구간의 latency를 Estimation"""
    num_layers = end_layer - start_layer
    if num_layers > total_num_layers:
        raise ValueError("end_layer must be less than or equal to total_num_layers")

    if num_layers <= 0:
        return 0.0

    batch_size = 1  # alpa serve 는 batch size 1 로 가정
    
    prefill_computation_latency = get_prefill_computation_latency_per_layer(
        gpu_type=gpu_type,
        gpu_count=tp_size,
        input_len=avg_input_len,
        hidden_dim=hidden_dim,
        num_attention_head=num_attention_head,
        num_kv_cache_head=num_kv_cache_head,
        batch_size=batch_size,
        intermediate_dim=intermediate_dim,
        dtype=dtype,
    ) * num_layers

    prefill_tp_communication_latency = get_tp_communication_latency_per_layer(
        tp_size=tp_size,
        batch_size=batch_size,
        sequence_len=avg_input_len,
        hidden_dim=hidden_dim,
        p2p_bandwidth=p2p_bandwidth,
        dtype=dtype
    ) * num_layers

    decoding_computation_latency = get_decoding_computation_latency_per_layer(
        gpu_type=gpu_type,
        gpu_count=tp_size,
        input_len=avg_input_len,
        output_len=avg_output_len,
        hidden_dim=hidden_dim,
        num_attention_head=num_attention_head,
        num_kv_cache_head=num_kv_cache_head,
        batch_size=batch_size,
        intermediate_dim=intermediate_dim,
        dtype=dtype,
    ) * num_layers

    decoding_tp_communication_latency = get_tp_communication_latency_per_layer(
        tp_size=tp_size,
        batch_size=batch_size,
        sequence_len=avg_output_len,
        hidden_dim=hidden_dim,
        p2p_bandwidth=p2p_bandwidth,
        dtype=dtype
    ) * num_layers

    prefill_logit_latency = 0.0
    prefill_logit_tp_communication_latency = 0.0
    decoding_logit_latency = 0.0
    decoding_logit_tp_communication_latency = 0.0

    if end_layer == total_num_layers:
        # 마지막 stage 에는 lm head 가 포함되어 있음
        prefill_logit_latency = get_prefill_compute_logit_latency(
            gpu_type=gpu_type,
            gpu_count=tp_size,
            input_len=avg_input_len,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            vocab_size=vocab_size,
            dtype=dtype
        )
        prefill_logit_tp_communication_latency = get_tp_communication_latency_per_layer(
            tp_size=tp_size,
            batch_size=batch_size,
            sequence_len=avg_input_len,
            hidden_dim=vocab_size,
            p2p_bandwidth=p2p_bandwidth,
            dtype=dtype
        )
        decoding_logit_latency = get_decoding_compute_logit_latency(
            gpu_type=gpu_type,
            gpu_count=tp_size,
            output_len=avg_output_len,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            vocab_size=vocab_size,
            dtype=dtype
        )
        decoding_logit_tp_communication_latency = get_tp_communication_latency_per_layer(
            tp_size=tp_size,
            batch_size=batch_size,
            sequence_len=avg_output_len,
            hidden_dim=vocab_size,
            p2p_bandwidth=p2p_bandwidth,
            dtype=dtype
        )

    total_latency = (
        prefill_computation_latency +
        prefill_tp_communication_latency +
        prefill_logit_latency +
        prefill_logit_tp_communication_latency +
        decoding_computation_latency +
        decoding_tp_communication_latency +
        decoding_logit_latency +
        decoding_logit_tp_communication_latency
    )

    return total_latency


class AlpaServeOptimizer:
    def __init__(self, instance_type: str, num_stage: int, config: dict):
        self.instance_type = instance_type
        self.num_stage = num_stage
        self.config = config
        self.latency_cache: np.ndarray = None
        self.best_split: np.ndarray = None
        self.layers_per_stage: np.ndarray = None

    # 수식
    # F(s, start, end) = min(max(F(s-1, start, i), latency(i, end)))
    # 이 F 는 각 stage 당 가장 높은 latency 값을 의미한다
    # 이 각 stage 당 가장 높은 latency 를 최소화 하는 것을 목표로 한다
    # 이는 pipeline parallelism 적용시 가장 느린 stage 가 전체 latency 를 결정하기 때문이다
    # 이를 uneven partitioning overhead 라고 한다.
    def F(self, num_stage, start_layer, end_layer):
        if num_stage == 0 or start_layer >= end_layer:
            return 0.0

        if self.latency_cache[num_stage][start_layer][end_layer] >= 0:
            return self.latency_cache[num_stage][start_layer][end_layer]

        if num_stage == 1:
            latency = get_latency(
                start_layer=start_layer,
                end_layer=end_layer,
                gpu_type=self.instance_type,
                tp_size=self.config['tp_size'],
                p2p_bandwidth=self.config['p2p_bandwidth'],
                avg_input_len=self.config['expected_input_len'],
                avg_output_len=self.config['expected_output_len'],
                hidden_dim=self.config['hidden_size'],
                num_attention_head=self.config['num_attention_heads'],
                num_kv_cache_head=self.config['num_key_value_heads'],
                total_num_layers=self.config['num_layers'],
                intermediate_dim=self.config['intermediate_size'],
                vocab_size=self.config['vocab_size'],
                dtype=self.config['dtype']
            )
            self.latency_cache[num_stage][start_layer][end_layer] = latency
            return latency

        min_latency = float('inf')
        best_i = -1
        for i in range(start_layer + 1, end_layer + 1):
            prev_latency = self.F(num_stage - 1, start_layer, i)
            curr_latency = get_latency(
                start_layer=i,
                end_layer=end_layer,
                gpu_type=self.instance_type,
                tp_size=self.config['tp_size'],
                p2p_bandwidth=self.config['p2p_bandwidth'],
                avg_input_len=self.config['expected_input_len'],
                avg_output_len=self.config['expected_output_len'],
                hidden_dim=self.config['hidden_size'],
                num_attention_head=self.config['num_attention_heads'],
                num_kv_cache_head=self.config['num_key_value_heads'],
                total_num_layers=self.config['num_layers'],
                intermediate_dim=self.config['intermediate_size'],
                vocab_size=self.config['vocab_size'],
                dtype=self.config['dtype']
            )
            max_stage_latency = max(prev_latency, curr_latency)
            if max_stage_latency < min_latency:
                min_latency = max_stage_latency
                best_i = i

        self.best_split[num_stage][start_layer][end_layer] = best_i
        self.latency_cache[num_stage][start_layer][end_layer] = min_latency
        return min_latency

    def optimize(self):
        num_layers = self.config['num_layers']
        self.latency_cache = np.full((self.num_stage + 1, num_layers + 1, num_layers + 1), -1.0)
        self.best_split = np.full((self.num_stage + 1, num_layers + 1, num_layers + 1), -1, dtype=int)
        optimal_latency = self.F(self.num_stage, 0, num_layers)

        # Backtrack to get actual layer distribution
        layers = []
        start = 0
        end = num_layers
        for s in range(self.num_stage, 0, -1):
            if s == 1:
                layers.append(end - start)
            else:
                split_point = self.best_split[s][start][end]
                layers.append(end - split_point)
                end = split_point

        self.layers_per_stage = np.array(layers[::-1])
        return optimal_latency


if __name__ == "__main__":
    instance_type = "g6e.xlarge"
    num_stage = 10
    gpu_type = INSTANCE_SPEC[instance_type]['gpu_type']
    tp_size = INSTANCE_SPEC[instance_type]['gpu_count']
    interconnect_bandwidth = INTERCONNECT_SPEC[INSTANCE_SPEC[instance_type]['interconnect']]['bandwidth']


    model_name = "meta-llama/Llama-3.1-70B-Instruct"
    model_config = AutoConfig.from_pretrained(model_name)

    config = {
        "expected_input_len": 512,
        "expected_output_len": 128,
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_hidden_layers,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": getattr(model_config, "num_key_value_heads", model_config.num_attention_heads),
        "intermediate_size": model_config.intermediate_size,
        "vocab_size": model_config.vocab_size,
        "max_position_embeddings": model_config.max_position_embeddings,
        "dtype": torch.float16,
        "max_model_len": 8192,
        "gpu_mem_utilization": 0.9,
        "tp_size": tp_size,
        "p2p_bandwidth": interconnect_bandwidth,
    }

    optimizer = AlpaServeOptimizer(gpu_type, num_stage, config)
    optimal_latency = optimizer.optimize()
    print(f"Optimal latency per stage: {optimal_latency:.2f} ms")
    print(f"Layers per stage: {optimizer.layers_per_stage[1:]}")

