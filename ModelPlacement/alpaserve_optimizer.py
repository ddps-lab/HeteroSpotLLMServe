"""
AlpaServe Optimizer — uses the original AlpaServe codebase (mms) for
placement decisions and pipeline partitioning.

Only the profiling data (per-layer latency & memory) is generated from
our analytical estimator, since real GPU profiling data is unavailable.
All algorithmic logic (DP layer partitioning, ModelParallelismSearch,
capability computation, simulator) comes from the original AlpaServe code.
"""
import sys
import os
import numpy as np
import torch
from typing import List

# ── Our estimator ──
from estimator_utils import (
    get_prefill_computation_latency_per_layer,
    get_decoding_computation_latency_per_layer,
    get_prefill_compute_logit_latency,
    get_decoding_compute_logit_latency,
    get_tp_communication_latency_per_layer,
    get_memory_size_decoder_layer_weight_bytes,
    get_memory_size_embedding_or_lm_head_weight_bytes,
)
from hardware_specs import GPU_SPEC, INTERCONNECT_SPEC, INSTANCE_SPEC

# ── Original AlpaServe code ──
# Use local copy at ModelPlacement/alpaserve_lib (from mms repository)
_alpaserve_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpaserve_lib")
_alpaserve_parent = os.path.dirname(_alpaserve_lib_path)

# Temporarily make alpaserve_lib importable as "alpa_serve"
if _alpaserve_parent not in sys.path:
    sys.path.insert(0, _alpaserve_parent)

from alpaserve_lib.profiling import ParallelConfig, ProfilingResult, LatencyMemData
from alpaserve_lib.placement_policy import ModelParallelismSearch, ClusterEnv, ModelData
from alpaserve_lib.placement_policy.base_policy import ModelPlacement
from alpaserve_lib.simulator.controller import approximate_one_case
from alpaserve_lib.simulator.executable import Executable
from alpaserve_lib.simulator.workload import GammaProcess
from alpaserve_lib.util import GB, ServingCase

from functools import partial


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
    dtype: torch.dtype = torch.float16,
    head_dim: int = None,
) -> float:
    """Estimate latency (ms) for layers [start_layer, end_layer)."""
    num_layers = end_layer - start_layer
    if num_layers <= 0:
        return 0.0

    batch_size = 1

    lat = (
        get_prefill_computation_latency_per_layer(
            gpu_type=gpu_type, gpu_count=tp_size, input_len=avg_input_len,
            hidden_dim=hidden_dim, num_attention_head=num_attention_head,
            num_kv_cache_head=num_kv_cache_head, batch_size=batch_size,
            intermediate_dim=intermediate_dim, dtype=dtype, head_dim=head_dim)
        + get_tp_communication_latency_per_layer(
            tp_size=tp_size, batch_size=batch_size, sequence_len=avg_input_len,
            hidden_dim=hidden_dim, p2p_bandwidth=p2p_bandwidth, dtype=dtype)
        + get_decoding_computation_latency_per_layer(
            gpu_type=gpu_type, gpu_count=tp_size, input_len=avg_input_len,
            output_len=avg_output_len, hidden_dim=hidden_dim,
            num_attention_head=num_attention_head,
            num_kv_cache_head=num_kv_cache_head, batch_size=batch_size,
            intermediate_dim=intermediate_dim, dtype=dtype, head_dim=head_dim)
        + get_tp_communication_latency_per_layer(
            tp_size=tp_size, batch_size=batch_size, sequence_len=avg_output_len,
            hidden_dim=hidden_dim, p2p_bandwidth=p2p_bandwidth, dtype=dtype)
    ) * num_layers

    # lm_head on last stage
    if end_layer == total_num_layers:
        lat += get_prefill_compute_logit_latency(
            gpu_type=gpu_type, gpu_count=tp_size, input_len=avg_input_len,
            hidden_dim=hidden_dim, batch_size=batch_size,
            vocab_size=vocab_size, dtype=dtype)
        lat += get_decoding_compute_logit_latency(
            gpu_type=gpu_type, gpu_count=tp_size, output_len=avg_output_len,
            hidden_dim=hidden_dim, batch_size=batch_size,
            vocab_size=vocab_size, dtype=dtype)

    return lat


def _build_profiling_result(
    gpu_type: str,
    tp_size: int,
    p2p_bandwidth: float,
    max_pp: int,
    config: dict,
) -> ProfilingResult:
    """
    Build an AlpaServe ProfilingResult for the given model & hardware.
    Uses our analytical estimator for per-layer latency, then runs the
    paper's DP (§4.1) to partition layers into pipeline stages.
    """
    num_layers = config["num_layers"]
    hidden_dim = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    intermediate_dim = config["intermediate_size"]
    vocab_size = config["vocab_size"]
    dtype = config["dtype"]
    head_dim = config.get("head_dim", None)

    batch_size = 1

    # Per-layer latency in seconds
    per_layer_lat_sec = get_latency(
        start_layer=0, end_layer=1,
        gpu_type=gpu_type, tp_size=tp_size, p2p_bandwidth=p2p_bandwidth,
        avg_input_len=config["expected_input_len"],
        avg_output_len=config["expected_output_len"],
        hidden_dim=hidden_dim, num_attention_head=num_heads,
        num_kv_cache_head=num_kv_heads, total_num_layers=num_layers + 1,  # prevent lm_head
        intermediate_dim=intermediate_dim, vocab_size=vocab_size,
        dtype=dtype, head_dim=head_dim,
    ) / 1000.0

    # lm_head latency in seconds
    lm_head_lat_sec = (
        get_prefill_compute_logit_latency(
            gpu_type=gpu_type, gpu_count=tp_size,
            input_len=config["expected_input_len"],
            hidden_dim=hidden_dim, batch_size=batch_size,
            vocab_size=vocab_size, dtype=dtype)
        + get_decoding_compute_logit_latency(
            gpu_type=gpu_type, gpu_count=tp_size,
            output_len=config["expected_output_len"],
            hidden_dim=hidden_dim, batch_size=batch_size,
            vocab_size=vocab_size, dtype=dtype)
    ) / 1000.0

    # Weight memory
    layer_weight_bytes = get_memory_size_decoder_layer_weight_bytes(
        hidden_dim=hidden_dim, num_attention_head=num_heads,
        num_key_value_head=num_kv_heads, intermediate_dim=intermediate_dim,
        dtype=dtype, head_dim=head_dim)
    embed_weight_bytes = get_memory_size_embedding_or_lm_head_weight_bytes(
        hidden_dim=hidden_dim, vocab_size=vocab_size, dtype=dtype)

    para_dict = {}
    for pp in range(1, max_pp + 1):
        # ── DP: minimize max stage latency (AlpaServe §4.1) ──
        INF = float("inf")
        F = [[INF] * (num_layers + 1) for _ in range(pp + 1)]
        split = [[-1] * (num_layers + 1) for _ in range(pp + 1)]

        def _stage_lat(start, end):
            lat = (end - start) * per_layer_lat_sec
            if end == num_layers:
                lat += lm_head_lat_sec
            return lat

        for k in range(1, num_layers + 1):
            F[1][k] = _stage_lat(0, k)

        for s in range(2, pp + 1):
            for k in range(s, num_layers + 1):
                for i in range(s - 1, k):
                    val = max(F[s - 1][i], _stage_lat(i, k))
                    if val < F[s][k]:
                        F[s][k] = val
                        split[s][k] = i

        # Backtrack
        boundaries = []
        k = num_layers
        for s in range(pp, 1, -1):
            boundaries.append((split[s][k], k))
            k = split[s][k]
        boundaries.append((0, k))
        boundaries.reverse()

        layer_counts = [e - s for s, e in boundaries]
        stage_latencies_sec = [_stage_lat(s, e) for s, e in boundaries]

        # Per-GPU weight memory (after TP split)
        stage_weight_mem = []
        cumulative = 0
        for i, nlayers in enumerate(layer_counts):
            mem = layer_weight_bytes * nlayers / tp_size
            if i == 0:
                mem += embed_weight_bytes / tp_size
            cumulative += nlayers
            if cumulative == num_layers:
                mem += embed_weight_bytes / tp_size
            stage_weight_mem.append(mem)

        para_dict[ParallelConfig(1, tp_size, pp)] = LatencyMemData(
            latency={1: stage_latencies_sec},
            act_mem={1: [0] * pp},
            weight_mem=stage_weight_mem,
        )

    return ProfilingResult(
        model_name="model",
        para_dict=para_dict,
        preprocess_cpu=0,
        postprocess_cpu=0,
    )


class AlpaServeOptimizer:
    """
    Drop-in replacement that uses the original AlpaServe codebase.

    Interface (unchanged):
        optimizer = AlpaServeOptimizer(gpu_type, num_stage, config)
        optimal_latency, layers_per_stage = optimizer.optimize()
    """

    def __init__(self, instance_type: str, num_stage: int, config: dict):
        self.instance_type = instance_type
        self.num_stage = num_stage
        self.config = config
        self.layers_per_stage: np.ndarray = None

    def optimize(self):
        cfg = self.config
        tp_size = cfg["tp_size"]
        p2p_bandwidth = cfg["p2p_bandwidth"]
        gpu_type = self.instance_type
        num_stage = self.num_stage
        total_gpus = num_stage * tp_size

        gpu_mem_bytes = GPU_SPEC[gpu_type]["memory_size"] * 1e6
        gpu_util = cfg.get("gpu_mem_utilization", 0.9)

        # 1. Build profiling data from our estimator
        prof = _build_profiling_result(
            gpu_type=gpu_type,
            tp_size=tp_size,
            p2p_bandwidth=p2p_bandwidth,
            max_pp=num_stage,
            config=cfg,
        )

        # 2. Set up AlpaServe cluster
        cluster_env = ClusterEnv(
            num_devices=total_gpus,
            mem_budget=gpu_mem_bytes * gpu_util,
            num_devices_per_node=tp_size,
        )

        # SLO: 10x single-request latency (effectively no constraint)
        pp1_config = ParallelConfig(1, tp_size, 1)
        if pp1_config in prof.para_dict:
            single_req_lat = sum(prof.para_dict[pp1_config].latency[1])
        else:
            max_pp_config = max(prof.para_dict.keys(), key=lambda c: c.pp)
            single_req_lat = sum(prof.para_dict[max_pp_config].latency[1])
        slo = single_req_lat * 10

        rate = cfg.get("rate", 5.52)
        model_data = ModelData("model", slo=slo, rate=rate, cv=1,
                               profiling_result=prof)

        # 3. Use AlpaServe's ModelParallelismSearch
        policy = ModelParallelismSearch(verbose=0)
        placement, _ = policy.solve_placement([model_data], cluster_env)

        # 4. Extract the chosen config
        chosen_config = None
        for gc, gm in zip(placement.group_configs, placement.group_models):
            if 0 in gm:  # model id 0 is placed here
                chosen_config = gc
                break

        # Fallback: if search couldn't place (OOM etc.), use max PP
        if chosen_config is None:
            chosen_config = ParallelConfig(1, tp_size, num_stage)

        if chosen_config not in prof.para_dict:
            # Should not happen, but fallback to direct DP
            chosen_config = max(prof.para_dict.keys(), key=lambda c: c.pp)

        lat_data = prof.para_dict[chosen_config]
        stage_latencies = lat_data.latency[1]  # in seconds

        # Reconstruct layers_per_stage from the DP backtrack stored in profiling
        # Re-run the DP backtrack for the chosen PP
        pp = chosen_config.pp
        num_layers = cfg["num_layers"]

        per_layer_lat_sec = get_latency(
            start_layer=0, end_layer=1,
            gpu_type=gpu_type, tp_size=tp_size, p2p_bandwidth=p2p_bandwidth,
            avg_input_len=cfg["expected_input_len"],
            avg_output_len=cfg["expected_output_len"],
            hidden_dim=cfg["hidden_size"], num_attention_head=cfg["num_attention_heads"],
            num_kv_cache_head=cfg["num_key_value_heads"],
            total_num_layers=num_layers + 1,
            intermediate_dim=cfg["intermediate_size"], vocab_size=cfg["vocab_size"],
            dtype=cfg["dtype"], head_dim=cfg.get("head_dim", None),
        ) / 1000.0

        lm_head_lat_sec = (
            get_prefill_compute_logit_latency(
                gpu_type=gpu_type, gpu_count=tp_size,
                input_len=cfg["expected_input_len"],
                hidden_dim=cfg["hidden_size"], batch_size=1,
                vocab_size=cfg["vocab_size"], dtype=cfg["dtype"])
            + get_decoding_compute_logit_latency(
                gpu_type=gpu_type, gpu_count=tp_size,
                output_len=cfg["expected_output_len"],
                hidden_dim=cfg["hidden_size"], batch_size=1,
                vocab_size=cfg["vocab_size"], dtype=cfg["dtype"])
        ) / 1000.0

        def _stage_lat(start, end):
            lat = (end - start) * per_layer_lat_sec
            if end == num_layers:
                lat += lm_head_lat_sec
            return lat

        # DP backtrack for chosen PP
        INF = float("inf")
        F = [[INF] * (num_layers + 1) for _ in range(pp + 1)]
        split_arr = [[-1] * (num_layers + 1) for _ in range(pp + 1)]

        for k in range(1, num_layers + 1):
            F[1][k] = _stage_lat(0, k)
        for s in range(2, pp + 1):
            for k in range(s, num_layers + 1):
                for i in range(s - 1, k):
                    val = max(F[s - 1][i], _stage_lat(i, k))
                    if val < F[s][k]:
                        F[s][k] = val
                        split_arr[s][k] = i

        boundaries = []
        k = num_layers
        for s in range(pp, 1, -1):
            boundaries.append((split_arr[s][k], k))
            k = split_arr[s][k]
        boundaries.append((0, k))
        boundaries.reverse()

        layers_per_stage = np.array([e - s for s, e in boundaries])
        optimal_latency_ms = F[pp][num_layers] * 1000.0  # seconds → ms

        self.layers_per_stage = layers_per_stage
        self.chosen_config = chosen_config

        return optimal_latency_ms, layers_per_stage


if __name__ == "__main__":
    from transformers import AutoConfig

    instance_type = "g6e.xlarge"
    num_stage = 4
    gpu_type = INSTANCE_SPEC[instance_type]["gpu_type"]
    tp_size = INSTANCE_SPEC[instance_type]["gpu_count"]
    interconnect_bandwidth = INTERCONNECT_SPEC[
        INSTANCE_SPEC[instance_type]["interconnect"]
    ]["bandwidth"]

    model_name = "meta-llama/Llama-3.1-70B-Instruct"
    model_config = AutoConfig.from_pretrained(model_name)

    config = {
        "expected_input_len": 763,
        "expected_output_len": 232,
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_hidden_layers,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": getattr(
            model_config, "num_key_value_heads", model_config.num_attention_heads
        ),
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
    optimal_latency, layers = optimizer.optimize()
    print(f"Chosen config: {optimizer.chosen_config}")
    print(f"Optimal stage latency: {optimal_latency:.2f} ms")
    print(f"Layers per stage: {layers}")
