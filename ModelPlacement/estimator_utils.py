from typing import List, Optional
import torch
import logging
from hardware_specs import GPU_SPEC, INSTANCE_SPEC

import sys
import os
# Add parent directory to path for protocols import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocols import OUT_OF_MEMORY


def _effective_head_dim(hidden_dim, num_attention_head, head_dim=None):
    """Return head_dim: explicit if provided, else hidden_dim // num_attention_head (Llama-style)."""
    return head_dim if head_dim is not None else (hidden_dim // num_attention_head)


def _h_kv(hidden_dim, num_attention_head, num_kv_cache_head, head_dim=None):
    """Total KV dimension = num_kv_heads * head_dim."""
    return num_kv_cache_head * _effective_head_dim(hidden_dim, num_attention_head, head_dim)


def _h_q(hidden_dim, num_attention_head, head_dim=None):
    """Total Q dimension = num_attention_heads * head_dim."""
    return num_attention_head * _effective_head_dim(hidden_dim, num_attention_head, head_dim)


def get_prefill_qkv_projection_ops_per_layer(
    input_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    tp_size: int,
    batch_size: int,
    head_dim: int = None,
):
    H_kv = _h_kv(hidden_dim, num_attention_head, num_kv_cache_head, head_dim)
    H_q = _h_q(hidden_dim, num_attention_head, head_dim)
    qkv_projection_ops = (
        2 * input_len * hidden_dim * H_q + 4 * input_len * hidden_dim * H_kv  # Q,O + K,V
    ) // tp_size * batch_size
    return qkv_projection_ops

def get_prefill_attention_ops_per_layer(
    input_len: int,
    hidden_dim: int,
    tp_size: int,
    batch_size: int,
    num_attention_head: int = None,
    head_dim: int = None,
):
    # H_kv is not actually used during attention. It is replicated anyway, so the computation amount is the same.
    # It only uses slightly less memory
    # When head_dim != hidden_dim // num_heads (e.g. Qwen3), use H_q for attention dimension
    H_q = _h_q(hidden_dim, num_attention_head, head_dim) if num_attention_head is not None else hidden_dim
    attention_ops = 2 * (
        input_len ** 2 * H_q * 2 + input_len * H_q ** 2
    ) // tp_size * batch_size
    return attention_ops

def get_prefill_ffn_ops_per_layer(
    input_len: int,
    hidden_dim: int,
    intermediate_dim: int,
    tp_size: int,
    batch_size: int,
):
    ffn_ops = 6 * input_len * hidden_dim * intermediate_dim // tp_size * batch_size
    return ffn_ops

def get_prefill_compute_logit_ops(
    input_len: int,
    hidden_dim: int,
    vocab_size: int,
    tp_size: int,
    batch_size: int,
):
    compute_logit_ops = (
        2 * batch_size * input_len * hidden_dim * vocab_size // tp_size
    )

    return compute_logit_ops

def get_decoding_qkv_projection_ops_per_layer(
    output_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    tp_size: int,
    batch_size: int,
    head_dim: int = None,
):
    H_kv = _h_kv(hidden_dim, num_attention_head, num_kv_cache_head, head_dim)
    H_q = _h_q(hidden_dim, num_attention_head, head_dim)
    qkv_projection_ops = 2 * (
        hidden_dim * H_q + 2 * (hidden_dim * H_kv)  # Q,O + K,V
    ) // tp_size * batch_size * output_len
    return qkv_projection_ops

def get_decoding_attention_ops_per_layer(
    input_len: int,
    output_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    tp_size: int,
    batch_size: int,
    head_dim: int = None,
):
    # Only in attention does the formula change slightly depending on output len
    # For qkv projection or ffn, output_len is simply multiplied at the end,
    # but here the kv cache length increases by 1 at each step,
    # so it needs to be expanded as a series.
    # The resulting formula is: 4*(s_in + t)H + 2H^2
    # where t ranges from 1 to output_len, and expanding as a series gives:
    # 4s_in*H*s_out + 4H * s_out(s_out+1)/2 + 2H^2*s_out
    H_kv = _h_kv(hidden_dim, num_attention_head, num_kv_cache_head, head_dim)
    H_q = _h_q(hidden_dim, num_attention_head, head_dim)
    attention_ops = (
        4 * input_len * output_len * H_q +
        2 * H_q * output_len * (output_len + 1) +
        2 * H_q ** 2 * output_len
    )
    attention_ops = attention_ops // tp_size * batch_size
    return attention_ops

def get_decoding_ffn_ops_per_layer(
    output_len: int,
    hidden_dim: int,
    intermediate_dim: int,
    tp_size: int,
    batch_size: int,
):
    ffn_ops = 6 * hidden_dim * intermediate_dim // tp_size * batch_size * output_len
    return ffn_ops
    
def get_decoding_compute_logit_ops(
    output_len: int,
    hidden_dim: int,
    vocab_size: int,
    tp_size: int,
    batch_size: int,
):
    compute_logit_ops = (
        2 * batch_size * output_len * hidden_dim * vocab_size // tp_size
    )

    return compute_logit_ops
    
def get_prefill_qkv_projection_memory_access_count_per_layer(
    input_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    tp_size: int,
    batch_size: int,
    head_dim: int = None,
):
    H_kv = _h_kv(hidden_dim, num_attention_head, num_kv_cache_head, head_dim)
    H_q = _h_q(hidden_dim, num_attention_head, head_dim)
    qkv_projection_memory_access_count = (
        batch_size * input_len * hidden_dim +  # input
        (hidden_dim * H_q + 2 * hidden_dim * H_kv) // tp_size  # weights (Q,O + K,V)
    )
    return qkv_projection_memory_access_count

def get_prefill_attention_memory_access_count_per_layer(
    input_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    tp_size: int,
    batch_size: int,
    head_dim: int = None,
):
    H_kv = _h_kv(hidden_dim, num_attention_head, num_kv_cache_head, head_dim)
    H_q = _h_q(hidden_dim, num_attention_head, head_dim)
    attention_memory_access_count = (
        batch_size * input_len * hidden_dim +  # input
        2 * input_len * H_kv +  # k, v
        H_q**2 # q
    ) // tp_size
    return attention_memory_access_count

def get_prefill_ffn_memory_access_count_per_layer(
    input_len: int,
    hidden_dim: int,
    intermediate_dim: int,
    tp_size: int,
    batch_size: int,
):
    ffn_memory_access_count = (
        batch_size * input_len * hidden_dim + # up&gate input
        2 * hidden_dim * intermediate_dim // tp_size + # up&gate weight
        2 * batch_size * input_len * intermediate_dim // tp_size + # up gate element-wise product
        # The following is input generated as an intermediate tensor, so it is affected by tp_size.
        batch_size * input_len * intermediate_dim // tp_size +
        intermediate_dim * hidden_dim // tp_size # down weight
    )
    return ffn_memory_access_count

def get_prefill_compute_logit_memory_access_count(
    input_len: int,
    hidden_dim: int,
    vocab_size: int,
    tp_size: int,
    batch_size: int,
):
    prefill_logit_memory_access_count = (
        batch_size * input_len * hidden_dim + vocab_size * hidden_dim / tp_size
    )

    return prefill_logit_memory_access_count

def get_decoding_qkv_projection_memory_access_count_per_layer(
    output_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    tp_size: int,
    batch_size: int,
    head_dim: int = None,
):
    # input_len is not needed here because it is always 1
    H_kv = _h_kv(hidden_dim, num_attention_head, num_kv_cache_head, head_dim)
    H_q = _h_q(hidden_dim, num_attention_head, head_dim)
    qkv_projection_memory_access_count = (
        batch_size * hidden_dim +  # input
        (hidden_dim * H_q + 2 * hidden_dim * H_kv) // tp_size  # weights (Q,O + K,V)
    ) * output_len
    return qkv_projection_memory_access_count

def get_decoding_attention_memory_access_count_per_layer(
    input_len: int,
    output_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    tp_size: int,
    batch_size: int,
    head_dim: int = None,
):
    # Only in attention does the formula change slightly depending on output len
    # For qkv projection or ffn, output_len is simply multiplied at the end,
    # but here the kv cache length increases by 1 at each step,
    # so it needs to be expanded as a series.
    # The resulting formula is: (BH + 2B(s_in + t)H_kv + H^2)
    # where t ranges from 1 to output_len, and expanding as a series gives:
    # s_out*BH + 2Bs_in*H_kv*s_out + 2BH_kv * s_out(s_out+1)/2 + s_out*H^2
    H_kv = _h_kv(hidden_dim, num_attention_head, num_kv_cache_head, head_dim)
    H_q = _h_q(hidden_dim, num_attention_head, head_dim)
    attention_memory_access_count = (
        batch_size * hidden_dim * output_len +  # input(BH) is fed output_len times
        2 * batch_size * input_len * H_kv * output_len +  # k, v
        batch_size * H_kv * output_len * (output_len + 1) +  # compounded kv cache
        H_q**2 * output_len # q
    ) // tp_size
    return attention_memory_access_count

def get_decoding_ffn_memory_access_count_per_layer(
    output_len: int,
    hidden_dim: int,
    intermediate_dim: int,
    tp_size: int,
    batch_size: int,
):
    ffn_memory_access_count = (
        batch_size * hidden_dim + # up&gate input
        2 * hidden_dim * intermediate_dim // tp_size + # up&gate weight
        2 * batch_size * intermediate_dim // tp_size + # up gate element-wise product
        # The following is input generated as an intermediate tensor, so it is affected by tp_size.
        batch_size * intermediate_dim // tp_size +
        intermediate_dim * hidden_dim // tp_size # down weight
    ) * output_len
    return ffn_memory_access_count

def get_decoding_compute_logit_memory_access_count(
    output_len: int,
    hidden_dim: int,
    vocab_size: int,
    tp_size: int,
    batch_size: int,
):
    decoding_logit_memory_access_count = (
        output_len * (batch_size * hidden_dim + vocab_size * hidden_dim / tp_size)
    )

    return decoding_logit_memory_access_count

def get_tp_communication_latency_per_layer(
    tp_size: int,
    batch_size: int,
    sequence_len: int,
    hidden_dim: int,
    p2p_bandwidth: float, # unit is Bytes/s
    dtype: torch.dtype = torch.float16,
    p2p_latency_ms: Optional[float] = None,
):
    if tp_size == 1:
        return 0

    if p2p_latency_ms is None:
        # Assume intra p2p latency is nearly zero since it is extremely fast
        p2p_latency_ms = 0

    element_size = dtype.itemsize  # item size of dtype
    latency_per_layer_ms = (batch_size * sequence_len * hidden_dim * element_size) / (tp_size * p2p_bandwidth / 1000)  # Bytes/s to ms
    latency_per_layer_ms += p2p_latency_ms
    latency_per_layer_ms *= 4 * (tp_size - 1)

    return latency_per_layer_ms

def get_tp_communication_latency_compute_logit(
    tp_size: int,
    batch_size: int,
    sequence_len: int,
    hidden_dim: int,
    vocab_size: int,
    p2p_bandwidth: float, # unit is Bytes/s
    dtype: torch.dtype = torch.float16,
    p2p_latency_ms: Optional[float] = None,
):
    if tp_size == 1:
        return 0

    if p2p_latency_ms is None:
        p2p_latency_ms = 0

    element_size = dtype.itemsize  # item size of dtype
    # After logit computation, the output needs to be all-gathered.
    # The size of the output is batch_size * sequence length * vocab size
    latency_compute_logit_ms = (batch_size * sequence_len * vocab_size * element_size) / (tp_size * p2p_bandwidth / 1000)  # Bytes/s to ms
    latency_compute_logit_ms += p2p_latency_ms
    latency_compute_logit_ms *= 2*(tp_size - 1)

    return latency_compute_logit_ms


def get_pp_communication_latency(
    batch_size: int,
    sequence_len: int,
    hidden_dim: int,
    tp_size_at_receiver: int,
    p2p_bandwidth: float,
    p2p_latency_ms: Optional[float] = None,
    dtype: torch.dtype = torch.float16,
    inter_node_latency_ms: Optional[float] = None,
    inter_node_bandwidth: Optional[float] = None,  # unit is Bytes/s
):
    """
    P2P Latency and Bandwidth should be given based on the receiver side.
    """
    latency_send_recv = get_pp_communication_latency_send_recv(
        batch_size=batch_size,
        sequence_len=sequence_len,
        hidden_dim=hidden_dim,
        dtype=dtype,
        inter_node_latency_ms=inter_node_latency_ms,
        inter_node_bandwidth=inter_node_bandwidth
    )
    latency_broadcast = get_pp_communication_latency_broadcast(
        batch_size=batch_size,
        sequence_len=sequence_len,
        hidden_dim=hidden_dim,
        tp_size=tp_size_at_receiver,
        p2p_bandwidth=p2p_bandwidth,
        p2p_latency_ms=p2p_latency_ms,
        dtype=dtype
    )
    
    total_latency = latency_send_recv + latency_broadcast
    
    return total_latency

def get_pp_communication_latency_send_recv(
    batch_size: int,
    sequence_len: int,
    hidden_dim: int,
    inter_node_latency_ms: Optional[float] = None,
    inter_node_bandwidth: Optional[float] = None,  # unit is Bytes/s
    dtype: torch.dtype = torch.float16,
):

    if inter_node_latency_ms is None:
        # Assume inter node latency is 1ms by default (intra region)
        inter_node_latency_ms = 1
    if inter_node_bandwidth is None:
        # Assume inter node bandwidth is 5GB/s by default (intra region)
        inter_node_bandwidth = 5 * 10**9  # 5GB/s

    element_size = dtype.itemsize

    latency_send_recv = (batch_size * sequence_len * hidden_dim * element_size) / (inter_node_bandwidth / 1000)  # Bytes/s to ms
    latency_send_recv += inter_node_latency_ms
    
    return latency_send_recv

def get_pp_communication_latency_broadcast(
    batch_size: int,
    sequence_len: int,
    hidden_dim: int,
    tp_size: int,
    p2p_bandwidth: float,  # unit is Bytes/s
    p2p_latency_ms: Optional[float] = None,
    dtype: torch.dtype = torch.float16,
):
    """
    This function should be called from the receiving side (next stage) in PP communication
    """

    if p2p_latency_ms is None:
        # Assume intra p2p latency is nearly zero since it is extremely fast
        p2p_latency_ms = 0

    element_size = dtype.itemsize

    # Broadcast occurs when tp size is greater than 1
    if tp_size > 1:
        latency_broadcast = (batch_size * sequence_len * hidden_dim * element_size) / (p2p_bandwidth / 1000)  # Bytes/s to ms
        latency_broadcast += p2p_latency_ms
        latency_broadcast *= (tp_size - 1) # Assuming sequential transmission to other nodes excluding itself
    else:
        latency_broadcast = 0
    
    return latency_broadcast

def get_prefill_computation_latency_per_layer(
    gpu_type: str,
    gpu_count: int,
    input_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    batch_size: int,
    intermediate_dim: int,
    dtype: torch.dtype = torch.float16,
    head_dim: int = None,
):
    prefill_qkv_ops = get_prefill_qkv_projection_ops_per_layer(
        input_len=input_len,
        hidden_dim=hidden_dim,
        num_attention_head=num_attention_head,
        num_kv_cache_head=num_kv_cache_head,
        tp_size=gpu_count,
        batch_size=batch_size,
        head_dim=head_dim,
    )
    prefill_attention_ops = get_prefill_attention_ops_per_layer(
        input_len=input_len,
        hidden_dim=hidden_dim,
        tp_size=gpu_count,
        batch_size=batch_size,
        num_attention_head=num_attention_head,
        head_dim=head_dim,
    )
    prefill_ffn_ops = get_prefill_ffn_ops_per_layer(
        input_len=input_len,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        tp_size=gpu_count,
        batch_size=batch_size,
    )
    prefill_qkv_memory_access_count = get_prefill_qkv_projection_memory_access_count_per_layer(
        input_len=input_len,
        hidden_dim=hidden_dim,
        num_attention_head=num_attention_head,
        num_kv_cache_head=num_kv_cache_head,
        tp_size=gpu_count,
        batch_size=batch_size,
        head_dim=head_dim,
    )
    prefill_attention_memory_access_count = get_prefill_attention_memory_access_count_per_layer(
        input_len=input_len,
        hidden_dim=hidden_dim,
        num_attention_head=num_attention_head,
        num_kv_cache_head=num_kv_cache_head,
        tp_size=gpu_count,
        batch_size=batch_size,
        head_dim=head_dim,
    )
    prefill_ffn_memory_access_count = get_prefill_ffn_memory_access_count_per_layer(
        input_len=input_len,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        tp_size=gpu_count,
        batch_size=batch_size,
    )
    key_list = ["qkv_proj", "attention", "ffn"]
    ops_list = [prefill_qkv_ops, prefill_attention_ops, prefill_ffn_ops]
    memory_access_list = [prefill_qkv_memory_access_count, prefill_attention_memory_access_count, prefill_ffn_memory_access_count]

    GPU_SPEC_info = GPU_SPEC[gpu_type]
    flops = GPU_SPEC_info["FLOPS"]
    memory_bandwidth = GPU_SPEC_info["memory_bandwidth"]
    device_arithmetic_intensity = flops / memory_bandwidth

    latency_per_layer = 0.0

    for key, computation_ops, computation_memory_access in zip(key_list, ops_list, memory_access_list):
        arithmetic_intensity = computation_ops / (computation_memory_access * dtype.itemsize)  # Convert to Bytes
        if arithmetic_intensity < device_arithmetic_intensity:
            latency = (computation_memory_access * dtype.itemsize) / (memory_bandwidth / 1000) # Bytes/s to ms
            # Memory bound
            logging.debug(f"{key} layer is memory bound in prefill with device {gpu_type}x{gpu_count} with {latency:.2f}ms (per layer)")
            latency_per_layer += latency
        else:
            latency = computation_ops / (flops / 1000)  # FLOPS to ms
            # Compute bound
            logging.debug(f"{key} layer is compute bound in prefill with device {gpu_type}x{gpu_count} with {latency:.2f}ms (per layer)")
            latency_per_layer += latency

    return latency_per_layer

def get_prefill_compute_logit_latency(
    gpu_type: str,
    gpu_count: int,
    input_len: int,
    hidden_dim: int,
    batch_size: int,
    vocab_size: int,
    dtype: torch.dtype = torch.float16,
):
    if batch_size <= 0:
        return 0

    compute_logit_ops = get_prefill_compute_logit_ops(
        1, hidden_dim, vocab_size, gpu_count, batch_size
    )
    compute_logit_memory_access = get_prefill_compute_logit_memory_access_count(
        1, hidden_dim, vocab_size, gpu_count, batch_size
    ) * dtype.itemsize  # Convert to Bytes

    GPU_SPEC_info = GPU_SPEC[gpu_type]
    flops = GPU_SPEC_info["FLOPS"]
    memory_bandwidth = GPU_SPEC_info["memory_bandwidth"]
    arithmetic_intensity = compute_logit_ops / compute_logit_memory_access
    device_arithmetic_intensity = flops / memory_bandwidth

    if arithmetic_intensity < device_arithmetic_intensity:
        # Memory bound
        latency = compute_logit_memory_access / (memory_bandwidth / 1000) # Bytes/s to ms
        logging.debug(f"Prefill compute logit layer is memory bound with device {gpu_type}x{gpu_count} with {latency:.2f}ms")
    else:
        # Compute bound
        latency = compute_logit_ops / (flops / 1000)  # FLOPS to ms
        logging.debug(f"Prefill compute logit layer is compute bound with device {gpu_type}x{gpu_count} with {latency:.2f}ms")

    return latency

def get_decoding_computation_latency_per_layer(
    gpu_type: str,
    gpu_count: int,
    input_len: int,
    output_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    batch_size: int,
    intermediate_dim: int,
    dtype: torch.dtype = torch.float16,
    head_dim: int = None,
):
    qkv_projection_ops_per_layer = get_decoding_qkv_projection_ops_per_layer(
        output_len=output_len,
        hidden_dim=hidden_dim,
        num_attention_head=num_attention_head,
        num_kv_cache_head=num_kv_cache_head,
        tp_size=gpu_count,
        batch_size=batch_size,
        head_dim=head_dim,
    )
    attention_ops_per_layer = get_decoding_attention_ops_per_layer(
        input_len=input_len,
        output_len=output_len,
        hidden_dim=hidden_dim,
        num_attention_head=num_attention_head,
        num_kv_cache_head=num_kv_cache_head,
        tp_size=gpu_count,
        batch_size=batch_size,
        head_dim=head_dim,
    )
    ffn_ops_per_layer = get_decoding_ffn_ops_per_layer(
        output_len=output_len,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        tp_size=gpu_count,
        batch_size=batch_size,
    )
    qkv_projection_memory_access_count_per_layer = get_decoding_qkv_projection_memory_access_count_per_layer(
        output_len=output_len,
        hidden_dim=hidden_dim,
        num_attention_head=num_attention_head,
        num_kv_cache_head=num_kv_cache_head,
        tp_size=gpu_count,
        batch_size=batch_size,
        head_dim=head_dim,
    )
    attention_memory_access_count_per_layer = get_decoding_attention_memory_access_count_per_layer(
        input_len=input_len,
        output_len=output_len,
        hidden_dim=hidden_dim,
        num_attention_head=num_attention_head,
        num_kv_cache_head=num_kv_cache_head,
        tp_size=gpu_count,
        batch_size=batch_size,
        head_dim=head_dim,
    )
    ffn_memory_access_count_per_layer = get_decoding_ffn_memory_access_count_per_layer(
        output_len=output_len,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        tp_size=gpu_count,
        batch_size=batch_size,
    )
    key_list = ["qkv_proj", "attention", "ffn"]
    ops_list = [qkv_projection_ops_per_layer, attention_ops_per_layer, ffn_ops_per_layer]
    memory_access_list = [qkv_projection_memory_access_count_per_layer, attention_memory_access_count_per_layer, ffn_memory_access_count_per_layer]
    
    GPU_SPEC_info = GPU_SPEC[gpu_type]
    flops = GPU_SPEC_info["FLOPS"]
    memory_bandwidth = GPU_SPEC_info["memory_bandwidth"]
    device_arithmetic_intensity = flops / memory_bandwidth

    latency_per_layer = 0.0
    for key, computation_ops_per_layer, computation_memory_access_per_layer in zip(key_list, ops_list, memory_access_list):
        arithmetic_intensity = computation_ops_per_layer / (computation_memory_access_per_layer * dtype.itemsize)  # Convert to Bytes
        if arithmetic_intensity < device_arithmetic_intensity:
            latency = (computation_memory_access_per_layer * dtype.itemsize) / (memory_bandwidth / 1000) # Bytes/s to ms
            logging.debug(f"{key} layer is memory bound in decoding with device {gpu_type}x{gpu_count} with {latency:.2f}ms (per layer)")
            # Memory bound
            latency_per_layer += latency
        else:
            latency = computation_ops_per_layer / (flops / 1000)  # FLOPS to ms
            logging.debug(f"{key} layer is compute bound in decoding with device {gpu_type}x{gpu_count} with {latency:.2f}ms (per layer)")
            # Compute bound
            latency_per_layer += latency
    return latency_per_layer

def get_decoding_compute_logit_latency(
    gpu_type: str,
    gpu_count: int,
    output_len: int,
    hidden_dim: int,
    batch_size: int,
    vocab_size: int,
    dtype: torch.dtype = torch.float16,
):
    if batch_size <= 0:
        return 0

    compute_logit_ops = get_decoding_compute_logit_ops(
        output_len, hidden_dim, vocab_size, gpu_count, batch_size
    )
    compute_logit_memory_access = get_decoding_compute_logit_memory_access_count(
        output_len, hidden_dim, vocab_size, gpu_count, batch_size
    ) * dtype.itemsize  # Convert to Bytes

    GPU_SPEC_info = GPU_SPEC[gpu_type]
    flops = GPU_SPEC_info["FLOPS"]
    memory_bandwidth = GPU_SPEC_info["memory_bandwidth"]
    arithmetic_intensity = compute_logit_ops / compute_logit_memory_access
    device_arithmetic_intensity = flops / memory_bandwidth

    if arithmetic_intensity < device_arithmetic_intensity:
        # Memory bound
        latency = compute_logit_memory_access / (memory_bandwidth / 1000) # Bytes/s to ms
        logging.debug(f"Decoding compute logit layer is memory bound with device {gpu_type}x{gpu_count} with {latency:.2f}ms")
    else:
        # Compute bound
        latency = compute_logit_ops / (flops / 1000)  # FLOPS to ms
        logging.debug(f"Decoding compute logit layer is compute bound with device {gpu_type}x{gpu_count} with {latency:.2f}ms")
    return latency

def get_forwarding_memory(
    max_model_len: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.float16,
    intermediate_dim: Optional[int] = None,
):
    if intermediate_dim is None:
        intermediate_dim = 4 * hidden_dim

    forwarding_memory = (
        2 * # up / gate / output
        max_model_len * # maximum input
        intermediate_dim * # intermediate dimension
        dtype.itemsize # element size
    ) # Unit : Bytes

    return forwarding_memory

def get_memory_size_embedding_or_lm_head_weight_bytes(hidden_dim: int, vocab_size: int, dtype: torch.dtype):
    return vocab_size * hidden_dim * dtype.itemsize  # Unit : Bytes

def get_memory_size_decoder_layer_weight_bytes(
    hidden_dim: int,
    num_attention_head: int,
    num_key_value_head: int,
    intermediate_dim: int,
    dtype: torch.dtype = torch.float16,
    head_dim: int = None,
):
    """
    A single Decoder Layer contains the following weights:
      - input layer norm
      - q projection
      - k projection
      - v projection
      - o projection
      - post attention layer norm
      - up projection
      - gate projection
      - down projection
    """
    assert num_attention_head % num_key_value_head == 0, "num_attention_head must be divisible by num_key_value_head"
    num_total_elem = 0
    eff_head_dim = _effective_head_dim(hidden_dim, num_attention_head, head_dim)
    H_q = num_attention_head * eff_head_dim
    kv_cache_dim = num_key_value_head * eff_head_dim

    # input layer norm
    input_layer_norm_elem = hidden_dim
    num_total_elem += input_layer_norm_elem

    # q, o projection: Q = hidden_dim → H_q, O = H_q → hidden_dim
    q_o_projection_elem = hidden_dim * H_q * 2
    num_total_elem += q_o_projection_elem

    # k, v projection: K = hidden_dim → kv_dim, V = hidden_dim → kv_dim
    k_v_projection_elem = hidden_dim * kv_cache_dim * 2
    num_total_elem += k_v_projection_elem

    # post attention layer norm
    post_attention_layer_norm_elem = hidden_dim
    num_total_elem += post_attention_layer_norm_elem

    # up, gate, down projection
    up_gate_down_projection_elem = hidden_dim * intermediate_dim * 3
    num_total_elem += up_gate_down_projection_elem

    return num_total_elem * dtype.itemsize  # Unit : Bytes

def get_single_request_latency(
    avg_input_len: int,
    avg_output_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    total_num_layers: int,
    intermediate_dim: int,
    vocab_size: int,
    node_layer_comb: List[tuple[str, str, int]],
    dtype: torch.dtype = torch.float16,
    head_dim: int = None,
    tp_sizes: List[int] = None,
):
    batch_size = 1 #for single request

    prefill_latencies = []
    decoding_latencies = []

    pp_communication_latency = 0
    tp_communication_latency = 0

    total_stages = len(node_layer_comb)

    processed_layers = 0

    for i, (node_type, az, layer_count) in enumerate(node_layer_comb):
        gpu_type = INSTANCE_SPEC[node_type]["gpu_type"]
        num_gpu = tp_sizes[i] if tp_sizes else INSTANCE_SPEC[node_type]["gpu_count"]
        p2p_bandwidth = INSTANCE_SPEC[node_type]["interconnect_bandwidth"]
        processed_layers += layer_count

        prefill_computation_laytency = get_prefill_computation_latency_per_layer(
            gpu_type=gpu_type,
            gpu_count=num_gpu,
            input_len=avg_input_len,
            hidden_dim=hidden_dim,
            num_attention_head=num_attention_head,
            num_kv_cache_head=num_kv_cache_head,
            batch_size=batch_size,
            intermediate_dim=intermediate_dim,
            dtype=dtype,
            head_dim=head_dim,
        ) * layer_count

        decoding_computation_latency = get_decoding_computation_latency_per_layer(
            gpu_type=gpu_type,
            gpu_count=num_gpu,
            input_len=avg_input_len,
            output_len=avg_output_len,
            hidden_dim=hidden_dim,
            num_attention_head=num_attention_head,
            num_kv_cache_head=num_kv_cache_head,
            head_dim=head_dim,
            batch_size=batch_size,
            intermediate_dim=intermediate_dim,
            dtype=dtype
        ) * layer_count

        prefill_logit_latency = 0
        decoding_logit_latency = 0
        if processed_layers == total_num_layers:
            # The last stage includes the lm head
            prefill_logit_latency = get_prefill_compute_logit_latency(
                gpu_type=gpu_type,
                gpu_count=num_gpu,
                input_len=avg_input_len,
                hidden_dim=hidden_dim,
                batch_size=batch_size,
                vocab_size=vocab_size,
                dtype=dtype
            )
            decoding_logit_latency = get_decoding_compute_logit_latency(
                gpu_type=gpu_type,
                gpu_count=num_gpu,
                output_len=avg_output_len,
                hidden_dim=hidden_dim,
                batch_size=batch_size,
                vocab_size=vocab_size,
                dtype=dtype
            )

        prefill_latencies.append(prefill_computation_laytency + prefill_logit_latency)
        decoding_latencies.append(decoding_computation_latency + decoding_logit_latency)

        # If not the last stage, need to send (PP)
        if i < total_stages - 1:
            prefill_pp_communication_latency_send = get_pp_communication_latency_send_recv(
                batch_size=batch_size,
                sequence_len=avg_input_len,
                hidden_dim=hidden_dim,
                dtype=dtype
            )
            decoding_pp_communication_latency_send = get_pp_communication_latency_send_recv(
                batch_size=batch_size,
                sequence_len=1,
                hidden_dim=hidden_dim,
                dtype=dtype
            ) * avg_output_len
            pp_communication_latency += prefill_pp_communication_latency_send
            pp_communication_latency += decoding_pp_communication_latency_send
        
        # If not the first stage, need to broadcast (PP)
        if i != 0:
            prefill_pp_communication_latency_broadcast = get_pp_communication_latency_broadcast(
                batch_size=batch_size,
                tp_size=num_gpu,
                sequence_len=avg_input_len,
                hidden_dim=hidden_dim,
                p2p_bandwidth=p2p_bandwidth,
                dtype=dtype
            )
            decoding_pp_communication_latency_broadcast = get_pp_communication_latency_broadcast(
                batch_size=batch_size,
                tp_size=num_gpu,
                sequence_len=1,
                hidden_dim=hidden_dim,
                p2p_bandwidth=p2p_bandwidth,
                dtype=dtype
            ) * avg_output_len
            pp_communication_latency += prefill_pp_communication_latency_broadcast
            pp_communication_latency += decoding_pp_communication_latency_broadcast

        # TP communication
        prefill_tp_communication_latency = get_tp_communication_latency_per_layer(
            tp_size=num_gpu,
            batch_size=batch_size,
            sequence_len=avg_input_len,
            hidden_dim=hidden_dim,
            p2p_bandwidth=p2p_bandwidth,
            dtype=dtype
        ) * layer_count
        decoding_tp_communication_latency = get_tp_communication_latency_per_layer(
            tp_size=num_gpu,
            batch_size=batch_size,
            sequence_len=1,
            hidden_dim=hidden_dim,
            p2p_bandwidth=p2p_bandwidth,
            dtype=dtype
        ) * avg_output_len

        prefill_compute_logit_tp_communication_latency = get_tp_communication_latency_compute_logit(
            tp_size=num_gpu,
            batch_size=batch_size,
            sequence_len=avg_input_len,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            p2p_bandwidth=p2p_bandwidth,
            dtype=dtype
        ) if processed_layers == total_num_layers else 0
        decoding_compute_logit_tp_communication_latency = get_tp_communication_latency_compute_logit(
            tp_size=num_gpu,
            batch_size=batch_size,
            sequence_len=1,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            p2p_bandwidth=p2p_bandwidth,
            dtype=dtype
        ) * avg_output_len if processed_layers == total_num_layers else 0

        tp_communication_latency += prefill_tp_communication_latency + prefill_compute_logit_tp_communication_latency
        tp_communication_latency += decoding_tp_communication_latency + decoding_compute_logit_tp_communication_latency

        logging.debug(f"Node Type: {node_type}, Layers: {layer_count}, Prefill Latency: {prefill_latencies[-1]:.2f}ms, Decoding Latency: {decoding_latencies[-1]:.2f}ms")
        logging.debug(f"    Logit Latency: Prefill {prefill_logit_latency:.2f}ms, Decoding {decoding_logit_latency:.2f}ms")
    
    total_computation_latency = sum(prefill_latencies)
    total_computation_latency += sum(decoding_latencies)
    total_communication_latency = pp_communication_latency + tp_communication_latency

    return total_computation_latency + total_communication_latency


def get_global_batch_size(
    avg_input_len: int,
    avg_output_len: int,
    max_model_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    total_num_layers: int,
    vocab_size: int,
    intermediate_dim: int,
    gpu_mem_utilization: float,
    node_layer_comb: List[tuple[str, str, int]], # node type, az, containing layer count,
    dtype: torch.dtype = torch.float16,
    block_size: int = 16,
    head_dim: int = None,
    tp_sizes: List[int] = None,
):
    global_batch_sizes = []
    cumulative_layers = 0  # Track cumulative layer count

    # Use explicit head_dim if provided (e.g. Qwen3 where head_dim != hidden_dim // num_heads),
    # otherwise fall back to hidden_dim // num_attention_head
    effective_head_dim = head_dim if head_dim is not None else (hidden_dim // num_attention_head)
    dim_kv_head = effective_head_dim * num_kv_cache_head

    for i, (node_type, az, layer_count) in enumerate(node_layer_comb):
        gpu_type = INSTANCE_SPEC[node_type]["gpu_type"]
        num_gpu = tp_sizes[i] if tp_sizes else INSTANCE_SPEC[node_type]["gpu_count"]
        GPU_SPEC_info = GPU_SPEC[gpu_type]
        memory_size_per_gpu= GPU_SPEC_info["memory_size"] * 10**6 # MB to Bytes

        # Get total memory available
        total_memory_available = memory_size_per_gpu * num_gpu * gpu_mem_utilization

        # Get model weight memory
        # Special handling is needed here for the first or last layer where the lm_head weight is included
        model_weight_memory = get_memory_size_decoder_layer_weight_bytes(
            hidden_dim=hidden_dim,
            num_attention_head=num_attention_head,
            num_key_value_head=num_kv_cache_head,
            intermediate_dim=intermediate_dim,
            dtype=dtype,
            head_dim=head_dim,
        ) * layer_count  # Multiply by layer count for this node
        
        # Add embedding weight for first node
        if i == 0:  # First node has embedding
            model_weight_memory += get_memory_size_embedding_or_lm_head_weight_bytes(
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                dtype=dtype
            )
        
        # Check if this node contains the last layer of the model
        cumulative_layers += layer_count
        if cumulative_layers == total_num_layers:  # This node has the actual last layer
            model_weight_memory += get_memory_size_embedding_or_lm_head_weight_bytes(
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                dtype=dtype
            )


        forwarding_memory = get_forwarding_memory(
            max_model_len=max_model_len,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dtype=dtype
        )
        available_kv_cache_memory = total_memory_available - model_weight_memory - forwarding_memory
        kv_memory_needed_per_layer = 2 * max_model_len * dim_kv_head * dtype.itemsize

        if available_kv_cache_memory <= kv_memory_needed_per_layer * layer_count:
            global_batch_sizes.append(0)
            # When KV cache cannot be allocated even for a single request,
            # there is no need to continue the remaining loop iterations.
            # This pipeline is an infeasible pipeline.
            logging.debug(f"Node Type {node_type} with {layer_count} layers cannot fit KV cache even for batch size 1.")
            logging.debug(f"Required Minimum KV Cache memory : {kv_memory_needed_per_layer * layer_count / (1000**3):.2f} GB")
            logging.debug(f"Total Memory Available ({node_type}, {layer_count}): {total_memory_available / (1000**3):.2f} GB")
            logging.debug(f"Model Weight Memory ({node_type}, {layer_count}): {model_weight_memory / (1000**3):.2f} GB")
            logging.debug(f"Forwarding Memory ({node_type}, {layer_count}): {forwarding_memory / (1000**3):.2f} GB")
            logging.debug(f"Available KV Cache Memory ({node_type}, {layer_count}): {available_kv_cache_memory / (1000**3):.2f} GB")
            logging.debug(f"KV Cache Memory when one time max model len forwarding ({node_type}, {layer_count}): {kv_memory_needed_per_layer * layer_count / (1000**3):.2f} GB")
            break

        global_batch_size = (
            available_kv_cache_memory //
            (2 * (avg_input_len + avg_output_len) * dim_kv_head * layer_count * dtype.itemsize)
        )

        global_batch_sizes.append(global_batch_size)

        logging.debug(f"Total Memory Available ({node_type}, {layer_count}): {total_memory_available / (1000**3):.2f} GB")
        logging.debug(f"Model Weight Memory ({node_type}, {layer_count}): {model_weight_memory / (1000**3):.2f} GB")
        logging.debug(f"Forwarding Memory ({node_type}, {layer_count}): {forwarding_memory / (1000**3):.2f} GB")
        logging.debug(f"Available KV Cache Memory ({node_type}, {layer_count}): {available_kv_cache_memory / (1000**3):.2f} GB")
        logging.debug(f"KV Cache Memory when one time max model len forwarding ({node_type}, {layer_count}): {kv_memory_needed_per_layer * layer_count / (1000**3):.2f} GB")

    logging.debug(f"Available Global Batch Sizes per Stage: {global_batch_sizes}")

    # min_global_batch_size = global_batch_sizes[0]
    # min_memory_index = 0
    # for i in range(1, len(global_batch_sizes)):
    #     if global_batch_sizes[i] < min_global_batch_size:
    #         min_global_batch_size = global_batch_sizes[i]
    #         min_memory_index = i
    
    # # Calculate the number of blocks
    # num_layers_with_minimum_memory_stage = node_layer_comb[min_memory_index][2]

    # cache_block_size_bytes = num_layers_with_minimum_memory_stage * block_size * 2*dim_kv_head * dtype.itemsize
    # # Need to retrieve the remaining memory amount from the global batch size
    # available_memory = min_global_batch_size * \
    #     (2 * (avg_input_len + avg_output_len) * dim_kv_head * num_layers_with_minimum_memory_stage * dtype.itemsize)
    # available_num_blocks = available_memory // cache_block_size_bytes

    # # The above process ultimately becomes the following, which can be pre-divided (cancellation)
    # available_num_blocks = \
    #     min_global_batch_size * \
    #     ((avg_input_len + avg_output_len) * 2*dim_kv_head * num_layers_with_minimum_memory_stage * dtype.itemsize) \
    #     // (num_layers_with_minimum_memory_stage * block_size * 2*dim_kv_head * dtype.itemsize)

    # As a result, it can be simplified as follows.

    min_global_batch_size = min(global_batch_sizes)

    available_num_blocks = min_global_batch_size * (avg_input_len + avg_output_len) // block_size

    return min_global_batch_size, int(available_num_blocks)

def get_throughput(
    avg_input_len: int,
    avg_output_len: int,
    max_model_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    total_num_layers: int,
    vocab_size: int,
    intermediate_dim: int,
    gpu_mem_utilization: float,
    node_layer_comb: List[tuple[str, str, int]], # node type, az, containing layer count,
    dtype: torch.dtype = torch.float16,
    block_size: int = 16,
    head_dim: int = None,
    tp_sizes: List[int] = None,
    batch_override: tuple = None,  # Optional (global_batch_size, num_blocks) to skip estimation
    detail: bool = False,  # If True, return additional dict with prefill/decode latency breakdown
):
    if batch_override is not None:
        global_batch_size, num_blocks = batch_override
    else:
        global_batch_size, num_blocks = get_global_batch_size(
            avg_input_len=avg_input_len,
            avg_output_len=avg_output_len,
            max_model_len=max_model_len,
            hidden_dim=hidden_dim,
            num_attention_head=num_attention_head,
            num_kv_cache_head=num_kv_cache_head,
            total_num_layers=total_num_layers,
            vocab_size=vocab_size,
            intermediate_dim=intermediate_dim,
            gpu_mem_utilization=gpu_mem_utilization,
            node_layer_comb=node_layer_comb,
            dtype=dtype,
            block_size=block_size,
            head_dim=head_dim,
            tp_sizes=tp_sizes,
        )
    if global_batch_size == 0:
        return OUT_OF_MEMORY, float("inf"), 0 # global_batch_size being 0 means the memory constraint is not satisfied.

    prefill_latencies = []
    decoding_latencies = []

    processed_layers = 0

    # In prefill, model_len acts as the constraint for batch size.
    max_prefill_batch_size = max_model_len // avg_input_len
    pp_size = len(node_layer_comb)
    
    for stage, (node_type, az, layer_count) in enumerate(node_layer_comb):
        gpu_type = INSTANCE_SPEC[node_type]["gpu_type"]
        num_gpu = tp_sizes[stage] if tp_sizes else INSTANCE_SPEC[node_type]["gpu_count"]
        p2p_bandwidth = INSTANCE_SPEC[node_type]["interconnect_bandwidth"]

        processed_layers += layer_count

        # The following two include computation and communication operations.
        prefill_computation_latency = 0
        prefill_pp_communication_latency = 0
        prefill_tp_communication_latency = 0
        decoding_computation_latency = 0
        decoding_pp_communication_latency = 0
        decoding_tp_communication_latency = 0

        compute_logit_prefill_latency = 0 # for debugging
        compute_logit_decoding_latency = 0 # for debugging

        # Need to apply the micro-batch technique.
        max_batch_iteration = global_batch_size // (max_prefill_batch_size * pp_size)
        logging.debug(f"{'=' * 40} Max Batch Iteration : {max_batch_iteration} {'=' * 40}")
        logging.debug(f"Stage {stage} ({node_type}, {az}, {layer_count}):")
        if max_batch_iteration != 0:
            num_max_batch_prefill_inference = max_batch_iteration * pp_size

            logging.debug(f"  Num Max Batch at Prefill: {num_max_batch_prefill_inference}, Batch Size: {max_prefill_batch_size}")

            # Computation latency of prefill (max_batch)
            max_batch_prefill_computation_latency = get_prefill_computation_latency_per_layer(
                gpu_type=gpu_type,
                gpu_count=num_gpu,
                input_len=avg_input_len,
                hidden_dim=hidden_dim,
                num_attention_head=num_attention_head,
                num_kv_cache_head=num_kv_cache_head,
                intermediate_dim=intermediate_dim,
                batch_size=max_prefill_batch_size,
                dtype=dtype,
                head_dim=head_dim,
            ) * layer_count

            # When reaching the last stage, the logit computation latency should also be included.
            max_batch_prefill_compute_logit_latency = 0
            if processed_layers == total_num_layers:
                max_batch_prefill_compute_logit_latency = get_prefill_compute_logit_latency(
                    gpu_type=gpu_type,
                    gpu_count=num_gpu,
                    input_len=avg_input_len,
                    hidden_dim=hidden_dim,
                    batch_size=max_prefill_batch_size,
                    vocab_size=vocab_size,
                    dtype=dtype
                )
                compute_logit_prefill_latency += (max_batch_prefill_compute_logit_latency * num_max_batch_prefill_inference)
            
            # PP Communication Latency of Prefill (max_batch)
            max_batch_prefill_pp_communication_latency = 0
            if stage != len(node_layer_comb) - 1: # If not the last stage, need to send.
                max_batch_prefill_pp_communication_send_latency = get_pp_communication_latency_send_recv(
                    batch_size=max_prefill_batch_size,
                    sequence_len=avg_input_len,
                    hidden_dim=hidden_dim,
                    dtype=dtype,
                    inter_node_latency_ms=None,  # intra region latency
                    inter_node_bandwidth=None  # intra region bandwidth
                )
                max_batch_prefill_pp_communication_latency += max_batch_prefill_pp_communication_send_latency
            
            # TP Communication Latency of Prefill (max_batch)
            max_batch_prefill_tp_communication_latency = get_tp_communication_latency_per_layer(
                tp_size=num_gpu,
                batch_size=max_prefill_batch_size,
                sequence_len=avg_input_len,
                hidden_dim=hidden_dim,
                p2p_bandwidth=p2p_bandwidth,
                dtype=dtype
            ) * layer_count

            max_batch_prefill_logit_allgather_latency = get_tp_communication_latency_compute_logit(
                tp_size=num_gpu,
                batch_size=max_prefill_batch_size,
                sequence_len=avg_input_len,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                p2p_bandwidth=p2p_bandwidth,
                dtype=dtype
            ) if processed_layers == total_num_layers else 0

            prefill_computation_latency += ((max_batch_prefill_computation_latency + max_batch_prefill_compute_logit_latency) * num_max_batch_prefill_inference)
            prefill_pp_communication_latency += (max_batch_prefill_pp_communication_latency * num_max_batch_prefill_inference)
            prefill_tp_communication_latency += ((max_batch_prefill_tp_communication_latency + max_batch_prefill_logit_allgather_latency) * num_max_batch_prefill_inference)

            logging.debug(f"  Prefill Computation Latency (max_batch): {(max_batch_prefill_computation_latency + max_batch_prefill_compute_logit_latency) * num_max_batch_prefill_inference:.2f} ms")
            logging.debug(f"  Prefill PP Communication Latency (max_batch): {max_batch_prefill_pp_communication_latency * num_max_batch_prefill_inference:.2f} ms")
            logging.debug(f"  Prefill TP Communication Latency (max_batch): {(max_batch_prefill_tp_communication_latency + max_batch_prefill_logit_allgather_latency) * num_max_batch_prefill_inference:.2f} ms")

        # Now handle the remaining batches.
        remaining_batch = global_batch_size - max_batch_iteration * max_prefill_batch_size * pp_size
        
        if remaining_batch != 0:
            for i in range(2):
                # When i == 0, the second term
                # When i == 1, the third term
                if i == 0:
                    tmp_iteration = pp_size - remaining_batch % pp_size
                    tmp_batch_size = (remaining_batch - remaining_batch % pp_size) // pp_size
                else:
                    tmp_iteration = remaining_batch % pp_size
                    tmp_batch_size = (remaining_batch - remaining_batch % pp_size) // pp_size + 1

                if tmp_iteration == 0 or tmp_batch_size == 0:
                    # Not needed if there are no iterations to run the batch.
                    continue

                logging.debug(f"  Num Remaining Batch at Prefill :{i} - {tmp_iteration}, Batch Size: {tmp_batch_size}")

                tmp_prefill_computation_latency = get_prefill_computation_latency_per_layer(
                    gpu_type=gpu_type,
                    gpu_count=num_gpu,
                    input_len=avg_input_len,
                    hidden_dim=hidden_dim,
                    num_attention_head=num_attention_head,
                    num_kv_cache_head=num_kv_cache_head,
                    intermediate_dim=intermediate_dim,
                    batch_size=tmp_batch_size,
                    dtype=dtype,
                    head_dim=head_dim,
                ) * layer_count


                tmp_prefill_compute_logit_latency = 0
                if processed_layers == total_num_layers:
                    tmp_prefill_compute_logit_latency = get_prefill_compute_logit_latency(
                        gpu_type=gpu_type,
                        gpu_count=num_gpu,
                        input_len=avg_input_len,
                        hidden_dim=hidden_dim,
                        batch_size=tmp_batch_size,
                        vocab_size=vocab_size,
                        dtype=dtype
                    )
                    compute_logit_prefill_latency += (tmp_prefill_compute_logit_latency * tmp_iteration)

                tmp_prefill_pp_communication_latency = 0
                if stage != len(node_layer_comb) - 1:  # If not the last stage, need to send.
                    tmp_prefill_pp_communication_send_latency = get_pp_communication_latency_send_recv(
                        batch_size=tmp_batch_size,
                        sequence_len=avg_input_len,
                        hidden_dim=hidden_dim,
                        inter_node_latency_ms=None,  # intra region latency
                        inter_node_bandwidth=None,  # intra region bandwidth
                        dtype=dtype
                    )
                    tmp_prefill_pp_communication_latency += tmp_prefill_pp_communication_send_latency
                
                tmp_prefill_tp_communication_latency = get_tp_communication_latency_per_layer(
                    tp_size=num_gpu,
                    batch_size=tmp_batch_size,
                    sequence_len=avg_input_len,
                    hidden_dim=hidden_dim,
                    p2p_bandwidth=p2p_bandwidth,
                    dtype=dtype
                ) * layer_count

                tmp_prefill_logit_allgather_latency = get_tp_communication_latency_compute_logit(
                    tp_size=num_gpu,
                    batch_size=tmp_batch_size,
                    sequence_len=avg_input_len,
                    hidden_dim=hidden_dim,
                    vocab_size=vocab_size,
                    p2p_bandwidth=p2p_bandwidth,
                    dtype=dtype
                ) if processed_layers == total_num_layers else 0

                prefill_computation_latency += ((tmp_prefill_computation_latency + tmp_prefill_compute_logit_latency) * tmp_iteration) # includes logit computation
                prefill_pp_communication_latency += (tmp_prefill_pp_communication_latency * tmp_iteration)
                prefill_tp_communication_latency += ((tmp_prefill_tp_communication_latency + tmp_prefill_logit_allgather_latency) * tmp_iteration)

                # logging for debugging
                logging.debug(f"  Prefill Computation Latency (remaining batch {i}): {(tmp_prefill_computation_latency + tmp_prefill_compute_logit_latency) * tmp_iteration:.2f} ms")
                logging.debug(f"  Prefill PP Communication Latency (remaining batch {i}): {tmp_prefill_pp_communication_latency * tmp_iteration:.2f} ms")
                logging.debug(f"  Prefill TP Communication Latency (remaining batch {i}): {(tmp_prefill_tp_communication_latency + tmp_prefill_logit_allgather_latency) * tmp_iteration:.2f} ms")


        # Now handle the decoding phase
        for i in range(2):
            if i == 0:
                num_iteration = pp_size - global_batch_size % pp_size
                decoding_batch_size = (global_batch_size - global_batch_size % pp_size) // pp_size
            else:
                num_iteration = global_batch_size % pp_size
                decoding_batch_size = (global_batch_size - global_batch_size % pp_size) // pp_size + 1

            if num_iteration == 0 or decoding_batch_size == 0:
                continue

            logging.debug(f"  Num Remaining Batch at Decoding :{i} - {num_iteration}, Batch Size: {decoding_batch_size}")

            tmp_decoding_computation_latency = get_decoding_computation_latency_per_layer(
                gpu_type=gpu_type,
                gpu_count=num_gpu,
                input_len=avg_input_len,
                output_len=avg_output_len,
                hidden_dim=hidden_dim,
                num_attention_head=num_attention_head,
                num_kv_cache_head=num_kv_cache_head,
                intermediate_dim=intermediate_dim,
                batch_size=decoding_batch_size,
                dtype=dtype,
                head_dim=head_dim,
            ) * layer_count

            tmp_decoding_compute_logit_latency = 0
            if processed_layers == total_num_layers:
                tmp_decoding_compute_logit_latency = get_decoding_compute_logit_latency(
                    gpu_type=gpu_type,
                    gpu_count=num_gpu,
                    output_len=avg_output_len,
                    hidden_dim=hidden_dim,
                    batch_size=decoding_batch_size,
                    vocab_size=vocab_size,
                    dtype=dtype
                )
                compute_logit_decoding_latency += (tmp_decoding_compute_logit_latency * num_iteration)

            tmp_decoding_pp_communication_latency = 0
            if stage != len(node_layer_comb) - 1:  # If not the last stage, need to send.
                tmp_decoding_pp_communication_send_latency = get_pp_communication_latency_send_recv(
                    batch_size=decoding_batch_size,
                    sequence_len=1,
                    hidden_dim=hidden_dim,
                    inter_node_latency_ms=None,  # intra region latency
                    inter_node_bandwidth=None,  # intra region bandwidth
                    dtype=dtype
                ) * avg_output_len
                tmp_decoding_pp_communication_latency += tmp_decoding_pp_communication_send_latency

            tmp_decoding_tp_communication_latency = get_tp_communication_latency_per_layer(
                tp_size=num_gpu,
                batch_size=decoding_batch_size,
                sequence_len=1,
                hidden_dim=hidden_dim,
                p2p_bandwidth=p2p_bandwidth,
                dtype=dtype
            ) * layer_count * avg_output_len

            tmp_decoding_logit_allgather_latency = get_tp_communication_latency_compute_logit(
                tp_size=num_gpu,
                batch_size=decoding_batch_size,
                sequence_len=1,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                p2p_bandwidth=p2p_bandwidth,
                dtype=dtype
            ) * avg_output_len if processed_layers == total_num_layers else 0

            decoding_computation_latency += ((tmp_decoding_computation_latency + tmp_decoding_compute_logit_latency) * num_iteration)
            decoding_pp_communication_latency += (tmp_decoding_pp_communication_latency * num_iteration)
            decoding_tp_communication_latency += ((tmp_decoding_tp_communication_latency + tmp_decoding_logit_allgather_latency) * num_iteration)

            logging.debug(f"  Decoding Computation Latency (remaining batch {i}): {(tmp_decoding_computation_latency + tmp_decoding_compute_logit_latency) * num_iteration:.2f} ms")
            logging.debug(f"  Decoding PP Communication Latency (remaining batch {i}): {tmp_decoding_pp_communication_latency * num_iteration:.2f} ms")
            logging.debug(f"  Decoding TP Communication Latency (remaining batch {i}): {(tmp_decoding_tp_communication_latency + tmp_decoding_logit_allgather_latency) * num_iteration:.2f} ms")
        
        # logging for debugging
        logging.debug(f"Stage {stage} ({node_type}, {az}, {layer_count}):")
        logging.debug(f"  Prefill Latency: {prefill_computation_latency + prefill_pp_communication_latency + prefill_tp_communication_latency:.2f} ms")
        logging.debug(f"  Prefill Total Compute Logit Latency: {compute_logit_prefill_latency:.2f} ms") if compute_logit_prefill_latency != 0 else None
        logging.debug(f"    Prefill Computation Latency: {prefill_computation_latency:.2f} ms")
        logging.debug(f"    Prefill PP Communication Latency: {prefill_pp_communication_latency:.2f} ms")
        logging.debug(f"    Prefill TP Communication Latency: {prefill_tp_communication_latency:.2f} ms")
        logging.debug(f"  Decoding Latency: {decoding_computation_latency + decoding_pp_communication_latency + decoding_tp_communication_latency:.2f} ms")
        logging.debug(f"  Decoding Total Compute Logit Latency: {compute_logit_decoding_latency:.2f} ms") if compute_logit_decoding_latency != 0 else None
        logging.debug(f"    Decoding Computation Latency: {decoding_computation_latency:.2f} ms")
        logging.debug(f"    Decoding PP Communication Latency: {decoding_pp_communication_latency:.2f} ms")
        logging.debug(f"    Decoding TP Communication Latency: {decoding_tp_communication_latency:.2f} ms")

        prefill_latencies.append(prefill_computation_latency + prefill_pp_communication_latency + prefill_tp_communication_latency)
        decoding_latencies.append(decoding_computation_latency + decoding_pp_communication_latency + decoding_tp_communication_latency)
    

    logging.debug(f"Prefill latencies per Stage : {prefill_latencies}")
    logging.debug(f"Decoding latencies per Stage : {decoding_latencies}")

    pp_size = len(node_layer_comb)
    max_prefill_latency = max(prefill_latencies)
    max_decoding_latency = max(decoding_latencies)

    # Pipeline bubble correction.
    # K = number of micro-batches (capped at pp_size).
    # When K < PP, a single stage cannot represent full E2E — scale by pp//K.
    # Pipeline filling overhead (one TTFT or TPOT) is always added when K > 1.
    K = min(global_batch_size, pp_size)
    stage_multiplier = pp_size // K
    max_prefill_latency *= stage_multiplier
    max_decoding_latency *= stage_multiplier

    if K > 1:
        num_prefill_mbs = max(int(global_batch_size // max_prefill_batch_size), K)
        per_mb_prefill = [lat / num_prefill_mbs for lat in prefill_latencies]
        per_mb_decode = [lat / K for lat in decoding_latencies]
        ttft = max(per_mb_prefill)
        tpot = max(per_mb_decode) / avg_output_len
        filling_overhead = (K - 1) * max(ttft, tpot)
        if ttft >= tpot:
            max_prefill_latency += filling_overhead
        else:
            max_decoding_latency += filling_overhead
        logging.debug(f"Pipeline bubble (K={K}, PP={pp_size}): stage_multiplier={stage_multiplier}, filling={filling_overhead:.2f}ms")

    total_latency_per_global_batch = max_prefill_latency + max_decoding_latency

    throughput = global_batch_size / (total_latency_per_global_batch / 1000)  # ms to seconds

    logging.debug(f"Global Batch Size: {global_batch_size}")
    logging.debug(f"Number of blocks : {num_blocks}")
    logging.debug(f"End to End Latency per Global Batch: {total_latency_per_global_batch:.2f} ms")
    logging.debug(f"Throughput: {throughput:.2f} reqs/s")

    if detail:
        return throughput, total_latency_per_global_batch, num_blocks, {
            "prefill_latency_ms": max_prefill_latency,
            "decode_latency_ms": max_decoding_latency,
        }
    return throughput, total_latency_per_global_batch, num_blocks


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    from transformers import AutoConfig
    model_name = "Qwen/Qwen3-32B"
    model_config = AutoConfig.from_pretrained(model_name)

    config = {
        "expected_input_len": 763,  # input sequence length
        "expected_output_len": 232,  # output sequence length
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_hidden_layers,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": getattr(model_config, "num_key_value_heads", model_config.num_attention_heads),
        "intermediate_size": model_config.intermediate_size,
        "vocab_size": model_config.vocab_size,
        "max_position_embeddings": model_config.max_position_embeddings,
        "dtype": torch.float16,
        "max_model_len": 8192,
        "gpu_mem_utilization": 0.85,
    }

    # Unit test for model weights

    decoder_layer_weight_bytes = get_memory_size_decoder_layer_weight_bytes(
        hidden_dim=config["hidden_size"],
        num_attention_head=config["num_attention_heads"],
        num_key_value_head=config["num_key_value_heads"],
        intermediate_dim=config["intermediate_size"],
        dtype=config["dtype"]
    )

    embedding_weight_bytes = get_memory_size_embedding_or_lm_head_weight_bytes(
        hidden_dim=config["hidden_size"],
        vocab_size=config["vocab_size"],
        dtype=config["dtype"]
    )

    lm_head_weight_bytes = get_memory_size_embedding_or_lm_head_weight_bytes(
        hidden_dim=config["hidden_size"],
        vocab_size=config["vocab_size"],
        dtype=config["dtype"]
    )

    total_model_mem = (
        decoder_layer_weight_bytes * config["num_layers"] + 
        embedding_weight_bytes + 
        lm_head_weight_bytes
    )

    logging.debug(f"Decoder Layer Weight Bytes: {decoder_layer_weight_bytes / (10**6):.2f} MB")
    logging.debug(f"Embedding Weight Bytes: {embedding_weight_bytes / (10**6):.2f} MB")
    logging.debug(f"LM Head Weight Bytes: {lm_head_weight_bytes / (10**6):.2f} MB")
    logging.debug(f"Total Model Memory: {total_model_mem / (10**6):.2f} MB")

    logging.debug(f"\n")

    # Unit Test : Get Global Batch Size
    
    node_layer_comb = [
        ("g5.12xlarge", "dummy-az", 64),
    ]

    global_batch_size = get_global_batch_size(
        avg_input_len=config["expected_input_len"],
        avg_output_len=config["expected_output_len"],
        max_model_len=config["max_model_len"],
        hidden_dim=config["hidden_size"],
        num_attention_head=config["num_attention_heads"],
        num_kv_cache_head=config["num_key_value_heads"],
        vocab_size=config["vocab_size"],
        intermediate_dim=config["intermediate_size"],
        gpu_mem_utilization=config["gpu_mem_utilization"],
        total_num_layers=config["num_layers"],
        node_layer_comb=node_layer_comb,
        dtype=config["dtype"]
    )

    logging.debug(f"Global Batch Size: {global_batch_size}")

    logging.debug(f"\n")


    # Unit Test : Get Throughput
    throughput, total_latency, num_blocks = get_throughput(
        avg_input_len=config["expected_input_len"],
        avg_output_len=config["expected_output_len"],
        max_model_len=config["max_model_len"],
        hidden_dim=config["hidden_size"],
        num_attention_head=config["num_attention_heads"],
        num_kv_cache_head=config["num_key_value_heads"],
        vocab_size=config["vocab_size"],
        intermediate_dim=config["intermediate_size"],
        gpu_mem_utilization=config["gpu_mem_utilization"],
        total_num_layers=config["num_layers"],
        node_layer_comb=node_layer_comb,
        dtype=config["dtype"]
    )

    # Unit Test : Get Single Request Latency
    latency = get_single_request_latency(
        avg_input_len=config["expected_input_len"],
        avg_output_len=config["expected_output_len"],
        hidden_dim=config["hidden_size"],
        num_attention_head=config["num_attention_heads"],
        num_kv_cache_head=config["num_key_value_heads"],
        total_num_layers=config["num_layers"],
        intermediate_dim=config["intermediate_size"],
        vocab_size=config["vocab_size"],
        node_layer_comb=node_layer_comb,
        dtype=config["dtype"]
    )

    logging.debug(f"Single Request Latency: {latency} ms")