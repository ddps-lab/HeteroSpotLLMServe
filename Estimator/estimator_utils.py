from typing import List, Optional
import torch
import logging
from hardware_specs import GPU_SPEC, INTERCONNECT_SPEC, INSTANCE_SPEC

import sys
import os
# Add parent directory to path for protocols import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocols import OUT_OF_MEMORY



def get_prefill_computation_ops_per_layer(
    input_len: int,
    hidden_dim: int,
    tp_size: int,
    batch_size: int,
    intermediate_dim: Optional[int] = None,
):
    if intermediate_dim is None:
        intermediate_dim = 4 * hidden_dim

    attention_ops_per_layer = (
        4 * batch_size * (input_len**2) * hidden_dim / tp_size +  # 4Bs_in^2*H/D_TP (QK^T + *V)
        8 * batch_size * input_len * (hidden_dim**2) / tp_size    # 8Bs_in*H^2/D_TP (Q,K,V proj + W_O)
    )
    
    ffn_ops_per_layer = (
        24 * batch_size * input_len * (hidden_dim**2) / tp_size   # 24Bs_in*H^2/D_TP (Up/Gate/Down proj)
    )

    return attention_ops_per_layer + ffn_ops_per_layer

def get_decoding_computation_ops_per_layer(
    input_len: int,
    output_len: int,
    hidden_dim: int,
    tp_size: int,
    batch_size: int,
    intermediate_dim: Optional[int] = None,
):
    if intermediate_dim is None:
        intermediate_dim = 4 * hidden_dim

    attention_ops_per_layer = (
        8 * batch_size * output_len * (hidden_dim**2) / tp_size +              # 8Bs_out*H^2/D_TP (Q,K,V proj + W_O)
        4 * batch_size * input_len * output_len * hidden_dim / tp_size +       # 4Bs_in*s_out*H/D_TP (qK^T + softmax*V)
        2 * batch_size * (output_len**2) * hidden_dim / tp_size +             # 2Bs_out^2*H/D_TP (summation term from t)
        2 * batch_size * output_len * hidden_dim / tp_size                    # 2Bs_out*H/D_TP (summation term from t)
    )

    ffn_ops_per_layer = (
        24 * batch_size * output_len * (hidden_dim**2) / tp_size              # 24Bs_out*H^2/D_TP (Up/Gate/Down proj)
    )

    return attention_ops_per_layer + ffn_ops_per_layer

def get_prefill_memory_access_count_per_layer(
    input_len: int,
    hidden_dim: int,
    tp_size: int,
    batch_size: int,
    intermediate_dim: Optional[int] = None,
):
    if intermediate_dim is None:
        intermediate_dim = 4 * hidden_dim

    attention_memory_access_count_per_layer = (
        2 * batch_size * input_len * hidden_dim +                             # 2Bs_in*H (input read + output write)
        2 * batch_size * (input_len**2) +                                     # 2Bs_in^2 (attention scores)
        4 * (hidden_dim**2) / tp_size +                                       # 4H^2/D_TP (Q,K,V,O weights)
        8 * batch_size * input_len * hidden_dim / tp_size                     # 8Bs_in*H/D_TP (Q,K,V,O activations)
    )
    
    ffn_memory_access_count_per_layer = (
        2 * batch_size * input_len * hidden_dim +                             # 2Bs_in*H (input read + output write)
        12 * (hidden_dim**2) / tp_size +                                      # 12H^2/D_TP (Up/Gate/Down weights)
        12 * batch_size * input_len * hidden_dim / tp_size                    # 12Bs_in*H/D_TP (intermediate activations)
    )

    return attention_memory_access_count_per_layer + ffn_memory_access_count_per_layer

def get_decoding_memory_access_count_per_layer(
    input_len: int,
    output_len: int,
    hidden_dim: int,
    tp_size: int,
    batch_size: int,
    intermediate_dim: Optional[int] = None,
):
    if intermediate_dim is None:
        intermediate_dim = 4 * hidden_dim

    attention_memory_access_count_per_layer = (
        2 * batch_size * output_len * hidden_dim +  # 2Bs_out*H (attention 부분만)
        4 * output_len * (hidden_dim**2) / tp_size +  # 4s_out*H^2/D_TP
        6 * batch_size * output_len * hidden_dim / tp_size +  # 6Bs_out*H/D_TP
        2 * batch_size * input_len * output_len +  # 2Bs_in*s_out
        2 * batch_size * input_len * output_len * hidden_dim / tp_size +  # 2Bs_in*s_out*H/D_TP
        (batch_size + batch_size * hidden_dim / tp_size) * (output_len**2 + output_len)  # (B + BH/D_TP)(s_out^2 + s_out)
    )

    ffn_memory_access_count_per_layer = (
        2 * batch_size * hidden_dim + 12 * (hidden_dim**2) / tp_size + 
        12 * batch_size * hidden_dim / tp_size
    )

    return attention_memory_access_count_per_layer + ffn_memory_access_count_per_layer


def get_tp_communication_latency_per_layer(
    tp_size: int,
    batch_size: int,
    sequence_len: int,
    hidden_dim: int,
    p2p_bandwidth: float, # 단위는 Bytes/s
    dtype: torch.dtype = torch.float16,
    p2p_latency_ms: Optional[float] = None,
):
    if p2p_latency_ms is None:
        # intra p2p latency 는 너무 빨라서 거의 없다고 가정
        p2p_latency_ms = 0

    element_size = dtype.itemsize  # dtype 의 item size
    latency_per_layer_ms = (batch_size * sequence_len * hidden_dim * element_size) / (tp_size * p2p_bandwidth / 1000)  # Bytes/s to ms
    latency_per_layer_ms += p2p_latency_ms
    latency_per_layer_ms *= 4 * (tp_size - 1)

    return latency_per_layer_ms


def get_pp_communication_latency(
    batch_size: int,
    sequence_len: int,
    hidden_dim: int,
    tp_size_at_receiver: int,
    p2p_bandwidth: float,
    p2p_latency_ms: Optional[float] = None,
    dtype: torch.dtype = torch.float16,
    inter_node_latency_ms: Optional[float] = None,
    inter_node_bandwidth: Optional[float] = None,  # 단위는 Bytes/s
):
    """
    P2P Latency 와 Bandwidth 는 receiver 기준으로 주어져야 한다.
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
    inter_node_bandwidth: Optional[float] = None,  # 단위는 Bytes/s
    dtype: torch.dtype = torch.float16,
):
    
    if inter_node_latency_ms is None:
        # inter node latency 는 기본적으로 (intra region) 2ms 라고 가정
        inter_node_latency_ms = 2
    if inter_node_bandwidth is None:
        # inter node bandwidth 는 기본적으로 (intra region) 5GB/s 라고 가정
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
    p2p_bandwidth: float,  # 단위는 Bytes/s
    p2p_latency_ms: Optional[float] = None,
    dtype: torch.dtype = torch.float16,
):
    """
    이 함수는 PP 통신에서 전송받는 입장 (next stage) 에서 호출되어야 함
    """

    if p2p_latency_ms is None:
        # intra p2p latency 는 너무 빨라서 거의 없다고 가정
        p2p_latency_ms = 0

    element_size = dtype.itemsize

    # tp size 가 1보다 큰 경우 broadcast 발생
    if tp_size > 1:
        latency_broadcast = (batch_size * sequence_len * hidden_dim * element_size) / (p2p_bandwidth / 1000)  # Bytes/s to ms
        latency_broadcast += p2p_latency_ms
        latency_broadcast *= (tp_size - 1) # 자신을 제외한 다른 노드에게 순차적 전송 가정
    else:
        latency_broadcast = 0
    
    return latency_broadcast

def get_prefill_computation_latency_per_layer(
    gpu_type: str,
    gpu_count: int,
    input_len: int,
    hidden_dim: int,
    batch_size: int,
    intermediate_dim: Optional[int] = None,
    dtype: torch.dtype = torch.float16,
):
    if intermediate_dim is None:
        intermediate_dim = 4 * hidden_dim

    computation_ops_per_layer = get_prefill_computation_ops_per_layer(
        input_len, hidden_dim, gpu_count, batch_size, intermediate_dim
    )
    computation_memory_access_per_layer = get_prefill_memory_access_count_per_layer(
        input_len, hidden_dim, gpu_count, batch_size, intermediate_dim
    ) * dtype.itemsize  # Bytes 단위로 변환
    arithmetic_intensity = computation_ops_per_layer / computation_memory_access_per_layer
    
    GPU_SPEC_info = GPU_SPEC[gpu_type]
    flops = GPU_SPEC_info["FLOPS"]
    memory_bandwidth = GPU_SPEC_info["memory_bandwidth"]
    device_arithmetic_intensity = flops / memory_bandwidth

    if arithmetic_intensity < device_arithmetic_intensity:
        # Memory bound
        latency_per_layer = computation_memory_access_per_layer / (memory_bandwidth / 1000) # Bytes/s to ms
    else:
        # Compute bound
        latency_per_layer = computation_ops_per_layer / (flops / 1000)  # FLOPS to ms

    return latency_per_layer

def get_decodeing_computation_latency_per_layer(
    gpu_type: str,
    gpu_count: int,
    input_len: int,
    output_len: int,
    hidden_dim: int,
    batch_size: int,
    intermediate_dim: Optional[int] = None,
    dtype: torch.dtype = torch.float16,
):
    if intermediate_dim is None:
        intermediate_dim = 4 * hidden_dim

    computation_ops_per_layer = get_decoding_computation_ops_per_layer(
        input_len, output_len, hidden_dim, gpu_count, batch_size, intermediate_dim
    )
    computation_memory_access_per_layer = get_decoding_memory_access_count_per_layer(
        input_len, output_len, hidden_dim, gpu_count, batch_size, intermediate_dim
    ) * dtype.itemsize  # Bytes 단위로 변환
    arithmetic_intensity = computation_ops_per_layer / computation_memory_access_per_layer
    
    GPU_SPEC_info = GPU_SPEC[gpu_type]
    flops = GPU_SPEC_info["FLOPS"]
    memory_bandwidth = GPU_SPEC_info["memory_bandwidth"]
    device_arithmetic_intensity = flops / memory_bandwidth

    if arithmetic_intensity < device_arithmetic_intensity:
        # Memory bound
        latency_per_layer = computation_memory_access_per_layer / (memory_bandwidth / 1000) # Bytes/s to ms
    else:
        # Compute bound
        latency_per_layer = computation_ops_per_layer / (flops / 1000)  # FLOPS to ms

    return latency_per_layer

def get_forwarding_memory(
    max_model_len: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.float16,
    intermediate_dim: Optional[int] = None,
):
    if intermediate_dim is None:
        intermediate_dim = 4 * hidden_dim

    forwarding_memory = (
        3 * # up / gate / output
        max_model_len * # maximum input
        intermediate_dim * # intermediate dimension
        dtype.itemsize # element size
    ) # Unit : Bytes

    return forwarding_memory


def get_global_batch_size(
    avg_input_len: int,
    avg_output_len: int,
    max_model_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    total_layer_num: int,
    total_model_mem: int,
    gpu_mem_utilization: float,
    node_layer_comb: List[tuple[str, str, int]], # node type, az, containing layer count,
    dtype: torch.dtype = torch.float16,
):
    global_batch_sizes = []

    for node_type, az, layer_count in node_layer_comb:
        gpu_type = INSTANCE_SPEC[node_type]["gpu_type"]
        num_gpu = INSTANCE_SPEC[node_type]["gpu_count"]
        GPU_SPEC_info = GPU_SPEC[gpu_type]
        memory_size_per_gpu= GPU_SPEC_info["memory_size"] * 10**6 # MB to Bytes
        memory_size = memory_size_per_gpu * num_gpu

        # Get total memory available
        total_memory_available = memory_size * gpu_mem_utilization
        # Get model weight memory
        model_weight_memory = total_model_mem * (layer_count / total_layer_num)
        forwarding_memory = get_forwarding_memory(
            max_model_len,
            hidden_dim,
            dtype=dtype
        )
        available_kv_cache_memory = total_memory_available - model_weight_memory - forwarding_memory
        kv_memory_needed_per_layer = 2 * max_model_len * hidden_dim * dtype.itemsize

        if available_kv_cache_memory <= kv_memory_needed_per_layer * layer_count:
            global_batch_sizes.append(0)
            # 이미 KV cache 를 하나의 request 에 대해서도 할당할 수가 없을때
            # 굳이 남은 for 문을 돌릴 필요가 없다.
            # 해당 파이프라인을 불가능한 파이프라인이기 때문에.
            break

        global_batch_size = (
            available_kv_cache_memory //
            (2 * (avg_input_len + avg_output_len) * (hidden_dim // num_attention_head * num_kv_cache_head) * layer_count * dtype.itemsize)
        )

        global_batch_sizes.append(global_batch_size)

        logging.debug(f"Total Memory Available ({node_type}, {layer_count}): {total_memory_available / (1000**3):.2f} GB")
        logging.debug(f"Model Weight Memory ({node_type}, {layer_count}): {model_weight_memory / (1000**3):.2f} GB")
        logging.debug(f"Forwarding Memory ({node_type}, {layer_count}): {forwarding_memory / (1000**3):.2f} GB")
        logging.debug(f"Available KV Cache Memory ({node_type}, {layer_count}): {available_kv_cache_memory / (1000**3):.2f} GB")
        logging.debug(f"KV Cache Memory when one time max model len forwarding ({node_type}, {layer_count}): {kv_memory_needed_per_layer * layer_count / (1000**3):.2f} GB")

    logging.debug(f"Available Global Batch Sizes per Stage: {global_batch_sizes}")

    return min(global_batch_sizes)

def get_throughput(
    avg_input_len: int,
    avg_output_len: int,
    max_model_len: int,
    hidden_dim: int,
    num_attention_head: int,
    num_kv_cache_head: int,
    total_layer_num: int,
    total_model_mem: int,
    gpu_mem_utilization: float,
    node_layer_comb: List[tuple[str, str, int]], # node type, az, containing layer count,
    dtype: torch.dtype = torch.float16,
):
    global_batch_size = get_global_batch_size(
        avg_input_len=avg_input_len,
        avg_output_len=avg_output_len,
        max_model_len=max_model_len,
        hidden_dim=hidden_dim,
        num_attention_head=num_attention_head,
        num_kv_cache_head=num_kv_cache_head,
        total_layer_num=total_layer_num,
        total_model_mem=total_model_mem,
        gpu_mem_utilization=gpu_mem_utilization,
        node_layer_comb=node_layer_comb,
        dtype=dtype
    )
    if global_batch_size == 0:
        return OUT_OF_MEMORY # global_batch_size 가 0이라는 것은 메모리 제약조건을 충족하지 못한다는 뜻.

    prefill_latencies = []
    decoding_latencies = []

    # prefill 에서의 batch size 는 model_len 이 constraint 로 작용한다.
    max_prefill_batch_size = max_model_len // avg_input_len
    pp_size = len(node_layer_comb)
    
    for stage, (node_type, az, layer_count) in enumerate(node_layer_comb):
        gpu_type = INSTANCE_SPEC[node_type]["gpu_type"]
        num_gpu = INSTANCE_SPEC[node_type]["gpu_count"]
        p2p_bandwidth = INTERCONNECT_SPEC[INSTANCE_SPEC[node_type]["interconnect"]]["bandwidth"]

        # 아래 두 개는 Computation 과 Communication 연산을 포함한다.
        prefill_computation_latency = 0
        prefill_pp_communication_latency = 0
        prefill_tp_communication_latency = 0
        decoding_computation_latency = 0
        decoding_pp_communication_latency = 0
        decoding_tp_communication_latency = 0

        # micro-batch 기법을 적용해야함.
        max_batch_iteration = global_batch_size // (max_prefill_batch_size * pp_size)
        if max_batch_iteration != 0:
            num_max_batch_prefill_inference = max_batch_iteration * pp_size

            # Computation latency of prefill (max_batch)
            max_batch_prefill_computation_latency = get_prefill_computation_latency_per_layer(
                gpu_type=gpu_type,
                gpu_count=num_gpu,
                input_len=avg_input_len,
                hidden_dim=hidden_dim,
                batch_size=max_prefill_batch_size,
                dtype=dtype
            ) * layer_count
            
            # PP Communication Latency of Prefill (max_batch)
            max_batch_prefill_pp_communication_latency = 0
            if stage != len(node_layer_comb) - 1: # 마지막 Stage 가 아니면 send 를 해야함.
                max_batch_prefill_pp_communication_send_latency = get_pp_communication_latency_send_recv(
                    batch_size=max_prefill_batch_size,
                    sequence_len=avg_input_len,
                    hidden_dim=hidden_dim,
                    dtype=dtype,
                    inter_node_latency_ms=None,  # intra region latency
                    inter_node_bandwidth=None  # intra region bandwidth
                )
                max_batch_prefill_pp_communication_latency += max_batch_prefill_pp_communication_send_latency
            if stage != 0 and num_gpu > 1: # 첫 번째 Stage 가 아니고 tp size 가 1보다 크면 broadcast 를 해야함.
                max_batch_prefill_pp_communication_broadcast_latency = get_pp_communication_latency_broadcast(
                    batch_size=max_prefill_batch_size,
                    sequence_len=avg_input_len,
                    hidden_dim=hidden_dim,
                    tp_size=num_gpu,
                    p2p_bandwidth=p2p_bandwidth,
                    p2p_latency_ms=None,
                    dtype=dtype
                )
                max_batch_prefill_pp_communication_latency += max_batch_prefill_pp_communication_broadcast_latency
            
            # TP Communication Latency of Prefill (max_batch)
            max_batch_prefill_tp_communication_latency = get_tp_communication_latency_per_layer(
                tp_size=num_gpu,
                batch_size=max_prefill_batch_size,
                sequence_len=avg_input_len,
                hidden_dim=hidden_dim,
                p2p_bandwidth=p2p_bandwidth,
                dtype=dtype
            ) * layer_count

            prefill_computation_latency += (max_batch_prefill_computation_latency * num_max_batch_prefill_inference)
            prefill_pp_communication_latency += (max_batch_prefill_pp_communication_latency * num_max_batch_prefill_inference)
            prefill_tp_communication_latency += (max_batch_prefill_tp_communication_latency * num_max_batch_prefill_inference)

        # 이제 나머지 처리해야 함.
        remaining_batch = global_batch_size - max_batch_iteration * max_prefill_batch_size * pp_size
        
        if remaining_batch != 0:
            for i in range(2):
                # i == 0 일 때 두 번째 항
                # i == 1 일 때 세 번째 항
                if i == 0:
                    tmp_iteration = pp_size - remaining_batch % pp_size
                    tmp_batch_size = (remaining_batch - remaining_batch % pp_size) // pp_size
                else:
                    tmp_iteration = remaining_batch % pp_size
                    tmp_batch_size = (remaining_batch - remaining_batch % pp_size) // pp_size + 1

                if tmp_iteration == 0 or tmp_batch_size == 0:
                    # batch 를 돌릴 iteration 이 없으면 필요 없음.
                    continue

                tmp_prefill_computation_latency = get_prefill_computation_latency_per_layer(
                    gpu_type=gpu_type,
                    gpu_count=num_gpu,
                    input_len=avg_input_len,
                    hidden_dim=hidden_dim,
                    batch_size=tmp_batch_size,
                    dtype=dtype
                ) * layer_count

                tmp_prefill_pp_communication_latency = 0
                if stage != len(node_layer_comb) - 1:  # 마지막 Stage 가 아니면 send 를 해야함.
                    tmp_prefill_pp_communication_send_latency = get_pp_communication_latency_send_recv(
                        batch_size=tmp_batch_size,
                        sequence_len=avg_input_len,
                        hidden_dim=hidden_dim,
                        inter_node_latency_ms=None,  # intra region latency
                        inter_node_bandwidth=None,  # intra region bandwidth
                        dtype=dtype
                    )
                    tmp_prefill_pp_communication_latency += tmp_prefill_pp_communication_send_latency
                if stage != 0 and num_gpu > 1:  # 첫 번째 Stage 가 아니고 tp size 가 1보다 크면 broadcast 를 해야함.
                    tmp_prefill_pp_communication_broadcast_latency = get_pp_communication_latency_broadcast(
                        batch_size=tmp_batch_size,
                        sequence_len=avg_input_len,
                        hidden_dim=hidden_dim,
                        tp_size=num_gpu,
                        p2p_bandwidth=p2p_bandwidth,
                        p2p_latency_ms=None,
                        dtype=dtype
                    )
                    tmp_prefill_pp_communication_latency += tmp_prefill_pp_communication_broadcast_latency
                
                tmp_prefill_tp_communication_latency = get_tp_communication_latency_per_layer(
                    tp_size=num_gpu,
                    batch_size=tmp_batch_size,
                    sequence_len=avg_input_len,
                    hidden_dim=hidden_dim,
                    p2p_bandwidth=p2p_bandwidth,
                    dtype=dtype
                ) * layer_count

                prefill_computation_latency += (tmp_prefill_computation_latency * tmp_iteration)
                prefill_pp_communication_latency += (tmp_prefill_pp_communication_latency * tmp_iteration)
                prefill_tp_communication_latency += (tmp_prefill_tp_communication_latency * tmp_iteration)
            


        # 이제 디코딩 처리하자
        for i in range(2):
            if i == 0:
                num_iteration = pp_size - global_batch_size % pp_size
                decoding_batch_size = (global_batch_size - global_batch_size % pp_size) // pp_size
            else:
                num_iteration = global_batch_size % pp_size
                decoding_batch_size = (global_batch_size - global_batch_size % pp_size) // pp_size + 1

            if num_iteration == 0 or decoding_batch_size == 0:
                continue

            tmp_decoding_computation_latency = get_decodeing_computation_latency_per_layer(
                gpu_type=gpu_type,
                gpu_count=num_gpu,
                input_len=avg_input_len,
                output_len=avg_output_len,
                hidden_dim=hidden_dim,
                batch_size=decoding_batch_size,
                dtype=dtype
            ) * layer_count

            tmp_decoding_pp_communication_latency = 0
            if stage != len(node_layer_comb) - 1:  # 마지막 Stage 가 아니면 send 를 해야함.
                tmp_decoding_pp_communication_send_latency = get_pp_communication_latency_send_recv(
                    batch_size=decoding_batch_size,
                    sequence_len=1,
                    hidden_dim=hidden_dim,
                    inter_node_latency_ms=None,  # intra region latency
                    inter_node_bandwidth=None,  # intra region bandwidth
                    dtype=dtype
                ) * avg_output_len
                tmp_decoding_pp_communication_latency += tmp_decoding_pp_communication_send_latency
            if stage != 0 and num_gpu > 1:  # 첫 번째 Stage 가 아니고 tp size 가 1보다 크면 broadcast 를 해야함.
                tmp_decoding_pp_communication_broadcast_latency = get_pp_communication_latency_broadcast(
                    batch_size=decoding_batch_size,
                    sequence_len=1,
                    hidden_dim=hidden_dim,
                    tp_size=num_gpu,
                    p2p_bandwidth=p2p_bandwidth,
                    p2p_latency_ms=None,
                    dtype=dtype
                ) * avg_output_len
                tmp_decoding_pp_communication_latency += tmp_decoding_pp_communication_broadcast_latency

            tmp_decoding_tp_communication_latency = get_tp_communication_latency_per_layer(
                tp_size=num_gpu,
                batch_size=decoding_batch_size,
                sequence_len=1,
                hidden_dim=hidden_dim,
                p2p_bandwidth=p2p_bandwidth,
                dtype=dtype
            ) * layer_count * avg_output_len

            decoding_computation_latency += (tmp_decoding_computation_latency * num_iteration)
            decoding_pp_communication_latency += (tmp_decoding_pp_communication_latency * num_iteration)
            decoding_tp_communication_latency += (tmp_decoding_tp_communication_latency * num_iteration)

        
        # debugging 용 logging
        logging.debug(f"Stage {stage} ({node_type}, {az}, {layer_count}):")
        logging.debug(f"  Prefill Latency: {prefill_computation_latency + prefill_pp_communication_latency + prefill_tp_communication_latency:.2f} ms")
        logging.debug(f"    Prefill Computation Latency: {prefill_computation_latency:.2f} ms")
        logging.debug(f"    Prefill PP Communication Latency: {prefill_pp_communication_latency:.2f} ms")
        logging.debug(f"    Prefill TP Communication Latency: {prefill_tp_communication_latency:.2f} ms")
        logging.debug(f"  Decoding Latency: {decoding_computation_latency + decoding_pp_communication_latency + decoding_tp_communication_latency:.2f} ms")
        logging.debug(f"    Decoding Computation Latency: {decoding_computation_latency:.2f} ms")
        logging.debug(f"    Decoding PP Communication Latency: {decoding_pp_communication_latency:.2f} ms")
        logging.debug(f"    Decoding TP Communication Latency: {decoding_tp_communication_latency:.2f} ms")

        prefill_latencies.append(prefill_computation_latency + prefill_pp_communication_latency + prefill_tp_communication_latency)
        decoding_latencies.append(decoding_computation_latency + decoding_pp_communication_latency + decoding_tp_communication_latency)
    

    max_prefill_latency = max(prefill_latencies)
    max_decoding_latency = max(decoding_latencies)

    total_latency_per_global_batch = max_prefill_latency + max_decoding_latency

    throughput = global_batch_size / (total_latency_per_global_batch / 1000)  # ms to seconds

    logging.debug(f"Global Batch Size: {global_batch_size}")
    logging.debug(f"System Throughput: {throughput:.2f} reqs/s")

    return throughput


if __name__ == "__main__":
    # Example usage
    avg_input_len = 900
    avg_output_len = 300
    max_model_len = 4096
    dtype = torch.float16
    
    # get config from meta-llama/Llama-3.1-8B
    hidden_dim = 4096
    num_attention_head = 32
    num_kv_cache_head = 8
    total_layer_num = 32
    total_model_mem = 8 * dtype.itemsize * 10**9
    gpu_memory_utilization = 0.9

    node_layer_comb = [
        ("g6.xlarge", "dummy-az", 32),
    ]

    system_throughput = get_throughput(
        avg_input_len=avg_input_len,
        avg_output_len=avg_output_len,
        max_model_len=max_model_len,
        hidden_dim=hidden_dim,
        num_attention_head=num_attention_head,
        num_kv_cache_head=num_kv_cache_head,
        total_layer_num=total_layer_num,
        total_model_mem=total_model_mem,
        gpu_mem_utilization=gpu_memory_utilization,
        node_layer_comb=node_layer_comb,
        dtype=dtype
    )

