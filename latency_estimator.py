from typing import Optional
import torch


# spec 은 float 16 기준
gpu_spec = {
    "T4": {"memory_size": 16, "FLOPS": 65 * 10**12, "memory_bandwidth": 320 * 10**9},
    "A10G": {"memory_size": 24, "FLOPS": 125 * 10**12, "memory_bandwidth": 600 * 10**9},
    "L4": {"memory_size": 24, "FLOPS": 121 * 10**12, "memory_bandwidth": 300 * 10**9},
    "L40S": {"memory_size": 48, "FLOPS": 362 * 10**12, "memory_bandwidth": 864 * 10**9},
    "A100_40GB": {"memory_size": 40, "FLOPS": 312 * 10**12, "memory_bandwidth": 1555 * 10**9},
    "A100_80GB": {"memory_size": 80, "FLOPS": 312 * 10**12, "memory_bandwidth": 2039 * 10**9},
    "H100": {"memory_size": 80, "FLOPS": 1979 * 10**12, "memory_bandwidth": 3350 * 10**9},
    "H200": {"memory_size": 141, "FLOPS": 1979 * 10**12, "memory_bandwidth": 4800 * 10**9},
}

interconnect_spec = {
    "PCIe Gen3x16": {"bandwidth": 32 * 10**9},
    "PCIe Gen4x16": {"bandwidth": 64 * 10**9},
    "NVSwitch 3.0": {"bandwidth": 600 * 10**9},
    "NVSwitch 4.0": {"bandwidth": 900 * 10**9},
}

static_instances_config = {
    "g4dn.xlarge": {"gpu_type": "T4", "gpu_count": 1, "interconnect": "PCIe Gen3x16", "ondemand_price": 0.526},
    "g4dn.12xlarge": {"gpu_type": "T4", "gpu_count": 4, "interconnect": "PCIe Gen3x16", "ondemand_price": 3.912},
    "g4dn.metal": {"gpu_type": "T4", "gpu_count": 8, "interconnect": "PCIe Gen3x16", "ondemand_price": 7.824},
    "g5.xlarge": {"gpu_type": "A10G", "gpu_count": 1, "interconnect": "PCIe Gen4x16", "ondemand_price": 1.006},
    "g5.12xlarge": {"gpu_type": "A10G", "gpu_count": 4, "interconnect": "PCIe Gen4x16", "ondemand_price": 5.672},
    "g5.48xlarge": {"gpu_type": "A10G", "gpu_count": 8, "interconnect": "PCIe Gen4x16", "ondemand_price": 16.288},
    "g6.xlarge": {"gpu_type": "L4", "gpu_count": 1, "interconnect": "PCIe Gen4x16", "ondemand_price": 0.805},
    "g6.12xlarge": {"gpu_type": "L4", "gpu_count": 4, "interconnect": "PCIe Gen4x16", "ondemand_price": 4.602},
    "g6.48xlarge": {"gpu_type": "L4", "gpu_count": 8, "interconnect": "PCIe Gen4x16", "ondemand_price": 13.35},
    "g6e.xlarge": {"gpu_type": "L40S", "gpu_count": 1, "interconnect": "PCIe Gen4x16", "ondemand_price": 1.861},
    "g6e.12xlarge": {"gpu_type": "L40S", "gpu_count": 4, "interconnect": "PCIe Gen4x16", "ondemand_price": 10.493},
    "g6e.48xlarge": {"gpu_type": "L40S", "gpu_count": 8, "interconnect": "PCIe Gen4x16", "ondemand_price": 30.131},
    "p4d.24xlarge": {"gpu_type": "A100_40GB", "gpu_count": 8, "interconnect": "NVSwitch 3.0", "ondemand_price": 32.773},
    "p4de.24xlarge": {"gpu_type": "A100_80GB", "gpu_count": 8, "interconnect": "NVSwitch 3.0", "ondemand_price": 40.966},
    "p5.48xlarge": {"gpu_type": "H100", "gpu_count": 8, "interconnect": "NVSwitch 4.0", "ondemand_price": 98.320},
    "p5e.48xlarge": {"gpu_type": "H200", "gpu_count": 8, "interconnect": "NVSwitch 4.0", "ondemand_price": 84.800}, # 온디맨드 가격이 존재하지 않음.
    "p5en.48xlarge": {"gpu_type": "H200", "gpu_count": 8, "interconnect": "NVSwitch 4.0", "ondemand_price": 84.800},
}


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
    tp_size: int,
    p2p_bandwidth: float,
    p2p_latency_ms: Optional[float] = None,
    dtype: torch.dtype = torch.float16,
    inter_node_latency_ms: Optional[float] = None,
    inter_node_bandwidth: Optional[float] = None,  # 단위는 Bytes/s
):
    """
    P2P Latency 와 Bandwidth 는 receiver 기준으로 주어져야 한다.
    """
    if inter_node_latency_ms is None:
        # inter node latency 는 기본적으로 (intra region) 2ms 라고 가정
        inter_node_latency_ms = 2
    if inter_node_bandwidth is None:
        # inter node bandwidth 는 기본적으로 (intra region) 5GB/s 라고 가정
        inter_node_bandwidth = 5 * 10**9  # 5GB/s
    if p2p_latency_ms is None:
        # intra p2p latency 는 너무 빨라서 거의 없다고 가정
        p2p_latency_ms = 0

    element_size = dtype.itemsize

    latency_send_recv = (batch_size * sequence_len * hidden_dim * element_size) / (inter_node_bandwidth / 1000)  # Bytes/s to ms
    latency_send_recv += inter_node_latency_ms

    # tp size 가 1보다 큰 경우 broadcast 발생
    if tp_size > 1:
        latency_broadcast = (batch_size * sequence_len * hidden_dim * element_size) / (p2p_bandwidth / 1000)  # Bytes/s to ms
        latency_broadcast += p2p_latency_ms
        latency_broadcast *= (tp_size - 1) # 자신을 제외한 다른 노드에게 순차적 전송 가정
    else:
        latency_broadcast = 0
    
    total_latency = latency_send_recv + latency_broadcast
    
    return total_latency