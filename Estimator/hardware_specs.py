# spec 은 float 16 기준
GPU_SPEC = {
    "T4": {"memory_size": 16000, "FLOPS": 65 * 10**12, "memory_bandwidth": 320 * 10**9},
    "A10G": {"memory_size": 24000, "FLOPS": 125 * 10**12, "memory_bandwidth": 600 * 10**9},
    "L4": {"memory_size": 24000, "FLOPS": 121 * 10**12, "memory_bandwidth": 300 * 10**9},
    "L40S": {"memory_size": 48000, "FLOPS": 362 * 10**12, "memory_bandwidth": 864 * 10**9},
    "A100_40GB": {"memory_size": 40000, "FLOPS": 312 * 10**12, "memory_bandwidth": 1555 * 10**9},
    "A100_80GB": {"memory_size": 80000, "FLOPS": 312 * 10**12, "memory_bandwidth": 2039 * 10**9},
    "H100": {"memory_size": 80000, "FLOPS": 1979 * 10**12, "memory_bandwidth": 3350 * 10**9},
    "H200": {"memory_size": 141000, "FLOPS": 1979 * 10**12, "memory_bandwidth": 4800 * 10**9},
}

INTERCONNECT_SPEC = {
    "PCIe Gen3x16": {"bandwidth": 32 * 10**9},
    "PCIe Gen4x16": {"bandwidth": 64 * 10**9},
    "NVSwitch 3.0": {"bandwidth": 600 * 10**9},
    "NVSwitch 4.0": {"bandwidth": 900 * 10**9},
}

INSTANCE_SPEC = {
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