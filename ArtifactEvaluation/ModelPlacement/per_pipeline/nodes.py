g6_12xlarge_node_ip_1 = "" 
g6_12xlarge_node_ip_2 = "" 
g6_12xlarge_node_ip_3 = "" 
g5_12xlarge_node_ip_1 = "" 
g5_12xlarge_node_ip_2 = "" 
g6e_xlarge_node_ip_1  = ""
g6e_xlarge_node_ip_2  = ""
g6e_xlarge_node_ip_3  = "" 
g6e_xlarge_node_ip_4  = ""

# Extra single-GPU instances for HexGen pipelines
# Ray can't share a physical node across pipelines,
# so extra standalone instances are needed for leftover GPU stages.
#
# Llama3-70B HexGen:
#   - g5_xlarge_node_ip_1: P2 stage[7] (g5.xlarge, 1× A10G, 3 layers)
#
# Qwen3-32B HexGen:
#   - g5_xlarge_node_ip_1: P3 stage[3] (g5.xlarge, 1× A10G, 5 layers)
#   - g5_xlarge_node_ip_2: P3 stage[4] (g5.xlarge, 1× A10G, 4 layers)
#   - g6_xlarge_node_ip_1: P3 stage[2] (g6.xlarge, 1× L4, 8 layers)
#
g5_xlarge_node_ip_1   = ""  # HexGen extra: standalone g5.xlarge (1× A10G)
g5_xlarge_node_ip_2   = ""  # HexGen extra: standalone g5.xlarge (1× A10G) — Qwen3 only
g6_xlarge_node_ip_1   = ""  # HexGen extra: standalone g6.xlarge (1× L4)   — Qwen3 only
