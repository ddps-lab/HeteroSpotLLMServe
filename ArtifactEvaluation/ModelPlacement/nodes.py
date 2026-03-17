# ─── Physical cluster nodes ──────────────────────────────────────────
# Cluster: g6.12xlarge×3, g5.12xlarge×2, g6e.xlarge×4
# Fill in IPs when instances are launched.

g6_12xlarge_node_ip_1 = ""   # g6.12xlarge #1 (4× L4)
g6_12xlarge_node_ip_2 = ""   # g6.12xlarge #2 (4× L4)
g6_12xlarge_node_ip_3 = ""   # g6.12xlarge #3 (4× L4)
g5_12xlarge_node_ip_1 = ""   # g5.12xlarge #1 (4× A10G)
g5_12xlarge_node_ip_2 = ""   # g5.12xlarge #2 (4× A10G)
g6e_xlarge_node_ip_1  = ""   # g6e.xlarge  #1 (1× L40S)
g6e_xlarge_node_ip_2  = ""   # g6e.xlarge  #2 (1× L40S)
g6e_xlarge_node_ip_3  = ""   # g6e.xlarge  #3 (1× L40S)
g6e_xlarge_node_ip_4  = ""   # g6e.xlarge  #4 (1× L40S)

# ─── Unit test node (example pipeline) ───────────────────────────────
g6_xlarge_node_ip_1   = ""   # g6.xlarge   #1 (1× L4) — used by llama3-70b/example/p1.py
g6_xlarge_node_ip_2   = ""   # g6.xlarge   #1 (1× L4) — used by qwen3-32b/example/p1.py

# ─── Extra standalone instances (HexGen) ─────────────────────────────
# HexGen GA splits multi-GPU nodes across pipelines at sub-node granularity.
# However, Ray does not allow two independent vLLM workers (from different
# pipelines) to coexist on the same physical node simultaneously.
#
# Resolution strategy: consolidate same-pipeline stages onto shared nodes
# where possible, then launch small standalone instances for remaining
# cross-pipeline conflicts.
#
# Llama-3.1-70B HexGen (2 pipelines):
#   - P1 TP=2 g5 stages consolidated onto g5.12xl#1 (4 GPUs)
#   - P2 TP=1 g5 stages consolidated onto g5.12xl#2 (2 GPUs)
#   - P1 stage 0 (L4 TP=1) conflicts with P2 on g6.12xl#2 → extra g6.xlarge
#   - P1 stages 1,3 (A10G TP=1) moved off g5.12xl → extra g5.xlarge ×2

EXTRA_G6_XLARGE_1  = ""   # Extra: standalone g6.xlarge  (1× L4)   — Llama HX-P1 stage 0
EXTRA_G5_XLARGE_1  = ""   # Extra: standalone g5.xlarge  (1× A10G) — Llama HX-P1 stage 1
EXTRA_G5_XLARGE_2  = ""   # Extra: standalone g5.xlarge  (1× A10G) — Llama HX-P1 stage 3

# Qwen3-32B HexGen (5 pipelines):
#   - P5 L4 stages consolidated onto g6.12xl#3 (4 GPUs)
#   - P3 g5 stages consolidated onto g5.12xl#1 (3 GPUs)
#   - P5 g5 stages consolidated onto g5.12xl#2 (2 GPUs)
#   - P1 stage 0, P2 stage 3, P4 stage 3 (A10G TP=1) → extra g5.xlarge ×3
#   - P4 stage 2 (L4 TP=1) conflicts with P5 on g6.12xl#3 → extra g6.xlarge

EXTRA_G5_XLARGE_3  = ""   # Extra: standalone g5.xlarge  (1× A10G) — Qwen3 HX-P1 stage 0
EXTRA_G5_XLARGE_4  = ""   # Extra: standalone g5.xlarge  (1× A10G) — Qwen3 HX-P2 stage 3
EXTRA_G5_XLARGE_5  = ""   # Extra: standalone g5.xlarge  (1× A10G) — Qwen3 HX-P4 stage 3
EXTRA_G6_XLARGE_2  = ""   # Extra: standalone g6.xlarge  (1× L4)   — Qwen3 HX-P4 stage 2
