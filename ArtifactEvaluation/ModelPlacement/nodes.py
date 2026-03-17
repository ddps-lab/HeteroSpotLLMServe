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
# Llama and Qwen run sequentially (different models), so extra nodes
# are shared across models. Physical instances = max(Llama, Qwen) per type.
#
# Llama needs: 1× g6.xlarge + 2× g5.xlarge
# Qwen  needs: 1× g6.xlarge + 3× g5.xlarge
# → Physical: 1× g6.xlarge + 3× g5.xlarge = 4 extra instances total
#
# Llama usage:
#   EXTRA_G6_XLARGE_1 → P1 stage 0 (L4 TP=1)
#   EXTRA_G5_XLARGE_1 → P1 stage 1 (A10G TP=1)
#   EXTRA_G5_XLARGE_2 → P1 stage 3 (A10G TP=1)
#
# Qwen usage (same physical nodes, reused after Llama finishes):
#   EXTRA_G6_XLARGE_1 → P4 stage 2 (L4 TP=1)
#   EXTRA_G5_XLARGE_1 → P1 stage 0 (A10G TP=1)
#   EXTRA_G5_XLARGE_2 → P2 stage 3 (A10G TP=1)
#   EXTRA_G5_XLARGE_3 → P4 stage 3 (A10G TP=1)  ← Qwen only

EXTRA_G6_XLARGE_1  = ""   # Extra: standalone g6.xlarge  (1× L4)   — Llama P1.s0 / Qwen P4.s2
EXTRA_G5_XLARGE_1  = ""   # Extra: standalone g5.xlarge  (1× A10G) — Llama P1.s1 / Qwen P1.s0
EXTRA_G5_XLARGE_2  = ""   # Extra: standalone g5.xlarge  (1× A10G) — Llama P1.s3 / Qwen P2.s3
EXTRA_G5_XLARGE_3  = ""   # Extra: standalone g5.xlarge  (1× A10G) — Qwen P4.s3 only
