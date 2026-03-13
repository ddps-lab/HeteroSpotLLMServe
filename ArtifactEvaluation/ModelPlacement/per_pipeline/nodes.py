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
g6_xlarge_node_ip_1   = ""   # g6.xlarge   #1 (1× L4) — used by example/p1.py

# ─── Extra standalone instances (HexGen) ─────────────────────────────
# HexGen splits multi-GPU nodes across pipelines at sub-node granularity.
# However, Ray does not allow two independent vLLM workers (from different
# pipelines) to coexist on the same physical node simultaneously.
#
# Example: HexGen assigns 1 A10G from g5.12xlarge #2 to P1 and
# the remaining 3 A10G to P3. Since P1 and P3 are separate vLLM
# instances, they cannot share g5_12xlarge_node_ip_2.
#
# Solution: launch a separate standalone g5.xlarge (1× A10G) for the
# single-GPU stage, so each pipeline owns its nodes exclusively.

EXTRA_G5_XLARGE_1 = ""  # Extra: standalone g5.xlarge (1× A10G) — used by Llama-3.1-70B HX-P1 stage 7

# ─── Extra standalone instances (Qwen3-32B HexGen) ──────────────────
# Qwen3 HexGen has 6 pipelines. All must run simultaneously for parallel
# benchmarking. The following cross-pipeline conflicts require extra nodes:
#
#   P1 ↔ P5: both use g5.12xl#1 (A10G)
#   P1 ↔ P6: both use g6.12xl#1 (L4)
#   P4 ↔ P6: both use g5.12xl#2 (A10G)
#
# P6 stage 0 is TP=2 (needs 2 L4 GPUs on same node) → requires g6.12xlarge.

EXTRA_G5_XLARGE_2 = ""    # Extra: standalone g5.xlarge (1× A10G) — Qwen3 HX-P5 stage 1
EXTRA_G5_XLARGE_3 = ""    # Extra: standalone g5.xlarge (1× A10G) — Qwen3 HX-P5 stage 2
