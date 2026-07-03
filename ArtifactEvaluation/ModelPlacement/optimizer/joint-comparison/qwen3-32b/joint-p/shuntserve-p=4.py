"""
JOINT vs GREEDY pipeline extraction — Qwen3-32B, K = p = 4.

Reviewer R2#11: compare the deployed GREEDY pipeline extraction against JOINTLY
optimizing all K pipelines.  Here p=4 matches the number of pipelines greedy
actually produces for Qwen3-32B, so the greedy solution is DIRECTLY comparable
to the joint optimum.

This script enumerates EVERY way to split the cluster (g5x2, g6x3, g6ex4 = 9
nodes) into exactly 4 non-empty node groups, optimizes one pipeline per group
with the same beam-search DP the paper uses, ranks every partition by total
throughput, and reports where the greedy solution falls in that ranking.

Greedy baseline (current code == paper): K=4, total = 9.4241 req/s.

Run:  python3 shuntserve-p=4.py [--refresh]
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from joint_p_common import run_joint_experiment  # noqa: E402

if __name__ == "__main__":
    run_joint_experiment("qwen3-32b", p=4, refresh_memo=("--refresh" in sys.argv))
