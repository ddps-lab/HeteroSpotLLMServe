"""
JOINT vs GREEDY pipeline extraction — Qwen3-32B, K = p = 2.

Reviewer R2#11: compare the deployed GREEDY pipeline extraction against JOINTLY
optimizing all K pipelines.  For K=2 this is the tractable case the reviewer
explicitly calls out.

This script enumerates EVERY way to split the cluster (g5x2, g6x3, g6ex4 = 9
nodes) into exactly 2 non-empty node groups, optimizes one pipeline per group
with the same beam-search DP the paper uses, ranks every partition by total
throughput, and compares against the greedy solution.

Greedy baseline (current code == paper): K=4, total = 9.4241 req/s.  Since
greedy's K (4) differs from p=2 here, this script compares the best achievable
2-pipeline JOINT placement against the greedy total.

Run:  python3 shuntserve-p=2.py [--refresh]
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from joint_p_common import run_joint_experiment  # noqa: E402

if __name__ == "__main__":
    run_joint_experiment("qwen3-32b", p=2, refresh_memo=("--refresh" in sys.argv))
