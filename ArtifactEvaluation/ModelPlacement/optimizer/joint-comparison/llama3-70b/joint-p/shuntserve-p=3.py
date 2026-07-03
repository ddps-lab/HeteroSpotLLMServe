"""
JOINT vs GREEDY pipeline extraction — Llama-3.1-70B, p = 3.

Reviewer R2#11: compare the deployed GREEDY pipeline extraction against JOINTLY
optimizing all K pipelines.  Greedy selects K=2 for Llama-3.1-70B, so p=3 forces
ONE MORE pipeline than greedy chooses — it probes whether over-splitting helps.

This script enumerates EVERY way to split the cluster (g5x2, g6x3, g6ex4 = 9
nodes) into exactly 3 non-empty node groups, optimizes one pipeline per group
with the same beam-search DP the paper uses, ranks every partition by total
throughput, and compares against the greedy total.

Greedy baseline (current code == paper): K=2, total = 2.8305 req/s.  Since
greedy's K (2) differs from p=3 here, this script compares the best achievable
3-pipeline JOINT placement against the greedy total.

Run:  python3 shuntserve-p=3.py [--refresh]
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from joint_p_common import run_joint_experiment  # noqa: E402

if __name__ == "__main__":
    run_joint_experiment("llama3-70b", p=3, refresh_memo=("--refresh" in sys.argv))
