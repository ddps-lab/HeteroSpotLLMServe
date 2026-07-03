"""
JOINT (single-pipeline) baseline — llama3-70b, p = 1.

p=1 means the WHOLE cluster is optimized as a SINGLE pipeline (one node group).
This equals greedy's FIRST iteration only; greedy then continues to K pipelines.
Included to complete the p-sweep (p=1..4) and to show that "no shunting" (one
pipeline over the whole cluster) is the worst point — motivating multiple pipelines.

Run:  python3 shuntserve-p=1.py [--refresh]
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from joint_p_common import run_joint_experiment  # noqa: E402

if __name__ == "__main__":
    run_joint_experiment("llama3-70b", p=1, refresh_memo=("--refresh" in sys.argv))
