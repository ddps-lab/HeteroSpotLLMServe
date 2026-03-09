"""
Node IP addresses for each instance type.
Edit this file once; all benchmark p files import from here.

Usage in p files:
    from nodes import get_node_ip
    NODE_IP = get_node_ip("g5.48xlarge")
"""

# ─── Instance IP mapping ─────────────────────────────────────────────
# Fill in the IP address for each instance type you plan to benchmark.
# Leave empty string for instance types not yet provisioned.

INSTANCE_IPS = {
    "g5.48xlarge":   "",   # A10G ×8
    "g6.48xlarge":   "",   # L4 ×8
    "g6e.48xlarge":  "",   # L40S ×8
    "p4d.24xlarge":  "",   # A100_40GB ×8
    "p5.48xlarge":   "",   # H100 ×8
}


def get_node_ip(instance_type: str) -> str:
    """Get the IP address for an instance type."""
    ip = INSTANCE_IPS.get(instance_type, "")
    if not ip:
        raise ValueError(
            f"Node IP not set for {instance_type}. "
            f"Edit ArtifactEvaluation/PerformanceEstimation/nodes.py"
        )
    return ip
