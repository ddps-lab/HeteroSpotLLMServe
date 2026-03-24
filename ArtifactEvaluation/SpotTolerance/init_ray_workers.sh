#!/bin/bash
HEAD_IP=""
RAY="/home/ubuntu/.local/bin/ray"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

WORKER_IPS=(
    # scenario A & B 공통 노드 (중복 제거, IP를 채워 넣으세요)
)

# Stop existing ray workers on all nodes first
echo "=== Stopping existing Ray workers ==="
for IP in "${WORKER_IPS[@]}"; do
    ssh $SSH_OPTS $IP "${RAY} stop" 2>/dev/null &
done
wait
echo "=== All workers stopped ==="

for PORT in 6379 6380; do
    echo "=== Joining workers to Ray cluster at ${HEAD_IP}:${PORT} ==="
    for IP in "${WORKER_IPS[@]}"; do
        echo "  Connecting ${IP}..."
        ssh $SSH_OPTS $IP "${RAY} start --address=${HEAD_IP}:${PORT} --disable-usage-stats" &
    done
    wait
    echo "=== Done for port ${PORT} ==="
done

echo "All workers joined both Ray clusters."
