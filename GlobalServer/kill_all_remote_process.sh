#!/bin/bash

# Kill all HeteroSpotLLMServe related processes on remote nodes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}🔴 Killing all HeteroSpotLLMServe processes${NC}"
echo "======================================"

# List of remote nodes - add all your nodes here
NODES=(
    "172.31.17.139"
    # Add more nodes as needed
)

# SSH options to skip host key verification
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Function to show current processes
show_processes() {
    local node=$1
    echo -e "\n${YELLOW}📊 Checking processes on $node...${NC}"
    local count=$(ssh $SSH_OPTS $node "ps aux | grep -E '(mt_tensor_store_server|s3_tensor_store_server|api_server\.py|vllm)' | grep -v grep | wc -l" 2>/dev/null || echo "0")
    if [ "$count" -gt 0 ]; then
        ssh $SSH_OPTS $node "ps aux | grep -E '(mt_tensor_store_server|s3_tensor_store_server|api_server\.py|vllm)' | grep -v grep" 2>/dev/null
    else
        echo "   No processes found"
    fi
}

# Function to kill processes on a single node
kill_node_processes() {
    local node=$1
    echo -e "\n${YELLOW}🎯 Killing processes on node: $node${NC}"
    
    # Kill TensorStore servers (both mt_tensor_store_server.py and s3_tensor_store_server.py)
    echo "   Killing TensorStore servers..."
    ssh $SSH_OPTS $node "pkill -f 'mt_tensor_store_server.py' 2>/dev/null || true"
    ssh $SSH_OPTS $node "pkill -f 's3_tensor_store_server.py' 2>/dev/null || true"
    
    # Kill API servers (api_server.py)
    echo "   Killing API servers..."
    ssh $SSH_OPTS $node "pkill -f 'InferenceServer/api_server.py' 2>/dev/null || true"
    
    # Wait a moment for processes to terminate
    echo "   Waiting for processes to terminate..."
    sleep 5
    
    # Clean up GPU memory
    echo "   Cleaning up GPU memory..."
    ssh $SSH_OPTS $node "/usr/bin/python -c 'import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect(); print(\"GPU cache cleared\")' 2>/dev/null || echo '   Warning: GPU memory cleanup failed'"
    
    # Check if any processes are still running
    local remaining=$(ssh $SSH_OPTS $node "ps aux | grep -E '(mt_tensor_store_server|s3_tensor_store_server|api_server\.py|vllm)' | grep -v grep | wc -l" 2>/dev/null || echo "0")
    
    if [ "$remaining" -eq 0 ]; then
        echo -e "   ${GREEN}✅ All processes killed successfully${NC}"
    else
        echo -e "   ${RED}⚠️  Warning: $remaining processes still running${NC}"
        # Force kill with SIGKILL
        echo "   Using force kill (SIGKILL)..."
        ssh $SSH_OPTS $node "pkill -9 -f 'mt_tensor_store_server.py' 2>/dev/null || true"
        ssh $SSH_OPTS $node "pkill -9 -f 's3_tensor_store_server.py' 2>/dev/null || true"
        ssh $SSH_OPTS $node "pkill -9 -f 'InferenceServer/api_server.py' 2>/dev/null || true"
    fi
}

# Show current status first
echo -e "${YELLOW}Current process status:${NC}"
for node in "${NODES[@]}"; do
    show_processes $node
done

# Kill processes on all nodes in parallel
echo -e "\n${RED}Killing processes on all nodes...${NC}"
for node in "${NODES[@]}"; do
    kill_node_processes $node &
done

# Wait for all background jobs to complete
wait

# Show final status
echo -e "\n${YELLOW}Final status:${NC}"
for node in "${NODES[@]}"; do
    remaining=$(ssh $SSH_OPTS $node "ps aux | grep -E '(mt_tensor_store_server|s3_tensor_store_server|api_server\.py|vllm)' | grep -v grep | wc -l" 2>/dev/null || echo "0")
    if [ "$remaining" -eq 0 ]; then
        echo -e "   $node: ${GREEN}Clean${NC}"
    else
        echo -e "   $node: ${RED}$remaining processes still running${NC}"
        # Show which processes are still running
        echo -e "   ${RED}Still running processes:${NC}"
        ssh $SSH_OPTS $node "ps aux | grep -E '(mt_tensor_store_server|s3_tensor_store_server|api_server\.py|vllm)' | grep -v grep" 2>/dev/null | while read line; do
            echo "      $line"
        done
    fi
done

echo -e "\n${GREEN}🏁 Done!${NC}"