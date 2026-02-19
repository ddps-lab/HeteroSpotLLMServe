#!/bin/bash

# Health check script for ShuntServe
API_HOST="172.31.17.139"
API_PORT="8001"
TENSOR_STORE_PORT="10001"

echo "=== ShuntServe Health Check ==="
echo "Checking API Server at $API_HOST:$API_PORT"
echo

# Check API Server Health
echo "1. API Server Health Check:"
HEALTH_RESPONSE=$(curl -s -w "HTTP_STATUS:%{http_code}" --connect-timeout 5 --max-time 10 "http://$API_HOST:$API_PORT/health")
HTTP_STATUS=$(echo "$HEALTH_RESPONSE" | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
RESPONSE_BODY=$(echo "$HEALTH_RESPONSE" | sed 's/HTTP_STATUS:[0-9]*$//')

if [ "$HTTP_STATUS" = "200" ]; then
    echo "   ✓ API Server is healthy"
    echo "   HTTP Status: $HTTP_STATUS"
    if [ -n "$RESPONSE_BODY" ]; then
        echo "   Response Body: $RESPONSE_BODY"
    else
        echo "   Response Body: (empty - this is normal for /health endpoint)"
    fi
else
    echo "   ✗ API Server is not responding"
    echo "   HTTP Status: $HTTP_STATUS"
    echo "   Response: $RESPONSE_BODY"
fi

# Check API Server Load
echo "2. API Server Load:"
LOAD_RESPONSE=$(curl -s --connect-timeout 5 "http://$API_HOST:$API_PORT/load")
if [ $? -eq 0 ]; then
    echo "   ✓ Load endpoint accessible"
    echo "   Response: $LOAD_RESPONSE"
else
    echo "   ✗ Load endpoint not accessible"
fi

# Check Models
echo "3. Available Models:"
MODELS_RESPONSE=$(curl -s --connect-timeout 5 "http://$API_HOST:$API_PORT/v1/models")
if [ $? -eq 0 ]; then
    echo "   ✓ Models endpoint accessible"
    echo "   Response: $MODELS_RESPONSE"
else
    echo "   ✗ Models endpoint not accessible"
fi

# Check TensorStore
echo "4. TensorStore Status:"
TENSOR_STATUS=$(echo "" | nc -w 2 $API_HOST $TENSOR_STORE_PORT)
if [ "$TENSOR_STATUS" = "1" ]; then
    echo "   ✓ TensorStore is ready"
elif [ "$TENSOR_STATUS" = "0" ]; then
    echo "   ⚠ TensorStore is not ready"
else
    echo "   ✗ TensorStore is not responding"
fi

# Test a simple completion
echo "5. Simple Completion Test:"
TEST_RESPONSE=$(curl -s --connect-timeout 10 --max-time 30 \
    -H "Content-Type: application/json" \
    -d '{"model": "meta-llama/Llama-3.1-8B", "prompt": "Hello", "max_tokens": 5}' \
    "http://$API_HOST:$API_PORT/v1/completions")

if [ $? -eq 0 ] && echo "$TEST_RESPONSE" | grep -q "choices"; then
    echo "   ✓ Completion test successful"
    echo "   Response: $TEST_RESPONSE"
else
    echo "   ✗ Completion test failed"
    echo "   Response: $TEST_RESPONSE"
fi

echo
echo "=== Health Check Complete ==="