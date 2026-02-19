#!/usr/bin/env python3
"""
Health check script for ShuntServe using requests library
"""

import requests
import socket
import sys

def check_api_server_health(host, port, timeout=10):
    """Check API server health endpoint"""
    try:
        url = f"http://{host}:{port}/health"
        response = requests.get(url, timeout=timeout)
        return {
            'success': True,
            'status_code': response.status_code,
            'response_body': response.text,
            'healthy': response.status_code == 200
        }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': str(e),
            'healthy': False
        }

def check_api_server_load(host, port, timeout=5):
    """Check API server load endpoint"""
    try:
        url = f"http://{host}:{port}/load"
        response = requests.get(url, timeout=timeout)
        return {
            'success': True,
            'status_code': response.status_code,
            'response_body': response.text,
            'data': response.json() if response.status_code == 200 else None
        }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': str(e)
        }

def check_api_server_models(host, port, timeout=5):
    """Check API server models endpoint"""
    try:
        url = f"http://{host}:{port}/v1/models"
        response = requests.get(url, timeout=timeout)
        return {
            'success': True,
            'status_code': response.status_code,
            'response_body': response.text,
            'data': response.json() if response.status_code == 200 else None
        }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': str(e)
        }

def check_tensor_store_status(host, port, timeout=2):
    """Check TensorStore status via TCP connection"""
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            response = sock.recv(4)
            status = response.strip()
            return {
                'success': True,
                'raw_response': response,
                'status': status,
                'ready': status == b"1"
            }
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        return {
            'success': False,
            'error': str(e),
            'ready': False
        }

def test_completion(host, port, timeout=30):
    """Test a simple completion request"""
    try:
        url = f"http://{host}:{port}/v1/completions"
        payload = {
            "model": "meta-llama/Llama-3.1-8B",
            "prompt": "Hello",
            "max_tokens": 5
        }
        response = requests.post(url, json=payload, timeout=timeout)
        return {
            'success': True,
            'status_code': response.status_code,
            'response_body': response.text,
            'data': response.json() if response.status_code == 200 else None,
            'working': response.status_code == 200 and 'choices' in response.json()
        }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': str(e),
            'working': False
        }

def main():
    # Configuration
    API_HOST = "172.31.17.139"
    API_PORT = 8001
    TENSOR_STORE_PORT = 10001
    
    print("=== ShuntServe Health Check (Python) ===")
    print(f"Checking API Server at {API_HOST}:{API_PORT}")
    print()
    
    # 1. Check API Server Health
    print("1. API Server Health Check:")
    health_result = check_api_server_health(API_HOST, API_PORT)
    if health_result['success'] and health_result['healthy']:
        print("   ✓ API Server is healthy")
        print(f"   HTTP Status: {health_result['status_code']}")
        if health_result['response_body']:
            print(f"   Response Body: {health_result['response_body']}")
        else:
            print("   Response Body: (empty - this is normal for /health endpoint)")
    else:
        print("   ✗ API Server is not healthy")
        if health_result['success']:
            print(f"   HTTP Status: {health_result['status_code']}")
            print(f"   Response: {health_result['response_body']}")
        else:
            print(f"   Error: {health_result['error']}")
    
    # 2. Check API Server Load
    print("\n2. API Server Load:")
    load_result = check_api_server_load(API_HOST, API_PORT)
    if load_result['success']:
        print("   ✓ Load endpoint accessible")
        print(f"   Response: {load_result['response_body']}")
    else:
        print("   ✗ Load endpoint not accessible")
        print(f"   Error: {load_result['error']}")
    
    # 3. Check Available Models
    print("\n3. Available Models:")
    models_result = check_api_server_models(API_HOST, API_PORT)
    if models_result['success']:
        print("   ✓ Models endpoint accessible")
        if models_result['data']:
            models = models_result['data'].get('data', [])
            print(f"   Found {len(models)} model(s):")
            for model in models:
                print(f"     - {model.get('id', 'Unknown')}")
        else:
            print(f"   Response: {models_result['response_body']}")
    else:
        print("   ✗ Models endpoint not accessible")
        print(f"   Error: {models_result['error']}")
    
    # 4. Check TensorStore Status
    print("\n4. TensorStore Status:")
    tensor_result = check_tensor_store_status(API_HOST, TENSOR_STORE_PORT)
    if tensor_result['success']:
        if tensor_result['ready']:
            print("   ✓ TensorStore is ready")
            print(f"   Raw response: {tensor_result['raw_response']}")
        else:
            print("   ⚠ TensorStore is not ready")
            print(f"   Status: {tensor_result['status']}")
    else:
        print("   ✗ TensorStore is not responding")
        print(f"   Error: {tensor_result['error']}")
    
    # 5. Test Simple Completion
    print("\n5. Simple Completion Test:")
    completion_result = test_completion(API_HOST, API_PORT)
    if completion_result['success'] and completion_result['working']:
        print("   ✓ Completion test successful")
        if completion_result['data'] and 'choices' in completion_result['data']:
            choice = completion_result['data']['choices'][0]
            print(f"   Generated text: '{choice.get('text', 'N/A')}'")
            print(f"   Usage: {completion_result['data'].get('usage', {})}")
    else:
        print("   ✗ Completion test failed")
        if completion_result['success']:
            print(f"   HTTP Status: {completion_result['status_code']}")
            print(f"   Response: {completion_result['response_body']}")
        else:
            print(f"   Error: {completion_result['error']}")
    
    print("\n=== Health Check Complete ===")
    
    # Return exit code based on overall health
    overall_healthy = (
        health_result.get('healthy', False) and
        load_result.get('success', False) and
        models_result.get('success', False) and
        tensor_result.get('ready', False) and
        completion_result.get('working', False)
    )
    
    if overall_healthy:
        print("✓ All systems are healthy!")
        sys.exit(0)
    else:
        print("⚠ Some systems are not healthy")
        sys.exit(1)

if __name__ == "__main__":
    main()