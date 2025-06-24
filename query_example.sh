#!/bin/bash

# 서버에게 요청을 보내는 예시
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "meta-llama/Llama-3.2-3B-Instruct", "prompt": "What is your name?", "max_tokens": 1024, "temperature": 0}'