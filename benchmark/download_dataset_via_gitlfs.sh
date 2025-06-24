#!/bin/bash

# Git 에서 직접 클론
# Url 인자와 output dir 인자를 반드시 받아야 함
# 인자가 두 개인지 확인하고 아니라면 Usage 출력

if [ $# -ne 2 ]; then
    echo "Usage: $0 <url> <output_dir>"
    echo "Example: $0 https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered ./dataset/sharegpt/ShareGPT_Vicuna_unfiltered"
    exit 1
fi

# git lfs 가 반드시 설치되어 있어야함.
if ! command -v git-lfs &> /dev/null; then
    echo "git-lfs is not installed. Please install it first."
    exit 1
fi

url=$1
output_dir=$2

mkdir -p $output_dir

git clone $url $output_dir