#!/bin/bash
# Qwen3.5 训练环境配置
# 使用 miniconda3 环境（transformers 5.9.0 + torch 2.6.0+cu124）

set -e

export PYTHON=/root/miniconda3/bin/python3
export PATH=/root/miniconda3/bin:$PATH

# HF 镜像
export HF_ENDPOINT=https://hf-mirror.com

# CUDA
export CUDA_VISIBLE_DEVICES=2,3,4,6

# 项目根目录
PROJECT_ROOT=/h3c/haoxiang/TTT-OPD
cd $PROJECT_ROOT

# 清空日志
> log.txt

echo "=== Env Setup: $(date) ==="
echo "Python: $($PYTHON --version)"
echo "CUDA: $(CUDA_VISIBLE_DEVICES=0 $PYTHON -c 'import torch; print(torch.version.cuda)') 2>/dev/null)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
