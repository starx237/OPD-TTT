#!/bin/bash
# Qwen3-0.6B + In-Place TTT 持续预训练启动脚本
#
# 用法:
#   bash scripts/train_qwen3_ttt.sh [GPU数量]
#
# 默认使用4个GPU (2,3,4,6)。确保已安装 .venv 环境。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

source env_setup.sh

NUM_GPUS=${1:-4}
CONFIG="configs/opdttt/qwen3_0.6b_ttt.yaml"

# 指定使用的GPU（避开被占用的0,1,5,7）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"2,3,4,6"}

echo "=========================================="
echo "Qwen3-0.6B + TTT 持续预训练"
echo "配置: $CONFIG"
echo "GPU数量: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 清空日志
> log.txt

NPROC_PER_NODE=$NUM_GPUS torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --master_port=29500 \
    tasks/train_opdttt.py \
    "$CONFIG" \
    2>&1 | tee log.txt
