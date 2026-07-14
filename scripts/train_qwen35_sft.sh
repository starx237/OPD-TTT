#!/bin/bash
# Qwen3.5-2B SFT 训练启动脚本（基于 CPT step5000 checkpoint）
# 数据: OpenThoughts-3 200K QA 对，conversation 格式
# 环境: miniconda3 (Python 3.13)
set -e

PROJECT_ROOT=/h3c/haoxiang/TTT-OPD
PYTHON=/root/miniconda3/bin/python3

export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

cd $PROJECT_ROOT

# 加载 .env 文件
if [ -f .env ]; then
    _SCRIPT_CUDA=${CUDA_VISIBLE_DEVICES:-}
    export $(grep -v '^#' .env | xargs)
    if [ -n "$_SCRIPT_CUDA" ]; then
        export CUDA_VISIBLE_DEVICES=$_SCRIPT_CUDA
    fi
    echo "Loaded .env file, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# 清空日志
> log.txt

echo "=== SFT Training Start: $(date) ===" | tee -a log.txt
echo "Python: $($PYTHON --version)" | tee -a log.txt
NPROC=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NPROC procs)" | tee -a log.txt

torchrun \
    --nproc_per_node=$NPROC \
    --master_port=29501 \
    tasks/train_opdttt.py \
    --config configs/opdttt/qwen35_2b_sft.yaml \
    2>&1 | tee -a log.txt

echo "=== SFT Training Finished: $(date) ===" | tee -a log.txt
