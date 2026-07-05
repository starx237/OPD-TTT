#!/bin/bash
# Qwen3.5-2B TTT CPT 训练启动脚本（独立运行，不依赖 tmux heredoc）
set -e

PROJECT_ROOT=/h3c/haoxiang/TTT-OPD
PYTHON=/root/miniconda3/bin/python3

export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

cd $PROJECT_ROOT

# 加载 .env 文件（包含 WANDB_API_KEY 等）
# .env 中的 CUDA_VISIBLE_DEVICES 作为默认值，但脚本上方已 export 的优先
if [ -f .env ]; then
    # 先保存脚本已设置的 CUDA_VISIBLE_DEVICES
    _SCRIPT_CUDA=${CUDA_VISIBLE_DEVICES:-}
    export $(grep -v '^#' .env | xargs)
    # 如果脚本已设置 CUDA_VISIBLE_DEVICES，则覆盖 .env 的值
    if [ -n "$_SCRIPT_CUDA" ]; then
        export CUDA_VISIBLE_DEVICES=$_SCRIPT_CUDA
    fi
    echo "Loaded .env file, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# 清空日志
> log.txt

echo "=== Training Start: $(date) ===" | tee -a log.txt
echo "Python: $($PYTHON --version)" | tee -a log.txt
# 动态计算 GPU 数量
NPROC=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NPROC procs)" | tee -a log.txt

torchrun \
    --nproc_per_node=$NPROC \
    --master_port=29501 \
    tasks/train_opdttt.py \
    --config configs/opdttt/qwen35_2b_ttt.yaml \
    2>&1 | tee -a log.txt

echo "=== Training Finished: $(date) ===" | tee -a log.txt
