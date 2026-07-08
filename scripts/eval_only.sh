#!/bin/bash
# 对存档点进行一次性 PPL 评估（不训练），用于验证评估修复正确性
# 用法: bash scripts/eval_only.sh
# 依赖 config 的 checkpoint.load_path 指向要评估的存档点
set -e

PROJECT_ROOT=/h3c/haoxiang/TTT-OPD
PYTHON=/root/miniconda3/bin/python3

export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_VISIBLE_DEVICES=2,3,4,6
export OPDTTT_EVAL_ONLY=1

cd $PROJECT_ROOT

> eval_log.txt
echo "=== Eval-Only Start: $(date) ===" | tee -a eval_log.txt
echo "GPUs: $CUDA_VISIBLE_DEVICES" | tee -a eval_log.txt

torchrun \
    --nproc_per_node=4 \
    --master_port=29502 \
    tasks/train_opdttt.py \
    --config configs/opdttt/qwen35_2b_ttt.yaml \
    2>&1 | tee -a eval_log.txt

echo "=== Eval-Only Finished: $(date) ===" | tee -a eval_log.txt
