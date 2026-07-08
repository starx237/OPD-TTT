#!/bin/bash
# 重放 dataloader N 步并保存状态（用于重建缺失的 dataloader checkpoint）
# 用法: bash scripts/replay_dataloader.sh [STEPS] [NUM_GPUS]
set -e

STEPS=${1:-1200}
NUM_GPUS=${2:-4}

PROJECT_ROOT=/h3c/haoxiang/TTT-OPD

export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export OPDTTT_REPLAY_STEPS=$STEPS
export OPDTTT_REPLAY_SAVE_DIR=$PROJECT_ROOT/data/output/dataloader_states

cd $PROJECT_ROOT

# 加载 .env 文件
if [ -f .env ]; then
    _REPLAY_CUDA=$CUDA_VISIBLE_DEVICES
    export $(grep -v '^#' .env | xargs)
    if [ -n "$_REPLAY_CUDA" ]; then
        export CUDA_VISIBLE_DEVICES=$_REPLAY_CUDA
    fi
fi

LOG=replay_${STEPS}.log
> $LOG
echo "=== Replay Start: $(date) ===" | tee -a $LOG
echo "Steps: $STEPS | GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS procs)" | tee -a $LOG

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29504 \
    tasks/train_opdttt.py \
    --config configs/opdttt/qwen35_2b_ttt_replay.yaml \
    2>&1 | tee -a $LOG

echo "=== Replay Finished: $(date) ===" | tee -a $LOG
