#!/bin/bash
# 单 GPU 评估脚本（FSDP2 模式，float32 参数 + autocast → PPL 可信）
#
# 修复说明：
#   DDP bf16 模型与 4GPU FSDP2 MixedPrecision 等价（forward 均在 bf16 下），
#   PPL 可信。之前尝试的 float32+autocast 方案不等价（autocast 只转 matmul，
#   LayerNorm/Softmax 保持 float32），导致 PPL 偏高。
#   ttt_lr=3.0（与训练一致，经设计决定暂不修改）
#
# 用法: bash scripts/eval_single.sh [GPU_ID] [STEP] [SAMPLES] [GROUP_SIZE]
#   GPU_ID:      使用的 GPU 编号（默认 1）
#   STEP:        评估的 checkpoint 步数（默认 1200）
#   SAMPLES:     评估样本数（默认 50）
#   GROUP_SIZE:  分组统计大小（默认 10）
#
# 示例:
#   bash scripts/eval_single.sh 1 1200 50 10
#   bash scripts/eval_single.sh 0 1500 20 5
set -e

GPU_ID=${1:-1}
STEP=${2:-1200}
SAMPLES=${3:-50}
GROUP_SIZE=${4:-10}

PROJECT_ROOT=/h3c/haoxiang/TTT-OPD
PYTHON=/root/miniconda3/bin/python3

export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_VISIBLE_DEVICES=$GPU_ID
export OPDTTT_EVAL_ONLY=1
export OPDTTT_EVAL_SAMPLES=$SAMPLES
export OPDTTT_EVAL_GROUP_SIZE=$GROUP_SIZE
export OPDTTT_EVAL_TTT_LR=3.0
export OPDTTT_LOAD_PATH="${PROJECT_ROOT}/data/output/qwen35_2b_ttt/checkpoints/global_step_${STEP}/global_step_${STEP}"

cd $PROJECT_ROOT

# 加载 .env 文件（包含 WANDB_API_KEY 等），但保留脚本设置的环境变量
if [ -f .env ]; then
    _EVAL_CUDA=$CUDA_VISIBLE_DEVICES
    _EVAL_LOAD=$OPDTTT_LOAD_PATH
    export $(grep -v '^#' .env | xargs)
    export CUDA_VISIBLE_DEVICES=$_EVAL_CUDA
    export OPDTTT_LOAD_PATH=$_EVAL_LOAD
fi

CONFIG=configs/opdttt/qwen35_2b_ttt_eval.yaml

LOG=eval_${STEP}_fsdp2_${SAMPLES}s.log
> $LOG
echo "=== Eval-Only (DDP bf16) Start: $(date) ===" | tee -a $LOG
echo "GPU: $GPU_ID | Step: $STEP | Samples: $SAMPLES (group=$GROUP_SIZE) | ttt_lr=3.0" | tee -a $LOG
echo "Checkpoint: $OPDTTT_LOAD_PATH" | tee -a $LOG

torchrun \
    --nproc_per_node=1 \
    --master_port=29503 \
    tasks/train_opdttt.py \
    --config $CONFIG \
    2>&1 | tee -a $LOG

echo "=== Eval-Only Finished: $(date) ===" | tee -a $LOG
