#!/bin/bash
# 阶段1：基础TTT预训练（无教师）
# 目标：建立基础语言建模能力和TTT机制
# 对应OPD论文中的pre-training阶段

set -e

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 加载环境设置（包括 .env 中的 CUDA_VISIBLE_DEVICES）
source "$PROJECT_ROOT/env_setup.sh"

# 日志文件
LOG_FILE="$PROJECT_ROOT/log.txt"

# 模型配置
MODEL_SIZE=${1:-"500m"}  # 500m 或 1b5

# 根据模型大小设置配置
case $MODEL_SIZE in
    "500m")
        CONFIG="configs/opdttt/llama3_sc_500m_stage1_pretrain.yaml"
        ;;
    "1b5")
        CONFIG="configs/opdttt/llama3_sc_1b5_stage1_pretrain.yaml"
        ;;
    *)
        echo "错误: 不支持的模型大小 $MODEL_SIZE，请使用 500m 或 1b5" | tee -a "$LOG_FILE"
        exit 1
        ;;
esac

# 清空日志文件
> "$LOG_FILE"

echo "==========================================" | tee -a "$LOG_FILE"
echo "阶段1：基础TTT预训练（无教师）" | tee -a "$LOG_FILE"
echo "模型大小: $MODEL_SIZE" | tee -a "$LOG_FILE"
echo "配置文件: $CONFIG" | tee -a "$LOG_FILE"
echo "可见GPU: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# 阶段1特定参数（通过命令行覆盖配置）
# 注意：教师模型路径设为空字符串以禁用教师
STAGE1_ARGS="--opdttt.teacher_model_path=\"\" \
             --opdttt.lambda_align_rep=0.0 \
             --opdttt.lambda_kl=0.0 \
             --opdttt.lambda_ntp=1.0 \
             --opdttt.lambda_lm=1.0"

# 启动训练，输出到日志文件（配置文件作为位置参数）
# 自动检测可见 GPU 数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ -z "$CUDA_VISIBLE_DEVICES" ] || [ "$CUDA_VISIBLE_DEVICES" = "0,1,2,3,4,5,6,7" ]; then
    NUM_GPUS=8
fi
echo "使用 $NUM_GPUS 个 GPU" | tee -a "$LOG_FILE"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    tasks/train_opdttt.py \
    "$CONFIG" \
    $STAGE1_ARGS 2>&1 | tee -a "$LOG_FILE"

echo "==========================================" | tee -a "$LOG_FILE"
echo "阶段1训练完成！" | tee -a "$LOG_FILE"
echo "Checkpoint位置: data/output/${MODEL_SIZE}_stage1_pretrain/checkpoints/" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
