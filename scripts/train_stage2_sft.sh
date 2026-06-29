#!/bin/bash
# 阶段2：Off-policy 监督微调 (SFT)
# 目标：使用高质量QA对数据进行标准监督微调

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
        CONFIG="configs/opdttt/llama3_sc_500m_stage2_sft.yaml"
        ;;
    "1b5")
        CONFIG="configs/opdttt/llama3_sc_1b5_stage2_sft.yaml"
        ;;
    *)
        echo "错误: 不支持的模型大小 $MODEL_SIZE，请使用 500m 或 1b5" | tee -a "$LOG_FILE"
        exit 1
        ;;
esac

# 清空日志文件
> "$LOG_FILE"

echo "==========================================" | tee -a "$LOG_FILE"
echo "阶段2：Off-policy 监督微调 (SFT)" | tee -a "$LOG_FILE"
echo "模型大小: $MODEL_SIZE" | tee -a "$LOG_FILE"
echo "配置文件: $CONFIG" | tee -a "$LOG_FILE"
echo "可见GPU: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# SFT特定参数（通过命令行覆盖配置）
# 注意：教师模型路径设为空字符串以禁用教师
SFT_ARGS="--opdttt.teacher_model_path=\"\" \
          --opdttt.enable_opd_sampling=false \
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

# 检查数据文件是否存在
if [ ! -f "data/opd_sft_raw.jsonl" ]; then
    echo "错误: 未找到数据文件 data/opd_sft_raw.jsonl" | tee -a "$LOG_FILE"
    echo "请先运行数据准备脚本：python scripts/convert_parquet_to_qa_pairs.py" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "data/opd_sft_val.jsonl" ]; then
    echo "警告: 未找到验证集文件 data/opd_sft_val.jsonl" | tee -a "$LOG_FILE"
    echo "将使用训练集的一部分作为验证集" | tee -a "$LOG_FILE"
fi

# 检查阶段1 checkpoint是否存在
STAGE1_CKPT=$(grep -P "^\s+model_path:" "$CONFIG" | awk '{print $2}' | tr -d '"')
if [ ! -d "$STAGE1_CKPT" ]; then
    echo "错误: 未找到阶段1 checkpoint: $STAGE1_CKPT" | tee -a "$LOG_FILE"
    echo "请先完成阶段1预训练" | tee -a "$LOG_FILE"
    exit 1
fi

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    tasks/train_opdttt.py \
    "$CONFIG" \
    $SFT_ARGS 2>&1 | tee -a "$LOG_FILE"

echo "==========================================" | tee -a "$LOG_FILE"
echo "阶段2 SFT训练完成！" | tee -a "$LOG_FILE"
echo "Checkpoint位置: data/output/${MODEL_SIZE}_stage2_sft/checkpoints/" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"