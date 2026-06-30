#!/bin/bash
# 阶段2：On-Policy Distillation
# 目标：使用教师模型的密集奖励信号优化学生策略
# 对应OPD论文中的on-policy distillation阶段

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
TEACHER_MODEL=${2:-"qwen2.5-7b"}  # 教师模型别名

# 根据模型大小设置配置
case $MODEL_SIZE in
    "500m")
        CONFIG="configs/opdttt/llama3_sc_500m_stage2_opd.yaml"
        ;;
    "1b5")
        CONFIG="configs/opdttt/llama3_sc_1b5_stage2_opd.yaml"
        ;;
    *)
        echo "错误: 不支持的模型大小 $MODEL_SIZE，请使用 500m 或 1b5" | tee -a "$LOG_FILE"
        exit 1
        ;;
esac

# 从配置文件读取模型路径（以配置文件为准）
STAGE1_CKPT=$(grep -P "^\s+model_path:" "$CONFIG" | awk '{print $2}' | tr -d '"')

# 解析教师模型路径
case $TEACHER_MODEL in
    "qwen2.5-7b")
        TEACHER_PATH="model_assets/teacher_qwen2.5_7b"
        ;;
    "qwen2.5-32b")
        TEACHER_PATH="model_assets/teacher_qwen2.5_32b"
        ;;
    "no-teacher")
        TEACHER_PATH=""
        ;;
    *)
        # 直接使用提供的路径
        TEACHER_PATH="$TEACHER_MODEL"
        ;;
esac

echo "==========================================" | tee -a "$LOG_FILE"
echo "阶段2：On-Policy Distillation (OPD)" | tee -a "$LOG_FILE"
echo "模型大小: $MODEL_SIZE" | tee -a "$LOG_FILE"
echo "配置文件: $CONFIG" | tee -a "$LOG_FILE"
echo "阶段1 Checkpoint: $STAGE1_CKPT" | tee -a "$LOG_FILE"
echo "教师模型: $TEACHER_PATH" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "OPD参数：" | tee -a "$LOG_FILE"
echo "  - 启用on-policy采样" | tee -a "$LOG_FILE"
echo "  - 采样温度: 1.0" | tee -a "$LOG_FILE"
echo "  - Top-p: 0.9" | tee -a "$LOG_FILE"
echo "  - 最大采样长度: 2048" | tee -a "$LOG_FILE"
echo "  - 每prompt轨迹数: 1" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# 检查阶段1 checkpoint是否存在
if [ ! -d "$STAGE1_CKPT" ]; then
    echo "错误: 阶段1 checkpoint 不存在: $STAGE1_CKPT" | tee -a "$LOG_FILE"
    echo "请先运行阶段1训练: bash scripts/train_stage1_pretrain.sh $MODEL_SIZE" | tee -a "$LOG_FILE"
    exit 1
fi

# 检查教师模型是否存在
if [ -n "$TEACHER_PATH" ] && [ ! -d "$TEACHER_PATH" ]; then
    echo "警告: 教师模型不存在: $TEACHER_PATH" | tee -a "$LOG_FILE"
    echo "下载教师: bash scripts/setup_teacher.sh $TEACHER_MODEL" | tee -a "$LOG_FILE"
    echo "继续使用无教师模式..." | tee -a "$LOG_FILE"
    TEACHER_PATH=""
fi

# 阶段2特定参数（启用真正的OPD采样）
# 注意：teacher_model_path 已在配置文件中设置，此处不再冗余覆盖
# 损失权重等参数也已在配置文件中设置，此处仅覆盖采样相关参数
STAGE2_ARGS="--opdttt.enable_opd_sampling=true"

# 自动检测可见 GPU 数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ -z "$CUDA_VISIBLE_DEVICES" ] || [ "$CUDA_VISIBLE_DEVICES" = "0,1,2,3,4,5,6,7" ]; then
    NUM_GPUS=8
fi
echo "使用 $NUM_GPUS 个 GPU" | tee -a "$LOG_FILE"

# 启动训练，输出到日志文件（配置文件作为位置参数）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    tasks/train_opdttt.py \
    "$CONFIG" \
    $STAGE2_ARGS 2>&1 | tee -a "$LOG_FILE"

echo "==========================================" | tee -a "$LOG_FILE"
echo "阶段2训练完成！" | tee -a "$LOG_FILE"
echo "最终Checkpoint位置: data/output/${MODEL_SIZE}_stage2_opd/checkpoints/" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
