#!/bin/bash
# =============================================================================
# OPD-TTT 训练启动脚本
# =============================================================================
#
# 简化的训练启动命令，支持通过别名指定教师模型
#
# 使用方法:
#   bash scripts/train.sh 500m qwen2.5-7b
#   bash scripts/train.sh 1b5 qwen2.5-32b
#   bash scripts/train.sh 500m --no-teacher
#
# =============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 预定义的教师模型路径
TEACHER_PATHS=(
    "qwen2.5-7b:model_assets/teacher_qwen2.5_7b"
    "qwen2-14b:model_assets/teacher_qwen2_14b"
    "qwen2.5-14b:model_assets/teacher_qwen2.5_14b"
    "qwen2.5-32b:model_assets/teacher_qwen2.5_32b"
    "qwen2.5-72b:model_assets/teacher_qwen2.5_72b"
    "llama2-13b:model_assets/teacher_llama2_13b"
    "llama3-8b:model_assets/teacher_llama3_8b"
)

# 显示帮助信息
show_help() {
    cat << 'HELP_EOF'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    OPD-TTT 训练启动脚本
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用法: bash scripts/train.sh <学生模型> [教师模型] [选项]

参数:
  学生模型          学生模型规模（500m 或 1b5）
  教师模型          教师模型别名（见下方列表）或 --no-teacher

教师模型别名:
  qwen2.5-7b       Qwen2.5-7B     (推荐用于 500M)
  qwen2-14b        Qwen2-14B      (用于 500M)
  qwen2.5-14b      Qwen2.5-14B   (用于 500M)
  qwen2.5-32b      Qwen2.5-32B   (推荐用于 1.5B)
  qwen2.5-72b      Qwen2.5-72B   (用于 1.5B)
  llama2-13b       LLaMA2-13B
  llama3-8b        LLaMA3-8B
  --no-teacher     不使用教师模型（纯 TTT）

选项:
  --gpus N             GPU 数量（默认: 8）
  --port N             主进程端口（默认: 29500）
  --config PATH        配置文件路径（覆盖默认）
  --help, -h           显示此帮助信息

示例:
  # 训练 500M 模型，使用 Qwen2.5-7B 作为教师
  bash scripts/train.sh 500m qwen2.5-7b

  # 训练 1.5B 模型，使用 Qwen2.5-32B 作为教师
  bash scripts/train.sh 1b5 qwen2.5-32b

  # 训练 500M 模型，不使用教师
  bash scripts/train.sh 500m --no-teacher

  # 使用 4 个 GPU
  bash scripts/train.sh 500m qwen2.5-7b --gpus 4

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HELP_EOF
}

# 解析参数
STUDENT_MODEL=""
TEACHER_ALIAS=""
CONFIG_FILE=""
NUM_GPUS=8
MASTER_PORT=29500

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        500m|500M)
            STUDENT_MODEL="500m"
            shift
            ;;
        1b5|1.5b|1B5)
            STUDENT_MODEL="1b5"
            shift
            ;;
        --no-teacher)
            TEACHER_ALIAS="--no-teacher"
            shift
            ;;
        -*)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            exit 1
            ;;
        *)
            if [[ -z "$TEACHER_ALIAS" ]] && [[ "$1" != "--no-teacher" ]]; then
                TEACHER_ALIAS="$1"
            else
                echo -e "${RED}错误: 额外的参数 $1${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# 检查学生模型参数
if [[ -z "$STUDENT_MODEL" ]]; then
    echo -e "${RED}错误: 请指定学生模型规模 (500m 或 1b5)${NC}"
    echo "使用 --help 查看帮助信息"
    exit 1
fi

# 设置默认配置文件
if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="$PROJECT_ROOT/configs/opdttt/llama3_sc_${STUDENT_MODEL}_opdttt.yaml"
fi

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

# 解析教师模型路径
TEACHER_PATH=""
if [[ "$TEACHER_ALIAS" == "--no-teacher" ]]; then
    TEACHER_PATH=""
elif [[ -n "$TEACHER_ALIAS" ]]; then
    # 在预定义列表中查找
    for entry in "${TEACHER_PATHS[@]}"; do
        IFS=':' read -ra PARTS <<< "$entry"
        if [[ "${PARTS[0]}" == "$TEACHER_ALIAS" ]]; then
            TEACHER_PATH="${PARTS[1]}"
            break
        fi
    done

    # 如果没找到，使用提供的别名作为路径
    if [[ -z "$TEACHER_PATH" ]]; then
        TEACHER_PATH="$TEACHER_ALIAS"
    fi
fi

# 检查教师模型路径（如果指定）
if [[ -n "$TEACHER_PATH" ]] && [[ ! -d "$TEACHER_PATH" ]]; then
    echo -e "${YELLOW}警告: 教师模型路径不存在: $TEACHER_PATH${NC}"
    echo "请先下载教师模型:"
    echo "  bash scripts/setup_teacher.sh $TEACHER_ALIAS"
    echo ""
    read -p "是否继续? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 输出训练信息
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}                    OPD-TTT 训练${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}配置信息:${NC}"
echo "  学生模型: ${STUDENT_MODEL}"
echo "  配置文件: $CONFIG_FILE"
if [[ -n "$TEACHER_PATH" ]]; then
    echo "  教师模型: $TEACHER_PATH"
else
    echo "  教师模型: (未使用)"
fi
echo "  GPU 数量: $NUM_GPUS"
echo "  主端口: $MASTER_PORT"
echo ""

# 检查 GPU 可用性
python3 -c "
import torch
import sys
device_count = torch.cuda.device_count()
if device_count == 0:
    print('错误: 未检测到 GPU')
    sys.exit(1)
if device_count < $NUM_GPUS:
    print(f'警告: 请求 $NUM_GPUS 个 GPU，但只检测到 {device_count} 个')
print(f'检测到 {device_count} 个 GPU')
"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 构建训练命令
CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT tasks/train_opdttt.py --config $CONFIG_FILE"

# 添加教师模型路径（如果指定）
if [[ -n "$TEACHER_PATH" ]]; then
    CMD="$CMD --opdttt.teacher_model_path $TEACHER_PATH"
fi

# 添加日志输出
OUTPUT_DIR="$PROJECT_ROOT/data/output/${STUDENT_MODEL}_opdttt"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train.log"
CMD="$CMD 2>&1 | tee $LOG_FILE"

echo -e "${BLUE}开始训练...${NC}"
echo "日志文件: $LOG_FILE"
echo ""

# 启动训练
eval "$CMD"

# 训练完成
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}训练完成!${NC}"
echo -e "${GREEN}结果保存在: $OUTPUT_DIR${NC}"
echo -e "${GREEN}日志文件: $LOG_FILE${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
