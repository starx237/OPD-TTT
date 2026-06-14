#!/bin/bash
# =============================================================================
# OPD-TTT 教师模型下载脚本
# =============================================================================
#
# 从 HuggingFace 下载教师模型权重
#
# 使用方法:
#   bash scripts/setup_teacher.sh qwen2.5-7b
#   bash scripts/setup_teacher.sh qwen2.5-7b model_assets/teacher_qwen2.5_7b
#   bash scripts/setup_teacher.sh --list
#
# =============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录（自动检测）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 默认参数
MIRROR="https://hf-mirror.com"
TOKEN=""
OUTPUT=""
SKIP_TOKENIZER=false

# 解析参数
MODEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --list|-l)
            cat << 'EOF'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    推荐的教师模型选择
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  学生模型          推荐教师              别名          参数比例
  ──────────────────────────────────────────────────────────────────
  500M        →  Qwen2.5-7B            qwen2.5-7b         14×
  500M        →  Qwen2-14B             qwen2-14b         28×
  500M        →  Llama2-13B           llama2-13b         26×
  1.5B        →  Qwen2.5-32B           qwen2.5-32b       21×
  1.5B        →  Qwen2.5-72B           qwen2.5-72b       48×

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

示例:
  bash scripts/setup_teacher.sh qwen2.5-7b
  bash scripts/setup_teacher.sh qwen2.5-7b model_assets/teacher_qwen2.5_7b
  bash scripts/setup_teacher.sh qwen2.5-7b --mirror https://hf-mirror.com
  bash scripts/setup_teacher.sh llama3-8b --token YOUR_TOKEN
EOF
            exit 0
            ;;
        --mirror|-m)
            MIRROR="$2"
            shift 2
            ;;
        --token|-t)
            TOKEN="$2"
            shift 2
            ;;
        --skip-tokenizer)
            SKIP_TOKENIZER=true
            shift
            ;;
        --only-config)
            ONLY_CONFIG=true
            shift
            ;;
        -h|--help)
            cat << 'EOF'
用法: bash scripts/setup_teacher.sh <模型> [输出目录] [选项]

参数:
  模型              HuggingFace 模型名称或别名（必需）
  输出目录          模型保存目录（可选，默认: model_assets/teacher_<模型>）

选项:
  --mirror, -m      HuggingFace 镜像站（默认: https://hf-mirror.com）
  --token, -t       HuggingFace 访问令牌（用于 gated 模型）
  --skip-tokenizer  跳过 tokenizer 下载
  --only-config     仅下载配置文件
  --list, -l        列出推荐的教师模型
  --help, -h        显示此帮助信息

示例:
  bash scripts/setup_teacher.sh qwen2.5-7b
  bash scripts/setup_teacher.sh qwen2.5-7b model_assets/teacher_qwen2.5_7b
  bash scripts/setup_teacher.sh --list
EOF
            exit 0
            ;;
        -*)
            echo "错误: 未知选项 $1"
            exit 1
            ;;
        *)
            if [[ -z "$MODEL" ]]; then
                MODEL="$1"
            elif [[ -z "$OUTPUT" ]]; then
                OUTPUT="$1"
            else
                echo "错误: 额外的参数 $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# 检查模型参数
if [[ -z "$MODEL" ]]; then
    echo -e "${RED}错误: 请指定要下载的教师模型${NC}"
    echo "使用 --list 查看推荐的教师模型"
    echo "使用 --help 查看帮助信息"
    exit 1
fi

# 设置默认输出目录
if [[ -z "$OUTPUT" ]]; then
    MODEL_SAFE=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '_' | tr '-' '_')
    OUTPUT="model_assets/teacher_${MODEL_SAFE}"
fi

# 构建 Python 命令
CMD="python scripts/setup_teacher.py --model $MODEL --output $OUTPUT --mirror $MIRROR"

if [[ -n "$TOKEN" ]]; then
    CMD="$CMD --token $TOKEN"
fi

if [[ "$SKIP_TOKENIZER" == true ]]; then
    CMD="$CMD --skip-tokenizer"
fi

if [[ "$ONLY_CONFIG" == true ]]; then
    CMD="$CMD --only-config"
fi

# 执行下载
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}           OPD-TTT 教师模型下载${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "模型: $MODEL"
echo "输出: $OUTPUT"
echo "镜像: $MIRROR"
echo ""

# 检查 Python 和依赖
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到 Python${NC}"
    exit 1
fi

# 日志文件路径
LOG_FILE="$OUTPUT/download.log"

# 创建输出目录
mkdir -p "$OUTPUT"

# 执行并记录日志
echo "日志文件: $LOG_FILE"
echo ""

eval "$CMD" 2>&1 | tee "$LOG_FILE"

# 检查执行结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ 下载完成!${NC}"
    echo -e "${GREEN}  模型保存在: $OUTPUT${NC}"
    echo -e "${GREEN}  日志保存在: $LOG_FILE${NC}"
else
    echo ""
    echo -e "${RED}✗ 下载失败，请查看日志: $LOG_FILE${NC}"
    exit 1
fi
