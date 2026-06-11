#!/bin/bash
# =============================================================================
# OPD-TTT Tokenizer 设置脚本
# 从 HuggingFace 下载 tokenizer 文件到指定目录
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/env_setup.sh"

echo "=========================================="
echo "  OPD-TTT Tokenizer 设置"
echo "=========================================="
echo ""

# 默认参数
MODEL_NAME=""
OUTPUT_DIR=""
HF_TOKEN="${HF_TOKEN:-}"

# 显示帮助信息
show_help() {
    cat << EOF
使用方法:
    bash scripts/setup_tokenizer.sh --model MODEL_NAME --output OUTPUT_DIR [--token TOKEN]

参数:
    --model MODEL_NAME      HuggingFace 模型名称或预定义别名 (必需)
    --output OUTPUT_DIR     输出目录路径 (必需)
    --token TOKEN           HuggingFace 访问令牌 (可选，用于 gated 模型)

预定义模型别名:
    llama2-7b     meta-llama/Llama-2-7b
    llama2-13b    meta-llama/Llama-2-13b
    llama3-8b     meta-llama/Meta-Llama-3-8B
    llama3-70b    meta-llama/Meta-Llama-3-70B
    qwen1.5-7b    Qwen/Qwen1.5-7B
    qwen2-7b      Qwen/Qwen2-7B
    qwen2.5-7b    Qwen/Qwen2.5-7B
    mistral-7b    mistralai/Mistral-7B-v0.1

示例:
    # 下载 LLaMA-2 tokenizer
    bash scripts/setup_tokenizer.sh --model llama2-7b --output model_assets/llama_500m_config

    # 下载 LLaMA-3 tokenizer (需要 token)
    bash scripts/setup_tokenizer.sh --model llama3-8b --output model_assets/llama_1b5_config --token YOUR_TOKEN

    # 直接使用 HuggingFace 模型名
    bash scripts/setup_tokenizer.sh --model meta-llama/Llama-2-7b --output model_assets/llama_500m_config

环境变量:
    HF_TOKEN    HuggingFace 访问令牌 (可设置在 .env 中)

注意:
    对于 gated 模型（如 LLaMA），需要:
    1. 在 https://huggingface.co/settings/tokens 获取访问令牌
    2. 使用 --token 参数或在 .env 中设置 HF_TOKEN
EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "错误: 未知参数 $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$MODEL_NAME" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "错误: 缺少必需参数"
    echo ""
    show_help
    exit 1
fi

# 转换为绝对路径
OUTPUT_DIR_ABS="$PROJECT_ROOT/$OUTPUT_DIR"
if [[ "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_DIR_ABS="$OUTPUT_DIR"
fi

echo "配置:"
echo "  模型: $MODEL_NAME"
echo "  输出: $OUTPUT_DIR_ABS"
echo "  Token: ${HF_TOKEN:+已设置}"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR_ABS"

# 构建 Python 命令
PYTHON_CMD="python $SCRIPT_DIR/setup_tokenizer.py --model $MODEL_NAME --output $OUTPUT_DIR_ABS"

# 添加 token 参数（如果提供）
if [ -n "$HF_TOKEN" ]; then
    PYTHON_CMD="$PYTHON_CMD --token $HF_TOKEN"
fi

# 执行
echo "正在下载 tokenizer..."
echo ""
cd "$PROJECT_ROOT"
eval "$PYTHON_CMD"

echo ""
echo "=========================================="
echo "  Tokenizer 设置完成！"
echo "=========================================="
