#!/bin/bash
# =============================================================================
# OPD-TTT 项目设置脚本
# On-Policy Distillation Enhanced Test-Time Training
#
# 此脚本创建 OPD-TTT 项目所需的目录结构和配置文件
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "  OPD-TTT 项目设置"
echo "=========================================="
echo ""
echo "项目根目录: $PROJECT_ROOT"
echo ""

# ---- 创建目录结构 ----
echo "创建目录结构..."

mkdir -p "$PROJECT_ROOT/model_assets"
mkdir -p "$PROJECT_ROOT/configs/opdttt"
mkdir -p "$PROJECT_ROOT/configs/baseline"
mkdir -p "$PROJECT_ROOT/configs/ttt"
mkdir -p "$PROJECT_ROOT/configs/other"
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/data/output"
mkdir -p "$PROJECT_ROOT/data/val"
mkdir -p "$PROJECT_ROOT/scripts"
mkdir -p "$PROJECT_ROOT/hf_models"
mkdir -p "$PROJECT_ROOT/.cache/pip"
mkdir -p "$PROJECT_ROOT/.cache/tmp"
mkdir -p "$PROJECT_ROOT/.cache/huggingface"

echo "✓ 目录结构创建完成"

# ---- 创建 LLaMA 500M 配置 ----
echo ""
echo "创建 LLaMA 500M 配置..."

mkdir -p "$PROJECT_ROOT/model_assets/llama_500m_config"
cat > "$PROJECT_ROOT/model_assets/llama_500m_config/config.json" << 'EOF'
{
  "architecture": "llama",
  "hidden_size": 768,
  "intermediate_size": 2048,
  "num_hidden_layers": 24,
  "num_attention_heads": 24,
  "num_key_value_heads": 8,
  "vocab_size": 128256,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "rope_scaling": null,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "attention_bias": false
}
EOF

# 创建基本的 tokenizer 配置
cat > "$PROJECT_ROOT/model_assets/llama_500m_config/tokenizer_config.json" << 'EOF'
{
  "bos_token": "<|begin_of_text|>",
  "eos_token": "<|end_of_text|>",
  "pad_token": "<|end_of_text|>",
  "unk_token": null,
  "tokenizer_type": "llama"
}
EOF

cat > "$PROJECT_ROOT/model_assets/llama_500m_config/special_tokens_map.json" << 'EOF'
{
  "bos_token": "<|begin_of_text|>",
  "eos_token": "<|end_of_text|>",
  "pad_token": "<|end_of_text|>"
}
EOF

# 如果存在原始 tokenizer.json，复制它
if [ -f "$PROJECT_ROOT/model_assets/tokenizer.json" ]; then
    cp "$PROJECT_ROOT/model_assets/tokenizer.json" \
       "$PROJECT_ROOT/model_assets/llama_500m_config/tokenizer.json"
    echo "  ✓ 复制 tokenizer.json"
fi

echo "✓ LLaMA 500M 配置创建完成"

# ---- 创建 LLaMA 1.5B 配置 ----
echo ""
echo "创建 LLaMA 1.5B 配置..."

mkdir -p "$PROJECT_ROOT/model_assets/llama_1b5_config"
cat > "$PROJECT_ROOT/model_assets/llama_1b5_config/config.json" << 'EOF'
{
  "architecture": "llama",
  "hidden_size": 1536,
  "intermediate_size": 4096,
  "num_hidden_layers": 24,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 128256,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "rope_scaling": null,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "attention_bias": false
}
EOF

cp "$PROJECT_ROOT/model_assets/llama_500m_config/tokenizer_config.json" \
   "$PROJECT_ROOT/model_assets/llama_1b5_config/tokenizer_config.json"
cp "$PROJECT_ROOT/model_assets/llama_500m_config/special_tokens_map.json" \
   "$PROJECT_ROOT/model_assets/llama_1b5_config/special_tokens_map.json"

if [ -f "$PROJECT_ROOT/model_assets/llama_500m_config/tokenizer.json" ]; then
    cp "$PROJECT_ROOT/model_assets/llama_500m_config/tokenizer.json" \
       "$PROJECT_ROOT/model_assets/llama_1b5_config/tokenizer.json"
fi

echo "✓ LLaMA 1.5B 配置创建完成"

# ---- 创建 TTT 配置副本 ----
echo ""
echo "创建 TTT 配置副本..."

# LLaMA 500M TTT
mkdir -p "$PROJECT_ROOT/model_assets/llama_500m_ttt_config"
cp "$PROJECT_ROOT/model_assets/llama_500m_config/config.json" \
   "$PROJECT_ROOT/model_assets/llama_500m_ttt_config/config.json"
cp "$PROJECT_ROOT/model_assets/llama_500m_config/tokenizer.json" \
   "$PROJECT_ROOT/model_assets/llama_500m_ttt_config/" 2>/dev/null || true
cp "$PROJECT_ROOT/model_assets/llama_500m_config/tokenizer_config.json" \
   "$PROJECT_ROOT/model_assets/llama_500m_ttt_config/" 2>/dev/null || true
cp "$PROJECT_ROOT/model_assets/llama_500m_config/special_tokens_map.json" \
   "$PROJECT_ROOT/model_assets/llama_500m_ttt_config/" 2>/dev/null || true

# LLaMA 1.5B TTT
mkdir -p "$PROJECT_ROOT/model_assets/llama_1b5_ttt_config"
cp "$PROJECT_ROOT/model_assets/llama_1b5_config/config.json" \
   "$PROJECT_ROOT/model_assets/llama_1b5_ttt_config/config.json"
cp "$PROJECT_ROOT/model_assets/llama_1b5_config/tokenizer.json" \
   "$PROJECT_ROOT/model_assets/llama_1b5_ttt_config/" 2>/dev/null || true
cp "$PROJECT_ROOT/model_assets/llama_1b5_config/tokenizer_config.json" \
   "$PROJECT_ROOT/model_assets/llama_1b5_ttt_config/" 2>/dev/null || true
cp "$PROJECT_ROOT/model_assets/llama_1b5_config/special_tokens_map.json" \
   "$PROJECT_ROOT/model_assets/llama_1b5_ttt_config/" 2>/dev/null || true

echo "✓ TTT 配置创建完成"

# ---- 创建 .env 文件示例 ----
echo ""
echo "创建 .env 文件示例..."

cat > "$PROJECT_ROOT/.env.example" << 'EOF'
# =============================================================================
# OPD-TTT 环境配置示例
# 复制此文件为 .env 并根据您的环境修改
# =============================================================================

# ---- Python 环境 ----
# 使用虚拟环境（推荐）
VENV_DIR="$PROJECT_ROOT/.venv"

# 或者使用 conda 环境
# CONDA_ENV_DIR="/path/to/conda/env"

# ---- GPU 配置 ----
# 可见的 GPU 设备（逗号分隔）
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# ---- 缓存目录 ----
PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
TMPDIR="$PROJECT_ROOT/.cache/tmp"
HF_HOME="$PROJECT_ROOT/.cache/huggingface"

# ---- 训练配置 ----
# WandB 项目名称
# WANDB_PROJECT="opdttt-experiments"
# WANDB_API_KEY="your-key-here"
EOF

echo "✓ .env 示例文件创建完成"

# ---- 创建 .gitignore ----
echo ""
echo "创建 .gitignore..."

cat > "$PROJECT_ROOT/.gitignore" << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# 虚拟环境
.venv/
venv/
ENV/

# 环境配置
.env

# 数据和输出
data/
!data/.gitkeep

# 缓存
.cache/
*.log

# IDE
.vscode/
.idea/
.cursor/
*.swp
*.swo

# 模型文件
*.safetensors
*.bin
*.pt

# 检查点
checkpoints/
output/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# 临时文件
tmp/
temp/
EOF

echo "✓ .gitignore 创建完成"

# ---- 创建数据目录占位文件 ----
echo ""
echo "创建数据目录结构..."

mkdir -p "$PROJECT_ROOT/data/pretrain"
mkdir -p "$PROJECT_ROOT/data/val"
touch "$PROJECT_ROOT/data/.gitkeep"

echo "✓ 数据目录结构创建完成"

# ---- 总结 ----
echo ""
echo "=========================================="
echo "  设置完成！"
echo "=========================================="
echo ""
echo "已创建以下目录结构："
echo "  model_assets/       - 模型配置和分词器"
echo "  configs/            - 训练配置文件"
echo "    ├── opdttt/       - OPD-TTT 配置"
echo "    ├── baseline/    - 基线配置"
echo "    ├── ttt/         - TTT 配置"
echo "    └── other/       - 其他配置"
echo "  data/              - 数据目录"
echo "  scripts/           - 脚本目录"
echo "  hf_models/         - HuggingFace 模型"
echo ""
echo "后续步骤："
echo "  1. 复制 .env.example 为 .env 并配置您的环境"
echo "  2. 运行验证脚本: python scripts/test_opdttt_setup.py"
echo "  3. 准备训练数据"
echo "  4. 开始训练: bash scripts/train_opdttt_500m.sh"
echo ""
