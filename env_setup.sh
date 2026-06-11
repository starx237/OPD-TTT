#!/bin/bash
# =============================================================================
# Environment setup for In-Place TTT training scripts.
# Source this file from any shell script in the project root:
#   source "$(cd "$(dirname "$0")" && pwd)/env_setup.sh"
# =============================================================================

# Determine project root (directory containing this script)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load machine-specific overrides from .env (if present)
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# ---- Python environment ----
# 优先使用 VENV_DIR（虚拟环境），否则使用 CONDA_ENV_DIR（conda 环境）
if [ -n "$VENV_DIR" ] && [ -d "$VENV_DIR" ]; then
    export PATH="$VENV_DIR/bin:$PATH"
elif [ -n "$CONDA_ENV_DIR" ]; then
    export PATH="$CONDA_ENV_DIR/bin:$PATH"
fi

# ---- PYTHONPATH ----
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VeOmni:${PROJECT_ROOT}/hf_models:${PYTHONPATH:-}"

# ---- Cache directories (with defaults) ----
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/.cache/pip}"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/.cache/tmp}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"

# ---- Training-optimized environment variables ----
export NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NTHREADS=64
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=1800
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# ---- Model loading backend ----
# Use HuggingFace backend for all model types (llama, lact_swiglu)
# This ensures custom models like LaCT are loaded via AutoModel registration
export MODELING_BACKEND=hf

# No InfiniBand fallback
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0,enp
