#!/bin/bash
set -eo pipefail

# RULER evaluation for original Qwen3.5-2B weights with forced SWA=4096.
# Uses eval_ruler_oc.py (with _load_model patch for Qwen3_5TTTForCausalLM).
# Supports task-level resume via --reuse latest.
#
# Usage:
#   bash scripts/run_ruler_original_swa.sh [GPU_ID] [LENGTHS...]
#   Example: bash scripts/run_ruler_original_swa.sh 1 "2k 4k 8k 16k 32k"
#
# Created: 2026-07-10

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${1:-1}"
LENGTHS="${2:-2k 4k 8k 16k 32k}"
PYTHON_BIN="/root/miniconda3/bin/python"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${PROJECT_ROOT}/results/ruler_original_swa"

echo "============================================"
echo "RULER Evaluation — Original Qwen3.5-2B + Forced SWA=4096"
echo "GPU: ${GPU_ID}  Lengths: ${LENGTHS}"
echo "Started: $(date)"
echo "============================================"

for len in ${LENGTHS}; do
    WORK_DIR="${PROJECT_ROOT}/results/ruler_original_swa/${len}"

    # Generate temporary config from existing RULER config
    TMP_CONFIG="${PROJECT_ROOT}/eval_config/ruler_original_swa_${len}.py"
    sed -e 's/from \.models import models/from .models_original_swa import models/' \
        -e "s|results/ruler/${len}|results/ruler_original_swa/${len}|g" \
        "${PROJECT_ROOT}/eval_config/ruler_${len}.py" > "${TMP_CONFIG}"

    # Check if a previous run exists for resume
    REUSE_FLAG=""
    if [ -d "${WORK_DIR}" ] && [ -n "$(ls -A "${WORK_DIR}" 2>/dev/null)" ]; then
        REUSE_FLAG="--reuse latest"
        echo "[$(date)] Resuming ruler_original_swa_${len} (--reuse latest)"
    else
        echo "[$(date)] Starting ruler_original_swa_${len} (fresh)"
    fi

    ${PYTHON_BIN} "${SCRIPT_DIR}/eval_ruler_oc.py" \
        "${TMP_CONFIG}" \
        --debug \
        ${REUSE_FLAG} \
        2>&1 | tee "results/ruler_original_swa/ruler_original_swa_${len}.log"

    echo "[$(date)] Finished ruler_original_swa_${len}"
done

echo "============================================"
echo "All done: $(date)"
echo "============================================"
