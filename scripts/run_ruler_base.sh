#!/bin/bash
# RULER evaluation for Qwen3.5-2B-Base (no CPT, no TTT, no instruct).
#
# Usage:
#   bash scripts/run_ruler_base.sh [GPU_ID] [LENGTHS...]
#   Example: bash scripts/run_ruler_base.sh 1 "16k 32k"
#
# Created: 2026-07-11

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${1:-1}"
LENGTHS="${2:-16k 32k}"
PYTHON_BIN="/root/miniconda3/bin/python"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${PROJECT_ROOT}/results/ruler_base"

echo "============================================"
echo "RULER Base Evaluation (Qwen3.5-2B-Base)"
echo "GPU: ${GPU_ID}  Lengths: ${LENGTHS}"
echo "Started: $(date)"
echo "============================================"

for len in ${LENGTHS}; do
    WORK_DIR="${PROJECT_ROOT}/results/ruler_base/${len}"
    CONFIG="${PROJECT_ROOT}/eval_config/ruler_base_${len}.py"

    if [ ! -f "${CONFIG}" ]; then
        echo "Config not found: ${CONFIG}, skipping"
        continue
    fi

    # Check if a previous run exists for resume
    REUSE_FLAG=""
    if [ -d "${WORK_DIR}" ] && [ -n "$(ls -A "${WORK_DIR}" 2>/dev/null)" ]; then
        REUSE_FLAG="--reuse latest"
        echo "[$(date)] Resuming ruler_base_${len} (--reuse latest)"
    else
        echo "[$(date)] Starting ruler_base_${len} (fresh)"
    fi

    ${PYTHON_BIN} "${SCRIPT_DIR}/eval_ruler_baseline.py" \
        "${CONFIG}" \
        --debug \
        ${REUSE_FLAG} \
        2>&1 | tee "results/ruler_base/ruler_base_${len}.log"

    echo "[$(date)] Finished ruler_base_${len}"
done

echo "============================================"
echo "All done: $(date)"
echo "============================================"
