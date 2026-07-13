#!/bin/bash
set -eo pipefail

# RULER evaluation for step 5000 checkpoint with OpenCompass (TTT inference).
# Same method as step 3000 (run_ruler_opencompass.sh), but outputs to
# results/ruler_step5000/ to avoid mixing with step 3000 results.
#
# Usage:
#   bash scripts/run_ruler_step5000.sh [GPU_ID] [LENGTHS...]
#   Example: bash scripts/run_ruler_step5000.sh 1 "2k 4k 8k 16k 32k"
#
# Created: 2026-07-12

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

OUT_BASE="${PROJECT_ROOT}/results/ruler_step5000"
mkdir -p "${OUT_BASE}"

echo "============================================"
echo "RULER Step 5000 Evaluation (OpenCompass + TTT inference)"
echo "GPU: ${GPU_ID}  Lengths: ${LENGTHS}"
echo "Started: $(date)"
echo "============================================"

for len in ${LENGTHS}; do
    WORK_DIR="${OUT_BASE}/${len}"

    # Check if a previous run exists for resume
    REUSE_FLAG=""
    if [ -d "${WORK_DIR}" ] && [ -n "$(ls -A "${WORK_DIR}" 2>/dev/null)" ]; then
        REUSE_FLAG="--reuse latest"
        echo "[$(date)] Resuming ruler_step5000_${len} (--reuse latest, skipping completed tasks)"
    else
        echo "[$(date)] Starting ruler_step5000_${len} (fresh run)"
    fi

    ${PYTHON_BIN} "${SCRIPT_DIR}/eval_ruler_oc.py" \
        eval_config/ruler_${len}.py \
        --debug \
        -w "${WORK_DIR}" \
        ${REUSE_FLAG} \
        2>&1 | tee "${OUT_BASE}/ruler_step5000_${len}.log"

    echo "[$(date)] Finished ruler_step5000_${len}"
done

echo "============================================"
echo "All done: $(date)"
echo "============================================"
