#!/bin/bash
set -eo pipefail

# RULER evaluation with OpenCompass, supporting task-level resume.
#
# Resume mechanism:
#   OpenCompass's local_api.py skips tasks whose prediction files already
#   exist (if osp.exists(out_path): continue). By passing --reuse latest,
#   OpenCompass reuses the latest timestamp directory instead of creating
#   a new one, so completed tasks are skipped and only missing ones run.
#
# Usage:
#   bash scripts/run_ruler_opencompass.sh [GPU_ID] [LENGTHS...]
#   Example: bash scripts/run_ruler_opencompass.sh 1 "2k 4k 8k 16k 32k"
#
# To force re-run a specific task, delete its prediction json:
#   results/ruler/{len}/{timestamp}/predictions/{model}/{task}.json
#
# Created: 2026-07-09

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

mkdir -p "${PROJECT_ROOT}/results/ruler"

echo "============================================"
echo "RULER Evaluation (OpenCompass + TTT inference)"
echo "GPU: ${GPU_ID}  Lengths: ${LENGTHS}"
echo "Started: $(date)"
echo "============================================"

for len in ${LENGTHS}; do
    WORK_DIR="${PROJECT_ROOT}/results/ruler/${len}"

    # Check if a previous run exists for resume
    REUSE_FLAG=""
    if [ -d "${WORK_DIR}" ] && [ -n "$(ls -A "${WORK_DIR}" 2>/dev/null)" ]; then
        REUSE_FLAG="--reuse latest"
        echo "[$(date)] Resuming ruler_${len} (--reuse latest, skipping completed tasks)"
    else
        echo "[$(date)] Starting ruler_${len} (fresh run)"
    fi

    ${PYTHON_BIN} "${SCRIPT_DIR}/eval_ruler_oc.py" \
        eval_config/ruler_${len}.py \
        --debug \
        ${REUSE_FLAG} \
        2>&1 | tee "results/ruler/ruler_${len}.log"

    echo "[$(date)] Finished ruler_${len}"
done

echo "============================================"
echo "All done: $(date)"
echo "============================================"
