#!/bin/bash

# RULER evaluation script using OpenCompass (100% standard, aligned with
# the official In-Place-TTT repository).
#
# This script launches RULER evaluation for multiple context lengths in
# parallel, each on a separate GPU. It supports resume via OpenCompass's
# --reuse flag.
#
# Usage:
#   bash scripts/eval_ruler_opencompass.sh [GPU_IDS] [LENGTHS] [EXTRA_ARGS]
#
# Arguments:
#   GPU_IDS   - Comma-separated GPU IDs (default: "1,2,3,4,6")
#   LENGTHS   - Comma-separated context lengths (default: "4k,8k,16k,32k,64k")
#   EXTRA_ARGS - Extra args passed to OpenCompass (e.g. "--reuse")
#
# Example:
#   bash scripts/eval_ruler_opencompass.sh 1,2,3,4,6 4k,8k,16k,32k,64k --reuse
#   bash scripts/eval_ruler_opencompass.sh 1 4k --debug
#
# Created: 2026-07-09

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source environment setup
source "$PROJECT_ROOT/env_setup.sh"

# Arguments
GPU_IDS="${1:-1,2,3,4,6}"
LENGTHS="${2:-4k,8k,16k,32k,64k}"
EXTRA_ARGS="${3:-}"

# Split comma-separated values
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
IFS=',' read -ra LEN_ARRAY <<< "$LENGTHS"

NUM_GPUS=${#GPU_ARRAY[@]}
NUM_LENS=${#LEN_ARRAY[@]}

if [ "$NUM_GPUS" -ne "$NUM_LENS" ]; then
    echo "Error: Number of GPUs ($NUM_GPUS) must match number of lengths ($NUM_LENS)"
    exit 1
fi

echo "============================================"
echo "RULER Evaluation (OpenCompass Standard)"
echo "============================================"
echo "GPUs:    $GPU_IDS"
echo "Lengths: $LENGTHS"
echo "Extra:   $EXTRA_ARGS"
echo "============================================"
echo ""

# Clear log.txt if no --reuse (fresh start)
if [[ "$EXTRA_ARGS" != *"--reuse"* ]]; then
    > "$PROJECT_ROOT/log.txt"
    echo "[$(date)] Cleared log.txt for fresh evaluation" | tee -a "$PROJECT_ROOT/log.txt"
fi

# Launch each context length on its own GPU
PIDS=()
for i in "${!LEN_ARRAY[@]}"; do
    LEN="${LEN_ARRAY[$i]}"
    GPU="${GPU_ARRAY[$i]}"
    CONFIG="eval_config/ruler_${LEN}.py"

    echo "[$(date)] Starting ruler_${LEN} on GPU ${GPU}" | tee -a "$PROJECT_ROOT/log.txt"

    CUDA_VISIBLE_DEVICES=$GPU "$VENV_DIR/bin/python" "$SCRIPT_DIR/eval_ruler_oc.py" \
        "$CONFIG" \
        --debug \
        $EXTRA_ARGS \
        >> "$PROJECT_ROOT/log.txt" 2>&1 &

    PIDS+=($!)
    echo "[$(date)] Launched ruler_${LEN} (PID: ${PIDS[-1]})" | tee -a "$PROJECT_ROOT/log.txt"
    sleep 5
done

echo "[$(date)] All launched. Waiting for completion..." | tee -a "$PROJECT_ROOT/log.txt"
wait
echo "[$(date)] All RULER evaluations completed." | tee -a "$PROJECT_ROOT/log.txt"
