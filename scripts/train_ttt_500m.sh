#!/bin/bash
# OPD-TTT 500M TTT 模型训练脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/env_setup.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}" bash "$SCRIPT_DIR/train.sh" tasks/train_torch.py \
  "$PROJECT_ROOT/configs/ttt/llama3_sc_500m_ttt.yaml" \
  --model.tokenizer_path "$PROJECT_ROOT/model_assets/llama_500m_config" \
  --model.attn_implementation flash_attention_2 \
  "$@"
