#!/bin/bash
# OPD-TTT 500M 基线模型训练脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/env_setup.sh"

export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=WARN
# Reduce logging output
export PYTHONWARNINGS=ignore
export VEOMNI_VERBOSITY=warning
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
# Memory optimization for PyTorch 2.5.1 FSDP2
# Using expandable_segments to help with memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,6,7}" bash "$SCRIPT_DIR/train.sh" tasks/train_torch.py \
  "$PROJECT_ROOT/configs/baseline/llama3_sc_500m_baseline.yaml" \
  --model.tokenizer_path "$PROJECT_ROOT/model_assets/llama_500m_config" \
  --model.attn_implementation flash_attention_2 \
  "$@"
