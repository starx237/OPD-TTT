#!/bin/bash
# OPD-TTT 冒烟测试脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/env_setup.sh"

mkdir -p "$PROJECT_ROOT/data"

python -c "
import json
with open('$PROJECT_ROOT/data/smoke_test.jsonl', 'w') as f:
    for i in range(500):
        f.write(json.dumps({'content_split': 'The quick brown fox jumps over the lazy dog. ' * 100}, ensure_ascii=False) + '\n')
"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}" bash "$SCRIPT_DIR/train.sh" tasks/train_torch.py \
  "$PROJECT_ROOT/configs/baseline/llama3_sc_500m_baseline.yaml" \
  --data.train_path smoke_test.jsonl \
  --data.train_size 100000 \
  --train.max_steps 1 \
  --train.rmpad false \
  --train.use_wandb false \
  --train.output_dir "$PROJECT_ROOT/data/output/smoke_test" \
  --model.tokenizer_path "$PROJECT_ROOT/model_assets/llama_500m_config" \
  --model.attn_implementation flash_attention_2
