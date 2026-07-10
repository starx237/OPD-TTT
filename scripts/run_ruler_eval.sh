#!/bin/bash
# RULER 评估启动脚本：TTT-on/off × 5 lengths × 8 tasks × 100 samples
#
# 断点续训：脚本会在 results 文件中检查已完成的 task×length，自动跳过
# 用法: bash scripts/run_ruler_eval.sh [GPU_ID] [STEP] [PYTHON]
#   GPU_ID:  GPU 编号（默认 1）
#   STEP:    checkpoint 步数（默认 2000）
#   PYTHON:  python 路径（默认 /root/miniconda3/bin/python3）
set -e

GPU_ID=${1:-1}
STEP=${2:-2000}
PYTHON=${3:-/root/miniconda3/bin/python3}

PROJECT_ROOT=/h3c/haoxiang/TTT-OPD
MODEL_PATH="data/output/qwen35_2b_ttt/hf_step${STEP}"
TOKENIZER_PATH="model_assets/qwen3.5-2b"
OUTPUT_DIR="results/ruler/step${STEP}"
LENGTHS="2048,4096,8192,16384,32768"
NUM_SAMPLES=100

cd $PROJECT_ROOT

export CUDA_VISIBLE_DEVICES=$GPU_ID
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export MODELING_BACKEND=hf

if [ -f .env ]; then
    _GPU=$CUDA_VISIBLE_DEVICES
    export $(grep -v '^#' .env | xargs)
    export CUDA_VISIBLE_DEVICES=$_GPU
fi

LOG="ruler_step${STEP}.log"
> $LOG
echo "=== RULER Eval Start: $(date) ===" | tee -a $LOG
echo "GPU: $GPU_ID | Step: $STEP | Samples: $NUM_SAMPLES" | tee -a $LOG
echo "Model: $MODEL_PATH" | tee -a $LOG
echo "Output: $OUTPUT_DIR" | tee -a $LOG

# 先跑 TTT-off（更快，不需要 prefill+extract）
echo "" | tee -a $LOG
echo ">>> TTT-OFF <<<" | tee -a $LOG
$PYTHON scripts/eval_ruler.py \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --lengths "$LENGTHS" \
    --num_samples $NUM_SAMPLES \
    --ttt_mode off \
    --gpu 0 \
    --max_new_tokens 128 \
    2>&1 | tee -a $LOG

# 再跑 TTT-on
echo "" | tee -a $LOG
echo ">>> TTT-ON <<<" | tee -a $LOG
$PYTHON scripts/eval_ruler.py \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --lengths "$LENGTHS" \
    --num_samples $NUM_SAMPLES \
    --ttt_mode on \
    --gpu 0 \
    --max_new_tokens 128 \
    2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== RULER Eval Finished: $(date) ===" | tee -a $LOG

# 汇总结果
echo "" | tee -a $LOG
echo "=== Summary ===" | tee -a $LOG
$PYTHON -c "
import json, os, collections
for mode in ['on', 'off']:
    fpath = os.path.join('$OUTPUT_DIR', f'results_{mode}.json')
    if not os.path.exists(fpath):
        continue
    results = [json.loads(l) for l in open(fpath)]
    print(f'\nTTT-{mode}:')
    by_task = collections.defaultdict(dict)
    for r in results:
        by_task[r['task']][r['length']] = r['accuracy']
    for task in sorted(by_task):
        accs = by_task[task]
        parts = [f'{l//1024}k={accs.get(l, 0):.3f}' for l in [2048,4096,8192,16384,32768] if l in accs]
        print(f'  {task:20s} {\" | \".join(parts)}')
" 2>&1 | tee -a $LOG
