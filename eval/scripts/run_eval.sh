#!/bin/bash
# HMMT 评估启动脚本
#
# 用法:
#   bash eval/scripts/run_eval.sh <model_type>
#
# 参数:
#   model_type:  qwen35_2b_base | qwen35_2b_trained | qwen35_9b
#
# 示例:
#   bash eval/scripts/run_eval.sh qwen35_2b_base
#   bash eval/scripts/run_eval.sh qwen35_2b_trained
#   bash eval/scripts/run_eval.sh qwen35_9b
#
# 可通过 CUDA_VISIBLE_DEVICES 环境变量指定 GPU:
#   CUDA_VISIBLE_DEVICES=0 bash eval/scripts/run_eval.sh qwen35_9b

set -e

PROJECT_ROOT=/h3c/haoxiang/TTT-OPD
PYTHON=/root/miniconda3/bin/python3

export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

cd $PROJECT_ROOT

# 加载 .env 文件
if [ -f .env ]; then
    _SCRIPT_CUDA=${CUDA_VISIBLE_DEVICES:-}
    export $(grep -v '^#' .env | xargs)
    if [ -n "$_SCRIPT_CUDA" ]; then
        export CUDA_VISIBLE_DEVICES=$_SCRIPT_CUDA
    fi
    echo "Loaded .env, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# 参数解析
MODEL_TYPE=${1:-qwen35_2b_base}

CONFIG="eval/config/${MODEL_TYPE}.yaml"
OUTPUT_DIR="eval/output"

# 从配置文件中读取 benchmark 名称
BENCHMARK=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('eval', {}).get('benchmark', 'hmmt'))
")
LOG="${OUTPUT_DIR}/${MODEL_TYPE}_${BENCHMARK}.log"

# 检查配置文件
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    echo "Available configs:"
    ls eval/config/*.yaml 2>/dev/null || echo "  (none)"
    exit 1
fi

mkdir -p $OUTPUT_DIR
> $LOG

echo "=== Eval Start: $(date) ===" | tee -a $LOG
echo "Python: $($PYTHON --version)" | tee -a $LOG
echo "Model: $MODEL_TYPE" | tee -a $LOG
echo "Benchmark: $BENCHMARK" | tee -a $LOG
echo "Config: $CONFIG" | tee -a $LOG
echo "GPU: $CUDA_VISIBLE_DEVICES" | tee -a $LOG
echo "" | tee -a $LOG

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PYTHON eval/scripts/eval_hmmt.py \
    --config $CONFIG \
    2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== Eval Finished: $(date) ===" | tee -a $LOG
echo "Log: $LOG" | tee -a $LOG
