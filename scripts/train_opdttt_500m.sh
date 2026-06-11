#!/bin/bash
# OPD-TTT 500M 模型训练脚本
# On-Policy Distillation Enhanced Test-Time Training

set -e

# 加载环境设置
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/env_setup.sh"

echo "=========================================="
echo "  OPD-TTT 500M 模型训练"
echo "=========================================="

# 检查 GPU 可用性
python3 -c "
import torch
print(f'可用 GPU 数量: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / 1e9
    print(f'  GPU {i}: {props.name}, {mem_gb:.1f}GB')
"

# 训练配置
CONFIG_FILE="$PROJECT_ROOT/configs/opdttt/llama3_sc_500m_opdttt.yaml"
OUTPUT_DIR="$PROJECT_ROOT/data/output/500m_opdttt"
NUM_GPUS=8

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查数据
if [ ! -f "$PROJECT_ROOT/data/pretrain_500m_packed.jsonl" ]; then
    echo "错误: 训练数据不存在: $PROJECT_ROOT/data/pretrain_500m_packed.jsonl"
    echo "请先运行数据准备:"
    echo "  python $SCRIPT_DIR/prepare_pretrain_data.py --output $PROJECT_ROOT/data/pretrain_500m.jsonl --target_tokens 20000000000 --data-dir $PROJECT_ROOT/data"
    echo "  python $SCRIPT_DIR/pack_pretrain_data.py"
    exit 1
fi

echo ""
echo "开始 OPD-TTT 训练 (500M 模型)..."
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "GPU 数量: $NUM_GPUS"
echo ""

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 启动训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    tasks/train_opdttt.py \
    --config "$CONFIG_FILE" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "=========================================="
echo "训练完成! 结果保存在: $OUTPUT_DIR"
echo "=========================================="
