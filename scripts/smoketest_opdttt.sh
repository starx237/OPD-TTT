#!/bin/bash
# =============================================================================
# OPD-TTT 冒烟测试脚本
# =============================================================================
#
# 验证 OPD-TTT 训练流程能正常工作
# - 使用少量测试数据
# - 训练 1 步
# - 验证教师模型加载
# - 验证输出正常
#
# =============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 加载环境设置
source "$PROJECT_ROOT/env_setup.sh"

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}           OPD-TTT 冒烟测试${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 检查测试数据
TEST_DATA="$PROJECT_ROOT/data/test_piles_smoketest.jsonl"
if [ ! -f "$TEST_DATA" ]; then
    echo -e "${RED}错误: 测试数据不存在: $TEST_DATA${NC}"
    exit 1
fi

echo -e "${YELLOW}检查环境...${NC}"

# 检查 CUDA
python -c "
import torch
if not torch.cuda.is_available():
    print('错误: CUDA 不可用')
    exit(1)
print(f'✓ CUDA 可用, 设备数: {torch.cuda.device_count()}')
"

# 检查 tokenizer
if [ ! -d "$PROJECT_ROOT/model_assets/tokenizer" ]; then
    echo -e "${RED}错误: tokenizer 不存在: $PROJECT_ROOT/model_assets/tokenizer${NC}"
    echo "请先运行: python scripts/setup_tokenizer.py --model qwen2-7b --output model_assets/tokenizer"
    exit 1
fi
echo "✓ Tokenizer 存在"

# 检查教师模型
TEACHER_MODEL="$PROJECT_ROOT/model_assets/teacher_qwen2.5_7b"
if [ ! -d "$TEACHER_MODEL" ]; then
    echo -e "${YELLOW}警告: 教师模型不存在: $TEACHER_MODEL${NC}"
    echo "将不使用教师模型进行测试"
    USE_TEACHER=false
else
    echo "✓ 教师模型存在"
    USE_TEACHER=true
fi

echo ""
echo -e "${YELLOW}开始冒烟测试...${NC}"
echo "  - 数据: $TEST_DATA"
echo "  - 训练步数: 1"
if [ "$USE_TEACHER" = true ]; then
    echo "  - 教师模型: $TEACHER_MODEL"
else
    echo "  - 教师模型: (未使用)"
fi
echo ""

# 创建输出目录
OUTPUT_DIR="$PROJECT_ROOT/data/output/smoke_test_opdttt"
mkdir -p "$OUTPUT_DIR"

# 限制 GPU 使用（只用 2 个 GPU）
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

# 使用专门的 smoketest 配置文件
CONFIG_FILE="$PROJECT_ROOT/configs/opdttt/smoke_test_500m_opdttt.yaml"

# 构建 torchrun 命令（配置文件作为位置参数）
CMD="torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    tasks/train_opdttt.py \
    $CONFIG_FILE"

# 如果不使用教师，覆盖配置（通过环境变量或修改配置文件）
if [ "$USE_TEACHER" = false ]; then
    # 创建临时配置文件
    TEMP_CONFIG="$OUTPUT_DIR/smoke_test_no_teacher.yaml"
    cp "$CONFIG_FILE" "$TEMP_CONFIG"
    sed -i 's/teacher_model_path:.*/teacher_model_path: ""/' "$TEMP_CONFIG"
    CONFIG_FILE="$TEMP_CONFIG"
    CMD="torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        tasks/train_opdttt.py \
        $CONFIG_FILE"
fi

echo -e "${YELLOW}执行命令:${NC}"
echo "$CMD"
echo ""

# 执行训练
if eval "$CMD" 2>&1 | tee "$OUTPUT_DIR/smoke_test.log"; then
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✓ 冒烟测试通过!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "输出目录: $OUTPUT_DIR"
    echo "日志文件: $OUTPUT_DIR/smoke_test.log"
    
    # 检查输出文件
    if [ -d "$OUTPUT_DIR/checkpoints/global_step_1" ]; then
        echo "✓ 检查点已保存"
    fi
    
    exit 0
else
    echo ""
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}✗ 冒烟测试失败!${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "请查看日志: $OUTPUT_DIR/smoke_test.log"
    exit 1
fi
