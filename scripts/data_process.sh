#!/bin/bash
# ============================================================================
# OPD-TTT 数据处理脚本
# ============================================================================
# 一站式完成 PILES 数据集的下载和打包
#
# 用法:
#   bash scripts/data_process.sh [full|test]
#
#   full - 完整数据处理（20B tokens，默认）
#   test - 烟雾测试（1M tokens，用于验证流程）
# ============================================================================

# 加载项目环境（激活 venv/conda，设置 PYTHONPATH 等）
source "$(cd "$(dirname "$0")/.." && pwd)/env_setup.sh"

# 设置日志文件
LOG_FILE="data/data_process_$(date +%Y%m%d_%H%M%S).log"
mkdir -p data

# 使用 exec 将所有输出重定向到日志文件和控制台
exec > >(tee -a "$LOG_FILE")
exec 2>&1

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# 开始处理
# ============================================================================
echo "=========================================="
echo "  OPD-TTT 数据处理"
echo "=========================================="
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"
echo ""

# 默认参数
MODE="${1:-full}"
DATA_DIR="data"
MIRROR="https://hf-mirror.com"

# 根据模式设置参数
if [ "$MODE" = "test" ]; then
    TARGET_TOKENS=1000000
    OUTPUT_BASE="test_piles"
    DRY_RUN="--dry_run"
    log_info "运行烟雾测试模式（1M tokens）"
else
    TARGET_TOKENS=20000000000
    OUTPUT_BASE="piles"
    DRY_RUN=""
    log_info "运行完整数据处理（20B tokens）"
fi

# 输出文件路径
RAW_OUTPUT="${DATA_DIR}/${OUTPUT_BASE}_long_text.jsonl"
PACKED_OUTPUT="${DATA_DIR}/${OUTPUT_BASE}_packed_32768.jsonl"
STATE_FILE="${RAW_OUTPUT}.state.json"

# 清理旧的状态文件（如果存在）
if [ -f "$STATE_FILE" ] && [ "$MODE" = "test" ]; then
    log_warn "清理旧的状态文件: $STATE_FILE"
    rm -f "$STATE_FILE"
fi

# 创建数据目录
mkdir -p "$DATA_DIR"

# ============================================================================
# 步骤 1: 检查依赖
# ============================================================================
log_info "检查依赖..."

python3 -c "
import sys
try:
    import datasets
    import zstandard
    print('✓ datasets 库已安装')
    print('✓ zstandard 库已安装')
except ImportError as e:
    print(f'✗ 缺少依赖: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    log_error "缺少必要依赖，请运行: pip install datasets zstandard"
    exit 1
fi

log_success "依赖检查通过"

# ============================================================================
# 步骤 2: 检查网络和镜像站连接
# ============================================================================
log_info "检查镜像站连接..."

python3 -c "
import os
import urllib.request

mirror = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
print(f'  使用镜像站: {mirror}')

# 简单的连接测试
try:
    response = urllib.request.urlopen(f'{mirror}/', timeout=10)
    print('✓ 镜像站连接正常')
except Exception as e:
    print(f'⚠ 镜像站连接失败: {e}')
    print('  将尝试继续下载...')
"

# ============================================================================
# 步骤 3: 下载 PILES 长文本数据
# ============================================================================
log_info "开始下载 PILES 数据集..."
log_info "  - 目标 tokens: $TARGET_TOKENS"
log_info "  - 最小长度: 30000 字符 (约 7.5k tokens)"
log_info "  - 最大长度: 131072 字符 (约 32k tokens)"
log_info "  - 镜像站: $MIRROR"
log_info "  - 输出: $RAW_OUTPUT"
if [ -n "$DRY_RUN" ]; then
    log_warn "  - DRY RUN 模式：仅处理少量数据"
fi

# 记录开始时间
DOWNLOAD_START=$(date +%s)

python3 scripts/prepare_pretrain_data.py \
    --dataset piles \
    --output "$RAW_OUTPUT" \
    --target_tokens "$TARGET_TOKENS" \
    --max_length 131072 \
    --min_length 30000 \
    --subsets "ArXiv,Books3,Wikipedia (en),PubMed Central,Pile-CC,Github" \
    --mirror "$MIRROR" \
    $DRY_RUN

DOWNLOAD_STATUS=$?
DOWNLOAD_END=$(date +%s)
DOWNLOAD_TIME=$((DOWNLOAD_END - DOWNLOAD_START))

if [ $DOWNLOAD_STATUS -ne 0 ]; then
    log_error "数据下载失败（耗时: ${DOWNLOAD_TIME}s）"
    log_error "请检查："
    log_error "  1. 网络连接是否正常"
    log_error "  2. 镜像站 $MIRROR 是否可访问"
    log_error "  3. 查看日志文件: $LOG_FILE"
    exit 1
fi

log_success "PILES 数据下载完成（耗时: $((${DOWNLOAD_TIME}/60))分钟）"

# 检查下载是否真正完成（达到目标tokens）
if [ -f "$STATE_FILE" ]; then
    DOWNLOADED_TOKENS=$(python3 -c "import json; print(int(json.load(open('$STATE_FILE')).get('total_tokens', 0)))")
    if [ "$DOWNLOADED_TOKENS" -lt "$TARGET_TOKENS" ]; then
        log_warn "下载未完成: ${DOWNLOADED_TOKENS} / ${TARGET_TOKENS} tokens"
        log_info "状态文件已保存，下次运行将继续"
        exit 0  # 正常退出，让看门狗重启
    fi
fi

# 检查输出文件
if [ ! -f "$RAW_OUTPUT" ]; then
    log_error "输出文件不存在: $RAW_OUTPUT"
    log_error "下载过程可能没有成功写入数据"
    exit 1
fi

# 检查文件大小
FILE_SIZE=$(stat -f%z "$RAW_OUTPUT" 2>/dev/null || stat -c%s "$RAW_OUTPUT" 2>/dev/null)
if [ "$FILE_SIZE" -lt 1000 ]; then
    log_warn "输出文件大小异常: ${FILE_SIZE} bytes"
    log_warn "文件可能为空或数据不足"
fi

# ============================================================================
# 步骤 4: 数据打包
# ============================================================================
log_info "开始数据打包..."
log_info "  - 输入: $RAW_OUTPUT"
log_info "  - 输出: $PACKED_OUTPUT"
log_info "  - 目标长度: ~32768 tokens"

PACK_START=$(date +%s)

python3 scripts/pack_pretrain_data.py \
    --input "$RAW_OUTPUT" \
    --output "$PACKED_OUTPUT" \
    --tokenizer model_assets/tokenizer

PACK_STATUS=$?
PACK_END=$(date +%s)
PACK_TIME=$((PACK_END - PACK_START))

if [ $PACK_STATUS -ne 0 ]; then
    log_error "数据打包失败（耗时: ${PACK_TIME}s）"
    exit 1
fi

log_success "数据打包完成（耗时: ${PACK_TIME}s）"

# ============================================================================
# 步骤 5: 验证结果
# ============================================================================
log_info "验证打包结果..."

python3 -c "
import json
import sys
import os

input_file = '$RAW_OUTPUT'
output_file = '$PACKED_OUTPUT'

# 检查输出文件是否存在
if not os.path.exists(output_file):
    print(f'✗ 输出文件不存在: {output_file}', file=sys.stderr)
    sys.exit(1)

total_chars = 0
count = 0
min_chars = float('inf')
max_chars = 0

with open(output_file, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            text_len = len(data['content_split'])
            total_chars += text_len
            count += 1
            min_chars = min(min_chars, text_len)
            max_chars = max(max_chars, text_len)
        except Exception as e:
            print(f'✗ 解析行失败: {e}', file=sys.stderr)
            sys.exit(1)

if count == 0:
    print('✗ 没有有效的数据行', file=sys.stderr)
    sys.exit(1)

avg_chars = total_chars / count
avg_tokens = avg_chars / 4

print(f'✓ 数据验证成功')
print(f'  总行数: {count:,}')
print(f'  平均字符数: {avg_chars:.0f}')
print(f'  平均 tokens: {avg_tokens:.0f}')
print(f'  最小字符数: {min_chars:.0f}')
print(f'  最大字符数: {max_chars:.0f}')

# 检查输入文件行数
with open(input_file, 'r') as f:
    input_count = sum(1 for _ in f)
print(f'  输入行数: {input_count:,}')
print(f'  打包比率: {input_count/count:.1f}x')
"

if [ $? -ne 0 ]; then
    log_error "数据验证失败"
    exit 1
fi

# ============================================================================
# 完成
# ============================================================================
TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - DOWNLOAD_START))

echo ""
log_success "数据处理完成！"
echo "总耗时: $((${TOTAL_TIME}/60))分钟"
echo ""
echo "输出文件:"
echo "  - 原始数据: $RAW_OUTPUT"
echo "  - 打包数据: $PACKED_OUTPUT"
echo "  - 日志文件: $LOG_FILE"
echo ""
echo "下一步:"
echo "  1. 检查数据质量: head -n 1 $PACKED_OUTPUT"
echo "  2. 查看日志: tail -n 50 $LOG_FILE"
echo "  3. 开始训练: bash scripts/train_opdttt_500m.sh"
echo ""
