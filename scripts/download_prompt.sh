#!/bin/bash
# =============================================================================
# OPD-TTT Prompt 数据下载脚本
# =============================================================================
#
# 从 HuggingFace 镜像站下载 OpenThoughts3-1.2M 数据集的 prompt 数据
#
# 使用方法:
#   bash scripts/download_prompt.sh                    # 下载全部 120 个文件
#   bash scripts/download_prompt.sh 10                 # 下载前 10 个文件
#   bash scripts/download_prompt.sh 1 5                # 下载文件 1-5
#
# =============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录（自动检测）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 默认参数
MIRROR="https://hf-mirror.com"
DATASET="open-thoughts/OpenThoughts3-1.2M"
DATA_DIR="data"
RAW_DIR="data/prompts_raw"  # 原始 parquet 文件目录
START_FILE=0
END_FILE=119  # 0-based, 119 = 第120个文件
CONVERT=true
LOG_ALL=false  # 是否将所有输出记录到日志

# 位置参数
POS_ARG=()

# 解析选项参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mirror|-m)
            MIRROR="$2"
            shift 2
            ;;
        --dir|-d)
            DATA_DIR="$2"
            shift 2
            ;;
        --no-convert)
            CONVERT=false
            shift
            ;;
        --log-all)
            LOG_ALL=true
            shift
            ;;
        -h|--help)
            cat << 'EOF'
用法: bash scripts/download_prompt.sh [选项] [数量] [结束]

选项:
  --mirror, -m      HuggingFace 镜像站（默认: https://hf-mirror.com）
  --dir, -d         数据输出目录（默认: data）
  --no-convert      不转换为 JSONL，保留原始 parquet 文件
  --log-all         将所有输出记录到日志文件（同时显示在控制台）
  --help, -h        显示此帮助信息

位置参数:
  数量              下载文件数量（如：10 表示下载前 10 个文件）
  结束              配合数量使用，表示范围（如：1 5 表示下载文件 1-5）

示例:
  bash scripts/download_prompt.sh          # 下载全部 120 个文件
  bash scripts/download_prompt.sh 10      # 下载前 10 个文件（文件1-10）
  bash scripts/download_prompt.sh 1 5      # 下载文件 1-5
  bash scripts/download_prompt.sh --log-all  # 记录所有输出到日志
EOF
            exit 0
            ;;
        -*)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            exit 1
            ;;
        *)
            POS_ARG+=("$1")
            shift
            ;;
    esac
done

# 处理位置参数（用户友好的 1-based 索引）
if [[ ${#POS_ARG[@]} -eq 1 ]]; then
    # download_prompt.sh 10  -> 下载前10个文件 (文件1-10，内部索引0-9)
    COUNT="${POS_ARG[0]}"
    START_FILE=0
    END_FILE=$((COUNT - 1))
elif [[ ${#POS_ARG[@]} -eq 2 ]]; then
    # download_prompt.sh 1 5  -> 下载文件1-5（内部索引0-4）
    START_FILE=$((POS_ARG[0] - 1))
    END_FILE=$((POS_ARG[1] - 1))
elif [[ ${#POS_ARG[@]} -gt 2 ]]; then
    echo -e "${RED}错误: 参数过多${NC}"
    exit 1
fi

# 验证范围
if [[ $START_FILE -lt 0 ]]; then
    echo -e "${RED}错误: 起始文件号不能小于 1${NC}"
    exit 1
fi

if [[ $START_FILE -ge 120 ]]; then
    echo -e "${RED}错误: 起始文件号不能大于 120${NC}"
    exit 1
fi

if [[ $END_FILE -lt 0 ]]; then
    echo -e "${RED}错误: 结束文件号不能小于 1${NC}"
    exit 1
fi

if [[ $END_FILE -ge 120 ]]; then
    echo -e "${RED}错误: 结束文件号不能大于 120（共120个文件）${NC}"
    exit 1
fi

if [[ $START_FILE -gt $END_FILE ]]; then
    echo -e "${RED}错误: 起始文件号不能大于结束文件号${NC}"
    exit 1
fi

# 计算显示用的编号（用户友好的 1-based）
DISPLAY_START=$((START_FILE + 1))
DISPLAY_END=$((END_FILE + 1))
TOTAL_FILES=$((END_FILE - START_FILE + 1))

# 日志文件
LOG_FILE="${DATA_DIR}/download_prompt.log"

# 创建数据目录
mkdir -p "$RAW_DIR"
mkdir -p "$DATA_DIR"

# 创建日志文件并重定向所有输出（无缓冲）
> "$LOG_FILE"
exec > >(stdbuf -i0 -o0 tee -a "$LOG_FILE") 2>&1

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}           OPD-TTT Prompt 数据下载${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "数据集: $DATASET"
echo "镜像站: $MIRROR"
echo "原始文件目录: $RAW_DIR"
echo "JSONL 输出目录: $DATA_DIR"
echo "下载范围: train-$(printf "%05d" $DISPLAY_START)-of-00120.parquet 到 train-$(printf "%05d" $DISPLAY_END)-of-00120.parquet"
echo "文件数量: $TOTAL_FILES 个"
echo "日志文件: $LOG_FILE"
echo ""

# 检查依赖
if ! command -v curl &> /dev/null; then
    echo -e "${RED}错误: 未找到 curl，请先安装 curl${NC}"
    exit 1
fi

# 开始下载
echo -e "${BLUE}开始下载...${NC}"
echo ""

download_success=0
download_failed=0
download_skipped=0

for ((i=START_FILE; i<=END_FILE; i++)); do
    display_num=$((i + 1))
    file_num_padded=$(printf "%05d" $display_num)
    url="${MIRROR}/datasets/${DATASET}/resolve/main/data/train-${file_num_padded}-of-00120.parquet?download=true"
    output="${RAW_DIR}/train-${file_num_padded}-of-00120.parquet"

    # 检查文件是否已存在
    if [[ -f "$output" && -s "$output" ]]; then
        size=$(du -h "$output" | cut -f1)
        echo -e "${YELLOW}[${display_num}/${TOTAL_FILES}]${NC} 跳过已存在: train-${file_num_padded}-of-00120.parquet ($size)"
        download_skipped=$((download_skipped + 1))
        continue
    fi

    echo -e "${BLUE}[${display_num}/${TOTAL_FILES}]${NC} 下载: train-${file_num_padded}-of-00120.parquet"

    # 使用 aria2c 多线程下载
    if aria2c -x 16 -s 16 --retry-wait=10 --max-tries=3 --timeout=60 --connect-timeout=30 --file-allocation=none -d "$RAW_DIR" -o "train-${file_num_padded}-of-00120.parquet" "$url"; then
        if [[ -f "$output" && -s "$output" ]]; then
            size=$(du -h "$output" | cut -f1)
            echo -e "  ${GREEN}✓ 完成: train-${file_num_padded}-of-00120.parquet ($size)${NC}"
            download_success=$((download_success + 1))
        else
            echo -e "  ${RED}✗ 失败: 文件为空${NC}"
            download_failed=$((download_failed + 1))
            rm -f "$output"
        fi
    else
        echo -e "  ${RED}✗ 失败: 下载错误（将重试）${NC}"
        download_failed=$((download_failed + 1))
        rm -f "$output"

        # 重试一次
        echo -e "  ${YELLOW}重试: train-${file_num_padded}-of-00120.parquet${NC}"
        if aria2c -x 16 -s 16 --retry-wait=15 --max-tries=5 --timeout=120 --connect-timeout=60 --file-allocation=none -d "$RAW_DIR" -o "train-${file_num_padded}-of-00120.parquet" "$url"; then
            if [[ -f "$output" && -s "$output" ]]; then
                size=$(du -h "$output" | cut -f1)
                echo -e "  ${GREEN}✓ 重试成功: train-${file_num_padded}-of-00120.parquet ($size)${NC}"
                download_success=$((download_success + 1))
                download_failed=$((download_failed - 1))
            else
                echo -e "  ${RED}✗ 重试失败: 文件为空${NC}"
                rm -f "$output"
            fi
        else
            echo -e "  ${RED}✗ 重试失败: train-${file_num_padded}-of-00120.parquet${NC}"
        fi
    fi

    # 记录到日志
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] train-${file_num_padded}-of-00120.parquet" >> "$LOG_FILE"
done

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}下载完成!${NC}"
echo -e "${GREEN}  成功: $download_success 个文件${NC}"
if [[ $download_failed -gt 0 ]]; then
    echo -e "${RED}  失败: $download_failed 个文件${NC}"
fi
if [[ $download_skipped -gt 0 ]]; then
    echo -e "${YELLOW}  跳过: $download_skipped 个已存在文件${NC}"
fi
echo -e "${GREEN}  保存位置: $DATA_DIR/${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 转换为 JSONL
if [[ "$CONVERT" == true && $download_success -gt 0 ]]; then
    echo -e "${BLUE}转换为 JSONL 格式...${NC}"
    echo ""

    python -c "
import json
import glob
import sys
from tqdm import tqdm

raw_dir = '${RAW_DIR}'
data_dir = '${DATA_DIR}'
parquet_files = sorted(glob.glob(f'{raw_dir}/train-*-of-00120.parquet'))
output_file = f'{data_dir}/opd_prompts_raw.jsonl'

if not parquet_files:
    print('错误: 未找到 parquet 文件')
    sys.exit(1)

try:
    import pyarrow.parquet as pq

    total_count = 0
    error_count = 0
    with open(output_file, 'w') as f:
        for parquet_file in tqdm(parquet_files, desc='处理文件'):
            try:
                table = pq.read_table(parquet_file)
                for batch in table.to_batches():
                    for row in zip(*[batch.column(i).to_pylist() for i in range(batch.num_columns)]):
                        # 假设第一列是 prompt
                        prompt = row[0] if row else ''
                        if prompt:
                            f.write(json.dumps({'prompt': prompt}, ensure_ascii=False) + '\\n')
                            total_count += 1
            except Exception as e:
                print(f'警告: 处理 {parquet_file} 时出错: {e}')
                error_count += 1

    print(f'\\n✓ 转换完成! 共 {total_count} 个 prompts')
    if error_count > 0:
        print(f'  警告: {error_count} 个文件处理失败')
    print(f'  输出文件: {output_file}')

except ImportError:
    print('警告: 未安装 pyarrow，跳过转换')
    print('请运行: pip install pyarrow')
" 2>&1 | tee -a "$LOG_FILE"

    echo ""
fi

if [[ $download_failed -gt 0 ]]; then
    echo -e "${YELLOW}警告: 有 $download_failed 个文件下载失败，请重新运行脚本继续下载${NC}"
    exit 1
fi
