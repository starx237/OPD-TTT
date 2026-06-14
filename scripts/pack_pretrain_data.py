#!/usr/bin/env python3
"""Pack texts into ~32768-token sequences using EOS concatenation.

Fast version: uses character estimation instead of precise tokenization.
Only tokenizes near the boundary for accuracy.

Usage:
    python scripts/pack_pretrain_data.py \\
        --input data/piles_long_text.jsonl \\
        --output data/piles_packed_32768.jsonl
"""
import json
import os
import sys
import time

# 打包配置
MAX_SEQ_TOKENS = 32768         # 目标序列长度（tokens）
CHARS_PER_TOKEN = 4.0          # 快速估算：平均每token字符数
MAX_SEQ_CHARS = int(MAX_SEQ_TOKENS * CHARS_PER_TOKEN)  # 约131k字符
MIN_INPUT_CHARS = 30000         # 最小输入字符数

def estimate_tokens(text: str) -> int:
    """快速估算token数"""
    return max(1, len(text) // CHARS_PER_TOKEN)

def pack_data(input_path: str, output_path: str):
    """快速打包数据"""
    # 检查输出文件是否已存在
    skip_lines = 0
    lines_out = 0  # 输出文件的行数

    if os.path.exists(output_path):
        state_file = output_path + ".state"
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                skip_lines = int(f.read().strip())
            if skip_lines > 0:
                print(f"📋 从断点续传，跳过 {skip_lines:,} 行", flush=True)
                # 计算输出文件已有多少行（用于断点续传时的计数）
                with open(output_path, 'r') as f:
                    lines_out = sum(1 for _ in f)
                print(f"📋 输出文件已有 {lines_out:,} 行", flush=True)
        else:
            os.remove(output_path)

    lines_in = 0
    batch_texts = []
    batch_chars = 0
    total_chars = 0
    split_count = 0
    start = time.time()

    print(f"开始快速打包到 {MAX_SEQ_TOKENS} tokens (~{MAX_SEQ_CHARS} 字符)...", flush=True)

    # 根据是否有断点续传选择打开模式
    mode = 'a' if skip_lines > 0 else 'w'

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, mode, encoding='utf-8') as fout:

        for line in fin:
            lines_in += 1

            # 跳过已处理的行
            if lines_in <= skip_lines:
                continue

            try:
                data = json.loads(line)
                text = data.get('content_split', '')
            except json.JSONDecodeError as e:
                print(f"⚠️  解析第 {lines_in} 行失败: {e}", flush=True)
                continue

            if not text:
                continue

            text_len = len(text)
            total_chars += text_len

            # 处理超长文本（按字符数分割）
            if text_len > MAX_SEQ_CHARS * 1.5:  # 超过1.5倍限制
                # 先输出当前批次
                if batch_texts:
                    out_text = '</s>'.join(batch_texts)
                    fout.write(json.dumps({'content_split': out_text}, ensure_ascii=False) + '\n')
                    lines_out += 1
                    batch_texts = []
                    batch_chars = 0

                # 分割超长文本
                num_splits = (text_len + MAX_SEQ_CHARS - 1) // MAX_SEQ_CHARS
                for i in range(num_splits):
                    start_idx = i * MAX_SEQ_CHARS
                    end_idx = min((i + 1) * MAX_SEQ_CHARS, text_len)
                    split_text = text[start_idx:end_idx]
                    split_data = {'content_split': split_text}
                    fout.write(json.dumps(split_data, ensure_ascii=False) + '\n')
                    lines_out += 1
                    split_count += 1

                if lines_in % 10000 == 0:
                    print(f'  已处理 {lines_in:,} 行，输出 {lines_out:,} 行，分割 {split_count} 个超长片段', flush=True)
                continue

            # 如果当前批次加上新文本会超过目标长度，输出当前批次
            if batch_texts and batch_chars + text_len > MAX_SEQ_CHARS:
                out_text = '</s>'.join(batch_texts)
                fout.write(json.dumps({'content_split': out_text}, ensure_ascii=False) + '\n')
                lines_out += 1

                # 开始新批次
                batch_texts = [text]
                batch_chars = text_len
            else:
                # 添加到当前批次
                batch_texts.append(text)
                batch_chars += text_len

            # 每处理 10000 行报告一次进度
            if lines_in % 10000 == 0:
                elapsed = time.time() - start
                rate = lines_in / elapsed if elapsed > 0 else 0
                # 保存状态
                with open(output_path + ".state", 'w') as sf:
                    sf.write(str(lines_in))
                print(f'  已处理 {lines_in:,} 行，输出 {lines_out:,} 行，{rate:.0f} 行/秒', flush=True)

        # 输出最后一个批次
        if batch_texts:
            out_text = '</s>'.join(batch_texts)
            fout.write(json.dumps({'content_split': out_text}, ensure_ascii=False) + '\n')
            lines_out += 1

    # 删除状态文件
    state_file = output_path + ".state"
    if os.path.exists(state_file):
        os.remove(state_file)

    elapsed = time.time() - start
    print(f'\n✅ 打包完成！耗时: {elapsed:.0f}秒', flush=True)
    print(f'  输入:  {lines_in:,} 行', flush=True)
    print(f'  输出:  {lines_out:,} 行', flush=True)
    print(f'  打包比率: {lines_in/lines_out:.1f}x', flush=True)
    if split_count > 0:
        print(f'  分割了 {split_count} 个超长文本片段', flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Pack long texts into ~32768-token sequences (fast version)'
    )
    parser.add_argument('--input', default='data/piles_long_text.jsonl',
                        help='Input JSONL path')
    parser.add_argument('--output', default='data/piles_packed_32768.jsonl',
                        help='Output JSONL path')
    args = parser.parse_args()

    pack_data(args.input, args.output)
