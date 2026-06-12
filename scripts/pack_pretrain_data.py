#!/usr/bin/env python3
"""Pack texts into ~32768-token sequences using EOS concatenation.

This script handles texts of varying lengths (30k+ chars) and concatenates
them with EOS separator to create fixed-length sequences.

Streaming approach: reads JSONL line by line, concatenates texts with </s>
separator, and writes packed sequences as new JSONL lines.

Usage:
    python scripts/pack_pretrain_data.py \\
        --input data/piles_long_text.jsonl \\
        --output data/piles_packed_32768.jsonl \\
        --tokenizer model_assets/tokenizer
"""
import json
import os
import sys
import time
from transformers import AutoTokenizer

# 打包配置
MAX_SEQ_LEN = 32768         # 目标序列长度（tokens）
CHARS_PER_TOKEN = 1.40      # 每个 token 平均字符数
SEP_CHARS = 4               # EOS 分隔符长度 ('</s>')
TARGET_CHARS = int(MAX_SEQ_LEN * CHARS_PER_TOKEN)  # 目标字符数
MIN_INPUT_CHARS = 30000     # 最小输入字符数（约 7.5k tokens）

def pack_data(input_path: str, output_path: str, tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    eos = tokenizer.eos_token or '</s>'

    lines_in = 0
    lines_out = 0
    batch_texts = []
    batch_chars = 0
    total_chars_in = 0
    start = time.time()

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            text = json.loads(line)['content_split']
            text_len = len(text)
            total_chars_in += text_len

            if batch_texts:
                needed = text_len + SEP_CHARS
            else:
                needed = text_len

            if batch_chars + needed > TARGET_CHARS and batch_texts:
                out_text = eos.join(batch_texts)
                fout.write(json.dumps({'content_split': out_text}, ensure_ascii=False) + '\n')
                lines_out += 1
                batch_texts = [text]
                batch_chars = text_len
            else:
                if batch_texts:
                    batch_chars += SEP_CHARS + text_len
                else:
                    batch_chars = text_len
                batch_texts.append(text)

            lines_in += 1
            if lines_in % 100000 == 0:
                elapsed = time.time() - start
                rate = lines_in / elapsed
                print(f'  Read {lines_in:,} lines, wrote {lines_out:,} lines, '
                      f'{rate:.0f} lines/s, {elapsed:.0f}s', flush=True)

        if batch_texts:
            out_text = eos.join(batch_texts)
            fout.write(json.dumps({'content_split': out_text}, ensure_ascii=False) + '\n')
            lines_out += 1

    elapsed = time.time() - start
    print(f'\nDone! {elapsed:.0f}s')
    print(f'  Input:  {lines_in:,} lines, {total_chars_in:,} chars')
    print(f'  Output: {lines_out:,} lines')
    print(f'  Pack ratio: {lines_in/lines_out:.1f}x')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Pack long texts into ~32768-token sequences for training'
    )
    parser.add_argument('--input', default='data/piles_long_text.jsonl',
                        help='Input JSONL path (texts, min 30k chars)')
    parser.add_argument('--output', default='data/piles_packed_32768.jsonl',
                        help='Output JSONL path (packed to ~32768 tokens)')
    parser.add_argument('--tokenizer', default='model_assets/tokenizer',
                        help='Tokenizer path or name')
    args = parser.parse_args()
    pack_data(
        input_path=args.input,
        output_path=args.output,
        tokenizer_path=args.tokenizer,
    )
