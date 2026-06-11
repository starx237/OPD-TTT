#!/usr/bin/env python3
"""Pack short texts into ~32768-token sequences using EOS concatenation.

Streaming approach: reads JSONL line by line, concatenates texts with </s>
separator, and writes packed sequences as new JSONL lines.
"""
import json
import os
import sys
import time
from transformers import AutoTokenizer

MAX_SEQ_LEN = 32768
CHARS_PER_TOKEN = 1.40  # measured from sample
SEP_CHARS = 4  # length of '</s>' separator
TARGET_CHARS = int(MAX_SEQ_LEN * CHARS_PER_TOKEN)

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
    parser = argparse.ArgumentParser(description='Pack short texts into long sequences')
    parser.add_argument('--input', default='data/pretrain_500m.jsonl',
                        help='Input JSONL path')
    parser.add_argument('--output', default='data/pretrain_500m_packed.jsonl',
                        help='Output JSONL path')
    parser.add_argument('--tokenizer', default='model_assets/llama_500m_config',
                        help='Tokenizer path or name')
    args = parser.parse_args()
    pack_data(
        input_path=args.input,
        output_path=args.output,
        tokenizer_path=args.tokenizer,
    )
