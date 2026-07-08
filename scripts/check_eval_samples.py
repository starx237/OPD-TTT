#!/usr/bin/env python3
"""检查评估样本的来源分布：是否随机，还是集中在某个 subset。"""
import json, sys, os

data_path = sys.argv[1] if len(sys.argv) > 1 else "data/piles_packed_32768_shuffled.jsonl"
num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 50
min_len = 18432

file_size = os.path.getsize(data_path)
chunk_size = 16 * 1024 * 1024
pos = file_size
buffer = ""
samples = []

with open(data_path, "rb") as f:
    while pos > 0 and len(samples) < num_samples:
        read_size = min(chunk_size, pos)
        pos -= read_size
        f.seek(pos)
        chunk = f.read(read_size).decode("utf-8", errors="ignore")
        buffer = chunk + buffer
        lines = buffer.split("\n")
        if pos > 0:
            buffer = lines[0]
            complete_lines = lines[1:]
        else:
            buffer = ""
            complete_lines = lines
        for line in reversed(complete_lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            text = data.get("content_split", data.get("content", ""))
            if not text:
                continue
            char_len = len(text)
            if char_len < min_len * 2:
                continue
            samples.append({
                "idx_from_end": len(samples),
                "char_len": char_len,
                "text_preview": text[:200].replace("\n", " "),
                "pos_range": f"~{pos}-{pos+read_size}",
                "keys": list(data.keys()),
            })
            if len(samples) >= num_samples:
                break

samples.reverse()
print(f"File size: {file_size:,} bytes ({file_size/1024**3:.1f} GB)")
print(f"Read range: pos={pos:,} to {file_size:,} ({(file_size-pos)/1024**2:.1f} MB)")
print(f"Samples found: {len(samples)}\n")
for s in samples:
    print(f"[{s['idx_from_end']:2d}] char_len={s['char_len']:>8} | pos={s['pos_range']} | {s['text_preview'][:120]}...")
print(f"\nChar len stats: min={min(s['char_len'] for s in samples)}, max={max(s['char_len'] for s in samples)}, mean={sum(s['char_len'] for s in samples)/len(samples):.0f}")
