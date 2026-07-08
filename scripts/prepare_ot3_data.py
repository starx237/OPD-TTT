#!/usr/bin/env python3
"""Phase 2: OT3 数据处理 — 创建 SFT 和 OPD 数据集

从 OpenThoughts-3 parquet 创建:
  - SFT: conversation 格式 ({"messages": [...]})，prompt masking（训练时由 VeOmni ChatmlTemplate 处理）
  - OPD: plaintext 格式 ({"prompt": "<chat-formatted>"})，预格式化 chat template
  - SFT 和 OPD prompt 不重叠
  - 按 domain 分层采样确保多样性

OT3 response 格式 <think>...</think>\\n\\n<answer> 与 Qwen3.5 thinking mode 完全兼容，
可直接作为 SFT 的 assistant content。

用法（miniconda3 环境）:
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
  export PYTHONPATH="$PWD:$PWD/VeOmni:$PWD/hf_models:$PYTHONPATH"
  python scripts/prepare_ot3_data.py \\
      --input_dir data/prompts_raw/data --output_dir data/ \\
      --num_sft 150000 --num_opd 100000 --num_val 500 \\
      --seed 42 --require_think_close \\
      --tokenizer_path model_assets/qwen3.5-9b
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def read_one_file(path):
    """读取单个 parquet 文件，返回 records 列表。"""
    t = pq.read_table(path, columns=["difficulty", "source", "domain", "conversations"])
    df = t.to_pandas()
    records = []
    for _, row in df.iterrows():
        conv = row["conversations"]
        if len(conv) < 2:
            continue
        prompt = conv[0]["value"]
        response = conv[1]["value"]
        if not prompt or not response:
            continue
        src = row["source"]
        if src is None or (isinstance(src, float) and src != src):  # NaN check
            src = "unknown"
        records.append(
            {
                "prompt": prompt,
                "response": response,
                "difficulty": row["difficulty"],
                "source": src,
                "domain": row["domain"] if row["domain"] else "unknown",
                "has_think_close": "</think>" in response,
            }
        )
    return records


def load_all_records(input_dir, num_workers=8):
    """多线程并行读取所有 parquet 文件。"""
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    print(f"Reading {len(files)} parquet files with {num_workers} workers...", flush=True)
    all_records = []
    done = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(read_one_file, f): f for f in files}
        for fut in as_completed(futures):
            all_records.extend(fut.result())
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(files)} files done, {len(all_records)} records", flush=True)
    print(f"  Done: {len(all_records)} records from {len(files)} files", flush=True)
    return all_records


def dedup_records(records):
    """按 prompt hash 去重，保留第一条。"""
    seen = set()
    unique = []
    dup_count = 0
    for r in records:
        h = hashlib.md5(r["prompt"].encode("utf-8")).hexdigest()
        if h in seen:
            dup_count += 1
            continue
        seen.add(h)
        r["prompt_hash"] = h
        unique.append(r)
    print(f"Dedup: {len(records)} -> {len(unique)} (removed {dup_count} duplicates)", flush=True)
    return unique


def stratified_sample_by_domain(records, num, seed, used_hashes=None):
    """按 domain 分层采样 num 条（按各组比例），排除 used_hashes 中的 prompt。

    返回选中的 records 列表。
    """
    rng = random.Random(seed)
    used_hashes = used_hashes or set()

    # 按 domain 分组（排除已用的）
    by_domain = defaultdict(list)
    for r in records:
        if r["prompt_hash"] in used_hashes:
            continue
        by_domain[r["domain"]].append(r)

    total_avail = sum(len(v) for v in by_domain.values())
    if total_avail == 0:
        return []

    # 按比例分配各组采样数
    selected = []
    remaining = num
    domains = sorted(by_domain.keys())
    for i, dom in enumerate(domains):
        if i == len(domains) - 1:
            # 最后一组取剩余，避免舍入误差
            n = remaining
        else:
            n = round(num * len(by_domain[dom]) / total_avail)
            n = min(n, len(by_domain[dom]))
            remaining -= n
        n = min(n, len(by_domain[dom]))
        sampled = rng.sample(by_domain[dom], n)
        selected.extend(sampled)
        print(f"  {dom}: avail={len(by_domain[dom])}, sampled={n}", flush=True)

    return selected


def format_sft(record):
    """格式化为 SFT conversation 格式。"""
    return {
        "messages": [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]
    }


def format_opd(record, tokenizer):
    """格式化为 OPD plaintext 格式（预格式化 chat template）。"""
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": record["prompt"]}],
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=False,
    )
    return {"prompt": formatted}


def write_jsonl(records, path, formatter, tokenizer=None):
    """写入 jsonl 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            if tokenizer is not None:
                obj = formatter(r, tokenizer)
            else:
                obj = formatter(r)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records to {path}", flush=True)


def print_distribution(records, name):
    """打印 domain/source 分布。"""
    dom = Counter(r["domain"] for r in records)
    src = Counter(r["source"] for r in records)
    think = sum(1 for r in records if r["has_think_close"])
    print(f"  [{name}] total={len(records)}, </think>={think}/{len(records)}", flush=True)
    print(f"    domain: {dict(dom.most_common())}", flush=True)
    print(f"    source: {dict(src.most_common())}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: OT3 数据处理")
    parser.add_argument("--input_dir", default="data/prompts_raw/data")
    parser.add_argument("--output_dir", default="data/")
    parser.add_argument("--num_sft", type=int, default=150000)
    parser.add_argument("--num_opd", type=int, default=100000)
    parser.add_argument("--num_val", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_think_close", action="store_true",
                        help="SFT 只选有 </think> 收尾的 response")
    parser.add_argument("--tokenizer_path", default="model_assets/qwen3.5-9b",
                        help="用于 OPD 预格式化的 tokenizer")
    parser.add_argument("--num_workers", type=int, default=8, help="并行读取文件数")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    print("=" * 70)
    print("  Phase 2: OT3 Data Processing")
    print(f"  num_sft={args.num_sft}, num_opd={args.num_opd}, num_val={args.num_val}")
    print(f"  require_think_close={args.require_think_close}, seed={args.seed}")
    print("=" * 70, flush=True)

    # 1. 读取所有 parquet
    print("\n[1/6] Loading parquet files...", flush=True)
    records = load_all_records(args.input_dir, args.num_workers)

    # 2. 去重
    print("\n[2/6] Deduplicating...", flush=True)
    records = dedup_records(records)
    print_distribution(records, "all (deduped)")

    # 3. 准备 SFT 候选池
    print("\n[3/6] Preparing SFT candidate pool...", flush=True)
    if args.require_think_close:
        sft_pool = [r for r in records if r["has_think_close"]]
        print(f"  SFT pool (with </think>): {len(sft_pool)} / {len(records)}", flush=True)
    else:
        sft_pool = list(records)
        print(f"  SFT pool (all): {len(sft_pool)}", flush=True)

    # 4. 分层采样 SFT（train + val）
    print(f"\n[4/6] Stratified sampling SFT ({args.num_sft} train + {args.num_val} val)...", flush=True)
    total_sft = args.num_sft + args.num_val
    sft_selected = stratified_sample_by_domain(sft_pool, total_sft, args.seed)
    rng.shuffle(sft_selected)
    sft_val = sft_selected[: args.num_val]
    sft_train = sft_selected[args.num_val :]
    print(f"  SFT train: {len(sft_train)}, val: {len(sft_val)}", flush=True)
    print_distribution(sft_train, "sft_train")

    # 5. 分层采样 OPD（train + val），排除 SFT 已用的 prompt
    print(f"\n[5/6] Stratified sampling OPD ({args.num_opd} train + {args.num_val} val)...", flush=True)
    sft_used_hashes = set(r["prompt_hash"] for r in sft_selected)
    total_opd = args.num_opd + args.num_val
    opd_selected = stratified_sample_by_domain(
        records, total_opd, args.seed + 1, used_hashes=sft_used_hashes
    )
    rng.shuffle(opd_selected)
    opd_val = opd_selected[: args.num_val]
    opd_train = opd_selected[args.num_val :]
    print(f"  OPD train: {len(opd_train)}, val: {len(opd_val)}", flush=True)
    print_distribution(opd_train, "opd_train")

    # 验证无重叠
    opd_hashes = set(r["prompt_hash"] for r in opd_selected)
    overlap = sft_used_hashes & opd_hashes
    print(f"\n  Overlap check: SFT ∩ OPD = {len(overlap)} (should be 0)", flush=True)
    assert len(overlap) == 0, f"SFT and OPD overlap: {len(overlap)} prompts!"

    # 6. 格式化输出
    print("\n[6/6] Formatting & writing...", flush=True)
    print("  Loading tokenizer for OPD formatting...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    write_jsonl(sft_train, os.path.join(args.output_dir, "sft_train.jsonl"), format_sft)
    write_jsonl(sft_val, os.path.join(args.output_dir, "sft_val.jsonl"), format_sft)
    write_jsonl(opd_train, os.path.join(args.output_dir, "opd_train.jsonl"), format_opd, tokenizer)
    write_jsonl(opd_val, os.path.join(args.output_dir, "opd_val.jsonl"), format_opd, tokenizer)

    # 汇总
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  SFT train: {len(sft_train)} -> {os.path.join(args.output_dir, 'sft_train.jsonl')}")
    print(f"  SFT val:   {len(sft_val)} -> {os.path.join(args.output_dir, 'sft_val.jsonl')}")
    print(f"  OPD train: {len(opd_train)} -> {os.path.join(args.output_dir, 'opd_train.jsonl')}")
    print(f"  OPD val:   {len(opd_val)} -> {os.path.join(args.output_dir, 'opd_val.jsonl')}")
    print(f"  Overlap:   0 (verified)")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
