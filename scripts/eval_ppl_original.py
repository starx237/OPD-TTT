#!/usr/bin/env python3
"""评估原始 Qwen3.5-2B 模型在多个 context 长度下的 PPL。

与训练评估 (_evaluate_ppl) 配置一致：
- 从训练数据文件末尾取 20 个长样本
- context_lengths = [2048, 4096, 8192, 16384]
- target_len = 2048（固定 target 向前追溯）
- use_cache=False, bf16

用法：
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_ppl_original.py

输出：
  - 控制台打印每个 ctx 的 mean/std/min/max
  - 保存 JSON 到 results/ppl_original.json

Created: 2026-07-10
"""

import json
import math
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===================== 配置（与训练评估一致）=====================
MODEL_PATH = os.path.join(PROJECT_ROOT, "model_assets/qwen3.5-2b")
TOKENIZER_PATH = MODEL_PATH
DATA_PATH = os.path.join(PROJECT_ROOT, "data/piles_packed_32768_shuffled.jsonl")
CONTEXT_LENGTHS = [2048, 4096, 8192, 16384]
TARGET_LEN = 2048
NUM_SAMPLES = 20
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results/ppl_original.json")
MAX_EVAL_LEN = max(CONTEXT_LENGTHS) + TARGET_LEN  # 18432


def load_eval_samples_from_tail(data_path, num_samples, min_len, tokenizer):
    """从数据文件末尾向前读取，取最后 num_samples 个 token 长度 >= min_len 的样本。

    与训练评估的 _load_eval_samples_from_tail 逻辑完全一致。
    """
    samples = []
    with open(data_path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        chunk_size = 16 * 1024 * 1024  # 16MB
        pos = file_size
        buffer = ""
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
                if len(text) < min_len * 2:
                    continue
                ids = tokenizer(text, return_tensors="pt")["input_ids"]
                if ids.shape[1] >= min_len:
                    samples.append(ids[:, -min_len:])
                    print(f"  样本 {len(samples)}/{num_samples}: {ids.shape[1]} tokens (取末尾 {min_len})")
                    if len(samples) >= num_samples:
                        break
    samples.reverse()
    print(f"从文件末尾取 {len(samples)}/{num_samples} 个样本 (min_len={min_len})")
    return samples


def compute_ppl(model, input_ids, ctx_len, target_len, device):
    """计算给定 context 长度下的 PPL。

    与训练评估逻辑一致：前 ctx_len 个 token 作为 context（label=-100），
    后 target_len 个 token 作为 target，计算平均 cross-entropy loss。
    """
    total_len = ctx_len + target_len
    ids = input_ids[:, -total_len:].to(device)
    labels = ids.clone()
    labels[:, :ctx_len] = -100
    with torch.no_grad():
        outputs = model(input_ids=ids, labels=labels, use_cache=False)
    loss = outputs.loss.item()
    ppl = math.exp(min(loss, 20.0))
    del outputs
    torch.cuda.empty_cache()
    return ppl


def main():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, ".cache/huggingface"))

    print("=" * 60)
    print("PPL vs Context Length — Original Qwen3.5-2B (no CPT, no TTT)")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Data:  {DATA_PATH}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print(f"Target length:   {TARGET_LEN}")
    print(f"Num samples:     {NUM_SAMPLES}")
    print(f"Max eval length: {MAX_EVAL_LEN}")
    print()

    # 加载 tokenizer（与训练 build_tokenizer 一致：padding_side="right"）
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH, padding_side="right", trust_remote_code=True
    )

    # 加载评估数据（与训练 _load_eval_samples_from_tail 完全一致）
    print(f"\nLoading {NUM_SAMPLES} eval samples from tail of {DATA_PATH}...")
    t0 = time.time()
    samples = load_eval_samples_from_tail(DATA_PATH, NUM_SAMPLES, MAX_EVAL_LEN, tokenizer)
    print(f"Loaded {len(samples)} samples in {time.time()-t0:.1f}s")

    if len(samples) == 0:
        print("ERROR: No eval samples found!")
        sys.exit(1)

    # 打印样本验证信息（前5/后5 token IDs + hash，便于与训练评估对比）
    print(f"\n--- Sample verification (first 5 / last 5 token IDs) ---")
    for i, s in enumerate(samples):
        ids = s[0].tolist()
        import hashlib
        h = hashlib.md5(str(ids).encode()).hexdigest()[:8]
        print(f"  Sample {i:2d}: [{', '.join(str(x) for x in ids[:5])}, ... , "
              f"{', '.join(str(x) for x in ids[-5:])}]  md5={h}  len={len(ids)}")

    # 加载模型
    print(f"\nLoading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # 评估 PPL
    device = next(model.parameters()).device
    all_ppls = {ctx: [] for ctx in CONTEXT_LENGTHS}

    print(f"\n{'='*60}")
    print("Evaluating PPL...")
    print(f"{'='*60}")

    for s_idx, sample in enumerate(samples):
        for ctx_len in CONTEXT_LENGTHS:
            try:
                ppl = compute_ppl(model, sample, ctx_len, TARGET_LEN, device)
                all_ppls[ctx_len].append(ppl)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                all_ppls[ctx_len].append(float("nan"))
                print(f"  Sample {s_idx} ctx={ctx_len}: OOM")
        valid = [p for p in all_ppls[CONTEXT_LENGTHS[0]] if p == p]
        parts = []
        for ctx_len in CONTEXT_LENGTHS:
            grp = [p for p in all_ppls[ctx_len] if p == p]
            g_mean = sum(grp) / len(grp) if grp else float("nan")
            parts.append(f"{ctx_len}:{g_mean:.3f}")
        print(f"  Sample {s_idx+1}/{len(samples)}: {' | '.join(parts)}")

    # 汇总
    print(f"\n{'='*60}")
    print("Results — Original Qwen3.5-2B")
    print(f"{'='*60}")
    results = {}
    for ctx_len in CONTEXT_LENGTHS:
        valid = [p for p in all_ppls[ctx_len] if p == p]
        if valid:
            mean = sum(valid) / len(valid)
            if len(valid) > 1:
                var = sum((p - mean) ** 2 for p in valid) / (len(valid) - 1)
                std = var ** 0.5
            else:
                std = 0.0
            results[ctx_len] = {
                "mean": mean,
                "std": std,
                "min": min(valid),
                "max": max(valid),
                "count": len(valid),
                "total": len(all_ppls[ctx_len]),
            }
            print(f"  ctx={ctx_len:5d}: mean={mean:.4f} std={std:.4f} "
                  f"min={min(valid):.4f} max={max(valid):.4f} "
                  f"({len(valid)}/{len(all_ppls[ctx_len])} samples)")
        else:
            results[ctx_len] = {"mean": float("nan")}
            print(f"  ctx={ctx_len:5d}: mean=nan ({len(valid)}/{len(all_ppls[ctx_len])} samples)")

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output = {
        "model": "qwen3.5-2b-original",
        "context_lengths": CONTEXT_LENGTHS,
        "target_len": TARGET_LEN,
        "num_samples": NUM_SAMPLES,
        "results": {str(k): v for k, v in results.items()},
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
