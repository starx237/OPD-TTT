#!/usr/bin/env python3
"""Phase 1: 9B Response Length Test

用 Qwen3.5-9B 对 OT3 的 medium/advanced 难度 prompt 生成（thinking mode），
对比 9B 生成长度 vs OT3 原始 response 长度，为 SFT response 来源提供决策依据。

用法（miniconda3 环境）:
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
  export PYTHONPATH="$PWD:$PWD/VeOmni:$PWD/hf_models:$PYTHONPATH"
  export CUDA_VISIBLE_DEVICES=0
  python scripts/test_9b_response_length.py \\
      --input_dir data/prompts_raw/data \\
      --num_samples 10 --min_difficulty 8 \\
      --model_path model_assets/qwen3.5-9b \\
      --out data/phase1_9b_length_test.json
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import pyarrow.parquet as pq
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """对已出现过的 token 施加固定惩罚（vLLM 语义）。batch=1 实现。"""

    def __init__(self, presence_penalty=1.5):
        self.penalty = presence_penalty

    def __call__(self, input_ids, scores):
        unique_tokens = input_ids[0].unique()
        scores[0, unique_tokens] -= self.penalty
        return scores


# 官方推荐 thinking mode 采样参数
SAMPLING = dict(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
    presence_penalty=1.5,
)


def sample_prompts(input_dir, num_samples, min_difficulty, seed=42):
    """从 OT3 parquet 采样 medium/advanced 难度的 prompt。"""
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    candidates = []
    for f in files:
        t = pq.read_table(f, columns=["difficulty", "source", "domain", "conversations"])
        df = t.to_pandas()
        for _, row in df.iterrows():
            diff = row["difficulty"]
            if diff is None or diff < min_difficulty:
                continue
            conv = row["conversations"]
            if len(conv) < 2:
                continue
            prompt = conv[0]["value"]
            response = conv[1]["value"]
            if not prompt or not response:
                continue
            candidates.append(
                {
                    "prompt": prompt,
                    "ot3_response": response,
                    "difficulty": int(diff),
                    "source": row["source"],
                    "domain": row["domain"],
                }
            )
        if len(candidates) >= num_samples * 20:
            break
    rng.shuffle(candidates)
    return candidates[:num_samples]


def main():
    parser = argparse.ArgumentParser(description="Phase 1: 9B Response Length Test")
    parser.add_argument("--input_dir", default="data/prompts_raw/data")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--min_difficulty", type=int, default=8)
    parser.add_argument("--model_path", default="model_assets/qwen3.5-9b")
    parser.add_argument("--max_new_tokens", type=int, default=81920)
    parser.add_argument("--out", default="data/phase1_9b_length_test.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("  Phase 1: 9B Response Length Test")
    print(f"  min_difficulty={args.min_difficulty}, num_samples={args.num_samples}")
    print(f"  max_new_tokens={args.max_new_tokens}")
    print(f"  sampling={SAMPLING}")
    print("=" * 70, flush=True)

    # 1. 采样 prompt
    print("\n[1/3] Sampling prompts from OT3...", flush=True)
    samples = sample_prompts(args.input_dir, args.num_samples, args.min_difficulty, args.seed)
    print(f"  Sampled {len(samples)} prompts (difficulty>={args.min_difficulty})", flush=True)
    for i, s in enumerate(samples):
        print(
            f"  [{i}] diff={s['difficulty']} src={s['source'][:40]} dom={s['domain']} "
            f"prompt[:60]={s['prompt'][:60]!r}",
            flush=True,
        )

    # 2. 加载 tokenizer + 统计 OT3 response 长度
    print("\n[2/3] Loading tokenizer & computing OT3 response lengths...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    for s in samples:
        s["ot3_tokens"] = len(tokenizer(s["ot3_response"], add_special_tokens=False)["input_ids"])
        s["ot3_has_think_close"] = "</think>" in s["ot3_response"]
    import numpy as np

    ot3_lens = np.array([s["ot3_tokens"] for s in samples])
    print(
        f"  OT3 response tokens: mean={ot3_lens.mean():.0f} median={np.median(ot3_lens):.0f} "
        f"max={ot3_lens.max()} min={ot3_lens.min()}",
        flush=True,
    )
    print(
        f"  OT3 has </think>: {sum(s['ot3_has_think_close'] for s in samples)}/{len(samples)}",
        flush=True,
    )

    # 3. 加载 9B 模型并生成
    print("\n[3/3] Loading 9B model & generating...", flush=True)
    print(f"  GPU mem before load: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    print(f"  GPU mem after load: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    gen_lens = []
    for i, s in enumerate(samples):
        messages = [{"role": "user", "content": s["prompt"]}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=True,
            return_tensors="pt",
        )["input_ids"].cuda()

        t0 = time.time()
        logits_processor = [PresencePenaltyLogitsProcessor(SAMPLING["presence_penalty"])]
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=SAMPLING["temperature"],
                top_p=SAMPLING["top_p"],
                top_k=SAMPLING["top_k"],
                repetition_penalty=SAMPLING["repetition_penalty"],
                logits_processor=logits_processor,
            )
        elapsed = time.time() - t0
        new_tokens = output_ids.shape[1] - input_ids.shape[1]
        full_output = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
        has_think_close = "</think>" in full_output

        gen_lens.append(new_tokens)
        s["gen_tokens"] = new_tokens
        s["gen_has_think_close"] = has_think_close
        s["gen_elapsed"] = elapsed
        s["gen_output_preview"] = full_output[:200]

        print(
            f"\n  [{i}] gen_tokens={new_tokens} (ot3={s['ot3_tokens']}) "
            f"</think>={'Y' if has_think_close else 'N'} time={elapsed:.1f}s",
            flush=True,
        )
        print(f"      preview: {full_output[:120]!r}", flush=True)

    # 释放模型
    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # 汇总
    gen_arr = np.array(gen_lens)
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  OT3 response:  mean={ot3_lens.mean():.0f} median={np.median(ot3_lens):.0f} max={ot3_lens.max()}")
    print(f"  9B generated:  mean={gen_arr.mean():.0f} median={np.median(gen_arr):.0f} max={gen_arr.max()}")
    print(f"  9B >32768: {(gen_arr > 32768).sum()}/{len(gen_arr)}")
    print(f"  9B </think>: {sum(s['gen_has_think_close'] for s in samples)}/{len(samples)}")
    print(f"  OT3 </think>: {sum(s['ot3_has_think_close'] for s in samples)}/{len(samples)}")

    avg_gen = gen_arr.mean()
    if avg_gen > 32768:
        decision = "9B avg response > 32768 tokens -> use 9B generation (Phase 3)"
    else:
        decision = "9B avg response <= 32768 tokens -> use OT3 original response (skip Phase 3)"
    print(f"\n  Decision hint: {decision}")
    print("=" * 70, flush=True)

    # 保存结果
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "sampling": SAMPLING,
                "summary": {
                    "ot3_mean": float(ot3_lens.mean()),
                    "ot3_median": float(np.median(ot3_lens)),
                    "ot3_max": int(ot3_lens.max()),
                    "gen_mean": float(gen_arr.mean()),
                    "gen_median": float(np.median(gen_arr)),
                    "gen_max": int(gen_arr.max()),
                    "gen_over_32768": int((gen_arr > 32768).sum()),
                    "gen_has_think_close": sum(s["gen_has_think_close"] for s in samples),
                    "ot3_has_think_close": sum(s["ot3_has_think_close"] for s in samples),
                    "decision": decision,
                },
                "samples": [
                    {
                        "prompt": s["prompt"][:500],
                        "difficulty": s["difficulty"],
                        "source": s["source"],
                        "domain": s["domain"],
                        "ot3_tokens": s["ot3_tokens"],
                        "ot3_has_think_close": s["ot3_has_think_close"],
                        "gen_tokens": s["gen_tokens"],
                        "gen_has_think_close": s["gen_has_think_close"],
                        "gen_elapsed": s["gen_elapsed"],
                        "gen_output_preview": s["gen_output_preview"],
                    }
                    for s in samples
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
