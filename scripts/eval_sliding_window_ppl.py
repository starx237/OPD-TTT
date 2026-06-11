#!/usr/bin/env python3
"""
Sliding Window Perplexity 评估脚本（复现 Figure 2）。

用法:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_sliding_window_ppl.py \
        --model_path /path/to/hf_ckpt \
        --output_path results/500m_ttt_ppl.json \
        --val_data_path data/val/pile_val_veomni.jsonl \
        --num_samples 100
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_val_texts(path: str, num_samples: int, max_length: int = 65536):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= num_samples:
                break
            data = json.loads(line)
            text = data.get("content_split", "")
            if len(text) > max_length:
                text = text[:max_length]
            texts.append(text)
    return texts


@torch.no_grad()
def compute_sliding_window_ppl(model, tokenizer, texts,
                                context_length, sliding_window, device):
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc=f"CTX={context_length}"):
        encodings = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=context_length + sliding_window
        )
        input_ids = encodings["input_ids"].to(device)

        if input_ids.shape[1] < sliding_window + 1:
            continue

        outputs = model(input_ids, labels=input_ids)
        nll = outputs.loss.item() * input_ids.shape[1]
        total_nll += nll
        total_tokens += input_ids.shape[1]

    if total_tokens == 0:
        return float("inf")
    return torch.exp(torch.tensor(total_nll / total_tokens)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--context_lengths", type=str,
                        default="2048,4096,8192,16384,32768")
    parser.add_argument("--sliding_window", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"加载模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载验证集: {args.val_data_path}")
    val_texts = load_val_texts(args.val_data_path, args.num_samples)
    print(f"加载了 {len(val_texts)} 条验证文本")

    results = {}
    for ctx_len in context_lengths:
        ppl = compute_sliding_window_ppl(
            model, tokenizer, val_texts, ctx_len, args.sliding_window, device,
        )
        results[str(ctx_len)] = ppl
        print(f"  Context {ctx_len}: PPL = {ppl:.4f}")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存至: {args.output_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
