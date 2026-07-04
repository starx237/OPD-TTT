#!/usr/bin/env python3
"""
快速 Long Context PPL 评估脚本

评估方式（无sliding window，每text每context只1次forward）:
1. 对每个文本完整 tokenization
2. 截取前 context_length 个 token
3. 一次forward，计算最后 target_length 个 token 的 PPL
4. 对所有文本取平均

用法:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_ppl_fast.py \
        --model_path /path/to/hf_ckpt \
        --tokenizer_path model_assets/tokenizer \
        --output_path results/ppl.json \
        --val_data_path data/val/pile_val.jsonl \
        --num_samples 100
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_val_texts(path, num_samples, min_length=32768):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= num_samples:
                break
            data = json.loads(line)
            text = data.get("content_split", "")
            if len(text) >= min_length:
                texts.append(text)
    return texts


@torch.no_grad()
def compute_ppl(model, tokenizer, texts, context_length, target_length, device):
    """每text只1次forward，计算最后target_length个token的PPL"""
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    n_valid = 0

    for text in tqdm(texts, desc=f"CTX={context_length}"):
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings["input_ids"].squeeze(0)
        seq_len = input_ids.shape[0]

        if seq_len < context_length + 1:
            continue

        # 截取前 context_length 个 token
        window_ids = input_ids[:context_length].to(device)
        input_ids_window = window_ids[:-1]   # [context_length-1]
        target_ids_window = window_ids[1:]   # [context_length-1]

        # 单次forward
        outputs = model(input_ids_window.unsqueeze(0))
        logits = outputs.logits.squeeze(0)  # [context_length-1, vocab_size]

        # 只计算最后 target_length 个 token
        num_targets = min(target_length, logits.shape[0])
        loss = ce_loss_fn(
            logits[-num_targets:],
            target_ids_window[-num_targets:]
        )
        total_loss += loss.item()
        total_tokens += num_targets
        n_valid += 1

    if total_tokens == 0:
        return float("inf"), 0
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl, n_valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--context_lengths", type=str, default="2048,4096,8192,16384,32768")
    parser.add_argument("--target_length", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--min_length", type=int, default=140000,
                        help="验证文本最小字符数（确保够长context）")
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"加载模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    tok_path = args.tokenizer_path or args.model_path
    print(f"加载tokenizer: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载验证集: {args.val_data_path}")
    val_texts = load_val_texts(args.val_data_path, args.num_samples, args.min_length)
    print(f"加载了 {len(val_texts)} 条验证文本")

    per_ctx_results = {}
    for ctx_len in context_lengths:
        ppl, n = compute_ppl(model, tokenizer, val_texts, ctx_len, args.target_length, device)
        per_ctx_results[str(ctx_len)] = ppl
        print(f"  Context {ctx_len}: PPL = {ppl:.4f} ({n} samples)")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(per_ctx_results, f, indent=2)
    print(f"\n结果已保存至: {args.output_path}")
    print(json.dumps(per_ctx_results, indent=2))


if __name__ == "__main__":
    main()
