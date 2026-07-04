#!/usr/bin/env python3
"""
Sliding Window Perplexity 评估脚本。

评估方式：
1. 对每个文本完整 tokenization
2. 按固定 context_length 截取序列
3. 使用标准交叉熵损失计算最后 target_length 个 token 的困惑度
4. 滑动窗口处理，对所有 target tokens 取平均

用法:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_sliding_window_ppl.py \
        --model_path /path/to/hf_ckpt \
        --output_path results/500m_opdttt_ppl.json \
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


def load_val_texts(path: str, num_samples: int, min_length: int = 32768):
    """加载验证文本，确保文本足够长"""
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
def compute_sliding_window_ppl(
    model,
    tokenizer,
    texts,
    context_length,
    stride,
    target_length,
    device
):
    """
    计算滑动窗口困惑度（使用标准交叉熵损失）

    Args:
        model: 语言模型
        tokenizer: 分词器
        texts: 验证文本列表
        context_length: 上下文长度
        stride: 滑动窗口步长
        target_length: 目标 token 数量（从每个序列末尾计算多少个 token）
        device: 设备

    评估方式：
        对于长度为 L 的文本，取一个长度为 context_length 的窗口
        计算窗口最后 target_length 个 token 的标准交叉熵损失
        然后滑动 stride 个 token，重复上述过程
        PPL = exp(平均交叉熵损失)
    """
    model.eval()
    # 使用标准交叉熵损失，reduction='sum' 方便累计
    ce_loss_fn = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc=f"CTX={context_length}"):
        # 完整 tokenization，不截断
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings["input_ids"].squeeze(0)  # [seq_len]
        seq_len = input_ids.shape[0]

        # 跳过太短的文本
        if seq_len < context_length + 1:
            continue

        # 滑动窗口处理
        for start_idx in range(0, seq_len - context_length + 1, stride):
            end_idx = start_idx + context_length

            # 取窗口（完整 context_length 个 token）
            window_ids = input_ids[start_idx:end_idx].to(device)

            # 模型接收完整窗口的前 n-1 个 token，输出所有位置的 logits
            input_ids_window = window_ids[:-1]   # [context_length-1] 作为输入
            target_ids_window = window_ids[1:]   # [context_length-1] 作为目标

            if input_ids_window.shape[0] == 0:
                continue

            # 前向传播：模型看完整 context
            outputs = model(input_ids_window.unsqueeze(0))
            logits = outputs.logits.squeeze(0)  # [context_length-1, vocab_size]

            # 只计算最后 target_length 个 token 的 loss
            num_targets = min(target_length, logits.shape[0])
            if num_targets == 0:
                continue

            loss = ce_loss_fn(
                logits[-num_targets:],
                target_ids_window[-num_targets:]
            )

            total_loss += loss.item()
            total_tokens += num_targets

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Tokenizer路径（默认与model_path相同）")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--context_lengths", type=str,
                        default="2048,4096,8192,16384,32768")
    parser.add_argument("--stride", type=int, default=0,
                        help="滑动窗口步长（0=自动设为context_length-target_length，避免target重叠）")
    parser.add_argument("--target_length", type=int, default=256,
                        help="每个窗口计算最后多少个 token 的困惑度")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--min_length", type=int, default=32768,
                        help="验证文本最小字符长度")
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"加载模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
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
        # 自动stride：避免target区域重叠
        stride = args.stride if args.stride > 0 else max(ctx_len - args.target_length, 1)
        print(f"  stride={stride}")
        ppl = compute_sliding_window_ppl(
            model,
            tokenizer,
            val_texts,
            ctx_len,
            stride,
            args.target_length,
            device,
        )
        per_ctx_results[str(ctx_len)] = ppl
        print(f"  Context {ctx_len}: PPL = {ppl:.4f}")

    # 输出flat dict格式（兼容plot_figure.py）
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(per_ctx_results, f, indent=2)
    print(f"\n结果已保存至: {args.output_path}")
    print(json.dumps(per_ctx_results, indent=2))


if __name__ == "__main__":
    main()
