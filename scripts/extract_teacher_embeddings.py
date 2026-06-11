#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取教师模型的嵌入用于 PCA 初始化

此脚本从教师模型中提取嵌入矩阵，用于初始化 OPD-TTT 的教师投影矩阵。
可以提取完整的词汇表嵌入或采样部分嵌入。

使用方法：
    python scripts/extract_teacher_embeddings.py \
        --model_path /path/to/teacher/model \
        --output_path teacher_embeddings.pt \
        --num_samples 10000
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def extract_embeddings(
    model_path: str,
    output_path: str,
    num_samples: int = None,
    sample_vocab: bool = True,
    device: str = "cuda",
):
    """
    提取教师模型的嵌入

    Args:
        model_path: 教师模型路径
        output_path: 输出文件路径
        num_samples: 采样数量（如果为 None，使用全部词汇表）
        sample_vocab: 是否从词汇表采样（如果为 False，从随机文本采样）
        device: 设备
    """
    print(f"加载教师模型：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # 获取嵌入矩阵
    vocab_size = model.config.vocab_size
    hidden_size = model.config.hidden_size

    print(f"词汇表大小：{vocab_size}, 隐藏层大小：{hidden_size}")

    if num_samples is None:
        # 使用全部词汇表嵌入
        print("提取全部词汇表嵌入...")
        with torch.no_grad():
            embeddings = model.get_input_embeddings().weight.float()  # [vocab_size, hidden_size]
    elif sample_vocab:
        # 从词汇表中随机采样
        print(f"从词汇表中采样 {num_samples} 个嵌入...")
        indices = torch.randperm(vocab_size)[:num_samples]
        with torch.no_grad():
            embeddings = model.get_input_embeddings().weight[indices].float()
    else:
        # 从随机文本采样嵌入（通过前向传播）
        print(f"从随机文本采样 {num_samples} 个嵌入...")
        embeddings = []
        model.eval()

        # 生成随机文本
        num_tokens_per_batch = 512
        num_batches = (num_samples + num_tokens_per_batch - 1) // num_tokens_per_batch

        for _ in tqdm(range(num_batches), desc="提取嵌入"):
            # 生成随机 token IDs
            input_ids = torch.randint(
                0, vocab_size, (1, num_tokens_per_batch), device=device
            )

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                # 使用最后一层的隐藏状态
                hidden_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
                embeddings.append(hidden_states.squeeze(0).cpu())

        embeddings = torch.cat(embeddings, dim=0)[:num_samples]

    # 保存嵌入
    print(f"保存嵌入到：{output_path}")
    print(f"嵌入形状：{embeddings.shape}")
    torch.save(embeddings, output_path)
    print("完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取教师模型嵌入用于 PCA 初始化")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="教师模型路径",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="teacher_embeddings.pt",
        help="输出文件路径",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="采样数量（如果为 None，使用全部词汇表）",
    )
    parser.add_argument(
        "--sample_vocab",
        action="store_true",
        help="从词汇表采样（默认从随机文本采样）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备（cuda 或 cpu）",
    )

    args = parser.parse_args()
    extract_embeddings(
        model_path=args.model_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        sample_vocab=args.sample_vocab,
        device=args.device,
    )
