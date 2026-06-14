#!/usr/bin/env python3
"""
创建验证数据集（未打包的完整文本）

从原始长文本数据中抽取一定比例作为验证集，确保：
1. 每条数据是完整的原始文本（未打包）
2. 文本长度足够长（>= 32768 tokens）
3. 不包含 padding 或截断

用法:
    python scripts/create_val_set.py \
        --input data/piles_long_text.jsonl \
        --output data/val/pile_val.jsonl \
        --num_samples 1000 \
        --ratio 0.001
"""
import argparse
import json
import os
import random


def create_val_set(
    input_path: str,
    output_path: str,
    num_samples: int = 100,
    ratio: float = 0.001,
    min_chars: int = 30000,
    seed: int = 42
):
    """从原始数据中创建验证集

    Args:
        input_path: 原始打包前数据路径
        output_path: 输出验证集路径
        num_samples: 采样数量（与 ratio 二选一）
        ratio: 采样比例（与 num_samples 二选一）
        min_chars: 最小字符数（过滤短文本）
        seed: 随机种子
    """
    random.seed(seed)

    # 读取所有数据
    print(f"读取数据: {input_path}")
    texts = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                text = data.get('content_split', '')
                if len(text) >= min_chars:
                    texts.append(text)
            except json.JSONDecodeError:
                continue

    print(f"总共 {len(texts)} 条有效数据")

    # 采样
    if num_samples and num_samples < len(texts):
        # 指定数量
        sample_size = min(num_samples, len(texts))
        val_texts = random.sample(texts, sample_size)
    elif ratio:
        # 按比例
        sample_size = max(1, int(len(texts) * ratio))
        val_texts = random.sample(texts, sample_size)
    else:
        # 默认 1000 条
        val_texts = random.sample(texts, min(1000, len(texts)))

    print(f"采样 {len(val_texts)} 条作为验证集")

    # 写入验证集
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in val_texts:
            f.write(json.dumps({'content_split': text}, ensure_ascii=False) + '\n')

    print(f"验证集已保存: {output_path}")

    # 统计信息
    total_chars = sum(len(t) for t in val_texts)
    avg_chars = total_chars / len(val_texts)
    print(f"平均长度: {avg_chars:.0f} 字符 (约 {avg_chars/4:.0f} tokens)")


def main():
    parser = argparse.ArgumentParser(description='创建验证数据集')
    parser.add_argument('--input', type=str, default='data/piles_long_text.jsonl',
                        help='原始数据路径（未打包）')
    parser.add_argument('--output', type=str, default='data/val/pile_val.jsonl',
                        help='输出验证集路径')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='采样数量（默认1000）')
    parser.add_argument('--ratio', type=float, default=None,
                        help='采样比例（与 num_samples 二选一）')
    parser.add_argument('--min_chars', type=int, default=30000,
                        help='最小字符数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    create_val_set(
        input_path=args.input,
        output_path=args.output,
        num_samples=args.num_samples,
        ratio=args.ratio,
        min_chars=args.min_chars,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
