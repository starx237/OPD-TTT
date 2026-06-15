#!/usr/bin/env python3
"""
从log.txt提取训练指标并绘制曲线

用法:
    python scripts/plot_training_curve.py --log_path log.txt
"""

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    """从log.txt解析训练指标"""
    steps = []
    losses = []
    grad_norms = []
    lrs = []

    with open(log_path, 'r') as f:
        for line in f:
            # 匹配格式: "loss: X.XXXX, grad_norm: X.XXXX, lr: X.XXe-XX"
            match = re.search(r'(\d+)\|.*loss: ([\d.]+).*grad_norm: ([\d.]+).*lr: ([\d.e-]+)', line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                grad_norm = float(match.group(3))
                lr = float(match.group(4))

                steps.append(step)
                losses.append(loss)
                grad_norms.append(grad_norm)
                lrs.append(lr)

    return steps, losses, grad_norms, lrs


def plot_metrics(steps, losses, grad_norms, lrs):
    """绘制训练曲线"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Loss曲线
    axes[0].plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 困惑度曲线
    ppl = np.exp(losses)
    axes[1].plot(steps, ppl, 'g-', linewidth=2, label='Perplexity')
    axes[1].set_ylabel('Perplexity', fontsize=12)
    axes[1].set_title('Training Perplexity (PPL = exp(loss))', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_yscale('log')

    # 梯度范数和学习率
    ax2 = axes[2].twinx()
    axes[2].plot(steps, grad_norms, 'r-', linewidth=2, label='Grad Norm')
    axes[2].set_ylabel('Gradient Norm', fontsize=12, color='r')
    axes[2].tick_params(axis='y', labelcolor='r')
    axes[2].grid(True, alpha=0.3)

    ax2.plot(steps, lrs, 'purple', linewidth=2, label='Learning Rate', alpha=0.7)
    ax2.set_ylabel('Learning Rate', fontsize=12, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_yscale('log')

    axes[2].set_xlabel('Training Step', fontsize=12)
    axes[2].set_title('Gradient Norm & Learning Rate', fontsize=14)

    # 合并图例
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("训练曲线已保存: training_curves.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="log.txt")
    args = parser.parse_args()

    print(f"解析日志文件: {args.log_path}")
    steps, losses, grad_norms, lrs = parse_log(args.log_path)

    if not steps:
        print("未找到训练指标数据")
        return

    print(f"解析到 {len(steps)} 步数据")
    print(f"Step 范围: {min(steps)} - {max(steps)}")
    print(f"Loss 范围: {min(losses):.4f} - {max(losses):.4f}")
    print(f"最终 Loss: {losses[-1]:.4f} (PPL: {np.exp(losses[-1]):.2f})")

    plot_metrics(steps, losses, grad_norms, lrs)


if __name__ == "__main__":
    main()
