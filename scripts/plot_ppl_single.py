#!/usr/bin/env python3
"""绘制单个模型的 Sliding Window PPL 折线图"""

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="PPL结果JSON路径")
    parser.add_argument("--output", type=str, default="ppl_plot.png", help="输出图片路径")
    parser.add_argument("--title", type=str, default="Sliding Window Perplexity", help="图表标题")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    lengths = sorted(int(k) for k in data.keys())
    ppls = [data[str(l)] for l in lengths]
    x_labels = [f"{l//1024}k" for l in lengths]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_labels, ppls, "o-", color="blue", linewidth=2, markersize=8)
    ax.set_xlabel("Context Length", fontsize=13)
    ax.set_ylabel("Perplexity", fontsize=13)
    ax.set_title(args.title, fontsize=14)
    ax.grid(True, alpha=0.3)

    for i, (l, p) in enumerate(zip(lengths, ppls)):
        ax.annotate(f"{p:.2f}", (i, p), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"图表已保存至: {args.output}")
    print(f"PPL: {dict(zip(x_labels, [f'{p:.4f}' for p in ppls]))}")


if __name__ == "__main__":
    main()
