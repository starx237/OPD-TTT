#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制 Sliding Window Perplexity 对比图

对比以下三种方法：
1. Baseline (SWA) - 滑动窗口基线
2. In-Place TTT - 原始测试时训练
3. OPD-TTT - On-Policy Distillation Enhanced Test-Time Training
"""

import argparse
import json
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    # 500M models
    parser.add_argument("--results_500m_ttt", type=str, default=None,
                        help="500M In-Place TTT 结果 JSON 路径")
    parser.add_argument("--results_500m_baseline", type=str, default=None,
                        help="500M Baseline 结果 JSON 路径")
    parser.add_argument("--results_500m_opdttt", type=str, default=None,
                        help="500M OPD-TTT 结果 JSON 路径")
    # 1.5B models
    parser.add_argument("--results_1b5_ttt", type=str, default=None,
                        help="1.5B In-Place TTT 结果 JSON 路径")
    parser.add_argument("--results_1b5_baseline", type=str, default=None,
                        help="1.5B Baseline 结果 JSON 路径")
    parser.add_argument("--results_1b5_opdttt", type=str, default=None,
                        help="1.5B OPD-TTT 结果 JSON 路径")
    # General options
    parser.add_argument("--output", type=str, default="opdttt_comparison.png",
                        help="输出图片路径")
    args = parser.parse_args()

    def load(path):
        with open(path) as f:
            data = json.load(f)
        lengths = sorted(int(k) for k in data.keys())
        ppls = [data[str(l)] for l in lengths]
        return lengths, ppls

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_labels = ["2k", "4k", "8k", "16k", "32k"]

    # 500M subplot
    ax500 = axes[0]
    if args.results_500m_baseline:
        _, p_base = load(args.results_500m_baseline)
        ax500.plot(x_labels, p_base, "s--", color="gray", label="Baseline (SWA)", linewidth=2)
    if args.results_500m_ttt:
        _, p_ttt = load(args.results_500m_ttt)
        ax500.plot(x_labels, p_ttt, "o-", color="blue", label="In-Place TTT", linewidth=2)
    if args.results_500m_opdttt:
        _, p_opdttt = load(args.results_500m_opdttt)
        ax500.plot(x_labels, p_opdttt, "^-.", color="red", label="OPD-TTT", linewidth=2)

    ax500.set_xlabel("Context Length")
    ax500.set_ylabel("Perplexity")
    ax500.set_title("(a) 滑动窗口困惑度对比 (500M 模型)")
    ax500.legend()
    ax500.grid(True, alpha=0.3)

    # 1.5B subplot
    ax1b5 = axes[1]
    if args.results_1b5_baseline:
        _, p_base = load(args.results_1b5_baseline)
        ax1b5.plot(x_labels, p_base, "s--", color="gray", label="Baseline (SWA)", linewidth=2)
    if args.results_1b5_ttt:
        _, p_ttt = load(args.results_1b5_ttt)
        ax1b5.plot(x_labels, p_ttt, "o-", color="blue", label="In-Place TTT", linewidth=2)
    if args.results_1b5_opdttt:
        _, p_opdttt = load(args.results_1b5_opdttt)
        ax1b5.plot(x_labels, p_opdttt, "^-.", color="red", label="OPD-TTT", linewidth=2)

    ax1b5.set_xlabel("Context Length")
    ax1b5.set_ylabel("Perplexity")
    ax1b5.set_title("(b) 滑动窗口困惑度对比 (1.5B 模型)")
    ax1b5.legend()
    ax1b5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"OPD-TTT 对比图表已保存至: {args.output}")
    print("\n对比内容:")
    print("  - Baseline (SWA): 滑动窗口基线")
    print("  - In-Place TTT: 原始测试时训练")
    print("  - OPD-TTT: On-Policy Distillation Enhanced Test-Time Training")


if __name__ == "__main__":
    main()
