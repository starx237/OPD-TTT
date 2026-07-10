#!/usr/bin/env python3
"""将 DCP 检查点转换为 HuggingFace safetensors 格式。

功能：加载 FSDP2 DCP 检查点（支持 resharding），保存为单文件 safetensors，
      供 OpenCompass / 独立推理脚本加载。

使用方法：
    python scripts/convert_dcp_to_hf.py \
        --dcp_path data/output/qwen35_2b_ttt/checkpoints/global_step_2000/global_step_2000 \
        --output_dir data/output/qwen35_2b_ttt/hf_step2000

创建时间：2026-07-09
"""
import argparse
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

os.environ.setdefault("MODELING_BACKEND", "hf")

from veomni.checkpoint import ckpt_to_state_dict
from veomni.models import save_model_weights


def main():
    parser = argparse.ArgumentParser(description="DCP → HF safetensors 转换")
    parser.add_argument("--dcp_path", required=True, help="DCP 检查点路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    args = parser.parse_args()

    print(f"加载 DCP 检查点: {args.dcp_path}")
    state_dict = ckpt_to_state_dict(args.dcp_path)
    print(f"  参数数量: {len(state_dict)}")
    sample_key = next(iter(state_dict))
    print(f"  示例 key: {sample_key}, shape: {state_dict[sample_key].shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"保存 safetensors 到: {args.output_dir}")
    save_model_weights(args.output_dir, state_dict)
    print("完成")


if __name__ == "__main__":
    main()
