#!/usr/bin/env python3
"""OpenCompass RULER evaluation for the original Qwen3.5-2B model (no TTT).

This wrapper only applies the opencompass_compat patch (batch_encode_plus)
and calls OpenCompass main(). No TTT model registration or _load_model patch
is needed — the original model loads fine with standard from_pretrained.

Usage:
    python scripts/eval_ruler_baseline.py <config.py> [--debug] [--reuse latest] [...]

Created: 2026-07-10
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Qwen3.5 GatedDeltaNet 使用 fla 的 FusedRMSNormGated，其初始化硬编码
# device=torch.cuda.current_device()，导致模型在 GPU 上初始化参数时 OOM。
# 设为 None 强制回退到纯 PyTorch 的 Qwen3_5RMSNormGated。
import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen35_mod
_qwen35_mod.FusedRMSNormGated = None

import opencompass_compat  # noqa: F401

from opencompass.cli.main import main

if __name__ == "__main__":
    main()
