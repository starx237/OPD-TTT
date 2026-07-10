#!/usr/bin/env python3
"""准备原始 Qwen3.5-2B 权重 + 强制 SWA=4096 的 eval checkpoint。

从 model_assets/qwen3.5-2b/ 读取原始权重，重映射 key 前缀
（model.language_model.* → model.*），生成带 SWA 配置的 HF checkpoint。

用法：
  python scripts/prepare_original_swa_checkpoint.py

Created: 2026-07-10
"""

import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "model_assets/qwen3.5-2b")
DST_DIR = os.path.join(PROJECT_ROOT, "data/output/qwen35_2b_ttt/hf_original_swa")


def main():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, ".cache/huggingface"))

    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "hf_models"))

    from inference_model.hf_qwen3_5.configuration_qwen3_5 import Qwen3_5TTTConfig
    from transformers import AutoConfig

    os.makedirs(DST_DIR, exist_ok=True)

    # 1. 创建修改后的 config
    print("Creating config with forced SWA=4096...")
    orig_config = AutoConfig.from_pretrained(SRC_DIR, trust_remote_code=True)
    text_config = orig_config.text_config if hasattr(orig_config, "text_config") else orig_config
    config = Qwen3_5TTTConfig(**text_config.to_dict())
    config.sliding_window = 4096
    config.opdttt_mode = False
    config.opdttt_layers = []
    config.model_type = "qwen3_5_opdttt"
    config.save_pretrained(DST_DIR)
    print(f"  Saved config to {DST_DIR}")

    # 2. 重映射 safetensors key
    print("Remapping safetensors keys (model.language_model.* -> model.*)...")
    import glob
    from safetensors import safe_open

    sd = {}
    for sf in sorted(glob.glob(os.path.join(SRC_DIR, "*.safetensors"))):
        with safe_open(sf, framework="pt") as f:
            for k in f.keys():
                if k.startswith("model.language_model."):
                    new_key = k.replace("model.language_model.", "model.")
                    sd[new_key] = f.get_tensor(k)
                elif k == "lm_head.weight":
                    sd[k] = f.get_tensor(k)

    print(f"  Remapped {len(sd)} weights (dropped visual/mtp weights)")
    save_file(sd, os.path.join(DST_DIR, "model.safetensors"), metadata={"format": "pt"})
    print(f"  Saved to {DST_DIR}/model.safetensors")

    # 3. 复制 tokenizer 文件
    print("Copying tokenizer files...")
    for fname in os.listdir(SRC_DIR):
        if fname.startswith("tokenizer") or fname in ("vocab.json", "merges.txt", "special_tokens_map.json"):
            src = os.path.join(SRC_DIR, fname)
            dst = os.path.join(DST_DIR, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"  Copied {fname}")

    # 4. 验证
    print("\nVerification:")
    from inference_model.hf_qwen3_5.modeling_qwen3_5 import Qwen3_5TTTForCausalLM
    config2 = Qwen3_5TTTConfig.from_pretrained(DST_DIR, trust_remote_code=True)
    print(f"  sliding_window: {config2.sliding_window}")
    print(f"  opdttt_mode: {config2.opdttt_mode}")
    print(f"  model_type: {config2.model_type}")

    model = Qwen3_5TTTForCausalLM._from_config(config2, torch_dtype=torch.bfloat16)
    sd_check = {}
    with safe_open(os.path.join(DST_DIR, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            sd_check[k] = f.get_tensor(k)
    missing, unexpected = model.load_state_dict(sd_check, strict=False)
    print(f"  Missing: {len(missing)} (lm_head.weight tied to embed_tokens)")
    print(f"  Unexpected: {len(unexpected)}")
    print(f"\nDone! Eval checkpoint at {DST_DIR}")


if __name__ == "__main__":
    main()
