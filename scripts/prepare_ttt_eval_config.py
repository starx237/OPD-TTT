#!/usr/bin/env python3
"""Prepare HF checkpoint config for TTT inference evaluation.

Reads the multimodal Qwen3_5ForConditionalGeneration config from a converted
HF checkpoint and writes a text-only Qwen3_5TTTConfig (model_type="qwen3_5_opdttt")
with all TTT parameters from the training config, so that the inference_model
module can load it via AutoModelForCausalLM.from_pretrained().

Usage:
    python scripts/prepare_ttt_eval_config.py \
        --src data/output/qwen35_2b_ttt/hf_step2000 \
        --dst data/output/qwen35_2b_ttt/hf_step2000_eval \
        --opdttt-layers 0 4 8 12 16 20 \
        --ttt-lr 3 --ttt-chunk 1024 --ttt-proj

Created: 2026-07-09
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare TTT eval config")
    parser.add_argument("--src", required=True, help="Source HF checkpoint dir")
    parser.add_argument("--dst", required=True, help="Destination eval checkpoint dir")
    parser.add_argument("--opdttt-layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20])
    parser.add_argument("--ttt-lr", type=float, default=3)
    parser.add_argument("--ttt-chunk", type=int, default=1024)
    parser.add_argument("--ttt-proj", action="store_true", default=True)
    parser.add_argument("--ttt-max-norm", type=float, default=0)
    parser.add_argument("--ttt-target", type=str, default="hidden_states")
    parser.add_argument("--lambda-ntp", type=float, default=1.0)
    parser.add_argument("--lambda-align-rep", type=float, default=0.0)
    parser.add_argument("--sliding-window", type=int, default=0,
                        help="Sliding window size for full attention layers (0=disabled)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Read source config
    with open(src / "config.json", "r") as f:
        src_config = json.load(f)

    # Extract text_config (the source is a multimodal Qwen3_5ForConditionalGeneration)
    text_config = src_config.get("text_config", src_config)

    # Build TTT config by merging text_config with TTT parameters
    ttt_config = dict(text_config)
    ttt_config["model_type"] = "qwen3_5_opdttt"
    ttt_config["architectures"] = ["Qwen3_5TTTForCausalLM"]
    ttt_config["opdttt_mode"] = True
    ttt_config["opdttt_layers"] = args.opdttt_layers
    ttt_config["ttt_lr"] = args.ttt_lr
    ttt_config["ttt_chunk"] = args.ttt_chunk
    ttt_config["ttt_proj"] = args.ttt_proj
    ttt_config["ttt_max_norm"] = args.ttt_max_norm
    ttt_config["ttt_target"] = args.ttt_target
    ttt_config["lambda_ntp"] = args.lambda_ntp
    ttt_config["lambda_align_rep"] = args.lambda_align_rep
    ttt_config["sliding_window"] = args.sliding_window
    ttt_config["transformers_version"] = "5.9.0"

    # Remove fields that don't belong to the text config
    for key in ["mtp_num_hidden_layers", "mtp_use_dedicated_embeddings",
                "mamba_ssm_dtype", "mlp_only_layers"]:
        ttt_config.pop(key, None)

    # Write config
    with open(dst / "config.json", "w") as f:
        json.dump(ttt_config, f, indent=4)
    print(f"Written config.json to {dst}")

    # Symlink model weights and tokenizer files
    for fname in ["model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
        src_file = src / fname
        dst_file = dst / fname
        if src_file.exists() and not dst_file.exists():
            os.symlink(src_file.resolve(), dst_file)
            print(f"Symlinked {fname}")

    # Also symlink additional tokenizer files if they exist
    for fname in ["vocab.json", "merges.txt", "chat_template.jinja",
                  "special_tokens_map.json", "tokenizer.model"]:
        src_file = src / fname
        dst_file = dst / fname
        if src_file.exists() and not dst_file.exists():
            os.symlink(src_file.resolve(), dst_file)
            print(f"Symlinked {fname}")

    print(f"\nTTT eval checkpoint ready at: {dst}")
    print(f"  model_type: qwen3_5_opdttt")
    print(f"  architectures: ['Qwen3_5TTTForCausalLM']")
    print(f"  opdttt_layers: {args.opdttt_layers}")
    print(f"  ttt_lr: {args.ttt_lr}")
    print(f"  ttt_chunk: {args.ttt_chunk}")
    print(f"  ttt_proj: {args.ttt_proj}")


if __name__ == "__main__":
    main()
