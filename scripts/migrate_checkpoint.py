#!/usr/bin/env python3
"""
Checkpoint 迁移工具

将旧版 OPDTTTMLP checkpoint（参数名 ntp_target_proj）迁移为新版（参数名 ttt_conv）。
同时可选择重新初始化 TTT 参数（ttt_conv 零初始化, ttt_proj 稀疏对角）。

用法:
    # 迁移参数名（保留训练的 ttt_proj 值）
    python scripts/migrate_checkpoint.py \
        --input data/output/500m_stage1_pretrain/checkpoints/global_step_9500/hf_ckpt \
        --output model_assets/500m_stage1_migrated

    # 迁移参数名 + 重新初始化 TTT 参数（推荐用于继续训练）
    python scripts/migrate_checkpoint.py \
        --input data/output/500m_stage1_pretrain/checkpoints/global_step_9500/hf_ckpt \
        --output model_assets/500m_stage1_migrated \
        --reinit-ttt
"""

import argparse
import os
import shutil
import torch
from safetensors.torch import load_file, save_file


def migrate_state_dict(state_dict, reinit_ttt=False, hidden_size=1024, ttt_layers=None, std=0.02):
    """
    迁移 state dict:
    1. ntp_target_proj.weight → ttt_conv.weight
    2. 可选: 重新初始化 ttt_conv（零）和 ttt_proj（稀疏对角）
    """
    if ttt_layers is None:
        ttt_layers = [0, 6, 12, 18]

    new_state_dict = {}
    migrated = 0

    for key, value in state_dict.items():
        if "ntp_target_proj" in key:
            new_key = key.replace("ntp_target_proj", "ttt_conv")
            new_state_dict[new_key] = value
            migrated += 1
        else:
            new_state_dict[key] = value

    print(f"Migrated {migrated} keys: ntp_target_proj → ttt_conv")

    if reinit_ttt:
        import torch.nn.init as init
        for layer_idx in ttt_layers:
            conv_key = f"model.layers.{layer_idx}.mlp.ttt_conv.weight"
            proj_key = f"model.layers.{layer_idx}.mlp.ttt_proj.weight"

            if conv_key in new_state_dict:
                new_state_dict[conv_key] = torch.zeros_like(new_state_dict[conv_key])
                print(f"  Reinitialized {conv_key} → zero")

            if proj_key in new_state_dict:
                w = torch.zeros_like(new_state_dict[proj_key])
                n = w.shape[0]
                diag = torch.randn(n, device=w.device, dtype=w.dtype) * std
                w[torch.arange(n), torch.arange(n)] = diag
                new_state_dict[proj_key] = w
                print(f"  Reinitialized {proj_key} → sparse diagonal (std={std})")

            teacher_key = f"model.layers.{layer_idx}.mlp.teacher_proj.weight"
            if teacher_key in new_state_dict:
                w = torch.zeros_like(new_state_dict[teacher_key])
                n = w.shape[0]
                if w.shape[0] == w.shape[1]:
                    diag = torch.randn(n, device=w.device, dtype=w.dtype) * std
                    w[torch.arange(n), torch.arange(n)] = diag
                    new_state_dict[teacher_key] = w
                    print(f"  Reinitialized {teacher_key} → sparse diagonal")

    return new_state_dict


def main():
    parser = argparse.ArgumentParser(description="Migrate OPDTTT checkpoint parameter names")
    parser.add_argument("--input", type=str, required=True, help="Input HF checkpoint directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--reinit-ttt", action="store_true",
                        help="Reinitialize TTT params (ttt_conv=zero, ttt_proj=sparse diagonal)")
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--ttt-layers", type=int, nargs="+", default=[0, 6, 12, 18])
    parser.add_argument("--std", type=float, default=0.02)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Copy config and tokenizer, ensure sliding_window is set
    for fname in os.listdir(args.input):
        if fname.endswith('.json') or fname.endswith('.txt') or fname.endswith('.jinja'):
            shutil.copy2(os.path.join(args.input, fname), os.path.join(args.output, fname))

    # Ensure sliding_window in config.json
    config_path = os.path.join(args.output, 'config.json')
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        if 'sliding_window' not in cfg or cfg['sliding_window'] is None:
            cfg['sliding_window'] = 2048
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=2)
            print(f"Added sliding_window=2048 to config.json")

    # Migrate safetensors
    for fname in sorted(os.listdir(args.input)):
        if not fname.endswith('.safetensors'):
            continue

        print(f"\nProcessing {fname}...")
        state_dict = load_file(os.path.join(args.input, fname))
        new_state_dict = migrate_state_dict(
            state_dict,
            reinit_ttt=args.reinit_ttt,
            hidden_size=args.hidden_size,
            ttt_layers=args.ttt_layers,
            std=args.std,
        )
        save_file(new_state_dict, os.path.join(args.output, fname))
        print(f"  Saved {len(new_state_dict)} params")

    print(f"\nDone! Migrated checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
