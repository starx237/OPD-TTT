#!/usr/bin/env python3
"""下载 Qwen3.5-2B-Base 权重到 model_assets/qwen3.5-2b-base

功能：通过 hf-mirror 下载 Qwen/Qwen3.5-2B-Base 模型权重
使用方法：python kilo/history_scripts/download_qwen35_2b_base.py
创建时间：2026-07-11
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen3.5-2B-Base"
LOCAL_DIR = "model_assets/qwen3.5-2b-base"

print(f"Downloading {REPO_ID} -> {LOCAL_DIR} ...")
snapshot_download(
    repo_id=REPO_ID,
    local_dir=LOCAL_DIR,
)
print("Done!")
