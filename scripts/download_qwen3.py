#!/usr/bin/env python3
"""Download Qwen3-8B and check config/tokenizer compatibility"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

print("Downloading Qwen3-8B...")
path = snapshot_download(
    "Qwen/Qwen3-8B",
    cache_dir="model_assets/.cache",
    ignore_patterns=["*.gguf", "*.ot", "*.msgpack", "*.h5", "original/*"],
)
print(f"Downloaded to: {path}")

from transformers import AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained(path)
tok = AutoTokenizer.from_pretrained(path)

print(f"vocab_size: {config.vocab_size}")
print(f"hidden_size: {config.hidden_size}")
print(f"eos_token_id: {config.eos_token_id}")
print(f"tokenizer vocab: {tok.vocab_size}")
print(f"tokenizer eos: {tok.eos_token_id}")
print(f"architectures: {config.architectures}")
print(f"model_type: {config.model_type}")

with open("model_assets/qwen3_8b_info.txt", "w") as f:
    f.write(f"path: {path}\n")
    f.write(f"vocab_size: {config.vocab_size}\n")
    f.write(f"hidden_size: {config.hidden_size}\n")
    f.write(f"eos_token_id: {config.eos_token_id}\n")
    f.write(f"tokenizer vocab: {tok.vocab_size}\n")
    f.write(f"tokenizer eos: {tok.eos_token_id}\n")
    f.write(f"architectures: {config.architectures}\n")
    f.write(f"model_type: {config.model_type}\n")

print("Info saved to model_assets/qwen3_8b_info.txt")
