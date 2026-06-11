#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPD-TTT Tokenizer 设置脚本

从 HuggingFace 下载 tokenizer 文件到指定目录。

支持以下模型:
- LLaMA 系列: meta-llama/Llama-2-7b, meta-llama/Llama-2-13b, meta-llama/Meta-Llama-3-8B
- Qwen 系列: Qwen/Qwen1.5-7B, Qwen/Qwen2-7B
- 其他 HuggingFace 上的 CausalLM 模型

使用方法:
    python scripts/setup_tokenizer.py --model meta-llama/Llama-2-7b --output model_assets/llama_500m_config
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer
    from huggingface_hub import login
except ImportError:
    print("错误: 缺少依赖包。请运行: pip install transformers huggingface_hub")
    sys.exit(1)


# 预定义的模型选项
PREDEFINED_MODELS = {
    "llama2-7b": "meta-llama/Llama-2-7b",
    "llama2-13b": "meta-llama/Llama-2-13b",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama3-70b": "meta-llama/Meta-Llama-3-70B",
    "qwen1.5-7b": "Qwen/Qwen1.5-7B",
    "qwen2-7b": "Qwen/Qwen2-7B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}


def setup_tokenizer(model_name: str, output_dir: str, token: str = None):
    """
    从 HuggingFace 下载 tokenizer 到指定目录

    Args:
        model_name: HuggingFace 模型名称或预定义别名
        output_dir: 输出目录
        token: HuggingFace 访问令牌（用于 gated 模型）
    """
    # 解析模型名称
    if model_name.lower() in PREDEFINED_MODELS:
        hf_model_name = PREDEFINED_MODELS[model_name.lower()]
        print(f"使用预定义模型: {model_name} -> {hf_model_name}")
    else:
        hf_model_name = model_name
        print(f"使用模型: {hf_model_name}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_path}")

    # 设置 HuggingFace token（如果提供）
    if token:
        login(token=token)
        print("已使用提供的 HuggingFace token 登录")

    try:
        print(f"\n正在下载 tokenizer 从 {hf_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name,
            token=token,
            trust_remote_code=True
        )

        print(f"Tokenizer 信息:")
        print(f"  - 类型: {type(tokenizer).__name__}")
        print(f"  - 词汇表大小: {len(tokenizer)}")
        print(f"  - BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
        print(f"  - EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
        print(f"  - PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
        print(f"  - UNK token: {tokenizer.unk_token} (id={tokenizer.unk_token_id})")

        # 保存 tokenizer
        print(f"\n保存 tokenizer 到 {output_path}...")
        tokenizer.save_pretrained(output_path)

        # 验证保存的文件
        saved_files = list(output_path.glob("tokenizer*.json"))
        saved_files.extend(output_path.glob("vocab.json"))
        saved_files.extend(output_path.glob("merges.txt"))
        saved_files = list(set(saved_files))

        print(f"\n✓ 成功保存以下文件:")
        for f in saved_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")

        # 生成 config.json 中的词汇表大小
        vocab_size = len(tokenizer)
        print(f"\n提示: 在模型配置文件中设置 vocab_size = {vocab_size}")

        return True

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        print("\n提示:")
        print("  1. 对于 gated 模型（如 LLaMA），需要 HuggingFace token")
        print("     获取 token: https://huggingface.co/settings/tokens")
        print("     并使用 --token 参数或在 ~/.huggingface/token 中设置")
        print("  2. 确保模型名称正确")
        print("  3. 检查网络连接")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="从 HuggingFace 下载 tokenizer 到指定目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
预定义模型别名:
  llama2-7b    meta-llama/Llama-2-7b
  llama2-13b   meta-llama/Llama-2-13b
  llama3-8b    meta-llama/Meta-Llama-3-8B
  llama3-70b   meta-llama/Meta-Llama-3-70B
  qwen1.5-7b   Qwen/Qwen1.5-7B
  qwen2-7b     Qwen/Qwen2-7B
  qwen2.5-7b   Qwen/Qwen2.5-7B
  mistral-7b   mistralai/Mistral-7B-v0.1

示例:
  # 使用预定义别名
  python scripts/setup_tokenizer.py --model llama2-7b --output model_assets/llama_500m_config

  # 直接使用 HuggingFace 模型名
  python scripts/setup_tokenizer.py --model meta-llama/Llama-2-7b --output model_assets/llama_500m_config

  # 使用 token（用于 gated 模型）
  python scripts/setup_tokenizer.py --model llama3-8b --output model_assets/llama_1b5_config --token YOUR_TOKEN
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace 模型名称或预定义别名"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录路径"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace 访问令牌（用于 gated 模型）"
    )

    args = parser.parse_args()

    # 执行设置
    success = setup_tokenizer(
        model_name=args.model,
        output_dir=args.output,
        token=args.token
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
