#!/usr/bin/env python3
"""Qwen3.5-2B-Base 推理能力测试脚本

功能：
  1. 基本续写测试（raw completion，不用 chat template）
  2. Chat template 测试（think mode on/off 对比）
  3. 数学题测试（think on/off 对比正确率）
  4. 自动检测 base 模型是否有 chat template，无则回退到 instruct tokenizer

使用方法：
    CUDA_VISIBLE_DEVICES=1 python kilo/history_scripts/test_qwen35_base.py
    CUDA_VISIBLE_DEVICES=1 python kilo/history_scripts/test_qwen35_base.py --model_path model_assets/qwen3.5-2b-base
    CUDA_VISIBLE_DEVICES=1 python kilo/history_scripts/test_qwen35_base.py --think_only  # 只跑 think mode 测试

创建时间：2026-07-11
"""
import argparse
import os
import sys
import re
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Qwen3.5 GatedDeltaNet 使用 fla 的 FusedRMSNormGated，其初始化硬编码
# device=torch.cuda.current_device()，导致模型在 GPU 上初始化参数时 OOM。
# 设为 None 强制回退到纯 PyTorch 的 Qwen3_5RMSNormGated。
import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen35_mod
_qwen35_mod.FusedRMSNormGated = None


def load_model_and_tokenizer(model_path, tokenizer_path, gpu_id):
    """加载 base 模型和 tokenizer。

    如果 base 模型的 tokenizer 没有 chat_template，回退到 instruct 版本的 tokenizer。
    """
    print(f"Loading model from {model_path} ...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(f"cuda:{gpu_id}")
    model.eval()

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 检查是否有 chat_template
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    print(f"  Tokenizer has chat_template: {has_chat_template}")

    if not has_chat_template and tokenizer_path != model_path:
        print(f"  Trying model's own tokenizer from {model_path} ...")
        tok2 = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if getattr(tok2, "chat_template", None):
            tokenizer = tok2
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"  Using chat_template from {model_path}")
            has_chat_template = True

    return model, tokenizer, has_chat_template


def generate_raw(model, tokenizer, prompt, gpu_id, max_new_tokens=256):
    """Raw completion（不用 chat template）。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{gpu_id}")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_chat(model, tokenizer, messages, gpu_id, enable_thinking, max_new_tokens=1024):
    """Chat template 生成，支持 think mode。"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32768).to(f"cuda:{gpu_id}")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    full = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # 分离 thinking 和 answer
    if "</think>" in full:
        thinking, answer = full.split("</think>", 1)
        thinking = thinking.strip()
        answer = answer.strip()
    else:
        thinking = ""
        answer = full.strip()

    return thinking, answer, full


# ====================== 测试用例 ======================

RAW_TESTS = [
    ("续写", "The capital of France is"),
    ("算术", "1 + 1 ="),
    ("常识", "Water boils at"),
    ("代码", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"),
]

MATH_PROBLEMS = [
    {
        "question": "A store sells pencils at 3 for $1.20. How much would 12 pencils cost?",
        "answer": "4.80",
        "check": lambda pred: "4.80" in pred.replace(" ", "") or "4.8" in pred.replace(" ", ""),
    },
    {
        "question": "If a train travels at 60 km/h for 2.5 hours, how far does it travel?",
        "answer": "150",
        "check": lambda pred: "150" in pred,
    },
    {
        "question": "What is 17 * 23?",
        "answer": "391",
        "check": lambda pred: "391" in pred,
    },
    {
        "question": "A rectangle has length 8 cm and width 5 cm. What is its area?",
        "answer": "40",
        "check": lambda pred: any(f"{a}" in pred for a in ["40", "40cm", "40 cm", "40 square"]),
    },
    {
        "question": "If x + 5 = 12, what is x?",
        "answer": "7",
        "check": lambda pred: re.search(r'\b7\b', pred) is not None,
    },
]

CHAT_TESTS = [
    "What is the capital of Japan?",
    "Explain what is a prime number in one sentence.",
    "Write a haiku about autumn.",
]


def run_raw_tests(model, tokenizer, gpu_id):
    """测试 1：基本续写能力。"""
    print("\n" + "=" * 60)
    print("测试 1：基本续写（Raw Completion）")
    print("=" * 60)
    for name, prompt in RAW_TESTS:
        result = generate_raw(model, tokenizer, prompt, gpu_id, max_new_tokens=128)
        result_display = result[:200].replace("\n", " ↵ ")
        print(f"\n[{name}]")
        print(f"  Prompt: {prompt}")
        print(f"  Output: {result_display}")


def run_chat_tests(model, tokenizer, gpu_id):
    """测试 2：Chat template think on/off 对比。"""
    print("\n" + "=" * 60)
    print("测试 2：Chat Template（Think OFF vs Think ON）")
    print("=" * 60)

    for question in CHAT_TESTS:
        messages = [{"role": "user", "content": question}]

        # Think OFF
        _, answer_off, _ = generate_chat(
            model, tokenizer, messages, gpu_id,
            enable_thinking=False, max_new_tokens=512,
        )

        # Think ON
        thinking_on, answer_on, _ = generate_chat(
            model, tokenizer, messages, gpu_id,
            enable_thinking=True, max_new_tokens=1024,
        )

        print(f"\n[Q] {question}")
        print(f"  Think OFF: {answer_off[:200].replace(chr(10), ' ')}")
        if thinking_on:
            print(f"  Think ON (thinking): {thinking_on[:200].replace(chr(10), ' ')}")
        print(f"  Think ON (answer): {answer_on[:200].replace(chr(10), ' ')}")


def run_math_tests(model, tokenizer, gpu_id):
    """测试 3：数学题 think on/off 对比。"""
    print("\n" + "=" * 60)
    print("测试 3：数学题（Think OFF vs Think ON）")
    print("=" * 60)

    results = {"off": {"correct": 0, "total": 0}, "on": {"correct": 0, "total": 0}}

    for i, prob in enumerate(MATH_PROBLEMS):
        messages = [{"role": "user", "content": prob["question"]}]

        # Think OFF
        _, answer_off, _ = generate_chat(
            model, tokenizer, messages, gpu_id,
            enable_thinking=False, max_new_tokens=512,
        )
        correct_off = prob["check"](answer_off)
        results["off"]["total"] += 1
        results["off"]["correct"] += int(correct_off)

        # Think ON
        thinking_on, answer_on, _ = generate_chat(
            model, tokenizer, messages, gpu_id,
            enable_thinking=True, max_new_tokens=1024,
        )
        correct_on = prob["check"](answer_on)
        results["on"]["total"] += 1
        results["on"]["correct"] += int(correct_on)

        print(f"\n[题 {i+1}] {prob['question']}")
        print(f"  正确答案: {prob['answer']}")
        print(f"  Think OFF: {'✓' if correct_off else '✗'} {answer_off[:150].replace(chr(10), ' ')}")
        if thinking_on:
            print(f"  Think ON (thinking): {thinking_on[:150].replace(chr(10), ' ')}")
        print(f"  Think ON (answer): {'✓' if correct_on else '✗'} {answer_on[:150].replace(chr(10), ' ')}")

    print(f"\n--- 数学题汇总 ---")
    print(f"  Think OFF: {results['off']['correct']}/{results['off']['total']}")
    print(f"  Think ON:  {results['on']['correct']}/{results['on']['total']}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-2B-Base 推理能力测试")
    parser.add_argument("--model_path", default="model_assets/qwen3.5-2b-base")
    parser.add_argument("--tokenizer_path", default="model_assets/qwen3.5-2b",
                        help="tokenizer 路径（默认用 instruct 版本，因为 base 可能没有 chat template）")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--think_only", action="store_true", help="只跑 think mode 测试")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.gpu = 0

    model, tokenizer, has_chat = load_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.gpu
    )

    if not args.think_only:
        run_raw_tests(model, tokenizer, args.gpu)

    if has_chat:
        run_chat_tests(model, tokenizer, args.gpu)
        run_math_tests(model, tokenizer, args.gpu)
    else:
        print("\n[警告] Tokenizer 没有 chat_template，跳过 chat/think 测试")
        print("  请检查 --tokenizer_path 是否指向有 chat_template 的版本")

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main()
