#!/usr/bin/env python3
"""
快速生成测试：检查模型是否学到有意义的东西

用法:
    CUDA_VISIBLE_DEVICES=0 python scripts/quick_generation_test.py \
        --model_path /path/to/checkpoint \
        --prompts "The meaning of life is" "Once upon a time"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def quick_test(model_path, prompts, max_new_tokens=128, temperature=0.8):
    """快速生成测试"""
    print(f"加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("\n" + "="*60)
    print("生成测试结果")
    print("="*60)

    for prompt in prompts:
        print(f"\n提示: {prompt}")
        print("-" * 60)

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"生成: {generated[len(prompt):]}")
        print("-" * 60)


def check_quality_checklist():
    """打印质量检查清单"""
    print("\n" + "="*60)
    print("生成质量检查清单")
    print("="*60)
    print("""
    ✅ 正常信号:
    - 语法基本正确
    - 内容与提示相关
    - 有一定的连贯性
    - 常见单词拼写正确

    ❌ 问题信号:
    - 重复循环 (the the the the...)
    - 完全无关的乱码
    - 切换语言 (应该是英文但出现中文等)
    - 只有标点符号
    - 空白或极短输出
    """)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompts", type=str, nargs="+",
                        default=["The meaning of life is", "Once upon a time",
                                "In recent years,", "The main reason is"])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    quick_test(args.model_path, args.prompts, args.max_new_tokens, args.temperature)
    check_quality_checklist()


if __name__ == "__main__":
    main()
