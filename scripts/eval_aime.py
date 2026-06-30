#!/usr/bin/env python3
"""AIME'24 数学推理评测脚本

评测方法（OPD论文式）:
1. Zero-shot prompt，greedy生成解题过程
2. 正则提取最终整数答案
3. 对比ground truth，计算准确率
"""

import os
import sys
import re
import json
import argparse
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from _model_loader import load_opdttt_model, OPD_DCP_CKPT

PROMPT_TEMPLATE = (
    "Solve the following math problem. Show your work step by step. "
    "At the end, write your final answer as \"The answer is: [integer]\".\n\n"
    "Problem: {problem}\n\nSolution:"
)


def extract_answer(text):
    """从生成文本中提取整数答案（AIME答案为0-999的整数）"""
    # 1. "The answer is: X"
    m = re.search(r"[Tt]he answer is[:\s]*\$(\d+)\$", text)
    if m:
        return int(m.group(1))
    m = re.search(r"[Tt]he answer is[:\s]*(\d+)", text)
    if m:
        return int(m.group(1))

    # 2. \boxed{X}
    m = re.search(r"\\boxed\{(\d+)\}", text)
    if m:
        return int(m.group(1))

    # 3. "answer is X" or "Answer: X"
    m = re.search(r"[Aa]nswer[:\s]+\$?(\d+)\$?", text)
    if m:
        return int(m.group(1))

    # 4. 最后一个整数
    numbers = re.findall(r"\b(\d+)\b", text)
    if numbers:
        return int(numbers[-1])

    return None


def evaluate_aime(model, tokenizer, problems, max_new_tokens=1024):
    results = []
    correct = 0

    for i, (pid, problem, gt_answer) in enumerate(problems):
        prompt = PROMPT_TEMPLATE.format(problem=problem)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        pred_answer = extract_answer(generated)
        is_correct = pred_answer is not None and pred_answer == gt_answer
        if is_correct:
            correct += 1

        results.append({
            "id": pid,
            "problem": problem[:200],
            "ground_truth": gt_answer,
            "predicted": pred_answer,
            "correct": is_correct,
            "response": generated[:500],
        })

        status = "✓" if is_correct else "✗"
        print(f"  [{i+1}/{len(problems)}] {pid}: GT={gt_answer} Pred={pred_answer} {status}")

    return {"accuracy": correct / len(problems), "correct": correct, "total": len(problems), "samples": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/aime_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Download AIME 2024 data
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    problems = [(d["ID"], d["Problem"], d["Answer"]) for d in ds]
    print(f"Loaded {len(problems)} AIME 2024 problems")

    all_results = {}

    # Evaluate SFT baseline
    print("\n=== Evaluating SFT baseline ===")
    model, tokenizer = load_opdttt_model(ckpt_path=None)
    all_results["SFT"] = evaluate_aime(model, tokenizer, problems)
    print(f"SFT: {all_results['SFT']['correct']}/{all_results['SFT']['total']} = {all_results['SFT']['accuracy']:.1%}")
    del model
    torch.cuda.empty_cache()

    # Evaluate OPD checkpoint
    print("\n=== Evaluating OPD-150 ===")
    model, tokenizer = load_opdttt_model(ckpt_path=OPD_DCP_CKPT)
    all_results["OPD-150"] = evaluate_aime(model, tokenizer, problems)
    print(f"OPD-150: {all_results['OPD-150']['correct']}/{all_results['OPD-150']['total']} = {all_results['OPD-150']['accuracy']:.1%}")
    del model
    torch.cuda.empty_cache()

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "=" * 50)
    print("AIME'24 Results")
    print("=" * 50)
    for name, r in all_results.items():
        print(f"  {name}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")


if __name__ == "__main__":
    main()
