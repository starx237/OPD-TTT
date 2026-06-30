#!/usr/bin/env python3
"""IF-eval 指令遵循评测脚本

评测方法（OPD论文式）:
1. 使用IF-eval数据集的prompt（包含可验证的指令约束）
2. Greedy生成回复
3. 用Google官方代码检查每条指令是否被遵循
4. 输出strict/loose accuracy
"""

import os
import sys
import json
import argparse
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from _model_loader import load_opdttt_model, OPD_DCP_CKPT
from instruction_following_eval import instructions_registry


def check_instruction_following(prompt, response, instruction_id_list, kwargs_list, strict=True):
    """检查response是否遵循所有指令"""
    is_following_list = []

    for index, instruction_id in enumerate(instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        instruction.build_description(**kwargs_list[index])

        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        if strict:
            is_following = bool(response.strip() and instruction.check_following(response))
        else:
            # Loose: try variations
            r = response.split("\n")
            variations = [
                response,
                response.replace("*", ""),
                "\n".join(r[1:]).strip(),
                "\n".join(r[:-1]).strip(),
                "\n".join(r[1:-1]).strip(),
                "\n".join(r[1:]).strip().replace("*", ""),
                "\n".join(r[:-1]).strip().replace("*", ""),
                "\n".join(r[1:-1]).strip().replace("*", ""),
            ]
            is_following = False
            for v in variations:
                if v.strip() and instruction.check_following(v):
                    is_following = True
                    break

        is_following_list.append(is_following)

    follow_all = all(is_following_list)
    return follow_all, is_following_list


def evaluate_ifeval(model, tokenizer, data, max_new_tokens=512):
    results = []
    strict_correct = 0
    loose_correct = 0

    for i, item in enumerate(data):
        prompt = item["prompt"]
        instruction_id_list = item["instruction_id_list"]
        kwargs_list = item["kwargs"]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        strict_follow, strict_list = check_instruction_following(
            prompt, response, instruction_id_list, kwargs_list, strict=True
        )
        loose_follow, loose_list = check_instruction_following(
            prompt, response, instruction_id_list, kwargs_list, strict=False
        )

        if strict_follow:
            strict_correct += 1
        if loose_follow:
            loose_correct += 1

        results.append({
            "key": item.get("key", i),
            "prompt": prompt[:200],
            "response": response[:300],
            "instruction_ids": instruction_id_list,
            "strict_followed": strict_list,
            "loose_followed": loose_list,
            "strict_all": strict_follow,
            "loose_all": loose_follow,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(data)}] strict={strict_correct}/{i+1} ({strict_correct/(i+1):.1%}) "
                  f"loose={loose_correct}/{i+1} ({loose_correct/(i+1):.1%})")

    return {
        "strict_accuracy": strict_correct / len(data),
        "loose_accuracy": loose_correct / len(data),
        "strict_correct": strict_correct,
        "loose_correct": loose_correct,
        "total": len(data),
        "samples": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ifeval_results.json")
    parser.add_argument("--num-samples", type=int, default=0, help="0=all, N=limit")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Download IF-eval data
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset
    ds = load_dataset("google/IFEval", split="train")
    data = list(ds)
    if args.num_samples > 0:
        data = data[:args.num_samples]
    print(f"Loaded {len(data)} IF-eval prompts")

    all_results = {}

    # Evaluate SFT baseline
    print("\n=== Evaluating SFT baseline ===")
    model, tokenizer = load_opdttt_model(ckpt_path=None)
    all_results["SFT"] = evaluate_ifeval(model, tokenizer, data)
    print(f"SFT: strict={all_results['SFT']['strict_accuracy']:.1%} loose={all_results['SFT']['loose_accuracy']:.1%}")
    del model
    torch.cuda.empty_cache()

    # Evaluate OPD checkpoint
    print("\n=== Evaluating OPD-150 ===")
    model, tokenizer = load_opdttt_model(ckpt_path=OPD_DCP_CKPT)
    all_results["OPD-150"] = evaluate_ifeval(model, tokenizer, data)
    print(f"OPD-150: strict={all_results['OPD-150']['strict_accuracy']:.1%} loose={all_results['OPD-150']['loose_accuracy']:.1%}")
    del model
    torch.cuda.empty_cache()

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "=" * 60)
    print("IF-eval Results")
    print("=" * 60)
    print(f"{'Model':<15} {'Strict':>10} {'Loose':>10} {'Total':>8}")
    print("-" * 60)
    for name, r in all_results.items():
        print(f"{name:<15} {r['strict_accuracy']:>10.1%} {r['loose_accuracy']:>10.1%} {r['total']:>8}")


if __name__ == "__main__":
    main()
