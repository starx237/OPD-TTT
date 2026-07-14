#!/usr/bin/env python3
"""将 OpenThoughts-3 QA 对转换为 VeOmni conversation 格式。

输入: data/opd_sft_raw.jsonl  ({"prompt": "...", "response": "..."})
输出: data/sft_train.jsonl    ({"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]})
      data/sft_val.jsonl      (同上，1000条验证集)

VeOmni ChatmlTemplate 会对 assistant 角色计算 loss，user 部分被 mask (-100)。
OT3 response 中的 <think>...</think> 格式与 Qwen3.5 thinking mode 兼容。

使用方法:
    python scripts/prepare_sft_conversation.py

Created: 2026-07-14
"""

import json
import random
import os


def main():
    input_path = "data/opd_sft_raw.jsonl"
    train_path = "data/sft_train.jsonl"
    val_path = "data/sft_val.jsonl"
    val_size = 1000
    seed = 42

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # 读取所有 QA 对
    print(f"Reading {input_path}...")
    all_data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get("prompt", "").strip()
            response = data.get("response", "").strip()
            if not prompt or not response:
                continue
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            all_data.append({"messages": messages})

    print(f"Total valid samples: {len(all_data)}")

    # 随机分割
    random.seed(seed)
    random.shuffle(all_data)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]

    # 写入训练集
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Train: {len(train_data)} samples -> {train_path}")

    # 写入验证集
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Val: {len(val_data)} samples -> {val_path}")

    # 验证格式
    print("\n=== Format verification ===")
    with open(train_path) as f:
        sample = json.loads(f.readline())
    print(f"Keys: {list(sample.keys())}")
    print(f"Messages: {len(sample['messages'])} turns")
    for msg in sample["messages"]:
        print(f"  role={msg['role']}, content_len={len(msg['content'])} chars")
    print(f"  user content (first 100): {sample['messages'][0]['content'][:100]}...")
    print(f"  assistant content (first 100): {sample['messages'][1]['content'][:100]}...")


if __name__ == "__main__":
    main()
