#!/usr/bin/env python3
"""Qwen3-8B AIME'24 evaluation"""
import os, re, json, torch
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

QWEN3_PATH = "model_assets/.cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
print("Loading Qwen3-8B for AIME...")
tok = AutoTokenizer.from_pretrained(QWEN3_PATH)
model = AutoModelForCausalLM.from_pretrained(QWEN3_PATH, torch_dtype=torch.bfloat16).cuda().eval()

ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
problems = [(d["ID"], d["Problem"], d["Answer"]) for d in ds]
print(f"Loaded {len(problems)} AIME problems")

PROMPT = 'Solve the following math problem. Show your work step by step. At the end, write your final answer as "The answer is: [integer]".\n\nProblem: {problem}\n\nSolution:'

def extract_answer(text):
    for pat in [r"[Tt]he answer is[:\s]*\$?(\d+)\$?", r"\\boxed\{(\d+)\}", r"[Aa]nswer[:\s]+\$?(\d+)\$?"]:
        m = re.search(pat, text)
        if m: return int(m.group(1))
    nums = re.findall(r"\b(\d+)\b", text)
    return int(nums[-1]) if nums else None

correct = 0
for i, (pid, problem, gt) in enumerate(problems):
    prompt = PROMPT.format(problem=problem)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=1024, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    pred = extract_answer(gen)
    ok = pred is not None and pred == gt
    if ok: correct += 1
    print(f"[{i+1}/30] {pid}: GT={gt} Pred={pred} {'OK' if ok else 'X'}", flush=True)

print(f"\nQwen3-8B AIME24: {correct}/{len(problems)} = {correct/len(problems):.1%}")
