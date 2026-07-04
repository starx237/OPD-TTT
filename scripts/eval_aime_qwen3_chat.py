#!/usr/bin/env python3
"""Qwen3-8B AIME'24 evaluation with chat template (thinking mode)"""
import os, re, json, torch
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

QWEN3_PATH = "model_assets/.cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
print("Loading Qwen3-8B...")
tok = AutoTokenizer.from_pretrained(QWEN3_PATH)
model = AutoModelForCausalLM.from_pretrained(QWEN3_PATH, torch_dtype=torch.bfloat16).cuda().eval()

# Check chat template
print(f"Has chat template: {tok.chat_template is not None}")

ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
problems = [(d["ID"], d["Problem"], d["Answer"]) for d in ds]
print(f"Loaded {len(problems)} AIME problems")

def extract_answer(text):
    # Try after </think> first
    if "</think>" in text:
        text = text.split("</think>")[-1]
    for pat in [r"[Tt]he answer is[:\s]*\$?(\d+)\$?", r"\\boxed\{(\d+)\}", r"[Aa]nswer[:\s]+\$?(\d+)\$?"]:
        m = re.search(pat, text)
        if m: return int(m.group(1))
    nums = re.findall(r"\b(\d+)\b", text)
    return int(nums[-1]) if nums else None

correct = 0
for i, (pid, problem, gt) in enumerate(problems):
    # Use chat template with thinking mode
    messages = [{"role": "user", "content": f"Solve the following math problem. Show your work step by step. At the end, write your final answer as \"The answer is: [integer]\".\n\nProblem: {problem}"}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")

    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=4096, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    pred = extract_answer(gen)
    ok = pred is not None and pred == gt
    if ok: correct += 1
    print(f"[{i+1}/30] {pid}: GT={gt} Pred={pred} {'OK' if ok else 'X'}", flush=True)

print(f"\nQwen3-8B (chat template) AIME24: {correct}/{len(problems)} = {correct/len(problems):.1%}")

# Save first sample
messages = [{"role": "user", "content": f"Solve the following math problem. Show your work step by step. At the end, write your final answer as \"The answer is: [integer]\".\n\nProblem: {problems[0][1]}"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inp = tok(text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=4096, do_sample=False, pad_token_id=tok.eos_token_id)
gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
with open("data/output/500m_stage2_opd/qwen3_8b_aime_chat_sample.txt", "w") as f:
    f.write(f"=== AIME Problem {problems[0][0]} (Answer: {problems[0][2]}) ===\n\n--- Qwen3-8B (chat) Response ---\n{gen}\n")
print("Sample saved")
