#!/usr/bin/env python3
"""RULER 评估脚本：支持 TTT on/off、batch 推理、断点续训。

功能：
  - 实现 RULER 核心 task（niah_single, niah_multikey, niah_multivalue,
    niah_multiquery, vt, cwe, fwe, qa）
  - TTT-on：prefill 全序列 forward 提取每层适配后的 down_proj 权重，
            生成阶段用 bmm 应用 per-batch 适配权重
  - TTT-off：标准 generate
  - 断点续训：每个 task×length 完成后保存 JSON，重启时跳过已完成项

使用方法：
    python scripts/eval_ruler.py \
        --model_path data/output/qwen35_2b_ttt/hf_step2000 \
        --tokenizer_path model_assets/qwen3.5-2b \
        --output_dir results/ruler/step2000 \
        --lengths 2048,4096,8192,16384,32768 \
        --num_samples 100 \
        --ttt_mode on \
        --gpu 1

创建时间：2026-07-09
"""
import argparse
import json
import os
import random
import string
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

os.environ["MODELING_BACKEND"] = "hf"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")

import hf_models.hf_qwen3_5  # noqa: F401
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ====================== RULER Task Generators ======================

_FILLER = "The quick brown fox jumps over the lazy dog. "


def _build_context(tokenizer, seq_len, needles, question, seed=0):
    """构造 haystack：filler 文本 + needles 分散插入 + 末尾 question。"""
    filler_tokens = tokenizer(_FILLER, return_tensors="pt")["input_ids"][0]
    question_tokens = tokenizer(question, return_tensors="pt")["input_ids"][0]
    needle_token_lists = [tokenizer(n, return_tensors="pt")["input_ids"][0] for n in needles]
    all_needle_tokens = torch.cat(needle_token_lists)
    target_ctx = seq_len - len(all_needle_tokens) - len(question_tokens)
    if target_ctx < 0:
        target_ctx = 0
    repeats = target_ctx // len(filler_tokens) + 1
    ctx = filler_tokens.repeat(repeats)[:target_ctx]
    # 分散插入 needles
    num_gaps = len(needle_token_lists) + 1
    chunk = len(ctx) // num_gaps if num_gaps > 0 else len(ctx)
    parts = [ctx[:chunk]]
    for i, nt in enumerate(needle_token_lists):
        start = chunk * (i + 1)
        end = chunk * (i + 2) if i < len(needle_token_lists) - 1 else len(ctx)
        parts.append(nt)
        parts.append(ctx[start:end])
    full = torch.cat(parts + [question_tokens])
    return full[:seq_len]


def gen_niah_single(n, seq_len, tok):
    samples = []
    for i in range(n):
        needle = "The magic number is 43827."
        q = "\n\nWhat is the magic number? Answer with only the number."
        full = _build_context(tok, seq_len, [needle], q, seed=i)
        samples.append({"input_ids": full.tolist(), "answer": "43827", "task": "niah_single"})
    return samples


def gen_niah_multikey(n, seq_len, tok):
    pairs = [("passcode", "7294"), ("secret phrase", "blue moon rising"),
             ("identification code", "XQ-8842"), ("access key", "frosted-glass-window")]
    samples = []
    for i in range(n):
        ki = i % len(pairs)
        key, val = pairs[ki]
        needles = [f"The {k} is {v}." for k, v in pairs]
        q = f"\n\nWhat is the {key}? Answer with only the value."
        full = _build_context(tok, seq_len, needles, q, seed=i)
        samples.append({"input_ids": full.tolist(), "answer": val, "task": "niah_multikey"})
    return samples


def gen_niah_multivalue(n, seq_len, tok):
    vals = ["100", "200", "300", "400", "500"]
    samples = []
    for i in range(n):
        needles = [f"The count is {v}." for v in vals]
        q = "\n\nWhat is the count? Answer with only the number."
        full = _build_context(tok, seq_len, needles, q, seed=i)
        samples.append({"input_ids": full.tolist(), "answer": vals[-1], "task": "niah_multivalue"})
    return samples


def gen_niah_multiquery(n, seq_len, tok):
    needles = ["The magic number is 43827.", "The secret code is alpha-7.", "The passcode is 9921."]
    q = "\n\nWhat are the magic number, secret code, and passcode?"
    samples = []
    for i in range(n):
        full = _build_context(tok, seq_len, needles, q, seed=i)
        samples.append({"input_ids": full.tolist(), "answer": "43827, alpha-7, 9921", "task": "niah_multiquery"})
    return samples


def gen_vt(n, seq_len, tok):
    samples = []
    for i in range(n):
        vals = [str(j) for j in range(1, 20)]
        needles = [f"var = {v}." for v in vals]
        q = "\n\nWhat is the final value of var? Answer with only the value."
        full = _build_context(tok, seq_len, needles, q, seed=i)
        samples.append({"input_ids": full.tolist(), "answer": vals[-1], "task": "vt"})
    return samples


def gen_cwe(n, seq_len, tok):
    words = ["time", "people", "water", "world", "music"]
    samples = []
    for i in range(n):
        target = words[i % len(words)]
        parts = [target] * 10
        for w in words:
            if w != target:
                parts.extend([w] * 3)
        rng = random.Random(i)
        rng.shuffle(parts)
        core = " ".join(parts)
        q = "\n\nWhat word appears most frequently in the above text? Answer with only the word."
        core_tokens = tok(core, return_tensors="pt")["input_ids"][0]
        q_tokens = tok(q, return_tensors="pt")["input_ids"][0]
        repeats = (seq_len - len(q_tokens)) // len(core_tokens) + 1
        full = torch.cat([core_tokens.repeat(repeats)[:seq_len - len(q_tokens)], q_tokens])
        samples.append({"input_ids": full.tolist(), "answer": target, "task": "cwe"})
    return samples


def gen_fwe(n, seq_len, tok):
    samples = []
    for i in range(n):
        targets = ["apple", "banana", "cherry"]
        fillers = ["the", "sky", "runs", "fast", "blue", "tree"]
        parts = []
        for w in targets:
            parts.extend([w] * 5)
        parts.extend(fillers * 10)
        rng = random.Random(i)
        rng.shuffle(parts)
        core = " ".join(parts)
        q = "\n\nList all words that appear 5 or more times in the above text."
        core_tokens = tok(core, return_tensors="pt")["input_ids"][0]
        q_tokens = tok(q, return_tensors="pt")["input_ids"][0]
        repeats = (seq_len - len(q_tokens)) // len(core_tokens) + 1
        full = torch.cat([core_tokens.repeat(repeats)[:seq_len - len(q_tokens)], q_tokens])
        samples.append({"input_ids": full.tolist(), "answer": ", ".join(targets), "task": "fwe"})
    return samples


def gen_qa(n, seq_len, tok):
    docs = [
        ("Paris is the capital of France.", "What is the capital of France?", "Paris"),
        ("Mount Everest is the tallest mountain in the world.", "What is the tallest mountain in the world?", "Mount Everest"),
        ("Water boils at 100 degrees Celsius at sea level.", "At what temperature does water boil at sea level?", "100"),
    ]
    samples = []
    for i in range(n):
        di = i % len(docs)
        doc_text, q, ans = docs[di]
        needles = [f"Document {j}: {dt}" for j, (dt, _, _) in enumerate(docs)]
        q = f"\n\n{q} Answer briefly."
        full = _build_context(tok, seq_len, needles, q, seed=i)
        samples.append({"input_ids": full.tolist(), "answer": ans, "task": "qa"})
    return samples


TASK_GENERATORS = {
    "niah_single": gen_niah_single,
    "niah_multikey": gen_niah_multikey,
    "niah_multivalue": gen_niah_multivalue,
    "niah_multiquery": gen_niah_multiquery,
    "vt": gen_vt,
    "cwe": gen_cwe,
    "fwe": gen_fwe,
    "qa": gen_qa,
}


# ====================== Scoring ======================

def score_answer(pred: str, gold: str) -> float:
    pred = pred.strip().lower()
    gold = gold.strip().lower()
    if not pred:
        return 0.0
    if gold in pred:
        return 1.0
    # 多值答案（逗号分隔）：检查所有 gold 词是否都出现在 pred 中
    gold_words = [w.strip() for w in gold.replace(",", " ").split()]
    if len(gold_words) > 1:
        pred_words = set(pred.replace(",", " ").split())
        if all(w in pred_words for w in gold_words):
            return 1.0
        # 部分匹配
        matched = sum(1 for w in gold_words if w in pred_words)
        return matched / len(gold_words)
    # 单值模糊匹配
    gold_words = set(gold.split())
    pred_words = set(pred.split())
    if gold_words and len(gold_words & pred_words) / len(gold_words) >= 0.5:
        return 1.0
    return 0.0


# ====================== TTT Patching ======================

def extract_ttt_adapted_weights(model, input_ids):
    """Prefill forward 提取每层 TTT 适配后的 down_proj 权重。

    修改 TTT MLP forward 临时捕获 d_down_proj_sum[:, -1]，
    forward 结束后恢复。返回 {layer_idx: [bs, d, h]}。
    """
    from einops import repeat

    captured = {}
    patched_layers = []

    for layer in model.model.layers:
        mlp = layer.mlp
        if not getattr(mlp, "enable_opdttt", False):
            continue
        idx = mlp.layer_idx
        orig_forward = mlp.forward
        patched_layers.append((mlp, orig_forward))

        def make_capturing_fwd(mlp_obj, layer_idx, original):
            def capturing_fwd(x, t=None, teacher_repr=None):
                result = original(x, t, teacher_repr)
                # 重新计算适配权重
                if t is not None and hasattr(mlp_obj, "ttt_conv"):
                    h = mlp_obj.act_fn(mlp_obj.gate_proj(x)) * mlp_obj.up_proj(x)
                    t_padded = mlp_obj.padding(t)
                    h_padded = mlp_obj.padding(h)
                    bs, cn, cs, _ = t_padded.shape
                    nt = (mlp_obj.ttt_conv(t_padded.transpose(-1, -2).reshape(bs * cn, -1, cs))
                          .transpose(-1, -2).reshape(bs, cn, cs, -1))
                    dt = h_padded.dtype
                    hf = h_padded[:, :-1].float() if dt == torch.bfloat16 else h_padded[:, :-1]
                    nf = nt[:, :-1].float() if dt == torch.bfloat16 else nt[:, :-1]
                    wf = mlp_obj.ttt_proj.weight.float() if mlp_obj.ttt_proj.weight.dtype == torch.bfloat16 else mlp_obj.ttt_proj.weight
                    ntp_proj = torch.einsum("b t c h, b t c d, d e -> b t e h", hf, nf, wf).to(dt)
                    wu = ntp_proj * mlp_obj.lambda_ntp * mlp_obj.ttt_lr
                    ddp = torch.cat([
                        repeat(mlp_obj.down_proj.weight, "d h -> b 1 d h", b=bs), wu
                    ], dim=1)
                    ddp_sum = ddp.cumsum(dim=1)
                    captured[layer_idx] = ddp_sum[:, -1].detach()
                return result
            return capturing_fwd

        mlp.forward = make_capturing_fwd(mlp, idx, orig_forward)

    # 执行 prefill
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)

    # 恢复原始 forward
    for mlp, orig_fwd in patched_layers:
        mlp.forward = orig_fwd

    return captured


def apply_ttt_weights(model, adapted_weights):
    """替换 TTT 层的 down_proj 为 per-batch bmm。"""
    originals = {}
    for layer_idx, w in adapted_weights.items():
        mlp = model.model.layers[layer_idx].mlp
        originals[layer_idx] = mlp.forward
        w_ref = w  # 闭包捕获

        def make_fwd(mlp_obj, adapted_w):
            def fwd(x, t=None, teacher_repr=None):
                h = mlp_obj.act_fn(mlp_obj.gate_proj(x)) * mlp_obj.up_proj(x)
                bs = h.shape[0]
                out = torch.bmm(h, adapted_w[:bs].transpose(-1, -2))
                if mlp_obj.down_proj.bias is not None:
                    out = out + mlp_obj.down_proj.bias
                return out, {}
            return fwd

        mlp.forward = make_fwd(mlp, w_ref)
    return originals


def restore_ttt_weights(model, originals):
    for layer_idx, orig_fwd in originals.items():
        model.model.layers[layer_idx].mlp.forward = orig_fwd


# ====================== Model Loading ======================

def load_model(model_path, tokenizer_path, ttt_enabled, gpu_id):
    from hf_models.hf_qwen3_5 import OPDQwen3_5ForCausalLM
    from safetensors.torch import load_file
    import glob as _glob

    config = AutoConfig.from_pretrained(model_path)
    tc = config.text_config if hasattr(config, "text_config") else config

    ttt_layers = [0, 4, 8, 12, 16, 20]
    tc.opdttt_mode = True
    tc.opdttt_layers = ttt_layers
    tc.ttt_mode = True
    tc.ttt_layers = ttt_layers
    tc.ttt_target = "hidden_states"
    tc.ttt_lr = 3.0
    tc.ttt_chunk = 1024
    tc.ttt_proj = True
    tc.ttt_max_norm = 0
    tc.lambda_ntp = 1.0
    tc.lambda_lm = 1.0
    tc.lambda_kl = 0.0
    tc.lambda_align_rep = 0.0

    # 与训练一致：_from_config 创建模型 + 手动加载 safetensors 权重
    # from_pretrained 对自定义 OPD-TTT 模块处理不正确
    model = OPDQwen3_5ForCausalLM._from_config(
        config, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    sd = model.state_dict()
    loaded = 0
    for sf in sorted(_glob.glob(os.path.join(model_path, "*.safetensors"))):
        ckpt_sd = load_file(sf)
        for k, v in ckpt_sd.items():
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v.to(sd[k].dtype)
                loaded += 1
    model.load_state_dict(sd, strict=False)
    print(f"已加载 {loaded}/{len(sd)} 个权重", flush=True)

    model = model.to(f"cuda:{gpu_id}")
    model.eval()

    for layer in model.model.layers:
        if hasattr(layer.mlp, "enable_opdttt"):
            layer.mlp.enable_opdttt = ttt_enabled

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


# ====================== Main ======================

def run_eval(args):
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, f"results_{args.ttt_mode}.json")
    completed = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                item = json.loads(line)
                completed.add((item["task"], item["length"]))
        print(f"断点续训: 已完成 {len(completed)} 个 task×length")

    model, tokenizer = load_model(args.model_path, args.tokenizer_path, args.ttt_mode == "on", args.gpu)
    lengths = [int(x) for x in args.lengths.split(",")]
    tasks = args.tasks.split(",") if args.tasks else list(TASK_GENERATORS.keys())

    results_fp = open(results_file, "a")

    for length in lengths:
        for task_name in tasks:
            if (task_name, length) in completed:
                print(f"跳过已完成: {task_name} @ {length}")
                continue

            print(f"\n{'='*60}")
            print(f"Task: {task_name} | Length: {length} | TTT: {args.ttt_mode}")
            print(f"{'='*60}")

            samples = TASK_GENERATORS[task_name](args.num_samples, length, tokenizer)
            correct = 0
            t_start = time.time()

            for s_idx, sample in enumerate(samples):
                input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long).to(f"cuda:{args.gpu}")
                gold = sample["answer"]

                try:
                    if args.ttt_mode == "on":
                        # TTT-on: prefill forward 提取适配权重，然后用适配权重生成
                        adapted = extract_ttt_adapted_weights(model, input_ids)
                        originals = apply_ttt_weights(model, adapted)
                        try:
                            with torch.no_grad():
                                out = model.generate(
                                    input_ids=input_ids, max_new_tokens=args.max_new_tokens,
                                    do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                )
                        finally:
                            restore_ttt_weights(model, originals)
                        torch.cuda.empty_cache()
                    else:
                        with torch.no_grad():
                            out = model.generate(
                                input_ids=input_ids, max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                            )
                    pred = tokenizer.decode(out[0, input_ids.shape[-1]:], skip_special_tokens=True)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    pred = ""

                sc = score_answer(pred, gold)
                correct += sc

                if (s_idx + 1) % 10 == 0 or s_idx == 0:
                    elapsed = time.time() - t_start
                    print(f"  [{s_idx+1}/{args.num_samples}] acc={correct/(s_idx+1):.3f} "
                          f"| pred='{pred[:50]}' gold='{gold}' | {elapsed:.0f}s")

            acc = correct / args.num_samples
            elapsed = time.time() - t_start
            result = {"task": task_name, "length": length, "ttt_mode": args.ttt_mode,
                      "accuracy": acc, "total": args.num_samples, "elapsed_s": elapsed}
            results_fp.write(json.dumps(result) + "\n")
            results_fp.flush()
            print(f"  => {task_name} @ {length} ({args.ttt_mode}): acc={acc:.4f} ({correct}/{args.num_samples}) {elapsed:.0f}s")

    results_fp.close()
    print(f"\n完成! 结果: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RULER 评估脚本")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lengths", default="2048,4096,8192,16384,32768")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--ttt_mode", choices=["on", "off"], default="on")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.gpu = 0
    run_eval(args)
