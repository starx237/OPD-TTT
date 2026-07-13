#!/usr/bin/env python3
"""HMMT 数学竞赛评估脚本。

支持三种模型:
  - base:   原始 Qwen3.5 权重（Full Attention，无 TTT）
  - trained: 训练后 Qwen3.5-2B + TTT（可设 TTT on/off）

用法:
  python eval/scripts/eval_hmmt.py --config eval/config/qwen35_2b_base.yaml
  python eval/scripts/eval_hmmt.py --config eval/config/qwen35_2b_trained.yaml
  python eval/scripts/eval_hmmt.py --config eval/config/qwen35_9b.yaml
"""
import argparse
import gc
import glob
import json
import os
import re
import sys
import time

import torch
import yaml
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    Qwen3_5Config,
    StoppingCriteria,
    TextStreamer,
)
from transformers.cache_utils import DynamicCache

PROJECT_ROOT = "/h3c/haoxiang/TTT-OPD"
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# HMMT 试题集
# ============================================================
def _load_hmmt_problems():
    """从 eval/data/hmmt_feb_2025.json 加载 HMMT Feb 2025 真题（30 道）。"""
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "hmmt_feb_2025.json")
    with open(json_path) as f:
        problems = json.load(f)
    return [{"problem": p["problem"], "answer": p["answer"]} for p in problems]


HMMT_PROBLEMS = _load_hmmt_problems()


# ============================================================
# PresencePenalty LogitsProcessor
# ============================================================
class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """对已出现过的 token 施加 presence penalty（vLLM 语义）。

    presence_penalty: flat 惩罚，出现过就 -penalty
    repetition_penalty 由 HF 内置 RepetitionPenaltyLogitsProcessor 处理（除法惩罚）
    """

    def __init__(self, presence_penalty=1.5):
        self.penalty = presence_penalty

    def __call__(self, input_ids, scores):
        unique_tokens = input_ids[0].unique()
        scores[0, unique_tokens] -= self.penalty
        return scores


class LoopDetectionStoppingCriteria(StoppingCriteria):
    """检测模型陷入短循环（如 "Done. Done. Done."）并停止生成。

    检查最近 N 个 token 是否由一个短序列（长度 L）重复构成。
    如果最近 check_window 个 token 中，某个长度 L 的子序列重复了 threshold 次，停止。
    """

    def __init__(self, min_len=2, max_len=10, check_window=60, threshold=4):
        self.min_len = min_len
        self.max_len = max_len
        self.check_window = check_window
        self.threshold = threshold

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0].tolist()
        if len(seq) < self.check_window:
            return False

        window = seq[-self.check_window:]

        for L in range(self.min_len, self.max_len + 1):
            pattern = window[-L:]
            count = 0
            for i in range(0, len(window) - L + 1, L):
                if window[i:i + L] == pattern:
                    count += 1
                else:
                    break
            if count >= self.threshold:
                return True

        return False


# ============================================================
# TTT 推理支持
# ============================================================
class TTTDynamicCache(DynamicCache):
    """支持 TTT 跨步权重传递的 Cache。"""

    def __init__(self, ddp_cache_data=None, config=None):
        super().__init__(ddp_cache_data=ddp_cache_data, config=config)
        self.ttt_states = [(None, None, None)] * 100

    def TTT_update(self, ttt_state, layer_idx):
        self.ttt_states[layer_idx] = ttt_state


def _setup_ttt_inference(tc):
    """安装 TTT 推理 monkey-patch（MLP + DecoderLayer）。"""
    from hf_models.hf_qwen3_5.modeling_qwen3_5_opdttt import OPDQwen3_5MLP
    from hf_models.hf_qwen3_5.modeling_qwen3_5_opdttt_full import (
        OPDQwen3_5DecoderLayer,
    )

    # --- MLP: 推理模式 ---
    def inference_mlp_forward(self, x, t=None, teacher_repr=None, past_w=None, **kwargs):
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        if not self.enable_opdttt or not hasattr(self, "ttt_conv"):
            return self.down_proj(h), {}

        present_w = self.down_proj.weight if past_w is None else past_w

        if t is None:
            out = torch.nn.functional.linear(h, present_w, self.down_proj.bias)
            return out, present_w

        # 有 target: 分块处理（与训练逻辑一致）
        seq_len = x.shape[1]
        chunk_size = self.ttt_chunk

        if seq_len % chunk_size != 0:
            pad_len = chunk_size - seq_len % chunk_size
            h_padded = torch.cat(
                [h, torch.zeros(1, pad_len, h.shape[-1], device=h.device, dtype=h.dtype)],
                dim=1,
            )
            t_padded = torch.cat(
                [t, torch.zeros(1, pad_len, t.shape[-1], device=t.device, dtype=t.dtype)],
                dim=1,
            )
        else:
            h_padded = h
            t_padded = t

        num_chunks = h_padded.shape[1] // chunk_size
        outs = []

        for i in range(num_chunks):
            h_chunk = h_padded[:, i * chunk_size : (i + 1) * chunk_size]
            t_chunk = t_padded[:, i * chunk_size : (i + 1) * chunk_size]

            out_chunk = torch.nn.functional.linear(h_chunk, present_w, self.down_proj.bias)
            outs.append(out_chunk)

            h_for_update = h_chunk[0, :-1].float()
            t_input = t_chunk[0].T.unsqueeze(0)
            t_conv = self.ttt_conv(t_input).squeeze(0).T
            t_conv_for_update = t_conv[:-1].float()

            if self.ttt_proj is not None:
                dw = (
                    torch.einsum(
                        "ch,cd,de->eh",
                        h_for_update,
                        t_conv_for_update,
                        self.ttt_proj.weight.float(),
                    )
                    * self.ttt_lr
                )
            else:
                dw = torch.einsum("ch,cd->dh", h_for_update, t_conv_for_update) * self.ttt_lr

            present_w = present_w + dw.to(present_w.dtype)

        out = torch.cat(outs, dim=1)[:, :seq_len, :]
        return out, present_w

    OPDQwen3_5MLP.forward = inference_mlp_forward

    # --- DecoderLayer: TTT cache 支持 ---
    def inference_layer_forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        target_states=None,
        teacher_repr=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        elif self.layer_type == "full_attention":
            if self.sliding_window > 0:
                kwargs["sliding_window"] = self.sliding_window
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states_normed = self.post_attention_layernorm(hidden_states)

        if self.is_opdttt_layer:
            target_states = hidden_states_normed

            past_h, past_t, past_w = (
                past_key_values.ttt_states[self.layer_idx]
                if past_key_values is not None and hasattr(past_key_values, "ttt_states")
                else (None, None, None)
            )

            if past_h is None:
                present_h = hidden_states_normed
                present_t = target_states
            else:
                present_h = torch.cat([past_h, hidden_states_normed], dim=1)
                present_t = torch.cat([past_t, target_states], dim=1)

            chunk_size = self.mlp.ttt_chunk
            total_len = present_h.shape[1]
            num_complete = total_len // chunk_size
            remaining_len = total_len % chunk_size

            if num_complete > 0:
                complete_h = present_h[:, : num_complete * chunk_size]
                complete_t = present_t[:, : num_complete * chunk_size]
                complete_out, present_w = self.mlp(complete_h, t=complete_t, past_w=past_w)

                if remaining_len > 0:
                    remaining_h = present_h[:, num_complete * chunk_size :]
                    remaining_t = present_t[:, num_complete * chunk_size :]
                    remaining_out, present_w = self.mlp(remaining_h, t=None, past_w=present_w)
                    mlp_out = torch.cat([complete_out, remaining_out], dim=1)
                else:
                    remaining_h = None
                    remaining_t = None
                    mlp_out = complete_out

                if hidden_states_normed.shape[1] == 1:
                    mlp_out = mlp_out[:, -1:]
            else:
                mlp_out, present_w = self.mlp(hidden_states_normed, t=None, past_w=past_w)
                remaining_h = present_h
                remaining_t = present_t

            if past_key_values is not None and hasattr(past_key_values, "ttt_states"):
                past_key_values.TTT_update((remaining_h, remaining_t, present_w), self.layer_idx)

            hidden_states = mlp_out
        else:
            hidden_states = self.mlp(hidden_states_normed)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        hidden_states = residual + hidden_states
        return hidden_states, {}

    OPDQwen3_5DecoderLayer.forward = inference_layer_forward


# ============================================================
# 模型加载
# ============================================================
def load_base_model(cfg):
    """加载原始 Qwen3.5 权重（Full Attention，无 TTT）。"""
    model_path = cfg["model"]["model_path"]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=cfg["model"].get("attn_implementation", "flash_attention_2"),
    ).cuda().eval()
    return model


def load_trained_model(cfg):
    """加载训练后 Qwen3.5-2B + TTT 权重。"""
    from hf_models.hf_qwen3_5.modeling_qwen3_5_opdttt_full import OPDQwen3_5ForCausalLM

    model_path = cfg["model"]["model_path"]
    ckpt_path = cfg["model"]["ckpt_path"]
    ttt_cfg = cfg.get("ttt", {})
    ttt_enabled = ttt_cfg.get("enabled", True)

    base_config = Qwen3_5Config.from_pretrained(model_path)
    tc = base_config.text_config
    tc._attn_implementation = cfg["model"].get("attn_implementation", "flash_attention_2")
    tc.opdttt_mode = True
    tc.opdttt_layers = ttt_cfg.get("opdttt_layers", [0, 4, 8, 12, 16, 20])
    tc.ttt_chunk = ttt_cfg.get("ttt_chunk", 1024)
    tc.ttt_target = ttt_cfg.get("ttt_target", "hidden_states")
    tc.sliding_window = ttt_cfg.get("sliding_window", 4096)
    tc.ttt_lr = ttt_cfg.get("ttt_lr", 0.3)
    tc.ttt_max_norm = ttt_cfg.get("ttt_max_norm", 0)
    tc.lambda_ntp = 1.0
    tc.lambda_align_rep = 0.0

    model = OPDQwen3_5ForCausalLM._from_config(tc)

    sd = {}
    for f in sorted(glob.glob(f"{ckpt_path}/*.safetensors")):
        sd.update(load_file(f))
    model.load_state_dict(sd, strict=False)
    model = model.cuda().bfloat16().eval()

    if ttt_enabled:
        _setup_ttt_inference(tc)
        for layer in model.model.layers:
            if hasattr(layer.mlp, "enable_opdttt"):
                layer.mlp.enable_opdttt = True
        print(f"  TTT: ON (layers={tc.opdttt_layers}, chunk={tc.ttt_chunk}, lr={tc.ttt_lr}, sw={tc.sliding_window})")
    else:
        for layer in model.model.layers:
            if hasattr(layer.mlp, "enable_opdttt"):
                layer.mlp.enable_opdttt = False
        print(f"  TTT: OFF (消融对比)")

    return model, tc, ttt_enabled


# ============================================================
# 答案提取与判定
# ============================================================
def extract_boxed_answer(text):
    """从文本中提取最后一个 \\boxed{} 内的答案，支持嵌套大括号。"""
    result = None
    idx = 0
    while True:
        pos = text.find("\\boxed{", idx)
        if pos == -1:
            break
        # 找到匹配的右大括号
        depth = 1
        start = pos + len("\\boxed{")
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            result = text[start:i-1].strip()
        idx = i
    return result


def _normalize_latex(s):
    """标准化 LaTeX 答案以便比较。"""
    s = s.strip()
    # 去除空格
    s = s.replace(" ", "")
    # 去除 \left \right
    s = s.replace("\\left", "").replace("\\right", "")
    # 去除 \displaystyle
    s = s.replace("\\displaystyle", "")
    # 统一分数写法
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    return s


def check_answer(extracted, expected):
    """检查提取的答案是否正确，支持整数、分数、LaTeX 表达式。"""
    if extracted is None:
        return False
    extracted = extracted.strip()
    expected = expected.strip()

    # 1. 精确字符串匹配
    if _normalize_latex(extracted) == _normalize_latex(expected):
        return True

    # 2. 数值比较
    try:
        if float(extracted) == float(expected):
            return True
    except (ValueError, TypeError):
        pass

    # 3. sympy 符号比较（处理 \frac, \sqrt 等）
    try:
        import sympy
        ext_expr = sympy.sympify(extracted.replace("\\frac", "Lambda").replace("{", "(").replace("}", ")").replace("^", "**"))
        exp_expr = sympy.sympify(expected.replace("\\frac", "Lambda").replace("{", "(").replace("}", ")").replace("^", "**"))
        if sympy.simplify(ext_expr - exp_expr) == 0:
            return True
    except Exception:
        pass

    # 4. 标准化后字符串匹配
    if _normalize_latex(extracted) == _normalize_latex(expected):
        return True

    return False


# ============================================================
# 单题评估
# ============================================================
def evaluate_problem(model, tokenizer, cfg, problem_info, model_ctx=None):
    """评估单道 HMMT 试题。"""
    problem = problem_info["problem"]
    expected = problem_info["answer"]
    prompt_suffix = cfg["eval"].get("prompt_suffix", "")
    enable_thinking = cfg["eval"].get("enable_thinking", True)
    sampling = cfg["sampling"]
    max_new_tokens = sampling.get("max_new_tokens", 32768)

    messages = [{"role": "user", "content": problem + prompt_suffix}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
    )["input_ids"].cuda()

    t0 = time.time()
    logits_processor = [PresencePenaltyLogitsProcessor(
        presence_penalty=sampling.get("presence_penalty", 1.5),
    )]

    use_ttt = model_ctx is not None and model_ctx.get("ttt_enabled", False)
    if use_ttt:
        past_key_values = TTTDynamicCache(config=model_ctx["tc"])
    else:
        past_key_values = None

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    stopping_criteria = [LoopDetectionStoppingCriteria()]
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=sampling.get("temperature", 1.0),
            top_p=sampling.get("top_p", 0.95),
            top_k=sampling.get("top_k", 20),
            repetition_penalty=sampling.get("repetition_penalty", 1.0),
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            streamer=streamer,
        )

    elapsed = time.time() - t0
    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    full_output = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)

    has_think_close = "</think>" in full_output
    extracted = extract_boxed_answer(full_output)
    correct = check_answer(extracted, expected)

    return {
        "problem": problem[:80],
        "expected": expected,
        "extracted": extracted,
        "correct": correct,
        "has_think_close": has_think_close,
        "new_tokens": new_tokens,
        "elapsed": elapsed,
        "full_output": full_output,
    }


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="HMMT 评估脚本")
    parser.add_argument("--config", required=True, help="配置文件路径")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    benchmark = cfg.get("eval", {}).get("benchmark", "hmmt")
    if benchmark != "hmmt":
        raise ValueError(f"Unsupported benchmark: {benchmark} (currently only 'hmmt' is supported)")

    model_type = cfg["model"]["type"]
    model_path = cfg["model"]["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["tokenizer_path"])

    print(f"{'=' * 70}")
    print(f"  HMMT Evaluation")
    print(f"  Config: {args.config}")
    print(f"  Model type: {model_type}")
    print(f"  Model path: {model_path}")
    print(f"{'=' * 70}\n")

    # 加载模型
    print("Loading model...")
    model_ctx = None
    if model_type == "base":
        model = load_base_model(cfg)
    elif model_type == "trained":
        model, tc, ttt_enabled = load_trained_model(cfg)
        model_ctx = {"tc": tc, "ttt_enabled": ttt_enabled}
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB\n")

    # 运行评估
    results = []
    for i, problem_info in enumerate(HMMT_PROBLEMS):
        print(f"\n{'─' * 70}")
        print(f"  Problem {i + 1}/{len(HMMT_PROBLEMS)}: {problem_info['problem'][:80]}...")
        print(f"  Expected: {problem_info['answer']}")
        print(f"{'─' * 70}\n")

        result = evaluate_problem(model, tokenizer, cfg, problem_info, model_ctx)
        results.append(result)

        status = "CORRECT" if result["correct"] else "WRONG"
        think_status = "YES" if result["has_think_close"] else "NO"
        print(f"\n  Result: {status}")
        print(f"  </think>: {think_status}")
        print(f"  Extracted: {result['extracted']}")
        print(f"  Tokens: {result['new_tokens']}, Time: {result['elapsed']:.1f}s")
        print()

    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 汇总
    correct_count = sum(1 for r in results if r["correct"])
    think_close_count = sum(1 for r in results if r["has_think_close"])
    total = len(results)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Correct:        {correct_count}/{total} ({correct_count / total * 100:.0f}%)")
    print(f"  </think> close: {think_close_count}/{total}")
    print(f"  Avg tokens:     {sum(r['new_tokens'] for r in results) / total:.0f}")
    print(f"  Avg time:       {sum(r['elapsed'] for r in results) / total:.1f}s")
    print()
    print(f"  {'#':>3}  {'Expected':>10}  {'Extracted':>10}  {'Correct':>8}  {'</think>':>8}  {'Tokens':>7}  {'Time':>6}")
    print(f"  {'─' * 70}")
    for i, r in enumerate(results):
        status = "YES" if r["correct"] else "NO"
        think = "YES" if r["has_think_close"] else "NO"
        ext = r["extracted"] if r["extracted"] else "(none)"
        print(f"  {i + 1:>3}  {r['expected']:>10}  {ext:>10}  {status:>8}  {think:>8}  {r['new_tokens']:>7}  {r['elapsed']:>5.1f}s")

    # 保存完整结果到 JSON
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    # 用配置文件名作为标识，避免 TTT-OFF 和 Original+SWA 冲突
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    json_path = os.path.join(output_dir, f"hmmt_feb2025_{config_stem}_results.json")
    save_data = {
        "config": args.config,
        "config_stem": config_stem,
        "total": total,
        "correct": correct_count,
        "accuracy": correct_count / total,
        "think_close_count": think_close_count,
        "problems": [],
    }
    for i, r in enumerate(results):
        save_data["problems"].append({
            "idx": i + 1,
            "problem": r["problem"],
            "expected": r["expected"],
            "extracted": r["extracted"],
            "correct": r["correct"],
            "has_think_close": r["has_think_close"],
            "new_tokens": r["new_tokens"],
            "elapsed_s": round(r["elapsed"], 1),
            "full_output": r["full_output"],
        })
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {json_path}")


if __name__ == "__main__":
    main()
