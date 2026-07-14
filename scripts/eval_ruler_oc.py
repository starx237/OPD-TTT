#!/usr/bin/env python3
"""OpenCompass RULER evaluation entry point for TTT models.

This wrapper:
1. Registers TTT inference model classes with AutoModel
2. Monkey-patches HuggingFaceBaseModel to load via _from_config + load_state_dict
   (matching the training model's loading path, avoiding from_pretrained issues)

Think mode handling:
  RULER prompts are completion-style (e.g. niah ends with "...mentioned in the
  provided text are"). Using chat template would wrap the prompt as a user
  message, causing the model to generate verbose conversational responses
  instead of directly continuing the sentence. Instead, we pass raw prompts
  directly — the model stays in completion mode and does not generate <think>
  tags. The extract_non_reasoning_content postprocessor (configured in
  models.py) serves as a safety net: if think tags ever appear, it extracts
  the content after </think>; otherwise it returns the raw text unchanged.

Usage:
    python scripts/eval_ruler_oc.py eval_config/ruler_4k.py [--reuse] [--debug] [other opencompass args]

Created: 2026-07-09
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
for extra in ["VeOmni", "hf_models"]:
    p = os.path.join(project_root, extra)
    if os.path.isdir(p):
        sys.path.insert(0, p)

import opencompass_compat  # noqa: F401

import inference_model  # noqa: F401
from inference_model.hf_qwen3_5.configuration_qwen3_5 import Qwen3_5TTTConfig
from inference_model.hf_qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TTTModel,
    Qwen3_5TTTForCausalLM,
)
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from safetensors import safe_open
import torch
import glob

AutoConfig.register("qwen3_5_opdttt", Qwen3_5TTTConfig, exist_ok=True)
AutoModel.register(Qwen3_5TTTConfig, Qwen3_5TTTModel, exist_ok=True)
AutoModelForCausalLM.register(Qwen3_5TTTConfig, Qwen3_5TTTForCausalLM, exist_ok=True)


# ============================================================================
# Patch 1: _load_model — use _from_config + load_state_dict (not from_pretrained)
# ============================================================================
def _patched_load_model(self, path, kwargs, peft_path=None, peft_kwargs=dict()):
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)

    dtype_str = kwargs.get("torch_dtype", kwargs.get("dtype", "bfloat16"))
    if isinstance(dtype_str, str):
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(dtype_str, torch.bfloat16)
    else:
        dtype = dtype_str

    model = Qwen3_5TTTForCausalLM._from_config(config)

    sd = {}
    for sf in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(sf, framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # teacher_proj.* 是训练模型有但推理模型不需要的参数，允许 unexpected
    real_unexpected = [k for k in unexpected if "teacher_proj" not in k]
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint (model expects but checkpoint lacks):\n{missing}")
    if real_unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint (non-teacher_proj):\n{real_unexpected}")
    if unexpected:
        self.logger.info(f"Ignoring expected unexpected keys (teacher_proj): {[k for k in unexpected if 'teacher_proj' in k]}")
    model = model.to("cuda:0", dtype=dtype)
    model.eval()
    model.generation_config.do_sample = False

    self.model = model
    self.logger.info(f"Loaded model via _from_config + load_state_dict from {path}")


# ============================================================================
# Patch 2: generate — wrap prompts in chat template with enable_thinking=False
# ============================================================================
# OpenCompass's HuggingFaceBaseModel.generate (miniconda3 version) tokenizes
# raw strings directly via batch_encode_plus, bypassing apply_chat_template
# entirely. This leaves the model in completion mode, where Qwen3.5 generates
# <think> tags. This patch wraps each prompt in chat template with
# enable_thinking=False before passing to the original generate, ensuring
# think mode is properly disabled.
_original_generate = None

def _patched_generate(self, inputs, max_out_len, min_out_len=None,
                      stopping_criteria=[], **kwargs):
    from opencompass.models.huggingface_above_v4_33 import _convert_base_messages
    messages = _convert_base_messages(inputs)
    formatted = []
    for msg in messages:
        chat_msgs = [{"role": "user", "content": msg}]
        chat_prompt = self.tokenizer.apply_chat_template(
            chat_msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        formatted.append(chat_prompt)
    return _original_generate(self, formatted, max_out_len, min_out_len,
                              stopping_criteria, **kwargs)


# Apply patches
from opencompass.models.huggingface_above_v4_33 import HuggingFaceBaseModel
HuggingFaceBaseModel._load_model = _patched_load_model
_original_generate = HuggingFaceBaseModel.generate
HuggingFaceBaseModel.generate = _patched_generate

from opencompass.cli.main import main

if __name__ == "__main__":
    main()
