#!/usr/bin/env python3
"""共享模型加载工具：支持 SFT (HF checkpoint) 和 OPD (DCP checkpoint) 两种加载方式"""

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoConfig, AutoTokenizer
import hf_models.hf_llama  # noqa: F401
from hf_models.hf_llama import OPDTTTForCausalLM


SFT_HF_CKPT = "data/output/500m_stage2_sft/checkpoints/global_step_3000/hf_ckpt"
CONFIG_PATH = "model_assets/llama_500m_config"
TOKENIZER_PATH = "model_assets/tokenizer"
TEACHER_CONFIG_PATH = "model_assets/teacher_qwen2.5_7b"

OPD_DCP_CKPT = "data/output/500m_stage2_opd/checkpoints/global_step_150/global_step_150"


def load_opdttt_model(ckpt_path=None, device="cuda"):
    """
    加载 OPD-TTT 模型。

    Args:
        ckpt_path: DCP checkpoint 路径。None 表示加载 SFT baseline。
        device: 加载设备。

    Returns:
        (model, tokenizer)
    """
    config = AutoConfig.from_pretrained(CONFIG_PATH)
    config.opdttt_mode = True
    config.opdttt_layers = [0, 6, 12, 18]
    config.lambda_kl = 0.5
    config.lambda_lm = 0.0
    config.lambda_ntp = 1.0
    config.lambda_align_rep = 0.0
    config.ttt_lr = 3
    config.ttt_chunk = 1024
    config.ttt_proj = True
    config.ttt_max_norm = 1e-5
    config.ttt_target = "input_embed"
    config.weight_adaptation = "fixed"
    config.teacher_proj_init = "random"

    try:
        tc = AutoConfig.from_pretrained(TEACHER_CONFIG_PATH)
        config.teacher_hidden_size = tc.hidden_size
    except Exception:
        config.teacher_hidden_size = config.hidden_size

    model = OPDTTTForCausalLM.from_pretrained(
        SFT_HF_CKPT,
        config=config,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )

    if ckpt_path:
        from veomni.checkpoint import ckpt_to_state_dict
        state_dict = ckpt_to_state_dict(
            save_checkpoint_path=ckpt_path,
            ckpt_manager="dcp",
        )
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    return model, tokenizer
