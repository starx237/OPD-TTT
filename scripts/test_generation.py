#!/usr/bin/env python3
"""从DCP checkpoint加载模型并生成文本测试"""

import os
import sys
import json
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from transformers import AutoConfig, AutoTokenizer
from veomni.checkpoint import ckpt_to_state_dict
import hf_models.hf_llama  # noqa: F401
from hf_models.hf_llama import OPDTTTForCausalLM


def main():
    ckpt_path = "data/output/500m_stage2_opd/checkpoints/global_step_150/global_step_150"
    config_path = "model_assets/llama_500m_config"
    tokenizer_path = "model_assets/tokenizer"

    # Load config
    config = AutoConfig.from_pretrained(config_path)
    config.opdttt_mode = True
    config.opdttt_layers = [0, 6, 12, 18]
    config.lambda_kl = 0.0
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

    # Set teacher_hidden_size from teacher config
    try:
        tc = AutoConfig.from_pretrained("model_assets/teacher_qwen2.5_7b")
        config.teacher_hidden_size = tc.hidden_size
    except Exception:
        config.teacher_hidden_size = config.hidden_size

    # Build model from config
    model = OPDTTTForCausalLM.from_pretrained(
        "data/output/500m_stage2_sft/checkpoints/global_step_3000/hf_ckpt",
        config=config,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )

    # Load DCP checkpoint
    print("Loading DCP checkpoint...")
    state_dict = ckpt_to_state_dict(
        save_checkpoint_path=ckpt_path,
        ckpt_manager="dcp",
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    model = model.cuda()
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Select test prompts
    test_prompts = []
    with open("data/opd_prompts_train.jsonl") as f:
        for i, line in enumerate(f):
            if i in [0, 1, 5, 6, 9]:
                test_prompts.append(json.loads(line)["prompt"])
            if len(test_prompts) >= 5:
                break

    # Generate
    output_file = "data/output/500m_stage2_opd/generation_step150.txt"
    with open(output_file, "w") as out:
        out.write("=" * 80 + "\n")
        out.write("OPD-TTT Generation Test - Step 150 Checkpoint\n")
        out.write("=" * 80 + "\n\n")

        for idx, prompt in enumerate(test_prompts):
            out.write(f"--- Prompt {idx + 1} ---\n")
            out.write(f"{prompt}\n\n")

            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            out.write(f"--- Generated ---\n")
            out.write(f"{generated}\n\n")
            out.write("=" * 80 + "\n\n")
            print(f"Prompt {idx + 1} done ({len(generated)} chars)")

    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
