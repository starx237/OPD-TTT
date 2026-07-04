#!/usr/bin/env python3
"""
OPD-TTT 模型聊天面板
支持: 500M Pretrain, 500M SFT-3000, Qwen3-8B (base/chat)

用法:
    source env_setup.sh
    CUDA_VISIBLE_DEVICES=0 python scripts/chat_ui.py --port 7860

然后在VS Code中做端口转发，在本地浏览器打开 http://localhost:7860
"""

import os
import sys
import json
import torch
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ---------- Model configs ----------
MODELS = {
    "500M Pretrain (step 9500)": {
        "path": "data/output/500m_stage1_pretrain/checkpoints/global_step_9500/hf_ckpt",
        "tokenizer": "model_assets/tokenizer",
        "type": "opdttt",
        "use_chat_template": False,
        "max_new_tokens": 512,
        "description": "阶段1预训练模型，从零训练20B tokens，基础语言能力",
    },
    "500M SFT-3000": {
        "path": "data/output/500m_stage2_sft/checkpoints/global_step_3000/hf_ckpt",
        "tokenizer": "model_assets/tokenizer",
        "type": "opdttt",
        "use_chat_template": False,
        "max_new_tokens": 512,
        "description": "阶段2 SFT模型，在预训练基础上做了指令微调",
    },
    "Qwen3-8B (chat)": {
        "path": "model_assets/.cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        "tokenizer": None,
        "type": "qwen3",
        "use_chat_template": True,
        "max_new_tokens": 2048,
        "description": "Qwen3-8B指令模型，支持thinking模式",
    },
    "Qwen3-0.6B (base)": {
        "path": "model_assets/.cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
        "tokenizer": None,
        "type": "qwen3",
        "use_chat_template": False,
        "max_new_tokens": 512,
        "description": "Qwen3-0.6B基础模型（预训练，非指令微调），TTT训练的起点",
    },
}

loaded_models = {}


def load_model(model_name, disable_ttt=False):
    """加载指定模型"""
    cache_key = f"{model_name}__ttt_{'off' if disable_ttt else 'on'}"
    if cache_key in loaded_models:
        return loaded_models[cache_key]

    cfg = MODELS[model_name]
    print(f"Loading {model_name} (TTT {'OFF' if disable_ttt else 'ON'}) from {cfg['path']}...")

    tok_path = cfg["tokenizer"] or cfg["path"]

    if cfg["type"] == "opdttt":
        import hf_models.hf_llama
        from hf_models.hf_llama import OPDTTTForCausalLM

        config = AutoConfig.from_pretrained("model_assets/llama_500m_config")
        if disable_ttt:
            config.opdttt_mode = False
            config.opdttt_layers = []
        else:
            config.opdttt_mode = True
            config.opdttt_layers = [0, 6, 12, 18]
        config.lambda_kl = 0.0; config.lambda_lm = 0.0; config.lambda_ntp = 1.0
        config.lambda_align_rep = 0.0
        config.ttt_lr = 3; config.ttt_chunk = 1024; config.ttt_proj = True
        config.ttt_max_norm = 1e-5; config.ttt_target = "input_embed"
        config.weight_adaptation = "fixed"; config.teacher_proj_init = "random"
        config.teacher_hidden_size = config.hidden_size

        model = OPDTTTForCausalLM.from_pretrained(
            cfg["path"], config=config, torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
        ).cuda().eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16,
        ).cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loaded_models[cache_key] = (model, tokenizer)
    print(f"  Loaded {model_name}")
    return model, tokenizer


def generate_response(model_name, message, history, temperature, max_tokens, disable_ttt):
    """生成回复"""
    cfg = MODELS[model_name]
    model, tokenizer = load_model(model_name, disable_ttt=disable_ttt)

    if cfg["use_chat_template"]:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for h in history:
            messages.append(h)
        messages.append({"role": "user", "content": message})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    else:
        prompt_parts = []
        for h in history:
            prompt_parts.append(f"User: {h['content']}")
        prompt_parts.append(f"User: {message}\nAssistant:")
        text = "\n\n".join(prompt_parts)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")

    max_new = max_tokens or cfg["max_new_tokens"]
    do_sample = temperature > 0.01

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=do_sample,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    if cfg["use_chat_template"] and "</think>" in gen:
        gen = gen.split("</think>")[-1].strip()

    return gen


def chat(message, history, model_choice, temperature, max_tokens, disable_ttt):
    if not message.strip():
        return "", history
    try:
        response = generate_response(model_choice, message, history, temperature, max_tokens, disable_ttt)
    except Exception as e:
        response = f"[Error] {str(e)}"
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    with gr.Blocks(title="OPD-TTT Chat") as demo:
        gr.Markdown("# OPD-TTT 模型聊天面板\n选择不同模型进行对话，直观对比效果。")

        with gr.Row():
            model_choice = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=list(MODELS.keys())[0],
                label="选择模型",
                info="切换模型（首次切换需等待加载）",
            )
            disable_ttt = gr.Checkbox(
                label="禁用TTT（推荐勾选）",
                value=True,
                info="TTT用非因果padding训练，推理时需禁用才能正常生成",
            )
            temperature = gr.Slider(0, 1.5, value=0.7, step=0.1, label="Temperature")
            max_tokens = gr.Slider(64, 4096, value=512, step=64, label="Max New Tokens")

        model_desc = gr.Markdown(f"**{MODELS[list(MODELS.keys())[0]]['description']}**")

        def update_desc(model_name):
            return f"**{MODELS[model_name]['description']}**"

        model_choice.change(update_desc, inputs=[model_choice], outputs=[model_desc])

        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(placeholder="输入消息...", label="消息")

        with gr.Row():
            send = gr.Button("发送", variant="primary")
            clear = gr.Button("清空对话")

        msg.submit(chat, [msg, chatbot, model_choice, temperature, max_tokens, disable_ttt], [msg, chatbot])
        send.click(chat, [msg, chatbot, model_choice, temperature, max_tokens, disable_ttt], [msg, chatbot])
        clear.click(lambda: [], None, chatbot, queue=False)

    demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
