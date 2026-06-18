#!/usr/bin/env python3
"""
快速生成测试：检查模型是否学到有意义的东西

用法:
    CUDA_VISIBLE_DEVICES=0 python scripts/test_checkpoint_generation.py \
        --checkpoint_path /path/to/checkpoint \
        --prompts "The meaning of life is" "Once upon a time"

支持从 DCP checkpoint 或 HuggingFace 格式加载模型权重。
"""

import argparse
import os
import sys
import torch
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from hf_models.hf_llama.modeling_llama_opdttt_full import OPDTTTForCausalLM
from veomni.checkpoint.dcp_checkpointer import dcp_to_torch_state_dict


def load_model_with_checkpoint(checkpoint_path, config_path, device="cuda"):
    """
    从 checkpoint 加载模型

    Args:
        checkpoint_path: Checkpoint 路径（DCP 或 HuggingFace 格式）
        config_path: 模型配置路径
        device: 设备

    Returns:
        model, tokenizer
    """
    print(f"加载模型配置: {config_path}")
    config = AutoConfig.from_pretrained(config_path)

    # 推理模式：完全禁用 OPD-TTT
    config.opdttt_mode = False  # 禁用OPD-TTT模式
    config.opdttt_layers = []  # 清空OPD-TTT层

    # 加载 tokenizer
    tokenizer_path = config_path.replace("_config", "") if "_config" in config_path else config_path
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "model_assets/tokenizer"

    print(f"加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # 判断 checkpoint 格式
    is_dcp = os.path.isdir(checkpoint_path) and any(
        f.startswith("meta.") or f.startswith(".metadata") or "shard" in f.lower()
        for f in os.listdir(checkpoint_path)
    ) if os.path.exists(checkpoint_path) else False

    if is_dcp:
        print(f"检测到 DCP checkpoint 格式")
        print("初始化模型...")
        model = OPDTTTForCausalLM._from_config(config)
        model = model.to(device)
        model.eval()

        print(f"加载 DCP checkpoint: {checkpoint_path}")
        try:
            state_dict = dcp_to_torch_state_dict(checkpoint_path)
            print(f"DCP checkpoint 转换成功，包含 {len(state_dict)} 个参数")

            # 加载权重
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"加载结果: 缺失 {len(missing)} 个，多余 {len(unexpected)} 个")

        except Exception as e:
            print(f"DCP checkpoint 加载失败: {e}")
            print("使用随机初始化的模型...")

    else:
        # 尝试作为 HuggingFace 格式加载
        print(f"检测到 HuggingFace checkpoint 格式")
        print(f"从 checkpoint 加载模型: {checkpoint_path}")

        try:
            model = OPDTTTForCausalLM.from_pretrained(
                checkpoint_path,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"从 checkpoint 加载失败: {e}")
            print("从配置初始化模型...")
            model = OPDTTTForCausalLM._from_config(config)
            model = model.to(device)

    model.eval()
    return model, tokenizer


def quick_test(model_path, prompts, max_new_tokens=128, temperature=0.8, device="cuda"):
    """快速生成测试"""
    print(f"\n{'='*60}")
    print("生成测试结果")
    print('='*60)
    print("使用标准的 HuggingFace generate 方法")
    print('='*60)

    for i, prompt in enumerate(prompts):
        print(f"\n{i+1}. 提示: {prompt}")
        print("-" * 60)

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        try:
            # 使用标准的 HuggingFace generate 方法
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"生成: {generated[len(prompt):]}")
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            # 如果 generate 失败，回退到手动生成
            print("回退到手动生成...")
            generated_ids = input_ids.clone()
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = model(generated_ids)
                    logits = outputs.logits[:, -1, :]
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

            generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"生成: {generated[len(prompt):]}")
        
        print("-" * 60)


def check_quality_checklist():
    """打印质量检查清单"""
    print("\n" + "="*60)
    print("生成质量检查清单")
    print("="*60)
    print("""
    ✅ 正常信号:
    - 语法基本正确
    - 内容与提示相关
    - 有一定的连贯性
    - 常见单词拼写正确

    ❌ 问题信号:
    - 重复循环 (the the the the...)
    - 完全无关的乱码
    - 切换语言 (应该是英文但出现中文等)
    - 只有标点符号
    - 空白或极短输出
    """)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="data/output/500m_stage1_pretrain/checkpoints/global_step_500/global_step_500",
        help="Checkpoint路径（支持简写：step_1000）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="model_assets/llama_500m_config",
        help="模型配置路径"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["The meaning of life is", "Once upon a time", "In recent years,", "The main reason is"],
        help="测试提示词列表"
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（可选）")

    args = parser.parse_args()

    # 支持简写格式
    checkpoint_path = args.checkpoint_path
    if checkpoint_path.startswith("step_"):
        step_num = checkpoint_path.replace("step_", "")
        checkpoint_path = f"data/output/500m_stage1_pretrain/checkpoints/global_step_{step_num}/global_step_{step_num}"

    device = args.device if torch.cuda.is_available() else "cpu"

    # 设置输出文件
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        step = checkpoint_path.split("global_step_")[-1].split("/")[0] if "global_step_" in checkpoint_path else "unknown"
        output_file = f"checkpoint_test_step_{step}_{timestamp}.txt"

    # 重定向输出到文件
    original_stdout = sys.stdout
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        
        print("=" * 70)
        print("OPD-TTT Checkpoint 生成测试")
        print("=" * 70)
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"设备: {device}")
        print(f"Checkpoint路径: {checkpoint_path}")
        print(f"配置路径: {args.config}")
        print(f"输出文件: {output_file}")
        print("=" * 70)

        # 加载模型
        global model, tokenizer
        model, tokenizer = load_model_with_checkpoint(checkpoint_path, args.config, device)

        # 执行测试
        quick_test(checkpoint_path, args.prompts, args.max_new_tokens, args.temperature, device)
        check_quality_checklist()

        print("\n" + "=" * 70)
        print("测试完成！")
        print("=" * 70)

    # 恢复标准输出
    sys.stdout = original_stdout
    
    print(f"✅ 测试完成！结果已保存到: {output_file}")
    print(f"📄 查看结果: cat {output_file}")


if __name__ == "__main__":
    main()
