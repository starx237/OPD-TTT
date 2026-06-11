#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPD-TTT 安装验证脚本

本脚本验证 OPD-TTT 的所有组件是否正确安装并能正常导入。
"""

import sys
import traceback


def test_imports():
    """
    测试所有必需的模块是否能正确导入

    Returns:
        list: 测试结果列表，每项为 (名称, 是否成功, 消息)
    """
    tests = []

    # 测试 1: PyTorch
    try:
        import torch
        assert torch.cuda.is_available(), "CUDA 不可用"
        tests.append(("PyTorch with CUDA", True, f"PyTorch {torch.__version__}, {torch.cuda.device_count()} 个 GPU"))
    except Exception as e:
        tests.append(("PyTorch with CUDA", False, str(e)))

    # 测试 2: Transformers
    try:
        import transformers
        tests.append(("Transformers", True, f"Transformers {transformers.__version__}"))
    except Exception as e:
        tests.append(("Transformers", False, str(e)))

    # 测试 3: Flash Attention
    try:
        from flash_attn import flash_attn_func
        tests.append(("Flash Attention", True, "OK"))
    except Exception as e:
        tests.append(("Flash Attention", False, str(e)))

    # 测试 4: VeOmni
    try:
        from veomni.models import build_tokenizer
        from veomni.checkpoint import build_checkpointer
        tests.append(("VeOmni", True, "OK"))
    except Exception as e:
        tests.append(("VeOmni", False, str(e)))

    # 测试 5: OPDTTT 模型
    try:
        from hf_models.hf_llama import (
            OPDTTTForCausalLM,
            OPDTTTModel,
            OPDTTTDecoderLayer,
            OPDTTTMLP,
            OPDTTTLoss,
        )
        tests.append(("OPDTTT 模型", True, "OK"))
    except Exception as e:
        tests.append(("OPDTTT 模型", False, str(e)))

    # 测试 6: AutoConfig
    try:
        from transformers import AutoConfig
        tests.append(("AutoConfig", True, "OK"))
    except Exception as e:
        tests.append(("AutoConfig", False, str(e)))

    # 测试 7: Einops
    try:
        import einops
        tests.append(("Einops", True, f"Einops {einops.__version__}"))
    except Exception as e:
        tests.append(("Einops", False, str(e)))

    # 测试 8: WandB（可选）
    try:
        import wandb
        tests.append(("WandB", True, f"WandB {wandb.__version__}"))
    except Exception as e:
        tests.append(("WandB", False, str(e)))

    return tests


def print_results(tests):
    """
    打印测试结果

    Args:
        tests: 测试结果列表
    """
    print("\n" + "=" * 80)
    print("OPD-TTT 安装验证结果")
    print("=" * 80)

    for name, success, message in tests:
        status = "✓ 通过" if success else "✗ 失败"
        color = "\033[92m" if success else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {name:30s} : {message}")

    print("=" * 80)

    passed = sum(1 for _, success, _ in tests if success)
    total = len(tests)
    print(f"\n总结: {passed}/{total} 个测试通过")

    if passed == total:
        print("\n✓ 所有测试通过！OPD-TTT 已准备就绪。")
        return 0
    else:
        print("\n✗ 部分测试失败。请检查上述错误。")
        return 1


def test_model_creation():
    """
    测试 OPDTTT 模型是否能正确实例化

    Returns:
        bool: 测试是否成功
    """
    print("\n" + "=" * 80)
    print("测试模型创建")
    print("=" * 80)

    try:
        from transformers import AutoConfig
        from hf_models.hf_llama import OPDTTTForCausalLM

        # 创建用于测试的最小化配置
        config = AutoConfig.for_model("llama")
        config.hidden_size = 128
        config.intermediate_size = 512
        config.num_hidden_layers = 4
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.vocab_size = 32000
        config.rms_norm_eps = 1e-6
        config.mlp_bias = False
        config.attention_bias = False

        # OPDTTT 设置
        config.opdttt_mode = True
        config.opdttt_layers = [0, 2]
        config.ttt_lr = 0.3
        config.ttt_chunk = 512
        config.ttt_proj = True
        config.ttt_target = "input_embed"
        config.lambda_kl = 0.1
        config.lambda_lm = 1.0
        config.lambda_ntp = 1.0
        config.lambda_align_rep = 0.5

        print("创建测试配置的 OPDTTT 模型...")
        model = OPDTTTForCausalLM(config)

        print(f"✓ 模型创建成功！")
        print(f"  - 总参数量: {sum(p.numel() for p in model.model.parameters()):,}")
        print(f"  - OPDTTT 层: {config.opdttt_layers}")
        print(f"  - TTT 分块大小: {config.ttt_chunk}")

        return True

    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        traceback.print_exc()
        return False


def test_forward_pass():
    """
    测试前向传播是否正常工作

    Returns:
        bool: 测试是否成功
    """
    print("\n" + "=" * 80)
    print("测试前向传播")
    print("=" * 80)

    try:
        import torch
        from transformers import AutoConfig
        from hf_models.hf_llama import OPDTTTForCausalLM

        # 创建模型
        config = AutoConfig.for_model("llama")
        config.hidden_size = 128
        config.intermediate_size = 512
        config.num_hidden_layers = 4
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.vocab_size = 32000
        config.rms_norm_eps = 1e-6
        config.mlp_bias = False
        config.attention_bias = False
        config.opdttt_mode = True
        config.opdttt_layers = [0, 2]
        config.ttt_lr = 0.3
        config.ttt_chunk = 512
        config.ttt_proj = True
        config.ttt_target = "input_embed"
        config.lambda_kl = 0.1
        config.lambda_lm = 1.0
        config.lambda_ntp = 1.0
        config.lambda_align_rep = 0.5

        model = OPDTTTForCausalLM(config)
        model.eval()

        # 测试前向传播
        batch_size = 2
        seq_len = 256
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        print(f"运行前向传播，输入形状: {input_ids.shape}...")

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        print(f"✓ 前向传播成功！")
        print(f"  - 损失: {outputs.loss.item():.4f}")
        print(f"  - Logits 形状: {outputs.logits.shape}")

        return True

    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        traceback.print_exc()
        return False


def main():
    """
    运行所有测试

    Returns:
        int: 退出代码（0 表示成功，1 表示失败）
    """
    print("OPD-TTT 安装验证")
    print("本脚本检查 OPD-TTT 是否正确安装。\n")

    # 运行导入测试
    tests = test_imports()
    result = print_results(tests)

    # 如果所有导入通过，测试模型创建和前向传播
    if result == 0:
        if test_model_creation() and test_forward_pass():
            print("\n" + "=" * 80)
            print("✓ OPD-TTT 功能完全正常！")
            print("=" * 80)
            return 0
        else:
            print("\n" + "=" * 80)
            print("✗ OPD-TTT 存在需要修复的问题。")
            print("=" * 80)
            return 1

    return result


if __name__ == "__main__":
    sys.exit(main())
