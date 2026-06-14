#!/usr/bin/env python3
"""
简单的训练测试脚本
用于验证配置和依赖是否正确
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试必要的导入"""
    print("测试导入...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU数量: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"✗ PyTorch导入失败: {e}")
        return False

    try:
        from transformers import AutoTokenizer
        print("✓ Transformers")
    except ImportError as e:
        print(f"✗ Transformers导入失败: {e}")
        return False

    try:
        import flash_attn
        print("✓ Flash Attention")
    except ImportError as e:
        print(f"✗ Flash Attention导入失败: {e}")
        return False

    return True

def test_tokenizer():
    """测试tokenizer加载"""
    print("\n测试tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("model_assets/tokenizer")
        print(f"✓ Tokenizer加载成功")
        print(f"  Vocab size: {len(tokenizer)}")
        return True
    except Exception as e:
        print(f"✗ Tokenizer加载失败: {e}")
        return False

def test_config():
    """测试配置加载"""
    print("\n测试配置加载...")
    try:
        from veomni.config import load_config
        config = load_config("configs/opdttt/llama3_sc_500m_stage1_pretrain.yaml")
        print(f"✓ 配置加载成功")
        print(f"  模型路径: {config.model.get('model_path', 'N/A')}")
        print(f"  数据路径: {config.data.get('train_path', 'N/A')}")
        print(f"  最大步数: {config.train.get('max_steps', 'N/A')}")
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data():
    """测试数据文件"""
    print("\n测试数据文件...")
    import os
    data_path = "data/piles_packed_32768.jsonl"
    if os.path.exists(data_path):
        print(f"✓ 数据文件存在: {data_path}")
        # 统计行数
        with open(data_path, 'r') as f:
            count = sum(1 for _ in f)
        print(f"  数据行数: {count}")
        return True
    else:
        print(f"✗ 数据文件不存在: {data_path}")
        return False

def test_model_imports():
    """测试模型导入"""
    print("\n测试模型导入...")
    try:
        from hf_models.hf_llama.modeling_llama_opdttt import OPDTTTMLP
        print("✓ OPDTTTMLP导入成功")

        from hf_models.hf_llama.modeling_llama_opdttt_full import OPDTTTForCausalLM
        print("✓ OPDTTTForCausalLM导入成功")

        from tasks.train_opdttt import main
        print("✓ train_opdttt导入成功")

        return True
    except Exception as e:
        print(f"✗ 模型导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("OPD-TTT 训练前检查")
    print("=" * 60)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_tokenizer()
    all_passed &= test_config()
    all_passed &= test_data()
    all_passed &= test_model_imports()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有检查通过！准备开始训练。")
        print("=" * 60)
        return 0
    else:
        print("✗ 部分检查失败！请修复错误后重试。")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
