#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPD-TTT 教师模型设置脚本

从 HuggingFace 下载教师模型（权重 + tokenizer）到指定目录。
支持使用镜像站加速下载。

支持的教师模型（推荐按学生模型规模选择）:
    学生500M: Qwen2.5-7B, Qwen2-14B, 1.5B预训练模型
    学生1.5B: Qwen2.5-32B, Qwen2.5-72B

使用方法:
    # 使用默认镜像站下载 Qwen2.5-7B 作为教师
    python scripts/setup_teacher.py --model qwen2.5-7b --output model_assets/teacher_qwen2.5_7b

    # 指定镜像站
    python scripts/setup_teacher.py --model qwen2.5-7b --output model_assets/teacher_qwen2.5_7b --mirror https://hf-mirror.com

    # 仅下载模型权重（不下载tokenizer，使用共享tokenizer）
    python scripts/setup_teacher.py --model qwen2.5-7b --output model_assets/teacher_qwen2.5_7b --skip-tokenizer

    # 使用自定义模型
    python scripts/setup_teacher.py --model meta-llama/Llama-3-8B-Instruct --output model_assets/teacher_llama3_8b --token YOUR_TOKEN
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 默认使用 HuggingFace 镜像站
DEFAULT_MIRROR = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = DEFAULT_MIRROR

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    from huggingface_hub import login
except ImportError:
    print("错误: 缺少依赖包。请运行: pip install transformers huggingface_hub")
    sys.exit(1)


# 预定义的推荐教师模型（按规模分组）
PREDEFINED_TEACHERS = {
    # 小型教师（用于500M学生）
    "qwen2-7b": "Qwen/Qwen2-7B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2-14b": "Qwen/Qwen2-14B",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "gemma-7b": "google/gemma-7b",
    "llama2-7b": "meta-llama/Llama-2-7b",
    "llama2-13b": "meta-llama/Llama-2-13b",

    # 中型教师（用于1.5B学生）
    "qwen2-32b": "Qwen/Qwen2-32B",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",  # 需要 token
    "llama3-70b": "meta-llama/Meta-Llama-3-70B",  # 需要 token

    # 大型教师（用于大模型学生）
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
}


def get_model_size(model_path: str) -> float:
    """
    估算模型参数量（单位：十亿）
    根据配置文件或权重文件估算
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # 估算参数量
        num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 32))
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'n_embd', 4096))
        vocab_size = getattr(config, 'vocab_size', getattr(config, 'padded_vocab_size', 32000))

        # 基础参数（embeddings）
        params = vocab_size * hidden_size

        # 每层的参数（attention + mlp）
        # 粗略估算：约 12 * hidden_size^2
        params += num_layers * 12 * hidden_size * hidden_size

        return params / 1e9  # 转换为十亿
    except:
        return 0


def setup_teacher_model(
    model_name: str,
    output_dir: str,
    token: str = None,
    mirror: str = None,
    skip_tokenizer: bool = False,
    skip_model: bool = False,
    only_config: bool = False,
    log_file: str = None
):
    """
    从 HuggingFace 下载教师模型到指定目录

    Args:
        model_name: HuggingFace 模型名称或预定义别名
        output_dir: 输出目录
        token: HuggingFace 访问令牌（用于 gated 模型）
        mirror: HuggingFace 镜像站 URL
        skip_tokenizer: 跳过 tokenizer 下载
        skip_model: 跳过模型权重下载（仅下载配置）
        only_config: 仅下载配置文件
        log_file: 日志文件路径
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 设置日志文件
    if log_file is None:
        log_file = output_path / "download.log"

    # 配置日志处理器
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 添加文件处理器到根日志记录器
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    logging.info(f"=== 教师模型下载开始: {model_name} ===")
    logging.info(f"输出目录: {output_path}")
    logging.info(f"日志文件: {log_file}")

    # 设置镜像站
    if mirror:
        os.environ["HF_ENDPOINT"] = mirror
        print(f"使用镜像站: {mirror}")
        logging.info(f"使用镜像站: {mirror}")

    # 解析模型名称
    model_key = model_name.lower().replace("-instruct", "").replace("-chat", "")
    if model_key in PREDEFINED_TEACHERS:
        hf_model_name = PREDEFINED_TEACHERS[model_key]
        print(f"使用预定义教师: {model_name} -> {hf_model_name}")
    else:
        hf_model_name = model_name
        print(f"使用教师模型: {hf_model_name}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_path}")

    # 设置 HuggingFace token（如果提供）
    if token:
        login(token=token)
        print("已使用提供的 HuggingFace token 登录")

    try:
        # 下载配置文件（总是需要）
        print(f"\n正在下载配置文件从 {hf_model_name}...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            hf_model_name,
            token=token,
            trust_remote_code=True
        )
        config.save_pretrained(output_path)
        print(f"✓ 配置文件已保存")

        # 下载 tokenizer
        if not skip_tokenizer:
            print(f"\n正在下载 tokenizer 从 {hf_model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(
                hf_model_name,
                token=token,
                trust_remote_code=True
            )

            print(f"Tokenizer 信息:")
            print(f"  - 类型: {type(tokenizer).__name__}")
            print(f"  - 词汇表大小: {len(tokenizer)}")

            tokenizer.save_pretrained(output_path)
            print(f"✓ Tokenizer 已保存")
        else:
            print("\n跳过 tokenizer 下载（将使用共享 tokenizer）")

        # 下载模型权重
        if not skip_model and not only_config:
            print(f"\n正在下载模型权重从 {hf_model_name}...")
            print("这可能需要一段时间，请耐心等待...")

            # 使用 snapshot_download 下载所有文件
            downloaded_path = snapshot_download(
                repo_id=hf_model_name,
                local_dir=output_path,
                local_dir_use_symlinks=False,
                token=token,
                resume_download=True,
            )

            # 计算模型大小
            print(f"\n✓ 模型已下载到: {downloaded_path}")

            # 估算参数量和文件大小
            param_size = get_model_size(downloaded_path)
            total_size = sum(
                f.stat().st_size for f in Path(downloaded_path).rglob("*.safetensors")
            ) + sum(
                f.stat().st_size for f in Path(downloaded_path).rglob("*.bin")
            )

            print(f"\n教师模型信息:")
            print(f"  - 参数量: ~{param_size:.1f}B")
            print(f"  - 文件大小: {total_size / (1024**3):.2f} GB")

            # 检查 vocab_size 一致性
            if not skip_tokenizer:
                model_vocab_size = getattr(config, 'vocab_size', getattr(config, 'padded_vocab_size', None))
                tokenizer_vocab_size = len(tokenizer)
                if model_vocab_size and abs(model_vocab_size - tokenizer_vocab_size) > 1000:
                    print(f"\n⚠ 警告: 模型 vocab_size ({model_vocab_size}) 与 tokenizer ({tokenizer_vocab_size}) 不匹配")
                    print(f"  这可能导致训练时出现问题")

        print(f"\n✓ 教师模型设置完成!")
        print(f"\n配置文件: {output_path}/config.json")
        if not skip_tokenizer:
            print(f"Tokenizer: {output_path}/tokenizer.json")
        if not skip_model and not only_config:
            print(f"模型权重: {output_path}/")

        print(f"\n在训练配置中设置:")
        print(f"  opdttt:")
        print(f"    teacher_model_path: \"{output_path}\"")

        logging.info("=== 教师模型下载完成 ===")

        # 关闭日志处理器
        file_handler.close()
        root_logger.removeHandler(file_handler)

        return True

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        print("\n提示:")
        print("  1. 对于 gated 模型（如 LLaMA3），需要 HuggingFace token")
        print("     获取 token: https://huggingface.co/settings/tokens")
        print("     并使用 --token 参数或在 ~/.huggingface/token 中设置")
        print("  2. 推荐使用公开模型，如 Qwen2, Qwen2.5, Mistral, Gemma")
        print("  3. 确保模型名称正确")
        print("  4. 检查网络连接")
        print("  5. 如果磁盘空间不足，可以使用 --only-config 先下载配置")

        logging.error(f"下载失败: {e}")
        logging.info("=== 教师模型下载失败 ===")

        # 关闭日志处理器
        file_handler.close()
        root_logger.removeHandler(file_handler)

        return False


def list_recommended_teachers():
    """打印推荐的教师模型列表"""
    print("\n推荐的教师模型选择:\n")
    print("┌" + "─" * 70 + "┐")
    print("│" + " " * 20 + "按学生模型规模选择" + " " * 27 + "│")
    print("├" + "─" * 70 + "┤")
    print("│ {:<15} │ {:<20} │ {:<20} │".format("学生模型", "推荐教师", "教师模型路径"))
    print("├" + "─" * 70 + "┤")

    recommendations = [
        ("500M", "Qwen2.5-7B (14x)", "qwen2.5-7b"),
        ("500M", "Qwen2-14B (28x)", "qwen2-14b"),
        ("500M", "Llama2-13B (26x)", "llama2-13b"),
        ("1.5B", "Qwen2.5-32B (21x)", "qwen2.5-32b"),
        ("1.5B", "Qwen2.5-72B (48x)", "qwen2.5-72b"),
    ]

    for student, teacher, alias in recommendations:
        print("│ {:<15} │ {:<20} │ {:<20} │".format(student, teacher, alias))

    print("└" + "─" * 70 + "┘")
    print("\n说明: 括号内为师生参数比例，一般建议 3x-30x")


def main():
    parser = argparse.ArgumentParser(
        description="从 HuggingFace 下载教师模型到指定目录（支持镜像站）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
预定义教师模型别名:
    小型教师 (500M学生):
      qwen2-7b         Qwen/Qwen2-7B
      qwen2.5-7b      Qwen/Qwen2.5-7B (推荐)
      qwen2-14b       Qwen/Qwen2-14B
      qwen2.5-14b     Qwen/Qwen2.5-14B
      mistral-7b      mistralai/Mistral-7B-v0.1
      gemma-7b        google/gemma-7b
      llama2-7b       meta-llama/Llama-2-7b
      llama2-13b      meta-llama/Llama-2-13b

    中型教师 (1.5B学生):
      qwen2-32b       Qwen/Qwen2-32B
      qwen2.5-32b     Qwen/Qwen2.5-32B (推荐)
      llama3-8b       meta-llama/Meta-Llama-3-8B (需要 token)

    大型教师:
      qwen2.5-72b     Qwen/Qwen2.5-72B-Instruct
      llama3-70b      meta-llama/Meta-Llama-3-70B (需要 token)

示例:
    # 下载 Qwen2.5-7B 作为教师（推荐）
    python scripts/setup_teacher.py --model qwen2.5-7b --output model_assets/teacher_qwen2.5_7b

    # 查看推荐配置
    python scripts/setup_teacher.py --list-recommended

    # 仅下载配置（用于测试）
    python scripts/setup_teacher.py --model qwen2.5-7b --output model_assets/teacher_qwen2.5_7b --only-config

    # 不下载 tokenizer（使用共享 tokenizer）
    python scripts/setup_teacher.py --model qwen2.5-7b --output model_assets/teacher_qwen2.5_7b --skip-tokenizer

    # 指定镜像站
    python scripts/setup_teacher.py --model qwen2.5-7b --output model_assets/teacher_qwen2.5_7b --mirror https://hf-mirror.com

    # 使用 HuggingFace 官方源
    python scripts/setup_teacher.py --model qwen2.5-7b --output model_assets/teacher_qwen2.5_7b --mirror https://huggingface.co

    # 使用 token（用于 gated 模型）
    python scripts/setup_teacher.py --model llama3-8b --output model_assets/teacher_llama3_8b --token YOUR_TOKEN

训练配置:
    在训练配置中添加:
      opdttt:
        teacher_model_path: "model_assets/teacher_qwen2.5_7b"
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace 模型名称或预定义别名"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录路径（默认: model_assets/teacher_<model>）"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace 访问令牌（用于 gated 模型）"
    )
    parser.add_argument(
        "--mirror",
        type=str,
        default=DEFAULT_MIRROR,
        help=f"HuggingFace 镜像站（默认: {DEFAULT_MIRROR}）"
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="跳过 tokenizer 下载（使用共享 tokenizer）"
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="跳过模型权重下载（仅下载配置和 tokenizer）"
    )
    parser.add_argument(
        "--only-config",
        action="store_true",
        help="仅下载配置文件"
    )
    parser.add_argument(
        "--list-recommended",
        action="store_true",
        help="列出推荐的教师模型选择"
    )

    args = parser.parse_args()

    # 列出推荐配置
    if args.list_recommended:
        list_recommended_teachers()
        return 0

    # 验证必需参数
    if not args.model:
        parser.error("--model 是必需参数（除非使用 --list-recommended）")

    # 设置默认输出目录
    if not args.output:
        model_alias = args.model.lower().replace("/", "_").replace("-", "_")
        args.output = f"model_assets/teacher_{model_alias}"

    # 执行设置
    success = setup_teacher_model(
        model_name=args.model,
        output_dir=args.output,
        token=args.token,
        mirror=args.mirror,
        skip_tokenizer=args.skip_tokenizer,
        skip_model=args.skip_model,
        only_config=args.only_config
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
