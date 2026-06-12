#!/usr/bin/env python3
"""
训练数据准备脚本

支持以下数据源:
1. WanJuanCC (万卷): ModelScope 上的中文数据集
2. PILES: HuggingFace 上的英文数据集（推荐，更适合长文本训练）

数据格式:
  - WanJuanCC: tar.gz 文件，内嵌 jsonl
  - PILES: 直接从 HuggingFace datasets 加载

特点:
  - 带断点续传
  - 边下载边处理
  - 支持长文本（32768 tokens）

用法:
    # 使用 PILES 数据集（推荐）
    python scripts/prepare_pretrain_data.py \
        --dataset piles \
        --output data/pretrain_piles.jsonl \
        --target_tokens 20000000000

    # 使用 WanJuanCC 数据集
    python scripts/prepare_pretrain_data.py \
        --dataset wanjuan \
        --output data/pretrain_wanjuan.jsonl \
        --target_tokens 20000000000

    # Dry run 测试
    python scripts/prepare_pretrain_data.py \
        --dataset piles \
        --output data/test.jsonl \
        --target_tokens 1000000 \
        --dry_run
"""

import argparse
import gzip
import json
import os
import tarfile
import time
import requests
import sys

# 设置 PILES 默认镜像（必须在导入 datasets 之前）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# WanJuanCC 相关
try:
    from modelscope.hub.api import HubApi
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False

# PILES 相关
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# WanJuanCC 配置
WANJUAN_NAMESPACE = "Shanghai_AI_Laboratory"
WANJUAN_NAME = "WanJuanCC"
TEXT_KEY = "content"
CONTENT_KEY = "content_split"
LANGUAGE_KEY = "language"
TAR_TEMP_DIR = "data/wanjuanc_tar"

# PILES 配置
# 数据集: ArmelR/the-pile-splitted (https://hf-mirror.com/datasets/ArmelR/the-pile-splitted)
PILE_SUBSETS = [
    "ArXiv",      # 学术论文，长文本
    "Books3",     # 图书，长文本
    "Wikipedia (en)",     # 维基百科
    "PubMed Central",     # 医学文献
    "Pile-CC",      # 网页
    "Github",       # 代码
]
DEFAULT_PILE_SUBSETS = ["ArXiv", "Books3", "Wikipedia (en)", "PubMed Central", "Pile-CC", "Github"]

# 子集权重配置（总和为 1.0）
# Books3 和 ArXiv 权重更高
PILE_SUBSET_WEIGHTS = {
    "ArXiv": 0.25,           # 25% - 学术论文，高质量
    "Books3": 0.25,          # 25% - 图书，长文本
    "Wikipedia (en)": 0.15,  # 15% - 维基百科
    "PubMed Central": 0.15,  # 15% - 医学文献
    "Pile-CC": 0.10,         # 10% - 网页
    "Github": 0.10,          # 10% - 代码
}
# 总权重验证
assert abs(sum(PILE_SUBSET_WEIGHTS.values()) - 1.0) < 0.01, "子集权重总和必须为 1.0"

# 长文本配置（最小 30k 字符 ≈ 7.5k tokens）
MIN_LONG_TEXT_CHARS = 30000  # 最小 30k 字符（约 7.5k tokens）
MAX_LONG_TEXT_CHARS = 131072  # 最大约 32k tokens
PACKED_TARGET_TOKENS = 32768  # 打包目标长度


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h{m:02d}m{s:02d}s"


class StateFile:
    """管理断点续传的状态文件"""
    def __init__(self, output_path: str):
        self.path = output_path + ".state.json"

    def load(self) -> dict:
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"processed": [], "total_tokens": 0, "total_lines": 0}

    def save(self, state: dict) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def mark_done(self, item: str, total_tokens: int, total_lines: int) -> None:
        state = self.load()
        if item not in state["processed"]:
            state["processed"].append(item)
        state["total_tokens"] = total_tokens
        state["total_lines"] = total_lines
        self.save(state)

    def is_done(self, item: str) -> bool:
        return item in self.load()["processed"]

    def delete(self) -> None:
        if os.path.isfile(self.path):
            os.remove(self.path)


# ========== PILES 数据集处理 ==========

def process_piles_dataset(
    output_path: str,
    target_tokens: int,
    subsets: list = None,
    max_length: int = MAX_LONG_TEXT_CHARS,
    min_length: int = MIN_LONG_TEXT_CHARS,
    dry_run: bool = False,
    mirror: str = "https://hf-mirror.com",
) -> tuple[int, int]:
    """
    处理 PILES 数据集

    Args:
        output_path: 输出文件路径
        target_tokens: 目标 token 数
        subsets: 数据子集列表
        max_length: 单条文本最大字符数
        min_length: 单条文本最小字符数
        dry_run: 是否为 dry run 模式
        mirror: HuggingFace 镜像站

    Returns:
        (处理的行数, 总 token 数)
    """
    if not DATASETS_AVAILABLE:
        print("错误: 需要安装 datasets 库")
        print("请运行: pip install datasets")
        return 0, 0

    if subsets is None:
        subsets = DEFAULT_PILE_SUBSETS

    # 设置镜像
    os.environ["HF_ENDPOINT"] = mirror

    print(f"{'=' * 60}")
    print(f"PILES 数据集处理")
    print(f"{'=' * 60}")
    print(f"子集: {', '.join(subsets)}")
    print(f"目标: {target_tokens / 1e9:.0f}B tokens")
    print(f"输出: {output_path}")
    print(f"镜像: {mirror}")
    if dry_run:
        print("🔷 DRY RUN: 仅处理少量数据")
    print()

    state = StateFile(output_path)
    loaded_state = state.load()
    total_tokens = loaded_state["total_tokens"]
    total_lines = loaded_state["total_lines"]
    processed_subsets = set(loaded_state["processed"])

    # 过滤未处理的子集
    remaining = [s for s in subsets if s not in processed_subsets]
    skipped = len(subsets) - len(remaining)

    if skipped > 0:
        print(f"📋 状态文件发现 {skipped} 个已处理子集")
        if not remaining:
            print("  ✅ 所有子集已处理完毕!")
            return total_lines, total_tokens
        print(f"  还需处理 {len(remaining)}/{len(subsets)} 个")

    if not dry_run and total_tokens >= target_tokens:
        print(f"✅ 目标 tokens ({target_tokens/1e9:.0f}B) 已达成，跳过")
        return total_lines, total_tokens

    overall_start = time.time()

    # 计算每个子集的配额
    subset_quotas = {}
    remaining_total = target_tokens - total_tokens
    if remaining_total <= 0:
        print(f"✅ 目标 tokens ({target_tokens/1e9:.0f}B) 已达成，跳过")
        return total_lines, total_tokens

    print(f"\n📊 子集配额分配 (目标: {remaining_total/1e9:.2f}B tokens):")
    for subset in subsets:
        if subset in PILE_SUBSET_WEIGHTS:
            quota = int(remaining_total * PILE_SUBSET_WEIGHTS[subset])
        else:
            # 默认权重（均分剩余部分）
            quota = int(remaining_total / len(subsets))
        subset_quotas[subset] = quota
        weight_pct = PILE_SUBSET_WEIGHTS.get(subset, 0) * 100
        print(f"  {subset:20s}: {quota/1e9:.2f}B ({weight_pct:.0f}%)")

    for subset in remaining:
        print(f"\n{'=' * 60}")
        print(f"处理子集: {subset}")
        print(f"配额: {subset_quotas.get(subset, 0)/1e9:.2f}B tokens")
        print(f"当前: {total_lines:,} 条, ~{total_tokens/1e9:.2f}B tokens")
        print(f"{'=' * 60}")

        subset_start = time.time()
        count = 0
        subset_tokens = 0
        subset_quota = subset_quotas.get(subset, remaining_total)

        try:
            # 加载数据集
            print(f"  加载数据集...")
            dataset = load_dataset(
                "ArmelR/the-pile-splitted",
                data_files=f"data/{subset}/train/*.arrow",
                split="train",
                streaming=True,
            )

            with open(output_path, "a", encoding="utf-8") as out_f:
                for item in dataset:
                    if not dry_run and subset_tokens >= subset_quota:
                        print(f"  子集配额已达 ({subset_tokens/1e9:.2f}B)，切换下一个")
                        break

                    text = item.get("text", "")
                    if not text or len(text) < min_length:
                        continue

                    text = text[:max_length]
                    record = {CONTENT_KEY: text}
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    total_lines += 1
                    text_tokens = estimate_tokens(text)
                    total_tokens += text_tokens
                    subset_tokens += text_tokens
                    count += 1

                    if total_lines % 10000 == 0:
                        elapsed = time.time() - subset_start
                        rate = total_lines / max(elapsed, 1)
                        pct = min(100, subset_tokens / subset_quota * 100) if subset_quota > 0 else 100
                        print(f"  已处理 {total_lines:,} 行, ~{total_tokens/1e9:.2f}B tokens, "
                              f"子集进度: {pct:.1f}%, {rate:.0f} 行/s", flush=True)

                    if dry_run and total_lines >= 100:
                        break

        except Exception as e:
            print(f"  警告: 处理子集 {subset} 失败: {e}")
            continue

        # 更新状态
        if not dry_run:
            state.mark_done(subset, total_tokens, total_lines)

        elapsed = time.time() - subset_start
        print(f"  子集完成: {count} 行, {subset_tokens/1e9:.2f}B tokens, {format_time(elapsed)}")

        if not dry_run and total_tokens >= target_tokens:
            print(f"\n  ✅ 目标 {target_tokens/1e9:.0f}B tokens 已达成, 停止")
            break

    overall_elapsed = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"PILES 处理完成")
    print(f"  耗时: {format_time(overall_elapsed)}")
    print(f"  条数: {total_lines:,}")
    print(f"  tokens: ~{total_tokens/1e9:.2f}B")

    return total_lines, total_tokens


# ========== WanJuanCC 数据集处理 ==========

def get_wanjuan_file_list() -> list:
    """获取 WanJuanCC 文件列表"""
    if not MODELSCOPE_AVAILABLE:
        print("错误: 需要安装 modelscope 库")
        print("请运行: pip install modelscope")
        return []

    api = HubApi()
    files = api.get_dataset_files(
        f"{WANJUAN_NAMESPACE}/{WANJUAN_NAME}",
        revision="master",
    )
    result = []
    for f in files:
        if f["Type"] != "blob" or not f["Name"].endswith(".tar.gz"):
            continue
        download_url = api.get_dataset_file_url(
            file_name=f["Path"],
            dataset_name=WANJUAN_NAME,
            namespace=WANJUAN_NAMESPACE,
            revision="master",
        )
        result.append({
            "name": f["Name"],
            "path": f["Path"],
            "size": f["Size"],
            "download_url": download_url,
        })
    result.sort(key=lambda x: x["name"])
    return result


def download_wanjuan_file(url: str, dest: str) -> None:
    """下载 WanJuanCC 文件"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    headers = {"User-Agent": "modelscope"}
    tmp_dest = dest + ".incomplete"

    if os.path.isfile(dest):
        return

    if os.path.isfile(tmp_dest):
        downloaded_size = os.path.getsize(tmp_dest)
        headers["Range"] = f"bytes={downloaded_size}-"
        mode = "ab"
    else:
        downloaded_size = 0
        mode = "wb"

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=(30, 300))
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            if "Range" in headers:
                total_size += downloaded_size

            downloaded = downloaded_size
            start_time = time.time()

            with open(tmp_dest, mode) as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            os.replace(tmp_dest, dest)
            return

        except Exception as e:
            if os.path.isfile(tmp_dest):
                downloaded_size = os.path.getsize(tmp_dest)
            print(f"  下载失败 (尝试 {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))

    if os.path.isfile(tmp_dest):
        os.remove(tmp_dest)
    raise Exception("下载失败")


def extract_wanjuan_lines(tar_path: str) -> iter:
    """从 WanJuanCC tar.gz 文件中提取行"""
    with gzip.open(tar_path, "rb") as zf:
        with tarfile.open(fileobj=zf, mode="r") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".jsonl"):
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    for line_bytes in f:
                        line = line_bytes.decode("utf-8", errors="replace").strip()
                        if line:
                            yield line


def process_wanjuan_dataset(
    output_path: str,
    target_tokens: int,
    max_length: int = 131072,
    min_length: int = 200,
    language: str = "en",
    dry_run: bool = False,
    data_dir: str = None,
) -> tuple[int, int]:
    """处理 WanJuanCC 数据集"""
    if not MODELSCOPE_AVAILABLE:
        print("错误: 需要安装 modelscope 库")
        print("请运行: pip install modelscope")
        return 0, 0

    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(output_path))
    tar_dir = os.path.join(data_dir, "wanjuanc_tar")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(tar_dir, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"WanJuanCC 数据集处理")
    print(f"{'=' * 60}")
    print(f"目标: {target_tokens / 1e9:.0f}B tokens")
    print(f"语言: {language}")
    print(f"输出: {output_path}")
    if dry_run:
        print("🔷 DRY RUN: 仅处理少量数据")
    print()

    state = StateFile(output_path)
    loaded_state = state.load()
    total_tokens = loaded_state["total_tokens"]
    total_lines = loaded_state["total_lines"]

    # 获取文件列表
    print("[1/2] 获取 WanJuanCC 文件列表...")
    file_list = get_wanjuan_file_list()
    if not file_list:
        print("  ❌ 未找到任何 tar.gz 文件!")
        return 0, 0

    total_files = len(file_list)
    remaining = [f for f in file_list if f["path"] not in set(loaded_state["processed"])]

    if not dry_run and total_tokens >= target_tokens:
        print(f"✅ 目标 tokens ({target_tokens/1e9:.0f}B) 已达成，跳过")
        return total_lines, total_tokens

    write_mode = "a" if total_lines > 0 else "w"
    print(f"[2/2] 下载并处理数据...")

    start_time = time.time()

    with open(output_path, write_mode, encoding="utf-8") as out_f:
        for idx, f_info in enumerate(remaining):
            if not dry_run and total_tokens >= target_tokens:
                break

            print(f"\n  [{idx + 1}/{len(remaining)}] {f_info['name']}")

            # 下载
            tar_local = os.path.join(tar_dir, f_info["name"])
            try:
                download_wanjuan_file(f_info["download_url"], tar_local)
            except Exception as e:
                print(f"  跳过: {e}")
                continue

            # 提取
            count = 0
            try:
                for line in extract_wanjuan_lines(tar_local):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    lang = data.get(LANGUAGE_KEY, "")
                    if lang and lang != language:
                        continue

                    text = data.get(TEXT_KEY, "")
                    if not text or len(text) < min_length:
                        continue

                    text = text[:max_length]
                    record = {CONTENT_KEY: text}
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    total_lines += 1
                    total_tokens += estimate_tokens(text)
                    count += 1

                    if dry_run and total_lines >= 10:
                        break
                    if not dry_run and total_tokens >= target_tokens:
                        break

            except Exception as e:
                print(f"  解压失败: {e}")

            # 删除压缩包
            if os.path.isfile(tar_local):
                os.remove(tar_local)

            # 更新状态
            if not dry_run:
                state.mark_done(f_info["path"], total_tokens, total_lines)

            if dry_run and total_lines >= 10:
                break

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"WanJuanCC 处理完成")
    print(f"  耗时: {format_time(elapsed)}")
    print(f"  条数: {total_lines:,}")
    print(f"  tokens: ~{total_tokens/1e9:.2f}B")

    return total_lines, total_tokens


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(
        description="准备训练数据（支持 PILES 和 WanJuanCC 数据集）"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["piles", "wanjuan"], default="piles",
        help="数据集类型（默认: piles，推荐用于长文本训练）"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="输出的 JSONL 文件路径"
    )
    parser.add_argument(
        "--target_tokens", type=int, default=20000000000,
        help="目标 token 数（默认: 20B）"
    )
    parser.add_argument(
        "--max_length", type=int, default=MAX_LONG_TEXT_CHARS,
        help=f"单条文本最大字符数（默认: {MAX_LONG_TEXT_CHARS}，约32k tokens）"
    )
    parser.add_argument(
        "--min_length", type=int, default=MIN_LONG_TEXT_CHARS,
        help=f"单条文本最小字符数（默认: {MIN_LONG_TEXT_CHARS}，约7.5k tokens）"
    )
    parser.add_argument(
        "--language", type=str, default="en",
        help="语言过滤（仅对 WanJuanCC 有效，默认: en）"
    )
    parser.add_argument(
        "--subsets", type=str, default=None,
        help=f"PILES 子集（逗号分隔，默认: {','.join(DEFAULT_PILE_SUBSETS)}）"
    )
    parser.add_argument(
        "--mirror", type=str, default="https://hf-mirror.com",
        help="HuggingFace 镜像站（仅对 PILES 有效）"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="数据存储基目录（仅对 WanJuanCC 有效）"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Dry run: 仅处理少量数据"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="忽略已有状态文件，从头开始"
    )

    args = parser.parse_args()

    # 清除状态文件
    if args.force:
        state = StateFile(args.output)
        state.delete()
        print("🔶 --force: 已清除状态文件")
        print()

    # 处理数据
    if args.dataset == "piles":
        # 解析子集
        if args.subsets:
            subsets = [s.strip() for s in args.subsets.split(",")]
        else:
            subsets = DEFAULT_PILE_SUBSETS

        process_piles_dataset(
            output_path=args.output,
            target_tokens=args.target_tokens,
            subsets=subsets,
            max_length=args.max_length,
            min_length=args.min_length,
            dry_run=args.dry_run,
            mirror=args.mirror,
        )

    elif args.dataset == "wanjuan":
        process_wanjuan_dataset(
            output_path=args.output,
            target_tokens=args.target_tokens,
            max_length=args.max_length,
            min_length=args.min_length,
            language=args.language,
            dry_run=args.dry_run,
            data_dir=args.data_dir,
        )


if __name__ == "__main__":
    main()
