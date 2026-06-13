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
import fcntl
import gzip
import json
import logging
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
    print("[准备中] datasets 库已加载...", flush=True)
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
    """管理断点续传的状态文件

    特性:
      - 使用 fcntl.flock 实现线程安全
      - 支持版本号和自动迁移
      - 原子写入 (tmp + rename)
      - 向后兼容 processed/processed_subsets 键
    """

    STATE_VERSION = 2  # 当前状态文件版本号

    def __init__(self, output_path: str):
        self.path = output_path + ".state.json"
        self._lock_fd = None

    def _lock_file(self, f):
        """获取文件锁 (线程安全)"""
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, BlockingIOError):
            # 非阻塞锁失败，尝试阻塞锁
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def _unlock_file(self, f):
        """释放文件锁"""
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass  # 忽略解锁失败

    def _get_default_state(self) -> dict:
        """获取默认状态结构"""
        return {
            "version": self.STATE_VERSION,
            "timestamp": time.time(),
            "processed_subsets": {},  # {subset: {"files": [file1, file2], "tokens": X, "lines": Y}}
            "processed": [],  # 兼容旧版本 (WanJuanCC)
            "current_subset": None,   # 当前正在处理的子集
            "total_tokens": 0,
            "total_lines": 0
        }

    def _migrate_state(self, state: dict) -> dict:
        """迁移旧版本状态到新版本"""
        # 迁移 processed -> processed_subsets (用于 WanJuanCC 兼容)
        if "processed" in state and isinstance(state["processed"], list):
            if "processed_subsets" not in state:
                state["processed_subsets"] = {}
            # 将旧的 processed 列表迁移到 wanjuan 子集
            if "wanjuan" not in state["processed_subsets"]:
                state["processed_subsets"]["wanjuan"] = {"files": [], "tokens": 0, "lines": 0}
            for file_path in state["processed"]:
                if file_path not in state["processed_subsets"]["wanjuan"]["files"]:
                    state["processed_subsets"]["wanjuan"]["files"].append(file_path)

        # 确保必要字段存在
        state.setdefault("version", self.STATE_VERSION)
        state.setdefault("timestamp", time.time())
        state.setdefault("processed_subsets", {})
        state.setdefault("processed", [])
        state.setdefault("current_subset", None)
        state.setdefault("total_tokens", 0)
        state.setdefault("total_lines", 0)

        # 更新版本号和时间戳
        state["version"] = self.STATE_VERSION
        state["timestamp"] = time.time()

        return state

    def load(self) -> dict:
        """加载状态文件，带错误日志和向后兼容"""
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r") as f:
                    self._lock_file(f)
                    try:
                        state = json.load(f)
                        return self._migrate_state(state)
                    finally:
                        self._unlock_file(f)
            except json.JSONDecodeError as e:
                logging.error(f"状态文件 JSON 解析失败: {e}")
                # 备份损坏的文件
                backup_path = self.path + ".corrupt"
                try:
                    os.replace(self.path, backup_path)
                    logging.info(f"已备份损坏的状态文件到: {backup_path}")
                except OSError:
                    pass
            except OSError as e:
                logging.error(f"读取状态文件失败: {e}")
            except Exception as e:
                logging.error(f"加载状态文件时发生未知错误: {e}")

        return self._get_default_state()

    def save(self, state: dict) -> None:
        """保存状态文件（原子写入 + 线程安全）"""
        # 更新时间戳
        state["timestamp"] = time.time()
        state["version"] = self.STATE_VERSION

        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                self._lock_file(f)
                try:
                    json.dump(state, f, ensure_ascii=False, indent=2)
                    f.flush()  # 确保写入磁盘
                    os.fsync(f.fileno())
                finally:
                    self._unlock_file(f)
            os.replace(tmp, self.path)
        except Exception as e:
            logging.error(f"保存状态文件失败: {e}")
            # 清理临时文件
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            raise

    def mark_file_done(self, subset: str, file_name: str, total_tokens: int, total_lines: int) -> None:
        """标记某个子集的文件已处理"""
        state = self.load()
        if subset not in state["processed_subsets"]:
            state["processed_subsets"][subset] = {"files": [], "tokens": 0, "lines": 0}
        if file_name not in state["processed_subsets"][subset]["files"]:
            state["processed_subsets"][subset]["files"].append(file_name)
        state["processed_subsets"][subset]["tokens"] = total_tokens
        state["processed_subsets"][subset]["lines"] = total_lines
        state["total_tokens"] = total_tokens
        state["total_lines"] = total_lines
        self.save(state)

    def is_file_done(self, subset: str, file_name: str) -> bool:
        """检查某个文件是否已处理"""
        state = self.load()
        if subset not in state["processed_subsets"]:
            return False
        return file_name in state["processed_subsets"][subset].get("files", [])

    def get_processed_files(self, subset: str) -> list:
        """获取某子集已处理的文件列表"""
        state = self.load()
        if subset not in state["processed_subsets"]:
            return []
        return state["processed_subsets"][subset].get("files", [])

    def set_current_subset(self, subset: str, total_tokens: int, total_lines: int) -> None:
        """设置当前正在处理的子集"""
        state = self.load()
        state["current_subset"] = subset
        state["total_tokens"] = total_tokens
        state["total_lines"] = total_lines
        self.save(state)

    def get_current_subset(self) -> str:
        """获取当前正在处理的子集"""
        state = self.load()
        return state.get("current_subset")

    def mark_subset_done(self, subset: str, total_tokens: int, total_lines: int) -> None:
        """标记子集完成（兼容旧接口）"""
        self.mark_file_done(subset, "__COMPLETE__", total_tokens, total_lines)

    def is_subset_done(self, subset: str) -> bool:
        """检查子集是否完成"""
        return self.is_file_done(subset, "__COMPLETE__")

    def get_subset_tokens(self, subset: str) -> int:
        """获取某子集已处理的 token 数"""
        state = self.load()
        if subset not in state["processed_subsets"]:
            return 0
        return state["processed_subsets"][subset].get("tokens", 0)

    def update_subset_tokens(self, subset: str, tokens: int) -> None:
        """更新某子集的 token 数"""
        state = self.load()
        if subset not in state["processed_subsets"]:
            state["processed_subsets"][subset] = {"files": [], "tokens": 0, "lines": 0}
        state["processed_subsets"][subset]["tokens"] = tokens
        self.save(state)

    def delete(self) -> None:
        """删除状态文件"""
        if os.path.isfile(self.path):
            try:
                os.remove(self.path)
            except OSError as e:
                logging.error(f"删除状态文件失败: {e}")
                raise

    # ========== WanJuanCC 兼容方法 ==========

    def mark_done(self, file_path: str, total_tokens: int, total_lines: int) -> None:
        """标记文件完成 (WanJuanCC 兼容接口)"""
        state = self.load()
        if file_path not in state["processed"]:
            state["processed"].append(file_path)
        # 同步到 processed_subsets
        if "wanjuan" not in state["processed_subsets"]:
            state["processed_subsets"]["wanjuan"] = {"files": [], "tokens": 0, "lines": 0}
        if file_path not in state["processed_subsets"]["wanjuan"]["files"]:
            state["processed_subsets"]["wanjuan"]["files"].append(file_path)
        state["total_tokens"] = total_tokens
        state["total_lines"] = total_lines
        self.save(state)


# ========== PILES 数据集处理 ==========

def configure_download_stability():
    """配置下载稳定性参数"""
    import socket
    import warnings

    # 设置更长的超时时间（秒）
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")  # 10分钟
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # 禁用 hf-transfer（不稳定）
    os.environ.setdefault("DATASETS_DOWNLOAD_TIMEOUT", "600")  # 数据集下载超时
    os.environ.setdefault("HF_DATASETS_OFFLINE", "0")  # 确保在线模式
    os.environ.setdefault("HF_HUB_RETRY_MAX_RETRIES", "10")  # 更多重试
    os.environ.setdefault("HF_HUB_RETRY_HTTP_TIMEOUT", "60")  # HTTP 超时

    # 设置全局 socket timeout（这会影响所有网络请求）
    socket.setdefaulttimeout(300)  # 5分钟

    # 抑制一些警告
    warnings.filterwarnings("ignore", category=UserWarning, module="datasets")

def process_piles_dataset(
    output_path: str,
    target_tokens: int,
    subsets: list = None,
    max_length: int = MAX_LONG_TEXT_CHARS,
    min_length: int = MIN_LONG_TEXT_CHARS,
    dry_run: bool = False,
    mirror: str = "https://hf-mirror.com",
) -> tuple[int, int, bool]:
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
        (处理的行数, 总 token 数, 是否完成)
    """
    if not DATASETS_AVAILABLE:
        print("错误: 需要安装 datasets 库")
        print("请运行: pip install datasets")
        return 0, 0, False

    if subsets is None:
        subsets = DEFAULT_PILE_SUBSETS

    # 设置镜像
    os.environ["HF_ENDPOINT"] = mirror

    print(f"\n[启动] 初始化数据处理...", flush=True)

    print(f"{'=' * 60}", flush=True)
    print(f"PILES 数据集处理", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"子集: {', '.join(subsets)}", flush=True)
    print(f"目标: {target_tokens / 1e9:.0f}B tokens", flush=True)
    print(f"输出: {output_path}", flush=True)
    print(f"镜像: {mirror}", flush=True)
    if dry_run:
        print("🔷 DRY RUN: 仅处理少量数据", flush=True)
    print(flush=True)

    # 配置下载稳定性
    configure_download_stability()

    state = StateFile(output_path)
    loaded_state = state.load()
    total_tokens = loaded_state["total_tokens"]
    total_lines = loaded_state["total_lines"]
    processed_data = loaded_state.get("processed_subsets", {})

    if not dry_run and total_tokens >= target_tokens:
        print(f"✅ 目标 tokens ({target_tokens/1e9:.0f}B) 已达成，跳过")
        return total_lines, total_tokens, True

    overall_start = time.time()

    # 获取每个子集的文件列表并过滤已处理的文件
    import glob
    subset_files = {}  # {subset: [file1, file2, ...]}
    all_files_count = 0
    skipped_files_count = 0

    # 使用数据集加载模式（不需要文件列表）
    # 直接通过 subset 名称加载数据
    # 映射 subset 名称到数据集路径
    subset_data_dirs = {
        "ArXiv": "data/ArXiv/train",
        "Books3": "data/Books3/train",
        "Wikipedia (en)": "data/Wikipedia (en)/train",
        "PubMed Central": "data/PubMed Central/train",
        "Pile-CC": "data/Pile-CC/train",
        "Github": "data/Github/train",
    }

    # 检查是否有已处理的文件状态
    print("  检查状态文件...")
    current_subset = state.get_current_subset()
    if current_subset:
        print(f"  📋 上次处理到子集: {current_subset}")

    # 按子集轮流处理（直接从数据集加载）
    print(f"\n🔄 开始处理子集...")

    last_subset = None
    start_from_current = False

    for subset in subsets:
        # 检查是否从当前子集开始
        if current_subset:
            if subset == current_subset:
                start_from_current = True
            elif not start_from_current:
                print(f"  跳过已完成子集: {subset}")
                continue

        # 检查是否达到目标
        if not dry_run and total_tokens >= target_tokens:
            print(f"\n✅ 目标 {target_tokens/1e9:.0f}B tokens 已达成")
            break

        # 获取该子集的配额
        subset_quota = int(target_tokens * PILE_SUBSET_WEIGHTS.get(subset, 0))
        subset_tokens_processed = state.get_subset_tokens(subset)

        # 如果该子集已完成，跳过
        if subset_tokens_processed >= subset_quota:
            print(f"  ✅ {subset} 已完成 ({subset_tokens_processed/1e9:.2f}B / {subset_quota/1e9:.2f}B)")
            continue

        # 更新状态
        if subset != last_subset:
            if last_subset is not None:
                print(f"  ✅ 子集 {last_subset} 处理完成")
            state.set_current_subset(subset, total_tokens, total_lines)
            last_subset = subset

        print(f"\n📦 处理子集: {subset}")
        print(f"  目标: {subset_quota/1e9:.2f}B tokens")
        print(f"  已处理: {subset_tokens_processed/1e9:.2f}B tokens")
        print(f"  进度: {total_tokens/1e9:.2f}B / {target_tokens/1e9:.0f}B")

        # 加载该子集的数据
        max_retries = 3
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            if retry_count > 0:
                wait_time = min(30, 2 ** retry_count)
                print(f"  等待 {wait_time} 秒后重试 (第 {retry_count}/{max_retries})...", flush=True)
                time.sleep(wait_time)

            try:
                # 下载稳定性配置
                configure_download_stability()

                import datasets as ds
                ds.config.DOWNLOAD_IN_THREAD = False
                ds.config.STREAMING_READ_MAX_RETRIES = 10
                ds.config.STREAMING_READ_TIMEOUT = 300

                download_config = ds.DownloadConfig(max_retries=5)

                # 使用镜像站加载数据集子集（流式模式，按需下载）
                data_dir = subset_data_dirs.get(subset, f"data/{subset}/train")
                dataset = load_dataset(
                    "ArmelR/the-pile-splitted",
                    data_dir=data_dir,
                    split="train",
                    streaming=True,  # 流式模式：逐文件下载，达到配额后停止
                    download_config=download_config,
                )

                # 处理数据
                subset_count = 0
                subset_tokens = 0
                with open(output_path, "a", encoding="utf-8") as out_f:
                    for item in dataset:
                        # 检查是否达到目标
                        if not dry_run and total_tokens >= target_tokens:
                            break
                        # 检查是否达到子集配额
                        if not dry_run and subset_tokens_processed + subset_tokens >= subset_quota:
                            break

                        text = item.get("text", "")
                        if not text or len(text) < min_length:
                            continue
                        if len(text) > max_length:
                            text = text[:max_length]

                        record = {CONTENT_KEY: text}
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()

                        total_lines += 1
                        text_tokens = estimate_tokens(text)
                        total_tokens += text_tokens
                        subset_tokens += text_tokens
                        subset_count += 1

                        # 每1000条保存一次状态
                        if subset_count % 1000 == 0 and not dry_run:
                            state.set_current_subset(subset, total_tokens, total_lines)

                        if subset_count % 10000 == 0:
                            print(f"    已处理 {subset_count:,} 行, ~{subset_tokens/1e6:.1f}M tokens", flush=True)

                # 更新子集处理状态
                if not dry_run:
                    state.update_subset_tokens(subset, subset_tokens_processed + subset_tokens)
                    print(f"  ✅ {subset} 本次处理: {subset_count:,} 行, ~{subset_tokens/1e6:.1f}M tokens", flush=True)

                success = True

            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                if any(x in error_msg.lower() for x in ["timeout", "broken pipe", "connection", "network"]):
                    print(f"  ⚠️  网络错误: {str(e)[:80]}...", flush=True)
                    if retry_count < max_retries:
                        continue
                print(f"  ❌ 处理 {subset} 失败: {e}")
                break

        if not success and retry_count >= max_retries:
            print(f"  ⚠️  子集 {subset} 多次重试失败，跳过")
            continue

    overall_elapsed = time.time() - overall_start
    progress_pct = min(100, total_tokens / target_tokens * 100) if target_tokens > 0 else 100

    print(f"\n{'=' * 60}")
    print(f"PILES 处理完成")
    print(f"  耗时: {format_time(overall_elapsed)}")
    print(f"  条数: {total_lines:,}")
    print(f"  tokens: ~{total_tokens/1e9:.2f}B / {target_tokens/1e9:.0f}B ({progress_pct:.1f}%)")

    if not dry_run and total_tokens < target_tokens:
        print(f"\n📋 下载未完成，已保存状态文件")
        print(f"  状态文件: {output_path}.state.json")
        print(f"  重新运行脚本将继续下载")
        # 返回None表示下载未完成
        return total_lines, total_tokens, False

    return total_lines, total_tokens, True


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
) -> tuple[int, int, bool]:
    """处理 WanJuanCC 数据集"""
    if not MODELSCOPE_AVAILABLE:
        print("错误: 需要安装 modelscope 库")
        print("请运行: pip install modelscope")
        return 0, 0, False

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
        return 0, 0, False

    total_files = len(file_list)
    # 使用 processed_subsets 而不是 processed
    processed_wanjuan = set()
    if "wanjuan" in loaded_state.get("processed_subsets", {}):
        processed_wanjuan = set(loaded_state["processed_subsets"]["wanjuan"].get("files", []))
    remaining = [f for f in file_list if f["path"] not in processed_wanjuan]

    if not dry_run and total_tokens >= target_tokens:
        print(f"✅ 目标 tokens ({target_tokens/1e9:.0f}B) 已达成，跳过")
        return total_lines, total_tokens, True

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
                state.mark_file_done("wanjuan", f_info["path"], total_tokens, total_lines)

            if dry_run and total_lines >= 10:
                break

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"WanJuanCC 处理完成")
    print(f"  耗时: {format_time(elapsed)}")
    print(f"  条数: {total_lines:,}")
    print(f"  tokens: ~{total_tokens/1e9:.2f}B")

    success = total_tokens >= target_tokens if not dry_run else True
    return total_lines, total_tokens, success


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

        lines, tokens, success = process_piles_dataset(
            output_path=args.output,
            target_tokens=args.target_tokens,
            subsets=subsets,
            max_length=args.max_length,
            min_length=args.min_length,
            dry_run=args.dry_run,
            mirror=args.mirror,
        )

        # 移除自动退出逻辑，让看门狗负责重启
        # 脚本正常退出（code 0），即使下载未完成

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
