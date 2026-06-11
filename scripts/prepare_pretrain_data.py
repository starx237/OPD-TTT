#!/usr/bin/env python3
"""
从 ModelScope 的 Shanghai_AI_Laboratory/WanJuanCC 下载英文数据并转为 JSONL。

数据格式:
  远程: tar.gz 文件 (每个 ~3.4GB), 内嵌 jsonl/part-*.jsonl
  字段: id, content(文本), title, language(语言), date, token_num, ...

特点:
  - 使用 ModelScope SDK 获取文件列表和下载 URL
  - 带断点续传: 每处理完一个 tar.gz 即记录到状态文件
  - 边下载边处理边删除: 不保留 tar.gz 压缩包
  - 中断后重跑自动继续

用法:
    # Dry run 测试
    python scripts/prepare_pretrain_data.py \
        --output data/pretrain_500m.jsonl \
        --target_tokens 20000000000 \
        --dry_run

    # 实际下载 500M 数据 (20B tokens)
    python scripts/prepare_pretrain_data.py \
        --output data/pretrain_500m.jsonl \
        --target_tokens 20000000000

    # 中断后续传 (自动检测状态文件)
    python scripts/prepare_pretrain_data.py \
        --output data/pretrain_500m.jsonl \
        --target_tokens 20000000000

    # 强制从头开始 (忽略状态文件)
    python scripts/prepare_pretrain_data.py \
        --output data/pretrain_500m.jsonl \
        --target_tokens 20000000000 \
        --force
"""

import argparse
import gzip
import json
import os
import shutil
import tarfile
import tempfile
import time
import requests

from modelscope.hub.api import HubApi

DATASET_NAMESPACE = "Shanghai_AI_Laboratory"
DATASET_NAME = "WanJuanCC"
TEXT_KEY = "content"            # jsonl 中的文本字段
CONTENT_KEY = "content_split"    # 输出 JSONL 中的 key
LANGUAGE_KEY = "language"        # 语言过滤字段
TAR_TEMP_DIR = "data/wanjuanc_tar"  # 下载临时目录 (可通过 --data-dir 改变基路径)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h{m:02d}m{s:02d}s"


class StateFile:
    """
    管理断点续传的状态文件。
    路径: {output}.state.json
    格式: {"processed": [...], "total_tokens": N, "total_lines": N}
    """

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

    def mark_done(self, file_path: str, total_tokens: int, total_lines: int) -> None:
        state = self.load()
        if file_path not in state["processed"]:
            state["processed"].append(file_path)
        state["total_tokens"] = total_tokens
        state["total_lines"] = total_lines
        self.save(state)

    def is_done(self, file_path: str) -> bool:
        return file_path in self.load()["processed"]

    def delete(self) -> None:
        if os.path.isfile(self.path):
            os.remove(self.path)


def get_file_list() -> list:
    """
    通过 ModelScope SDK 获取数据集文件列表。
    返回: [{"name": ..., "path": ..., "size": ..., "download_url": ...}, ...]
    """
    api = HubApi()
    files = api.get_dataset_files(
        f"{DATASET_NAMESPACE}/{DATASET_NAME}",
        revision="master",
    )
    result = []
    for f in files:
        if f["Type"] != "blob" or not f["Name"].endswith(".tar.gz"):
            continue
        download_url = api.get_dataset_file_url(
            file_name=f["Path"],
            dataset_name=DATASET_NAME,
            namespace=DATASET_NAMESPACE,
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


def download_file(url: str, dest: str) -> None:
    """
    流式下载单个文件，支持断点续传和进度显示。
    先下载到 .incomplete 后缀，完成后 rename，避免不完整文件。
    """
    import sys
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    headers = {"User-Agent": "modelscope"}
    tmp_dest = dest + ".incomplete"

    # 检查是否已存在完整文件
    if os.path.isfile(dest):
        print(f"  📦 已有本地缓存")
        return

    # 检查不完整文件（断点续传）
    if os.path.isfile(tmp_dest):
        downloaded_size = os.path.getsize(tmp_dest)
        headers["Range"] = f"bytes={downloaded_size}-"
        mode = "ab"
        print(f"  📄 续传 ({downloaded_size/1e9:.2f}GB 已下载)...")
    else:
        downloaded_size = 0
        mode = "wb"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=(30, 300))
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            if "Range" in headers:
                # 断点续传时，content-length 是剩余大小
                total_size += downloaded_size

            downloaded = downloaded_size
            start_time = time.time()
            last_update = start_time
            last_size = downloaded_size

            with open(tmp_dest, mode) as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # 每秒更新一次进度
                        now = time.time()
                        if now - last_update >= 1.0:
                            elapsed = now - start_time
                            chunk_speed = (downloaded - last_size) / (now - last_update)
                            progress = downloaded / total_size * 100 if total_size > 0 else 0
                            mb_downloaded = downloaded / 1e6
                            mb_total = total_size / 1e6
                            sys.stdout.write(
                                f"\r  ⬇️  {mb_downloaded:.1f}/{mb_total:.1f}MB ({progress:.1f}%) @ {chunk_speed/1e6:.1f}MB/s"
                            )
                            sys.stdout.flush()
                            last_update = now
                            last_size = downloaded

            os.replace(tmp_dest, dest)
            # 下载完成，显示最终信息
            elapsed = time.time() - start_time
            avg_speed = (downloaded - downloaded_size) / max(elapsed, 1)
            print(f"\r  ⬇️  完成 ({downloaded/1e9:.2f}GB, {elapsed:.0f}s, {avg_speed/1e6:.1f}MB/s)     ")
            return

        except Exception as e:
            if os.path.isfile(tmp_dest):
                downloaded_size = os.path.getsize(tmp_dest)
            print(f"\n  ⚠️  下载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"  🔄 {wait_time}秒后重试...")
                time.sleep(wait_time)
                # 重置 headers 和 mode
                headers = {"User-Agent": "modelscope"}
                mode = "wb"
                tmp_dest = dest + ".incomplete"
                if os.path.isfile(tmp_dest):
                    downloaded_size = os.path.getsize(tmp_dest)
                    headers["Range"] = f"bytes={downloaded_size}-"
                    mode = "ab"
            else:
                if os.path.isfile(tmp_dest):
                    # 保留不完整文件，以便下次续传
                    print(f"  📄 不完整文件已保存: {tmp_dest}")
                raise


def extract_jsonl_lines(tar_path: str) -> iter:
    """从本地 tar.gz 文件中逐行 yield jsonl 文本。"""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True,
                        help="输出的 JSONL 文件路径")
    parser.add_argument("--target_tokens", type=int, required=True,
                        help="目标 token 数")
    parser.add_argument("--max_length", type=int, default=131072,
                        help="单条文本最大字符数 (默认 131072)")
    parser.add_argument("--min_length", type=int, default=200,
                        help="单条文本最小字符数 (默认 200)")
    parser.add_argument("--language", type=str, default="en",
                        help="语言过滤 (默认 en)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Dry run: 仅处理 10 条")
    parser.add_argument("--force", action="store_true",
                        help="忽略已有状态文件，从头开始")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="数据存储基目录 (默认使用 output 所在目录下的 data/)")
    args = parser.parse_args()

    # 如果指定了 --data-dir，data 相关目录和文件都在其下
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.dirname(os.path.abspath(args.output))
    tar_dir = os.path.join(data_dir, "wanjuanc_tar")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(tar_dir, exist_ok=True)

    state = StateFile(args.output)
    if args.force:
        print("🔶 --force: 忽略已有状态，从头开始")
        state.delete()

    print(f"数据集:   {DATASET_NAMESPACE}/{DATASET_NAME}")
    print(f"目标:     {args.target_tokens / 1e9:.0f}B tokens")
    print(f"输出:     {args.output}")
    if args.dry_run:
        print("🔷 DRY RUN: 仅处理 10 条")
    print()

    # [1/3] 获取文件列表
    print("[1/3] 获取 WanJuanCC 文件列表...")
    file_list = get_file_list()
    if not file_list:
        print("  ❌ 未找到任何 tar.gz 文件!")
        return
    total_files = len(file_list)
    total_gb = sum(f["size"] for f in file_list) / 1e9
    print(f"  找到 {total_files} 个 tar.gz 文件 (共 {total_gb:.1f}GB 压缩)")
    print(f"  首个: {file_list[0]['name']}")
    print(f"  末尾: {file_list[-1]['name']}")

    # 加载已处理状态
    loaded_state = state.load()
    processed_set = set(loaded_state["processed"])
    total_tokens = loaded_state["total_tokens"]
    total_lines = loaded_state["total_lines"]

    # 过滤未处理的文件
    remaining = [f for f in file_list if f["path"] not in processed_set]
    skipped = total_files - len(remaining)

    if skipped > 0:
        print(f"\n📋 状态文件发现 {skipped} 个已处理完成")
        if not remaining:
            print("  ✅ 所有文件已处理完毕!")
            return
        print(f"  还需处理 {len(remaining)}/{total_files} 个")

    if not args.dry_run and total_tokens >= args.target_tokens:
        print(f"✅ 目标 tokens ({args.target_tokens/1e9:.0f}B) 已达成，跳过")
        return

    if args.dry_run:
        print("\n🔷 DRY RUN 模式: 不记录状态文件，不保留压缩包")

    if total_tokens > 0:
        print(f"📋 从上次继续: {total_lines:,} 条, ~{total_tokens/1e9:.2f}B tokens")
        write_mode = "a"
    else:
        print("📋 全新开始")
        write_mode = "w"

    # ========== 主循环 ==========
    print("\n[2/3] 下载并处理数据...")
    start_time = time.time()

    with open(args.output, write_mode, encoding="utf-8") as out_f:
        for idx, f_info in enumerate(remaining):
            if not args.dry_run and total_tokens >= args.target_tokens:
                print(f"\n  ✅ 目标 {args.target_tokens/1e9:.0f}B tokens 已达成, 停止")
                break

            elapsed = time.time() - start_time
            print(f"\n  [{idx + 1 + skipped}/{total_files}] {f_info['name']}")
            print(f"  当前: {total_lines:,} 条, ~{total_tokens/1e9:.2f}B tok, {format_time(elapsed)}")

            # Step A: 下载 tar.gz 到临时目录
            tar_local = os.path.join(tar_dir, f_info["name"])
            incomplete_local = tar_local + ".incomplete"

            # 检查不完整文件
            if os.path.isfile(incomplete_local):
                incomplete_size = os.path.getsize(incomplete_local) / 1e9
                total_size = f_info["size"] / 1e9
                print(f"  📄 发现未完成下载 ({incomplete_size:.2f}/{total_size:.2f}GB)，继续...")

            if os.path.isfile(tar_local):
                print(f"  📦 已有本地缓存 {tar_local}")
            else:
                print(f"  ⬇️  下载中 ({f_info['size']/1e9:.2f}GB)...")
                t0 = time.time()
                try:
                    download_file(f_info["download_url"], tar_local)
                    dl_time = time.time() - t0
                    # 下载完成信息已在 download_file 中打印
                except Exception as e:
                    print(f"❌ 下载失败: {e}")
                    print(f"  提示: 可以重新运行脚本，它会自动续传")
                    if not args.dry_run:
                        state.mark_done(f_info["path"], total_tokens, total_lines)
                    continue

            # Step B: 解压提取 jsonl (streaming from tar.gz)
            print(f"  📖 解压提取...", end=" ", flush=True)
            t0 = time.time()
            count = 0
            try:
                lines_iter = extract_jsonl_lines(tar_local)
                for line in lines_iter:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    lang = data.get(LANGUAGE_KEY, "")
                    if lang and lang != args.language:
                        continue

                    text = data.get(TEXT_KEY, "")
                    if not text or len(text) < args.min_length:
                        continue

                    text = text[:args.max_length]
                    record = {CONTENT_KEY: text}
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    total_lines += 1
                    total_tokens += estimate_tokens(text)
                    count += 1

                    if args.dry_run and total_lines >= 10:
                        break
                    if not args.dry_run and total_tokens >= args.target_tokens:
                        break

                dt = time.time() - t0
                print(f"{count} 条英文行 ({dt:.1f}s)")
            except Exception as e:
                print(f"❌ 解压失败: {e}")
                if not args.dry_run:
                    state.mark_done(f_info["path"], total_tokens, total_lines)
                # 清理损坏文件
                if os.path.isfile(tar_local):
                    os.remove(tar_local)
                continue

            # Step C: 删除 tar.gz
            if os.path.isfile(tar_local):
                os.remove(tar_local)
                print(f"  🗑️  已删除压缩包")

            # Step D: 更新状态文件 (非 dry run 时)
            if not args.dry_run:
                state.mark_done(f_info["path"], total_tokens, total_lines)

            if args.dry_run and total_lines >= 10:
                print("  🔷 DRY RUN: 10 条完成, 停止")
                break

    # ========== 汇总 ==========
    elapsed = time.time() - start_time
    file_size = os.path.getsize(args.output) / 1e9
    print(f"\n{'─' * 55}")
    tag = "DRY RUN ✅" if args.dry_run else "完成 ✅"
    print(f"  {tag}")
    print(f"  耗时:     {format_time(elapsed)}")
    print(f"  条数:     {total_lines:,}")
    print(f"  tokens:   ~{total_tokens/1e9:.2f}B / {args.target_tokens/1e9:.0f}B")
    print(f"  文件:     {file_size:.2f}GB")
    print(f"  平均:     {total_tokens / max(total_lines, 1):.1f} tok/条")
    print(f"  状态:     {state.path}")

    if not args.dry_run and total_tokens < args.target_tokens:
        shortfall = (args.target_tokens - total_tokens) / 1e9
        print(f"\n⚠️  数据不足! 还差 ~{shortfall:.1f}B tokens")
        print(f"  已处理完 WanJuanCC 全部 {total_files} 个文件")
        print(f"  可能需要补充其他数据集")

    # 清理空临时目录
    if os.path.isdir(tar_dir) and not os.listdir(tar_dir):
        os.rmdir(tar_dir)
        print(f"  已清理临时目录 {tar_dir}")


if __name__ == "__main__":
    main()
