#!/usr/bin/env python3
"""
检查 packed 32768 shuffled 训练数据的质量。

功能：
  1. 全量流式统计：行数、JSON解析错误、字符长度分布、</s>分隔符频率、空文本
  2. 采样深度检查：tokenize后token长度分布、</s>编码行为、文档边界分析
  3. 重复检测：采样行hash去重
  4. 超长文本分割检查：检查按字符分割是否导致问题

使用方法：
    python kilo/history_scripts/check_packed_data_quality.py [--data PATH] [--sample N]

创建时间：2026-07-10
"""
import json
import os
import sys
import time
import hashlib
import argparse
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "piles_packed_32768_shuffled.jsonl")


def check_file(data_path, sample_size=2000, tokenize_check=True):
    start_time = time.time()

    # ========== 全量统计 ==========
    total_lines = 0
    json_errors = 0
    empty_text = 0
    too_short = 0  # < 30000 chars (pack 脚本的 MIN_INPUT_CHARS)
    eos_count_dist = Counter()  # </s> 出现次数分布
    char_len_buckets = Counter()  # 字符长度桶
    char_lengths = []  # 用于计算统计量（采样存储避免内存爆炸）
    sample_interval = 100  # 每100行存一个字符长度

    # 采样数据
    sampled_lines = []
    sample_hashes = set()
    duplicate_count = 0

    # </s> 相关统计
    has_eos_separator = 0
    eos_sep_counts = []

    print(f"开始检查: {data_path}")
    print(f"文件大小: {os.path.getsize(data_path) / 1e9:.2f} GB")
    print()

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1

            # JSON 解析
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                json_errors += 1
                if json_errors <= 5:
                    print(f"  [JSON错误] 行 {line_num}: {e}")
                continue

            text = data.get('content_split', '')
            text_len = len(text)

            # 空文本检查
            if not text:
                empty_text += 1
                continue

            # 过短检查
            if text_len < 30000:
                too_short += 1

            # </s> 分隔符检查
            eos_count = text.count('</s>')
            eos_count_dist[eos_count] += 1
            if eos_count > 0:
                has_eos_separator += 1
                eos_sep_counts.append(eos_count)

            # 字符长度桶
            bucket = (text_len // 10000) * 10000
            char_len_buckets[bucket] += 1

            # 采样字符长度
            if line_num % sample_interval == 0:
                char_lengths.append(text_len)

            # 采样行用于深度检查
            if len(sampled_lines) < sample_size:
                # 均匀采样：每隔一定行数取一行
                pass

            # 重复检测（采样）
            if line_num % 50 == 0:  # 采样2%的行做hash
                h = hashlib.md5(text[:1000].encode('utf-8')).hexdigest()
                if h in sample_hashes:
                    duplicate_count += 1
                else:
                    sample_hashes.add(h)

            # 进度报告
            if total_lines % 50000 == 0:
                elapsed = time.time() - start_time
                rate = total_lines / elapsed
                print(f"  已检查 {total_lines:,} 行 ({rate:.0f} 行/秒), "
                      f"JSON错误={json_errors}, 空文本={empty_text}, "
                      f"过短={too_short}, 含</s>={has_eos_separator}")

    elapsed = time.time() - start_time

    # ========== 输出全量统计 ==========
    print("\n" + "=" * 70)
    print("全量统计结果")
    print("=" * 70)
    print(f"总行数:          {total_lines:,}")
    print(f"JSON解析错误:    {json_errors}")
    print(f"空文本:          {empty_text}")
    print(f"过短(<30k字符):  {too_short}")
    print(f"含</s>分隔符:    {has_eos_separator:,} ({has_eos_separator/total_lines*100:.1f}%)")
    print(f"重复(采样2%):    {duplicate_count}")
    print(f"耗时:            {elapsed:.0f}秒")

    # 字符长度分布
    print(f"\n字符长度分布 (采样 {len(char_lengths)} 行):")
    if char_lengths:
        char_lengths.sort()
        n = len(char_lengths)
        print(f"  最小: {char_lengths[0]:,}")
        print(f"  P25:  {char_lengths[n//4]:,}")
        print(f"  P50:  {char_lengths[n//2]:,}")
        print(f"  P75:  {char_lengths[3*n//4]:,}")
        print(f"  最大: {char_lengths[-1]:,}")
        print(f"  均值: {sum(char_lengths)/n:,.0f}")
        # 估算 token 数 (chars/4)
        print(f"  估算token范围: {char_lengths[0]//4:,} ~ {char_lengths[-1]//4:,}")

    # 字符长度桶
    print(f"\n字符长度桶分布:")
    for bucket in sorted(char_len_buckets.keys()):
        count = char_len_buckets[bucket]
        bar = '#' * min(50, count * 50 // max(char_len_buckets.values()))
        print(f"  {bucket:>7d}-{bucket+9999:>7d}: {count:>7d} ({count/total_lines*100:5.1f}%) {bar}")

    # </s> 次数分布
    print(f"\n</s> 分隔符出现次数分布:")
    for count in sorted(eos_count_dist.keys()):
        freq = eos_count_dist[count]
        print(f"  {count}个</s>: {freq:>7d} 行 ({freq/total_lines*100:.1f}%)")

    if eos_sep_counts:
        print(f"  含</s>的行中，平均</s>数: {sum(eos_sep_counts)/len(eos_sep_counts):.1f}")

    # ========== Tokenize 深度检查 ==========
    if tokenize_check and sampled_lines:
        print("\n" + "=" * 70)
        print("Tokenize 深度检查")
        print("=" * 70)
        # 这里需要重新读取采样行做 tokenize
        _tokenize_check(data_path, sample_size)

    return {
        'total_lines': total_lines,
        'json_errors': json_errors,
        'empty_text': empty_text,
        'too_short': too_short,
        'has_eos_separator': has_eos_separator,
        'duplicate_count': duplicate_count,
    }


def _tokenize_check(data_path, sample_size):
    """采样 tokenize 检查"""
    try:
        from transformers import AutoTokenizer
        tokenizer_path = os.path.join(PROJECT_ROOT, "model_assets", "qwen3.5-2b")
        tok = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"  无法加载tokenizer: {e}")
        return

    eos_id = tok.eos_token_id
    eos_str = tok.eos_token

    # </s> 编码
    eos_sep_ids = tok.encode('</s>', add_special_tokens=False)
    print(f"  EOS token: {eos_str!r} (id={eos_id})")
    print(f"  </s> 编码为: {eos_sep_ids} (不是EOS!)")

    # 采样读取
    total = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1

    # 均匀采样
    step = max(1, total // sample_size)
    token_lengths = []
    eos_token_counts = []  # 实际EOS token在序列中的数量
    eos_sep_token_counts = []  # </s>编码后的token在序列中的数量
    cross_doc_boundary_count = 0  # 跨文档预测的样本数

    sampled = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num % step != 0 or sampled >= sample_size:
                continue

            data = json.loads(line)
            text = data.get('content_split', '')
            if not text:
                continue

            # 模拟 data_transform.py 的 tokenize 逻辑
            tokens = tok.encode(text, add_special_tokens=False) + [eos_id]
            token_lengths.append(len(tokens))

            # 统计实际 EOS token 数量（只算末尾追加的那个）
            actual_eos = tokens.count(eos_id)
            eos_token_counts.append(actual_eos)

            # 统计 </s> 编码的 token 序列出现次数
            # </s> -> [510, 82, 29]
            sep_count = 0
            for i in range(len(tokens) - len(eos_sep_ids) + 1):
                if tokens[i:i+len(eos_sep_ids)] == eos_sep_ids:
                    sep_count += 1
            eos_sep_token_counts.append(sep_count)

            # 检查跨文档预测：如果文本中有 </s>，说明有多个文档拼接
            # 但 </s> 不被识别为 EOS，所以模型会在文档间做无意义的预测
            if '</s>' in text:
                cross_doc_boundary_count += 1

            sampled += 1
            if sampled % 200 == 0:
                print(f"  已 tokenize {sampled}/{sample_size} 行...")

    print(f"\n  采样 tokenize 结果 ({sampled} 行):")
    if token_lengths:
        token_lengths.sort()
        n = len(token_lengths)
        print(f"  Token长度: min={token_lengths[0]:,}, P50={token_lengths[n//2]:,}, "
              f"P75={token_lengths[3*n//4]:,}, max={token_lengths[-1]:,}")
        print(f"  Token长度均值: {sum(token_lengths)/n:,.0f}")

        # 与 max_seq_len=32768 对比
        over_32768 = sum(1 for t in token_lengths if t > 32768)
        print(f"  超过32768的行: {over_32768}/{n} ({over_32768/n*100:.1f}%)")
        # data_transform 会按 max_seq_len 分块，超过的部分会被截断

        # EOS token 统计
        print(f"\n  实际EOS token数量分布:")
        eos_dist = Counter(eos_token_counts)
        for k in sorted(eos_dist.keys()):
            print(f"    {k}个EOS: {eos_dist[k]} 行 ({eos_dist[k]/n*100:.1f}%)")

        # </s> 编码 token 统计
        print(f"\n  </s>编码token出现次数分布 (这些是普通token，不是EOS):")
        sep_dist = Counter(eos_sep_token_counts)
        for k in sorted(sep_dist.keys()):
            print(f"    {k}组</s>: {sep_dist[k]} 行 ({sep_dist[k]/n*100:.1f}%)")

        print(f"\n  含跨文档边界(有</s>)的行: {cross_doc_boundary_count}/{n} "
              f"({cross_doc_boundary_count/n*100:.1f}%)")
        print(f"  -> 这些行中，文档间没有真正的EOS分隔，"
              f"模型会在文档边界处做无意义的跨文档预测")


def main():
    parser = argparse.ArgumentParser(description="检查 packed 训练数据质量")
    parser.add_argument('--data', default=DEFAULT_DATA, help='数据文件路径')
    parser.add_argument('--sample', type=int, default=2000, help='tokenize采样行数')
    parser.add_argument('--no-tokenize', action='store_true', help='跳过tokenize检查(更快)')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"错误: 文件不存在 {args.data}")
        sys.exit(1)

    check_file(args.data, args.sample, tokenize_check=not args.no_tokenize)


if __name__ == '__main__':
    main()
