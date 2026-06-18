"""
从 OpenThoughts-3 Parquet 文件转换为 JSONL 格式
支持去重处理，仅提取 prompts 用于 OPD-TTT 训练
"""
import pandas as pd
import json
import numpy as np
import glob
import argparse

def extract_prompts_from_parquet(
    parquet_files: list,
    output_file: str,
    max_prompts: int = 200000,
    enable_dedup: bool = True
) -> dict:
    """
    从 parquet 文件中提取 prompts
    
    Args:
        parquet_files: parquet 文件列表
        output_file: 输出文件路径
        max_prompts: 最大 prompts 数量
        enable_dedup: 是否启用去重
    
    Returns:
        统计信息字典
    """
    print(f'找到 {len(parquet_files)} 个 parquet 文件')
    
    # 根据是否启用去重选择数据结构
    if enable_dedup:
        unique_prompts = set()
    else:
        unique_prompts = []
    
    seen_count = 0
    total_count = 0
    duplicate_count = 0
    
    for i, file in enumerate(parquet_files):
        try:
            df = pd.read_parquet(file)
            print(f'[{i+1}/{len(parquet_files)}] 处理 {file} (包含 {len(df)} 条数据)')
            
            for idx, row in df.iterrows():
                total_count += 1
                
                # 检查是否达到最大数量
                if enable_dedup and len(unique_prompts) >= max_prompts:
                    break
                elif not enable_dedup and len(unique_prompts) >= max_prompts:
                    break
                
                # 提取conversations
                conv = row.get('conversations', np.array([]))
                if isinstance(conv, np.ndarray):
                    conv = conv.tolist()
                
                # 提取human消息作为prompt
                prompt = None
                if isinstance(conv, list) and len(conv) > 0:
                    for msg in conv:
                        if isinstance(msg, dict) and msg.get('from') == 'human':
                            prompt = msg.get('value', '')
                            break
                
                # 去重处理
                if prompt and len(str(prompt).strip()) > 0:
                    prompt_str = str(prompt)
                    
                    if enable_dedup:
                        if prompt_str in unique_prompts:
                            seen_count += 1
                            duplicate_count += 1
                        else:
                            unique_prompts.add(prompt_str)
                    else:
                        unique_prompts.append(prompt_str)
            
            if (i + 1) % 10 == 0 or i == len(parquet_files) - 1:
                if enable_dedup:
                    print(f'  进度: 总计={total_count}, 唯一={len(unique_prompts)}, 重复={seen_count}')
                else:
                    print(f'  进度: 总计={total_count}, 已提取={len(unique_prompts)}')
                    
        except Exception as e:
            print(f'处理文件 {file} 时出错: {e}')
            continue
    
    print(f'\n=== 处理完成 ===')
    print(f'总样本数: {total_count}')
    if enable_dedup:
        print(f'唯一prompts: {len(unique_prompts)}')
        print(f'重复数量: {duplicate_count}')
        print(f'去重率: {duplicate_count/total_count*100:.2f}%' if total_count > 0 else '去重率: 0%')
    else:
        print(f'提取prompts: {len(unique_prompts)}')
    
    # 转换为列表（如果使用了集合）
    if enable_dedup:
        final_prompts = list(unique_prompts)
    else:
        final_prompts = unique_prompts
    
    # 保存为JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt in final_prompts:
            f.write(json.dumps({'prompt': prompt}, ensure_ascii=False) + '\n')
    
    file_size = len(open(output_file, encoding='utf-8').read()) / (1024*1024)
    print(f'\n已保存到: {output_file}')
    print(f'文件大小: {file_size:.2f} MB')
    print(f'prompts数量: {len(final_prompts)}')
    
    return {
        'total_samples': total_count,
        'unique_prompts': len(final_prompts),
        'duplicate_count': duplicate_count if enable_dedup else 0,
        'file_size_mb': file_size,
        'deduplication_rate': (duplicate_count/total_count*100) if total_count > 0 and enable_dedup else 0
    }


def main():
    parser = argparse.ArgumentParser(description='从 Parquet 文件提取 prompts')
    parser.add_argument('--input_dir', type=str, default='data/prompts_raw/data',
                       help='parquet 文件目录')
    parser.add_argument('--output_file', type=str, default='data/opd_prompts_raw.jsonl',
                       help='输出 JSONL 文件路径')
    parser.add_argument('--max_prompts', type=int, default=200000,
                       help='最大提取的 prompts 数量')
    parser.add_argument('--no_dedup', action='store_true',
                       help='禁用去重')
    
    args = parser.parse_args()
    
    # 查找所有 parquet 文件
    parquet_files = sorted(glob.glob(f'{args.input_dir}/*.parquet'))
    
    if not parquet_files:
        print(f'在 {args.input_dir} 中没有找到 parquet 文件')
        return
    
    # 提取 prompts
    stats = extract_prompts_from_parquet(
        parquet_files=parquet_files,
        output_file=args.output_file,
        max_prompts=args.max_prompts,
        enable_dedup=not args.no_dedup
    )
    
    print(f'\n=== 统计信息 ===')
    for key, value in stats.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()