"""
从 OpenThoughts-3 Parquet 文件转换为完整的问答对 JSONL 格式
用于 Off-policy SFT 训练，直接使用原始数据中的回答
"""
import pandas as pd
import json
import numpy as np
import glob
import argparse

def extract_qa_pairs_from_parquet(
    parquet_files: list,
    output_file: str,
    max_pairs: int = 200000,
    enable_dedup: bool = True
) -> dict:
    """
    从 parquet 文件中提取完整的问答对
    
    Args:
        parquet_files: parquet 文件列表
        output_file: 输出文件路径
        max_pairs: 最大问答对数量
        enable_dedup: 是否启用去重
    
    Returns:
        统计信息字典
    """
    print(f'找到 {len(parquet_files)} 个 parquet 文件')
    
    # 根据是否启用去重选择数据结构
    if enable_dedup:
        unique_qa_pairs = set()
    else:
        unique_qa_pairs = []
    
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
                if enable_dedup and len(unique_qa_pairs) >= max_pairs:
                    break
                elif not enable_dedup and len(unique_qa_pairs) >= max_pairs:
                    break
                
                # 提取conversations
                conv = row.get('conversations', np.array([]))
                if isinstance(conv, np.ndarray):
                    conv = conv.tolist()
                
                # 提取prompt和response
                prompt = None
                response = None
                
                if isinstance(conv, list):
                    for msg in conv:
                        if isinstance(msg, dict):
                            if msg.get('from') == 'human':
                                prompt = msg.get('value', '')
                            elif msg.get('from') == 'gpt':
                                response = msg.get('value', '')
                
                # 去重处理
                if prompt and response and len(str(prompt).strip()) > 0 and len(str(response).strip()) > 0:
                    qa_pair = {'prompt': str(prompt), 'response': str(response)}
                    qa_key = str(prompt) + '|||' + str(response)
                    
                    if enable_dedup:
                        if qa_key in unique_qa_pairs:
                            seen_count += 1
                            duplicate_count += 1
                        else:
                            unique_qa_pairs.add(qa_key)
                    else:
                        unique_qa_pairs.append(qa_pair)
            
            if (i + 1) % 10 == 0 or i == len(parquet_files) - 1:
                if enable_dedup:
                    print(f'  进度: 总计={total_count}, 唯一={len(unique_qa_pairs)}, 重复={seen_count}')
                else:
                    print(f'  进度: 总计={total_count}, 已提取={len(unique_qa_pairs)}')
                    
        except Exception as e:
            print(f'处理文件 {file} 时出错: {e}')
            continue
    
    print(f'\n=== 处理完成 ===')
    print(f'总样本数: {total_count}')
    if enable_dedup:
        print(f'唯一问答对: {len(unique_qa_pairs)}')
        print(f'重复数量: {duplicate_count}')
        print(f'去重率: {duplicate_count/total_count*100:.2f}%' if total_count > 0 else '去重率: 0%')
    else:
        print(f'提取问答对: {len(unique_qa_pairs)}')
    
    # 转换为最终格式
    final_qa_pairs = []
    if enable_dedup:
        for qa_key in unique_qa_pairs:
            prompt, response = qa_key.split('|||', 1)
            final_qa_pairs.append({'prompt': prompt, 'response': response})
    else:
        final_qa_pairs = unique_qa_pairs
    
    # 保存为JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa in final_qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    file_size = len(open(output_file, encoding='utf-8').read()) / (1024*1024)
    print(f'\n已保存到: {output_file}')
    print(f'文件大小: {file_size:.2f} MB')
    print(f'问答对数量: {len(final_qa_pairs)}')
    
    return {
        'total_samples': total_count,
        'unique_qa_pairs': len(final_qa_pairs),
        'duplicate_count': duplicate_count if enable_dedup else 0,
        'file_size_mb': file_size,
        'deduplication_rate': (duplicate_count/total_count*100) if total_count > 0 and enable_dedup else 0
    }


def main():
    parser = argparse.ArgumentParser(description='从 Parquet 文件提取问答对')
    parser.add_argument('--input_dir', type=str, default='data/prompts_raw/data',
                       help='parquet 文件目录')
    parser.add_argument('--output_file', type=str, default='data/opd_sft_raw.jsonl',
                       help='输出 JSONL 文件路径')
