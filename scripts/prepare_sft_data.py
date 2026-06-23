#!/usr/bin/env python3
"""
将 QA 对数据转换为预训练格式
将 prompt 和 response 合并为单个文本字段
"""

import json
import argparse

def convert_qa_to_pretrain_format(input_path, output_path, max_samples=None):
    """
    将 QA 对转换为预训练格式
    
    输入格式: {"prompt": "...", "response": "..."}
    输出格式: {"text": "prompt\n\nresponse"}
    """
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        count = 0
        for line in f_in:
            if max_samples and count >= max_samples:
                break
            
            data = json.loads(line)
            
            # 合并 prompt 和 response
            prompt = data.get('prompt', '')
            response = data.get('response', '')
            
            # 使用换行符分隔，让模型学习到 prompt 和 response 的关系
            text = f"{prompt}\n\n{response}"
            
            # 写入预训练格式
            f_out.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            
            count += 1
            if count % 1000 == 0:
                print(f"已处理 {count} 条数据")
        
        print(f"完成！共处理 {count} 条数据")
        print(f"输出文件: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 QA 对转换为预训练格式")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--max-samples", type=int, default=None, help="最大处理样本数")
    
    args = parser.parse_args()
    
    convert_qa_to_pretrain_format(args.input, args.output, args.max_samples)