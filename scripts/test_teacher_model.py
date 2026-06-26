#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
教师模型测试脚本

本脚本用于验证教师模型（qwen2.5-7b）是否正确配置并能正常工作，
包括权重加载、tokenizer兼容性和prompt生成功能测试。
"""

import os
import sys
import json
import torch
import traceback
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime


class TeacherModelTester:
    """教师模型测试器"""
    
    def __init__(self, teacher_model_path="model_assets/teacher_qwen2.5_7b"):
        self.teacher_model_path = teacher_model_path
        self.model = None
        self.tokenizer = None
        self.test_results = []
        
    def log_test(self, test_name, success, message, details=""):
        """记录测试结果"""
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "message": message,
            "details": details,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        status = "✓ 通过" if success else "✗ 失败"
        color = "\033[92m" if success else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {test_name:35s} : {message}")
        if details:
            print(f"    详情: {details}")
    
    def check_file_integrity(self):
        """检查文件完整性"""
        print("\n" + "=" * 80)
        print("1. 检查文件完整性")
        print("=" * 80)
        
        required_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json"
        ]
        
        # 检查目录是否存在
        if not os.path.exists(self.teacher_model_path):
            self.log_test("目录检查", False, f"教师模型目录不存在: {self.teacher_model_path}")
            return False
        
        self.log_test("目录检查", True, f"目录存在: {self.teacher_model_path}")
        
        # 检查必需文件
        missing_files = []
        for file in required_files:
            file_path = os.path.join(self.teacher_model_path, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                self.log_test(f"文件检查: {file}", True, f"存在 (大小: {size:,} bytes)")
            else:
                missing_files.append(file)
                self.log_test(f"文件检查: {file}", False, "文件缺失")
        
        # 检查模型权重文件
        weight_files = sorted([f for f in os.listdir(self.teacher_model_path) 
                              if f.startswith("model-") and f.endswith(".safetensors")])
        
        if weight_files:
            total_size = sum(os.path.getsize(os.path.join(self.teacher_model_path, f)) 
                            for f in weight_files)
            self.log_test("权重文件检查", True, 
                         f"找到 {len(weight_files)} 个权重文件 (总大小: {total_size/(1024**3):.2f} GB)")
        else:
            self.log_test("权重文件检查", False, "未找到权重文件")
            missing_files.append("model weights")
        
        if missing_files:
            self.log_test("文件完整性总结", False, f"缺失文件: {', '.join(missing_files)}")
            return False
        else:
            self.log_test("文件完整性总结", True, "所有必需文件都存在")
            return True
    
    def check_config(self):
        """检查模型配置"""
        print("\n" + "=" * 80)
        print("2. 检查模型配置")
        print("=" * 80)
        
        config_path = os.path.join(self.teacher_model_path, "config.json")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 关键配置项
            key_configs = {
                "model_type": config.get("model_type", "unknown"),
                "hidden_size": config.get("hidden_size", "unknown"),
                "num_hidden_layers": config.get("num_hidden_layers", "unknown"),
                "num_attention_heads": config.get("num_attention_heads", "unknown"),
                "vocab_size": config.get("vocab_size", "unknown"),
                "max_position_embeddings": config.get("max_position_embeddings", "unknown")
            }
            
            for key, value in key_configs.items():
                self.log_test(f"配置项: {key}", True, str(value))
            
            # 检查vocab_size是否与学生模型一致
            student_vocab_size = 151665  # 来自配置文件
            teacher_vocab_size = config.get("vocab_size", 0)
            
            if teacher_vocab_size == student_vocab_size:
                self.log_test("词汇表兼容性", True, 
                             f"教师模型vocab_size ({teacher_vocab_size}) 与学生模型一致")
            else:
                self.log_test("词汇表兼容性", False, 
                             f"教师模型vocab_size ({teacher_vocab_size}) 与学生模型 ({student_vocab_size}) 不一致",
                             "这可能导致tokenization问题")
            
            return True
            
        except Exception as e:
            self.log_test("配置检查", False, f"读取配置失败: {str(e)}")
            return False
    
    def test_tokenizer_loading(self):
        """测试tokenizer加载"""
        print("\n" + "=" * 80)
        print("3. 测试Tokenizer加载")
        print("=" * 80)
        
        try:
            print(f"正在加载tokenizer: {self.teacher_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.teacher_model_path,
                trust_remote_code=True
            )
            
            self.log_test("Tokenizer加载", True, "成功加载")
            
            # 测试tokenizer基本信息
            vocab_size = len(self.tokenizer)
            self.log_test("词汇表大小", True, f"{vocab_size:,} tokens")
            
            # 测试编码/解码
            test_texts = [
                "Hello, world!",
                "你好，世界！",
                "What is the meaning of life?",
                "Solve: 2+2="
            ]
            
            for text in test_texts:
                try:
                    encoded = self.tokenizer.encode(text)
                    decoded = self.tokenizer.decode(encoded)
                    
                    if text.strip() == decoded.strip():
                        self.log_test(f"编解码测试: '{text[:30]}...'", True, 
                                    f"编码长度: {len(encoded)} tokens")
                    else:
                        self.log_test(f"编解码测试: '{text[:30]}...'", False, 
                                    f"解码不匹配: 原文='{text}', 解码='{decoded}'")
                        
                except Exception as e:
                    self.log_test(f"编解码测试: '{text[:30]}...'", False, str(e))
            
            # 测试特殊token
            special_tokens = {
                "pad_token": self.tokenizer.pad_token,
                "eos_token": self.tokenizer.eos_token, 
                "bos_token": self.tokenizer.bos_token,
                "unk_token": self.tokenizer.unk_token
            }
            
            for token_name, token_value in special_tokens.items():
                if token_value is not None:
                    self.log_test(f"特殊token: {token_name}", True, f"'{token_value}'")
                else:
                    self.log_test(f"特殊token: {token_name}", False, "未定义")
            
            return True
            
        except Exception as e:
            self.log_test("Tokenizer加载", False, f"加载失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_model_loading(self):
        """测试模型加载"""
        print("\n" + "=" * 80)
        print("4. 测试模型权重加载")
        print("=" * 80)
        
        if not torch.cuda.is_available():
            self.log_test("CUDA可用性", False, "CUDA不可用，无法测试GPU加载")
            return False
        
        try:
            print(f"正在加载模型: {self.teacher_model_path}")
            print("使用半精度 (bf16) 加载...")
            
            # 先加载到CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.teacher_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 如果有CUDA，移动到GPU
            if torch.cuda.is_available():
                print("移动模型到GPU...")
                self.model = self.model.cuda()
            
            self.log_test("模型加载", True, "成功加载")
            
            # 检查模型参数
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.log_test("参数统计", True, 
                         f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
            
            # 检查模型设备
            device = next(self.model.parameters()).device
            self.log_test("模型设备", True, f"模型在设备: {device}")
            
            # 检查GPU内存使用
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                self.log_test("GPU内存使用", True, 
                             f"已分配: {memory_allocated:.2f} GB, 已保留: {memory_reserved:.2f} GB")
            
            # 测试模型eval模式
            self.model.eval()
            self.log_test("Eval模式设置", True, "成功设置为eval模式")
            
            return True
            
        except Exception as e:
            self.log_test("模型加载", False, f"加载失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_prompt_generation(self):
        """测试prompt生成功能"""
        print("\n" + "=" * 80)
        print("5. 测试Prompt生成功能")
        print("=" * 80)
        
        if self.model is None or self.tokenizer is None:
            self.log_test("生成测试", False, "模型或tokenizer未加载")
            return False
        
        # 测试prompts
        test_prompts = [
            "What is the meaning of life?",
            "Solve: 2+2=",
            "Explain quantum computing in simple terms.",
            "Write a short poem about technology."
        ]
        
        generation_params = [
            {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 50},
            {"temperature": 0.9, "top_p": 0.95, "max_new_tokens": 30},
        ]
        
        for i, prompt in enumerate(test_prompts):
            for j, params in enumerate(generation_params):
                test_name = f"生成测试 {i+1}.{j+1}: '{prompt[:20]}...'"
                
                try:
                    # Tokenize
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    
                    # Generate
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=params["max_new_tokens"],
                            temperature=params["temperature"],
                            top_p=params["top_p"],
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # Decode
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = generated_text[len(prompt):].strip()
                    
                    if response and len(response) > 0:
                        self.log_test(test_name, True, 
                                    f"生成 {len(response)} 字符, 温度={params['temperature']}",
                                    f"响应: '{response[:50]}...'")
                    else:
                        self.log_test(test_name, False, "生成为空或太短")
                        
                except Exception as e:
                    self.log_test(test_name, False, f"生成失败: {str(e)}")
        
        return True
    
    def test_batch_inference(self):
        """测试批量推理"""
        print("\n" + "=" * 80)
        print("6. 测试批量推理性能")
        print("=" * 80)
        
        if self.model is None or self.tokenizer is None:
            self.log_test("批量推理测试", False, "模型或tokenizer未加载")
            return False
        
        try:
            # 准备批量prompts
            batch_prompts = [
                "What is AI?",
                "Explain machine learning.",
                "Define deep learning.",
                "What is neural network?"
            ]
            
            # Tokenize批量
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            self.log_test("批量Tokenize", True, f"成功处理 {len(batch_prompts)} 个prompts")
            
            # 测试推理时间
            import time
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            inference_time = time.time() - start_time
            
            # 解码结果
            responses = []
            for i, output in enumerate(outputs):
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                responses.append(response)
            
            self.log_test("批量推理", True, 
                         f"完成 {len(batch_prompts)} 个推理, 耗时: {inference_time:.2f}s",
                         f"平均每个: {inference_time/len(batch_prompts):.2f}s")
            
            return True
            
        except Exception as e:
            self.log_test("批量推理测试", False, f"批量推理失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "=" * 80)
        print("测试报告总结")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests} ✓")
        print(f"失败: {failed_tests} ✗")
        print(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print("\n失败的测试:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  ✗ {result['test_name']}: {result['message']}")
                    if result['details']:
                        print(f"    {result['details']}")
        
        # 保存详细报告到文件
        report_path = "teacher_model_test_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "teacher_model_path": self.teacher_model_path,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": f"{passed_tests/total_tests*100:.1f}%",
                "test_results": self.test_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细报告已保存到: {report_path}")
        
        # 最终结论
        print("\n" + "=" * 80)
        if failed_tests == 0:
            print("✓ 教师模型测试全部通过！可以开始stage2_opd训练。")
        else:
            print("✗ 教师模型测试存在问题，请修复后再开始训练。")
        print("=" * 80)
        
        return failed_tests == 0
    
    def run_all_tests(self):
        """运行所有测试"""
        print("教师模型测试脚本")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"教师模型路径: {self.teacher_model_path}")
        
        try:
            # 运行测试
            success = True
            success &= self.check_file_integrity()
            success &= self.check_config()
            success &= self.test_tokenizer_loading()
            success &= self.test_model_loading()
            success &= self.test_prompt_generation()
            success &= self.test_batch_inference()
            
            # 生成报告
            final_success = self.generate_report()
            
            return 0 if final_success else 1
            
        except KeyboardInterrupt:
            print("\n\n测试被用户中断")
            return 1
        except Exception as e:
            print(f"\n\n测试过程中发生错误: {str(e)}")
            traceback.print_exc()
            return 1


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="教师模型测试脚本")
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        default="model_assets/teacher_qwen2.5_7b",
        help="教师模型路径"
    )
    
    args = parser.parse_args()
    
    # 创建测试器并运行测试
    tester = TeacherModelTester(args.teacher_model_path)
    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())