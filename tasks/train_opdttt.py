#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OPD-TTT 训练脚本

本脚本实现了 On-Policy Distillation Enhanced Test-Time Training (OPD-TTT)
的训练流程，将 In-Place TTT 与教师模型指导相结合。

主要功能：
1. 教师学生架构的训练
2. 四层损失函数：NTP对齐、教师表示对齐、KL散度、语言建模
3. 分块并行处理
4. 教师模型输出缓存
5. FSDP2 分布式训练支持

使用方法：
    torchrun --nproc_per_node=8 tasks/train_opdttt.py --config configs/opdttt/llama3_sc_500m_opdttt.yaml
"""

import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import wandb
from transformers import AutoConfig, AutoModelForCausalLM

# 强制使用 HuggingFace 后端
os.environ["MODELING_BACKEND"] = "hf"

# 禁用 expandable_segments（在 torch 导入前）
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
elif "expandable_segments" not in os.environ["PYTORCH_CUDA_ALLOC_CONF"]:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] += ",expandable_segments:False"

# 添加项目根目录到 Python 路径（用于导入本地模块）
_current_file_path = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_current_file_path))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 导入自定义模型
import hf_models.hf_llama  # noqa: F401

# VeOmni 框架导入
from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_chat_template,
    build_dataloader,
    build_dataset,
)
from veomni.data import data_transform as _data_transform
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    is_nccl_backend,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce

# 兼容不同的 VeOmni 版本
try:
    from veomni.utils.arguments import (
        DataArguments,
        ModelArguments,
        TrainingArguments,
        parse_args,
        save_args,
    )
except ImportError:
    from veomni.arguments import (
        DataArguments,
        ModelArguments,
        TrainingArguments,
        parse_args,
        save_args,
    )

logger = helper.create_logger(__name__)


@dataclass
class OPDTTTArguments:
    """
    OPD-TTT 训练的额外参数
    """

    # 教师模型配置
    teacher_model_path: str = field(
        default="", metadata={"help": "教师模型检查点路径"}
    )
    opdttt_layers: List[int] = field(
        default_factory=lambda: [0, 6, 12, 18],
        metadata={"help": "应用 OPD-TTT 的层索引"},
    )
    enable_teacher_cache: bool = field(
        default=True,
        metadata={"help": "启用教师输出缓存"},
    )

    # 损失权重
    lambda_kl: float = field(
        default=0.1,
        metadata={"help": "KL 散度损失权重"},
    )
    lambda_lm: float = field(
        default=1.0,
        metadata={"help": "语言建模损失权重"},
    )
    lambda_ntp: float = field(
        default=1.0,
        metadata={"help": "NTP 对齐损失权重"},
    )
    lambda_align_rep: float = field(
        default=0.5,
        metadata={"help": "教师表示对齐损失权重"},
    )

    # TTT/OPD-TTT 特定参数
    ttt_lr: float = field(
        default=0.3,
        metadata={"help": "快速权重学习率"},
    )
    ttt_chunk: int = field(
        default=4096,
        metadata={"help": "TTT 处理的分块大小"},
    )
    ttt_proj: bool = field(
        default=True,
        metadata={"help": "启用 NTP 目标投影"},
    )

    # 自适应权重和 PCA 初始化参数
    weight_adaptation: str = field(
        default="fixed",
        metadata={"help": "权重调整方式：'fixed' 使用固定权重，'adaptive' 根据梯度相似度动态调整"},
    )
    teacher_proj_init: str = field(
        default="random",
        metadata={"help": "教师投影矩阵初始化方式：'random' 随机初始化，'pca' 使用 PCA 初始化"},
    )
    teacher_embeddings_path: str = field(
        default="",
        metadata={"help": "用于 PCA 初始化的教师嵌入文件路径（.pt 格式）"},
    )

    # OPD采样参数（阶段2：On-Policy Distillation）
    enable_opd_sampling: bool = field(
        default=False,
        metadata={"help": "启用OPD on-policy采样（阶段2）"},
    )
    opd_disable_ttt: bool = field(
        default=True,
        metadata={"help": "OPD采样时禁用TTT（保持采样和训练一致）"},
    )
    opd_temperature: float = field(
        default=1.0,
        metadata={"help": "OPD采样温度"},
    )
    opd_top_p: float = field(
        default=0.9,
        metadata={"help": "OPD采样top-p（nucleus sampling）"},
    )
    opd_max_sample_length: int = field(
        default=2048,
        metadata={"help": "OPD最大采样长度"},
    )
    opd_num_trajectories: int = field(
        default=1,
        metadata={"help": "每个prompt的采样轨迹数"},
    )
    opd_prompt_field: str = field(
        default="prompt",
        metadata={"help": "OPD数据集中的prompt字段名"},
    )


@dataclass
class Arguments:
    """完整的训练参数集合"""
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)
    opdttt: "OPDTTTArguments" = field(default_factory=OPDTTTArguments)


class OPDTTTTrainer:
    """
    OPD-TTT 训练器

    该训练器处理：
    1. 教师学生前向传播
    2. 组合损失计算
    3. FSDP2 分布式训练
    4. 梯度检查点和混合精度
    """

    def __init__(
        self,
        student_model: nn.Module,          # 学生模型
        teacher_model: Optional[nn.Module], # 教师模型
        tokenizer,
        args: Arguments,
        train_dataloader,
        optimizer,
        lr_scheduler,
        checkpointer,
    ):
        """
        初始化 OPD-TTT 训练器

        Args:
            student_model: 学生模型
            teacher_model: 教师模型（可选）
            tokenizer: 分词器
            args: 训练参数
            train_dataloader: 训练数据加载器
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            checkpointer: 检查点管理器
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.args = args
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.checkpointer = checkpointer

        # 使用设备对象而不是模块，避免 tensor() 创建时的错误
        device_str = f"{get_device_type()}:{args.train.local_rank}"
        self.device = torch.device(device_str)
        self.parallel_state = get_parallel_state()

        # 设置激活卸载上下文
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            args.train.enable_activation_offload,
            args.train.enable_gradient_checkpointing,
            args.train.activation_gpu_limit,
        )

        # 为 DDP 模式添加 autocast 标志
        self.use_ddp_autocast = args.train.data_parallel_mode == "ddp" and args.train.enable_mixed_precision
        if self.use_ddp_autocast:
            logger.info("为 DDP 模式启用 autocast")

        # 教师缓存
        self.teacher_cache = {} if args.opdttt.enable_teacher_cache else None

    def compute_teacher_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        计算教师模型输出用于蒸馏

        Args:
            input_ids: 输入 Token ID
            attention_mask: 注意力掩码
            position_ids: 位置 ID

        Returns:
            教师输出字典，包含 logits、hidden_states、embeddings
        """
        if self.teacher_model is None:
            return None

        # 检查缓存
        cache_key = None
        if self.teacher_cache is not None:
            # 使用第一个batch item作为缓存键
            cache_key = tuple(input_ids[0].cpu().tolist())
            if cache_key in self.teacher_cache:
                return self.teacher_cache[cache_key]

        # 教师前向传播
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        result = {
            "logits": teacher_outputs.logits,
            "hidden_states": teacher_outputs.hidden_states,
            "embeddings": self.teacher_model.get_input_embeddings()(input_ids),
        }

        # 缓存结果
        if self.teacher_cache is not None and cache_key is not None:
            self.teacher_cache[cache_key] = result
            if len(self.teacher_cache) > 100:
                oldest_key = next(iter(self.teacher_cache))
                del self.teacher_cache[oldest_key]

        return result

    def sample_from_student(
        self,
        prompts: Dict[str, torch.Tensor],
        num_trajectories: int = 1,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        从学生模型采样轨迹（OPD核心步骤）

        注意：TTT快速权重会在采样过程中更新以适应prompt，
        但这些更新不参与梯度计算（TTT作为inference时适应机制）

        Args:
            prompts: 包含prompt的批次数据
            num_trajectories: 每个prompt的采样轨迹数

        Returns:
            采样轨迹列表，每个轨迹包含input_ids和采样logprobs
        """
        self.student_model.eval()
        trajectories = []

        prompt_ids = prompts["input_ids"].to(self.device)
        attention_mask = prompts.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        batch_size = prompt_ids.shape[0]

        for i in range(num_trajectories):
            for b in range(batch_size):
                # 单个prompt采样
                single_prompt = {
                    "input_ids": prompt_ids[b:b+1],
                    "attention_mask": attention_mask[b:b+1] if attention_mask is not None else None,
                }

                with torch.no_grad():
                    # 使用学生模型的generate方法进行采样
                    # 注意：这里TTT快速权重会更新以适应prompt
                    # 但这些更新不参与梯度计算（TTT作为inference适应机制）
                    # 在后续重新计算logprobs时，TTT会重新从头开始
                    sampled_ids = []
                    sampled_logprobs = []
                    current_input = single_prompt["input_ids"]

                    max_length = min(
                        single_prompt["input_ids"].shape[1] + self.args.opdttt.opd_max_sample_length,
                        self.args.data.max_seq_len,
                    )

                    # 自回归采样
                    for step in range(max_length - single_prompt["input_ids"].shape[1]):
                        # 学生前向传播获取logits（TTT快速权重会更新）
                        outputs = self.student_model(
                            input_ids=current_input,
                            attention_mask=single_prompt["attention_mask"],
                            use_cache=False,
                        )

                        next_token_logits = outputs.logits[:, -1, :]  # [1, vocab_size]

                        # 应用温度和top-p采样
                        if self.args.opdttt.opd_temperature > 0:
                            next_token_logits = next_token_logits / self.args.opdttt.opd_temperature

                        # 计算logprobs（仅用于采样，不保留梯度）
                        next_token_logprobs = F.log_softmax(next_token_logits, dim=-1)

                        # Top-p (nucleus) sampling
                        if self.args.opdttt.opd_top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > self.args.opdttt.opd_top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            next_token_logits[:, indices_to_remove] = float('-inf')

                        # 采样下一个token
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
                        next_token_logprob = next_token_logprobs.gather(-1, next_token).squeeze(-1)

                        sampled_ids.append(next_token)
                        sampled_logprobs.append(next_token_logprob)

                        # 检查是否生成EOS
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break

                        # 更新输入
                        current_input = torch.cat([current_input, next_token], dim=-1)

                        # 更新attention mask
                        if single_prompt["attention_mask"] is not None:
                            single_prompt["attention_mask"] = torch.cat(
                                [single_prompt["attention_mask"], torch.ones_like(next_token)], dim=-1
                            )

                # 构建完整轨迹
                full_input_ids = torch.cat([single_prompt["input_ids"], torch.cat(sampled_ids, dim=-1)], dim=-1)
                full_logprobs = torch.cat(sampled_logprobs, dim=-1) if sampled_logprobs else None

                trajectories.append({
                    "input_ids": full_input_ids,
                    "sampled_logprobs": full_logprobs,  # 采样时的logprobs（no_grad）
                    "prompt_length": single_prompt["input_ids"].shape[1],
                })

        self.student_model.train()
        return trajectories

    def compute_teacher_logprobs_on_sampled(
        self,
        trajectories: List[Dict[str, torch.Tensor]],
    ) -> List[torch.Tensor]:
        """
        计算教师在学生采样token上的logprobs（OPD核心步骤）

        Args:
            trajectories: 学生采样轨迹列表

        Returns:
            教师logprobs列表，与轨迹对应
        """
        if self.teacher_model is None:
            return None

        teacher_logprobs_list = []

        for trajectory in trajectories:
            input_ids = trajectory["input_ids"].to(self.device)
            prompt_length = trajectory["prompt_length"]

            # 只需要计算采样部分的教师logprobs
            if input_ids.shape[1] <= prompt_length:
                continue

            sampled_ids = input_ids[:, prompt_length:]

            with torch.no_grad():
                # 教师前向传播
                outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                )

                # 获取采样位置的logits
                sampled_logits = outputs.logits[:, prompt_length-1:-1, :]  # [1, sampled_length-1, vocab_size]

                # 计算教师logprobs
                teacher_logprobs = F.log_softmax(sampled_logits, dim=-1)

                # 提取实际采样token的logprob
                # teacher_logprobs: [1, sampled_length-1, vocab_size]
                # sampled_ids[1:]: [1, sampled_length-1] (去掉第一个生成的token，因为它的logprob来自prompt)
                sampled_tokens = sampled_ids[:, 1:] if sampled_ids.shape[1] > 1 else sampled_ids
                if sampled_tokens.shape[1] > 0:
                    sampled_tokens_reshaped = sampled_tokens.unsqueeze(-1)  # [1, sampled_length-1, 1]
                    # 将teacher_logprobs调整到匹配维度
                    if teacher_logprobs.shape[1] != sampled_tokens.shape[1]:
                        # 处理维度不匹配的情况
                        min_len = min(teacher_logprobs.shape[1], sampled_tokens.shape[1])
                        teacher_logprobs = teacher_logprobs[:, :min_len, :]
                        sampled_tokens_reshaped = sampled_tokens_reshaped[:, :min_len, :]

                    teacher_logprobs_on_sampled = teacher_logprobs.gather(-1, sampled_tokens_reshaped).squeeze(-1)
                    teacher_logprobs_list.append(teacher_logprobs_on_sampled)

        return teacher_logprobs_list

    def compute_importance_sampling_loss(
        self,
        trajectories: List[Dict[str, torch.Tensor]],
        teacher_logprobs_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        计算重要性采样损失（OPD核心）

        正确的实现：
        1. reverse KL作为advantage，必须detach（reward信号）
        2. 重要性权重rho：分子保留梯度，分母detach
           rho = exp(student_logprob - student_logprob.detach())
           在OPD中，因为theta_old = theta，所以rho = 1，但梯度保留

        Args:
            trajectories: 学生采样轨迹
            teacher_logprobs_list: 教师在采样token上的logprobs

        Returns:
            重要性采样损失
        """
        total_loss = 0.0
        total_tokens = 0

        for trajectory, teacher_logprobs in zip(trajectories, teacher_logprobs_list):
            if teacher_logprobs is None:
                continue

            # 学生的采样logprobs（需要保留梯度）
            student_logprobs = trajectory["sampled_logprobs"]

            if student_logprobs is None or teacher_logprobs is None:
                continue

            # 对齐维度
            min_len = min(student_logprobs.shape[0], teacher_logprobs.shape[1])
            if min_len == 0:
                continue

            student_logprobs = student_logprobs[:min_len]
            teacher_logprobs = teacher_logprobs[0, :min_len]

            # 计算reverse KL作为advantage（必须detach，作为reward信号）
            # reverse_KL = student_logprob - teacher_logprob
            # advantage = -reverse_KL = teacher_logprob - student_logprob
            reverse_kl = (student_logprobs.detach() - teacher_logprobs.detach())
            advantages = -reverse_kl  # = teacher_logprob - student_logprob

            # 重要性采样权重（关键：分子保留梯度，分母detach）
            # 在标准RL中：rho = pi_theta(a|s) / pi_theta_old(a|s)
            # 在OPD中：theta_old = theta，所以：
            # rho = exp(logprob) / exp(logprob.detach()) = exp(logprob - logprob.detach())
            # 数值上rho = 1，但梯度保留在分子中
            rho = torch.exp(student_logprobs - student_logprobs.detach())

            # 重要性采样策略梯度损失
            # loss = -rho * advantage
            # 这确保梯度能正确传播到student_logprobs
            loss_per_token = -rho * advantages

            total_loss += loss_per_token.sum()
            total_tokens += min_len

        if total_tokens > 0:
            return total_loss / total_tokens
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def train_step_opd(
        self,
        prompts_batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        OPD模式的训练步骤（真正的on-policy distillation）

        正确的两阶段流程：
        1. 采样阶段（no_grad）：从学生模型采样轨迹
        2. 训练阶段（保留梯度）：
           a. 重新计算采样token的logprobs（保留梯度）
           b. 计算教师logprobs（detach）
           c. 计算reverse KL作为advantage（detach）
           d. 使用重要性采样损失进行梯度更新

        Args:
            prompts_batch: 包含prompt的批次数据

        Returns:
            该批次的损失值
        """
        # 准备批次数据
        prompts_batch = {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in prompts_batch.items()
        }

        # 阶段1: 采样（no_grad）- 获取轨迹
        trajectories = self.sample_from_student(
            prompts=prompts_batch,
            num_trajectories=self.args.opdttt.opd_num_trajectories,
        )

        if not trajectories:
            return 0.0

        # 阶段2: 训练（保留梯度）
        # 2a. 重新计算采样token的logprobs（一次前向传播，保留梯度！）
        # 注意：这里使用一次前向传播计算整个序列的logprobs，避免重复自回归
        # TTT更新可能会有微小差异，但避免了巨大的计算开销
        for trajectory in trajectories:
            input_ids = trajectory["input_ids"]
            prompt_length = trajectory["prompt_length"]

            if input_ids.shape[1] <= prompt_length:
                trajectory["student_logprobs_with_grad"] = None
                continue

            # 重新计算学生logprobs（一次前向传播，保留梯度！）
            outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )

            # 获取采样位置的logits
            # 注意：采样位置的第一个token（prompt后第一个）的logprob在position prompt_length-1
            sampled_logits = outputs.logits[:, prompt_length-1:-1, :]
            sampled_logprobs = F.log_softmax(sampled_logits, dim=-1)

            # 提取采样token的logprob
            # 采样token从prompt_length+1开始（因为我们预测下一个token）
            sampled_ids = input_ids[:, prompt_length+1:] if input_ids.shape[1] > prompt_length + 1 else input_ids[:, prompt_length:]

            if sampled_ids.shape[1] > 0 and sampled_logprobs.shape[1] == sampled_ids.shape[1]:
                sampled_ids_reshaped = sampled_ids.unsqueeze(-1)
                student_logprobs_with_grad = sampled_logprobs.gather(-1, sampled_ids_reshaped).squeeze(-1)
                trajectory["student_logprobs_with_grad"] = student_logprobs_with_grad
            else:
                trajectory["student_logprobs_with_grad"] = None

        # 2b. 计算教师logprobs（detach）
        teacher_logprobs_list = self.compute_teacher_logprobs_on_sampled(trajectories)

        # 2c & 2d. 计算重要性采样损失（使用有梯度的学生logprobs）
        loss = self.compute_importance_sampling_loss_with_grad(
            trajectories,
            teacher_logprobs_list,
        )

        # 反向传播
        with self.model_bwd_context:
            loss.backward()

        return loss.item()

    def compute_importance_sampling_loss_with_grad(
        self,
        trajectories: List[Dict[str, torch.Tensor]],
        teacher_logprobs_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        计算重要性采样损失（使用重新计算的、带梯度的logprobs）

        Args:
            trajectories: 学生采样轨迹（包含重新计算的logprobs）
            teacher_logprobs_list: 教师在采样token上的logprobs

        Returns:
            重要性采样损失
        """
        total_loss = 0.0
        total_tokens = 0

        for trajectory, teacher_logprobs in zip(trajectories, teacher_logprobs_list):
            if teacher_logprobs is None:
                continue

            # 使用重新计算的、带梯度的学生logprobs
            student_logprobs = trajectory.get("student_logprobs_with_grad")

            if student_logprobs is None or teacher_logprobs is None:
                continue

            # student_logprobs形状: [sampled_length]
            # teacher_logprobs形状: [1, sampled_length]

            # 对齐维度
            min_len = min(student_logprobs.shape[0], teacher_logprobs.shape[1])
            if min_len == 0:
                continue

            student_logprobs = student_logprobs[:min_len]
            teacher_logprobs = teacher_logprobs[0, :min_len]

            # 计算reverse KL作为advantage（必须detach）
            # reverse_KL = student_logprob - teacher_logprob
            # advantage = -reverse_KL = teacher_logprob - student_logprob
            reverse_kl = (student_logprobs.detach() - teacher_logprobs.detach())
            advantages = -reverse_kl

            # 重要性采样权重（分子保留梯度，分母detach）
            # rho = exp(student_logprob - student_logprob.detach())
            # 在OPD中theta_old=theta，所以rho数值上=1，但梯度保留
            rho = torch.exp(student_logprobs - student_logprobs.detach())

            # 重要性采样策略梯度损失
            # loss = -rho * advantage
            loss_per_token = -rho * advantages

            total_loss += loss_per_token.sum()
            total_tokens += min_len

        if total_tokens > 0:
            return total_loss / total_tokens
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def train_step(
        self,
        micro_batch: Dict[str, torch.Tensor],
        length_in_batch: torch.Tensor,
    ) -> float:
        """
        执行单个训练步骤

        Args:
            micro_batch: 小批量数据
            length_in_batch: 批次中的有效 Token 数量

        Returns:
            该小批量的损失值
        """
        # 准备批次数据
        micro_batch = {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in micro_batch.items()
        }

        # 计算教师输出（如果使用教师）
        teacher_outputs = None
        if self.teacher_model is not None:
            teacher_outputs = self.compute_teacher_outputs(
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch.get("attention_mask"),
                position_ids=micro_batch.get("position_ids"),
            )

        # 学生前向传播（带教师指导）
        # 当 lambda_align_rep 为 0 时，不传递教师表示以避免维度不匹配
        # 当 lambda_kl 为 0 时，不传递教师 logits 以避免 vocab_size 不匹配
        use_teacher_rep = self.args.opdttt.lambda_align_rep > 0
        use_teacher_logits = self.args.opdttt.lambda_kl > 0

        # 使用 autocast 上下文（如果启用）
        if self.use_ddp_autocast:
            with self.model_fwd_context:
                student_outputs = self.student_model(
                    input_ids=micro_batch["input_ids"],
                    attention_mask=micro_batch.get("attention_mask"),
                    position_ids=micro_batch.get("position_ids"),
                    labels=micro_batch["labels"],
                    teacher_logits=teacher_outputs["logits"] if teacher_outputs and use_teacher_logits else None,
                    teacher_hidden_states=teacher_outputs["hidden_states"] if teacher_outputs and use_teacher_rep else None,
                    teacher_embeddings=teacher_outputs["embeddings"] if teacher_outputs and use_teacher_rep else None,
                    use_cache=False,
                )
        else:
            student_outputs = self.student_model(
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch.get("attention_mask"),
                position_ids=micro_batch.get("position_ids"),
                labels=micro_batch["labels"],
                teacher_logits=teacher_outputs["logits"] if teacher_outputs and use_teacher_logits else None,
                teacher_hidden_states=teacher_outputs["hidden_states"] if teacher_outputs and use_teacher_rep else None,
                teacher_embeddings=teacher_outputs["embeddings"] if teacher_outputs and use_teacher_rep else None,
                use_cache=False,
            )

        # 计算加权损失
        length_in_micro_batch = torch.sum(micro_batch["labels"] != -100)
        loss = (
            student_outputs.loss
            * length_in_micro_batch
            / length_in_batch
            * self.parallel_state.dp_size
        )

        # 反向传播
        with self.model_bwd_context:
            loss.backward()

        return loss.item()

    def train(self, train_steps: int, start_step: int = 0):
        """
        主训练循环

        Args:
            train_steps: 总训练步数
            start_step: 起始步数（用于恢复训练）
        """
        global_step = start_step
        last_ckpt_path = None

        self.student_model.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()

        # 检查是否启用OPD模式
        enable_opd = self.args.opdttt.enable_opd_sampling
        if enable_opd:
            logger.info("========== OPD模式：On-Policy Distillation ==========")
            logger.info(f"采样温度: {self.args.opdttt.opd_temperature}")
            logger.info(f"Top-p: {self.args.opdttt.opd_top_p}")
            logger.info(f"最大采样长度: {self.args.opdttt.opd_max_sample_length}")
            logger.info(f"轨迹数: {self.args.opdttt.opd_num_trajectories}")
        else:
            logger.info("========== 标准模式：Teacher-Student训练 ==========")

        logger.info(
            f"开始训练: 步数={train_steps}, "
            f"world_size={self.parallel_state.world_size}"
        )

        for epoch in range(self.args.train.num_train_epochs):
            if hasattr(self.train_dataloader, "set_epoch"):
                self.train_dataloader.set_epoch(epoch)

            data_loader_tqdm = trange(
                train_steps,
                desc=f"Epoch {epoch + 1}/{self.args.train.num_train_epochs}",
                total=train_steps,
                initial=start_step,
                disable=self.args.train.local_rank != 0,
            )
            data_iterator = iter(self.train_dataloader)

            for _ in range(start_step, train_steps):
                global_step += 1
                synchronize()
                start_time = time.time()

                try:
                    micro_batches: List[Dict[str, Any]] = next(data_iterator)
                except StopIteration:
                    logger.info(f"数据加载器在步骤 {global_step} 完成")
                    break

                # 计算批次统计（OPD模式不需要）
                if not enable_opd:
                    length_in_batch = torch.tensor(0, dtype=torch.int32, device=self.device)
                    for micro_batch in micro_batches:
                        length_in_batch += torch.sum(micro_batch["labels"] != -100)
                    length_in_batch = all_reduce(
                        length_in_batch, op="sum", group=self.parallel_state.fsdp_group
                    )

                # 处理小批量
                total_loss = 0.0
                if enable_opd:
                    # OPD模式：从学生采样并使用重要性采样
                    for micro_batch in micro_batches:
                        loss = self.train_step_opd(prompts_batch=micro_batch)
                        total_loss += loss
                        del micro_batch
                else:
                    # 标准模式：使用训练数据的teacher-student训练
                    for micro_batch in micro_batches:
                        loss = self.train_step(micro_batch, length_in_batch)
                        total_loss += loss
                        del micro_batch

                # 梯度裁剪和优化
                grad_norm = veomni_clip_grad_norm(
                    self.student_model, self.args.train.max_grad_norm
                )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if hasattr(grad_norm, "full_tensor"):
                    grad_norm = grad_norm.full_tensor().item()

                # 收集指标
                total_loss, grad_norm = all_reduce(
                    (total_loss, grad_norm), group=self.parallel_state.fsdp_group
                )
                synchronize()
                delta_time = time.time() - start_time
                lr = max(self.lr_scheduler.get_last_lr())

                # 更新进度条
                data_loader_tqdm.set_postfix_str(
                    f"loss: {total_loss:.4f}, grad_norm: {grad_norm:.4f}, lr: {lr:.2e}",
                    refresh=False,
                )
                data_loader_tqdm.update()

                # 日志记录
                if self.args.train.global_rank == 0 and self.args.train.use_wandb:
                    wandb.log(
                        {
                            "training/loss": total_loss,
                            "training/perplexity": math.exp(total_loss),
                            "training/grad_norm": grad_norm,
                            "training/lr": lr,
                        },
                        step=global_step,
                    )

                # 保存检查点
                if (
                    self.args.train.save_steps
                    and global_step % self.args.train.save_steps == 0
                ):
                    last_ckpt_path = self._save_checkpoint(global_step, last_ckpt_path)

            start_step = 0

        # 最终检查点
        if self.args.train.global_rank == 0:
            self._save_hf_weights(global_step, last_ckpt_path)

    def _save_checkpoint(self, global_step: int, last_ckpt_path: Optional[str]):
        """保存分布式检查点"""
        helper.empty_cache()
        save_path = os.path.join(
            self.args.train.save_checkpoint_path, f"global_step_{global_step}"
        )

        state = {
            "model": self.student_model,
            "optimizer": self.optimizer,
            "extra_state": {
                "global_step": global_step,
                "lr_scheduler": self.lr_scheduler.state_dict(),
            },
        }

        self.checkpointer.save(save_path, state, global_steps=global_step)
        dist.barrier()

        # DCP checkpointer 创建嵌套目录结构：save_path/global_step_X
        # 更新 last_ckpt_path 为实际保存的检查点路径
        actual_ckpt_path = os.path.join(save_path, f"global_step_{global_step}")
        logger.info(f"检查点已保存: {actual_ckpt_path}")

        # 轮转旧检查点
        if last_ckpt_path is not None and os.path.isdir(last_ckpt_path):
            shutil.rmtree(last_ckpt_path, ignore_errors=True)
            logger.info(f"删除旧检查点: {last_ckpt_path}")

        return actual_ckpt_path

    def _save_hf_weights(self, global_step: int, last_ckpt_path: Optional[str]):
        """保存 HuggingFace 格式权重"""
        save_path = os.path.join(
            self.args.train.save_checkpoint_path, f"global_step_{global_step}", "hf_ckpt"
        )

        # 在保存HuggingFace权重前清空缓存，避免OOM
        # 因为ckpt_to_state_dict会聚合所有分片参数到一起
        helper.empty_cache()
        synchronize()

        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=last_ckpt_path,
            ckpt_manager=self.args.train.ckpt_manager,
        )

        save_model_weights(save_path, model_state_dict, model_assets=None)
        logger.info(f"HuggingFace 权重已保存: {save_path}")


def build_teacher_model(
    teacher_path: str,
    device: torch.device,
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    tokenizer=None,
) -> Optional[nn.Module]:
    """
    构建教师模型用于蒸馏

    Args:
        teacher_path: 教师模型路径
        device: 设备
        torch_dtype: 数据类型
        attn_implementation: 注意力实现类型
        tokenizer: 统一的 tokenizer（如果提供，将检查 vocab_size 一致性）

    Returns:
        教师模型（如果路径有效），否则返回 None
    """
    # 处理空字符串和字面量 "" 的情况
    teacher_path = teacher_path.strip() if teacher_path else ""
    if teacher_path in ['""', "''", '']:
        return None

    logger.info(f"加载教师模型: {teacher_path}")
    config = AutoConfig.from_pretrained(teacher_path)

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        # 不使用 device_map 以避免依赖 accelerate，手动移动到设备
    )
    # 手动移动模型到指定设备
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # 检查 tokenizer vocab_size 一致性
    if tokenizer is not None:
        unified_vocab_size = len(tokenizer)
        teacher_vocab_size = config.vocab_size

        if unified_vocab_size != teacher_vocab_size:
            logger.warning(
                f"教师模型 vocab_size ({teacher_vocab_size}) 与 "
                f"统一 tokenizer vocab_size ({unified_vocab_size}) 不匹配。"
            )
            logger.warning(
                f"请确保使用相同的 tokenizer。"
                f"建议: bash scripts/setup_teacher.sh --skip-tokenizer <model>"
            )
        else:
            logger.info(f"✓ 教师 vocab_size 与统一 tokenizer 一致: {unified_vocab_size}")

    logger.info(f"教师模型已加载: {config.num_hidden_layers} 层, "
                f"{config.num_attention_heads} 头, hidden_size={config.hidden_size}")

    return teacher_model


def main():
    """主训练函数"""
    # 初始化进程组
    nccl_timeout = os.getenv("NCCL_TIMEOUT", None)
    pg_timeout = None
    if nccl_timeout is not None and is_nccl_backend():
        pg_timeout = timedelta(seconds=int(nccl_timeout))

    dist.init_process_group(backend=get_dist_comm_backend(), timeout=pg_timeout)

    # 解析参数
    args = parse_args(Arguments)
    logger.info(f"进程 rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(vars(args), indent=2, default=str))

    # 设置设备
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)

    if args.train.local_rank == 0:
        helper.enable_third_party_logging()
        save_args(args, args.train.output_dir)

    # 初始化并行状态
    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    # 构建分词器
    logger.info_rank0("构建分词器")
    tokenizer = build_tokenizer(args.model.tokenizer_path)

    # 准备数据
    logger.info_rank0("准备数据")
    transform = partial(
        _data_transform.process_plaintext_example,
        tokenizer=tokenizer,
        max_seq_len=args.data.max_seq_len,
        text_keys=args.data.text_keys,
    )

    train_dataset = build_dataset(
        transform=transform,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        **vars(args.data),
    )

    dataset_length = len(train_dataset) if hasattr(train_dataset, "__len__") else None
    if dataset_length is not None:
        dataset_length = dataset_length / args.train.data_parallel_size

    train_steps = args.train.max_steps or math.ceil(
        args.data.train_size / (args.train.global_batch_size * args.data.max_seq_len)
    )

    # 构建 collate_fn_kwargs，启用 pad_to_length 确保序列长度一致
    collate_fn_kwargs = {
        "pad_to_length": args.data.max_seq_len,  # 将序列 padding 到 max_seq_len
    }

    train_dataloader = build_dataloader(
        dataloader_type=args.data.dataloader.type,
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        max_seq_len=args.data.max_seq_len,
        train_steps=train_steps,
        # 注意: native dataloader 不支持 rmpad 和 rmpad_with_pos_ids 参数
        dyn_bsz_buffer_size=getattr(args.data, "dyn_bsz_buffer_size", 500),
        dyn_bsz=getattr(args.train, "dyn_bsz", True),
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        collate_fn_kwargs=collate_fn_kwargs,
    )

    # 构建学生模型
    logger.info_rank0("构建学生模型")
    from hf_models.hf_llama import OPDTTTForCausalLM

    config_path = args.model.config_path or args.model.model_path
    config = AutoConfig.from_pretrained(config_path)

    # 应用 OPD-TTT 设置
    config.opdttt_mode = True
    config.opdttt_layers = args.opdttt.opdttt_layers
    config.lambda_kl = args.opdttt.lambda_kl
    config.lambda_lm = args.opdttt.lambda_lm
    config.lambda_ntp = args.opdttt.lambda_ntp
    config.lambda_align_rep = args.opdttt.lambda_align_rep
    config.ttt_lr = args.opdttt.ttt_lr
    config.ttt_chunk = args.opdttt.ttt_chunk
    config.ttt_proj = args.opdttt.ttt_proj
    config.ttt_target = "input_embed"

    # 新增：自适应权重和 PCA 初始化设置
    config.weight_adaptation = args.opdttt.weight_adaptation
    config.teacher_proj_init = args.opdttt.teacher_proj_init

    # 如果使用 PCA 初始化，加载教师嵌入
    teacher_embeddings_for_init = None
    if args.opdttt.teacher_proj_init == "pca" and args.opdttt.teacher_embeddings_path:
        logger.info_rank0(f"加载教师嵌入用于 PCA 初始化：{args.opdttt.teacher_embeddings_path}")
        teacher_embeddings_for_init = torch.load(args.opdttt.teacher_embeddings_path)
        config.teacher_embeddings_for_init = teacher_embeddings_for_init

    # 自动从教师模型配置读取 teacher_hidden_size
    # 处理空字符串和字面量 "" 的情况
    teacher_path = args.opdttt.teacher_model_path.strip() if args.opdttt.teacher_model_path else ""
    # 移除可能的引号包裹
    if teacher_path in ['""', "''"]:
        teacher_path = ""

    if teacher_path:
        from transformers import AutoConfig as AutoConfigTeacher
        try:
            teacher_config = AutoConfigTeacher.from_pretrained(teacher_path)
            config.teacher_hidden_size = teacher_config.hidden_size
            logger.info_rank0(f"从教师配置读取 teacher_hidden_size: {config.teacher_hidden_size}")
        except Exception as e:
            logger.info_rank0(f"无法从教师配置读取 teacher_hidden_size: {e}，使用学生模型 hidden_size: {config.hidden_size}")
            config.teacher_hidden_size = config.hidden_size
    else:
        # 如果没有教师模型，使用学生模型的 hidden_size
        config.teacher_hidden_size = config.hidden_size
        logger.info_rank0(f"没有配置教师模型，使用学生模型 hidden_size 作为 teacher_hidden_size: {config.teacher_hidden_size}")

    # 检查是否存在模型权重，如果不存在则从配置初始化
    model_path = args.model.model_path
    has_weights = any(
        f.endswith(('.safetensors', '.bin', '.pt'))
        for f in os.listdir(model_path)
        if os.path.isfile(os.path.join(model_path, f))
    ) if os.path.isdir(model_path) else False

    if has_weights:
        logger.info_rank0(f"从预训练权重加载模型: {model_path}")
        student_model = OPDTTTForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype="bfloat16" if args.train.enable_mixed_precision else "float32",
            attn_implementation=args.model.attn_implementation,
        )
    else:
        logger.info_rank0(f"从配置初始化模型 (训练从开始): {config_path}")
        student_model = OPDTTTForCausalLM._from_config(
            config,
            torch_dtype="bfloat16" if args.train.enable_mixed_precision else "float32",
            attn_implementation=args.model.attn_implementation,
        )

    # 对于从配置初始化的模型，需要先移动到目标设备
    # build_parallelize_model 在 DDP 模式下不会自动移动设备
    from veomni.distributed.parallel_state import get_parallel_state
    parallel_state = get_parallel_state()
    if not has_weights and parallel_state.dp_mode == "ddp":
        device = torch.device(f"cuda:{parallel_state.local_rank}")
        logger.info_rank0(f"将模型移动到设备: {device}")
        student_model = student_model.to(device)

    # 并行化模型
    # 注意：当从配置初始化时（has_weights=False），weights_path 应为 None
    student_model = build_parallelize_model(
        student_model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path if has_weights else None,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=student_model._no_split_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
    )

    helper.print_device_mem_info("构建学生模型后的显存使用")

    # 构建教师模型（仅在 rank 0 且有教师路径时）
    teacher_model = None
    # 检查 teacher_model_path 是否非空且不是仅包含空格
    teacher_path = args.opdttt.teacher_model_path.strip() if args.opdttt.teacher_model_path else ""
    if teacher_path and args.train.global_rank == 0:
        # 使用设备字符串而不是模块，避免 pickle 错误
        device_str = f"{get_device_type()}:{args.train.local_rank}"
        teacher_model = build_teacher_model(
            teacher_path,
            device_str,  # 使用设备字符串
            torch_dtype="bfloat16" if args.train.enable_mixed_precision else "float32",
            attn_implementation=args.model.attn_implementation,
            tokenizer=tokenizer,  # 传递统一 tokenizer 用于检查 vocab_size 一致性
        )
        logger.info_rank0(f"教师模型已加载: {teacher_path}")
    else:
        logger.info_rank0("未配置教师模型，使用无教师模式训练")

    # 构建优化器和调度器
    optimizer = build_optimizer(
        student_model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    # 构建检查点管理器
    checkpointer = build_checkpointer(
        dist_backend=args.train.data_parallel_mode,
        ckpt_manager=args.train.ckpt_manager,
    )

    # 初始化 wandb
    if args.train.global_rank == 0 and args.train.use_wandb:
        wandb.init(
            project=args.train.wandb_project,
            name=args.train.wandb_name,
            config={**vars(args.model), **vars(args.data), **vars(args.train), **vars(args.opdttt)},
        )

    # 创建训练器并开始训练
    trainer = OPDTTTTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        args=args,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpointer=checkpointer,
    )

    trainer.train(train_steps=train_steps)

    # 清理
    synchronize()
    del optimizer, lr_scheduler
    helper.empty_cache()

    dist.barrier()
    dist.destroy_process_group()

    logger.info("训练完成!")


if __name__ == "__main__":
    main()
