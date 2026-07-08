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
from typing import Any, Dict, List, Optional, Tuple, Union

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

    # 模型类型
    model_type: str = field(
        default="llama",
        metadata={"help": "模型类型: llama, qwen3, 或 qwen3_5"},
    )
    sliding_window: int = field(
        default=0,
        metadata={"help": "SWA窗口大小（0=不启用，论文500M=2048）"},
    )

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
        default=1024,
        metadata={"help": "TTT 处理的分块大小"},
    )
    ttt_proj: bool = field(
        default=True,
        metadata={"help": "启用 NTP 目标投影"},
    )
    ttt_max_norm: float = field(
        default=1e-5,
        metadata={"help": "快速权重更新的 Frobenius 范数裁剪"},
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
    # 已移除 opd_disable_ttt 参数，因为禁用TTT会导致训练-推理不一致
    # 采样和训练时都应该使用TTT以保持一致性
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

    # 检查点策略
    keep_all_checkpoints: bool = field(
        default=True,
        metadata={"help": "True=保留所有checkpoint，False=只保留最新的（删除旧的）"},
    )
    save_best: bool = field(
        default=True,
        metadata={"help": "保存loss最低的checkpoint为best"},
    )


@dataclass
class Arguments:
    """完整的训练参数集合"""
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)
    opdttt: "OPDTTTArguments" = field(default_factory=OPDTTTArguments)


def _get_model_class(model_type: str = "llama"):
    """根据 model_type 返回对应的 OPD-TTT 模型类"""
    if model_type == "qwen3":
        from hf_models.hf_qwen3 import OPDQwen3ForCausalLM
        return OPDQwen3ForCausalLM
    elif model_type == "qwen3_5":
        from hf_models.hf_qwen3_5 import OPDQwen3_5ForCausalLM
        return OPDQwen3_5ForCausalLM
    else:
        from hf_models.hf_llama import OPDTTTForCausalLM
        return OPDTTTForCausalLM


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
        self.model_config = student_model.config if hasattr(student_model, 'config') else None

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
        # DDP 模式下模型为 bf16，autocast 确保 forward 一致性
        # FSDP2 多 GPU 由 MixedPrecision 自动处理，无需手动 autocast
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

    def _build_sampling_model(self):
        if hasattr(self, '_sampling_model') and self._sampling_model is not None:
            return self._sampling_model
        ModelClass = _get_model_class(self.args.opdttt.model_type)
        config_path = self.args.model.config_path or self.args.model.model_path
        model_path = self.args.model.model_path

        if model_path and os.path.isdir(model_path):
            sampling_model = ModelClass.from_pretrained(
                model_path,
                config=AutoConfig.from_pretrained(config_path),
                torch_dtype=torch.bfloat16,
                ignore_mismatched_sizes=True,
            )
        else:
            sampling_model = ModelClass.from_pretrained(
                config_path,
                torch_dtype=torch.bfloat16,
                ignore_mismatched_sizes=True,
            )

        sampling_model = sampling_model.to(self.device)
        sampling_model.eval()
        self._sampling_model = sampling_model
        self._update_sampling_model()
        return sampling_model

    def _update_sampling_model(self):
        if not hasattr(self, '_sampling_model') or self._sampling_model is None:
            return
        fsdp_params = dict(self.student_model.named_parameters())
        for name, param in self._sampling_model.named_parameters():
            if name in fsdp_params:
                src = fsdp_params[name]
                if src.shape == param.shape:
                    full = src.full_tensor() if hasattr(src, 'full_tensor') else src
                    param.data.copy_(full.detach())

    def sample_from_student(
        self,
        prompts: Dict[str, torch.Tensor],
        num_trajectories: int = 1,
    ) -> List[Dict[str, torch.Tensor]]:
        self.student_model.eval()
        trajectories = []

        sampling_model = self._build_sampling_model()
        self._update_sampling_model()

        input_ids = prompts["input_ids"].to(self.device)
        labels = prompts.get("labels")
        if labels is not None:
            labels = labels.to(self.device)

        batch_size = input_ids.shape[0]
        eos_token_id = self.tokenizer.eos_token_id
        max_sample_length = self.args.opdttt.opd_max_sample_length
        max_seq_len = self.args.data.max_seq_len
        temperature = self.args.opdttt.opd_temperature
        top_p = self.args.opdttt.opd_top_p

        prompt_lens = []
        prompt_list = []
        for batch_idx in range(batch_size):
            row_ids = input_ids[batch_idx]  # [seq_len]
            row_labels = labels[batch_idx] if labels is not None else None

            if row_labels is not None:
                # Packed sequence: split by IGNORE_INDEX (-100) boundary markers
                # -100 appears at the first token of each sub-prompt (except the first)
                # and at padding positions. Find actual packed length and boundaries.
                non_ignore = (row_labels != -100).nonzero(as_tuple=True)[0]
                if len(non_ignore) == 0:
                    continue
                packed_len = non_ignore[-1].item() + 1

                # Find boundary positions within packed region
                boundary_positions = []
                for i in range(1, packed_len):
                    if row_labels[i].item() == -100:
                        boundary_positions.append(i)

                if not boundary_positions:
                    # Single prompt (no packing)
                    p_ids = row_ids[:packed_len].unsqueeze(0)
                else:
                    # Multiple prompts: split at boundaries
                    boundaries = [0] + boundary_positions + [packed_len]
                    for bi in range(len(boundaries) - 1):
                        start = boundaries[bi]
                        end = boundaries[bi + 1]
                        p = row_ids[start:end]
                        if len(p) > 0:
                            prompt_list.append(p)
                            plen = p.shape[0]
                            if plen > 1 and p[-1].item() == eos_token_id:
                                p = p[:-1]
                                plen = p.shape[0]
                            prompt_lens.append(plen)
                    continue
            else:
                p_ids = row_ids.unsqueeze(0)

            if p_ids.shape[1] > 1 and p_ids[0, -1].item() == eos_token_id:
                p_ids = p_ids[:, :-1]
            prompt_lens.append(p_ids.shape[1])
            prompt_list.append(p_ids.squeeze(0))

        for i in range(num_trajectories):
            with torch.no_grad():
                from transformers.cache_utils import DynamicCache

                max_prompt_len = max(prompt_lens)
                bs = len(prompt_list)

                # Pad all prompts to max_prompt_len
                padded = torch.full((bs, max_prompt_len), self.tokenizer.pad_token_id,
                                    dtype=torch.long, device=self.device)
                attn_mask = torch.zeros(bs, max_prompt_len, dtype=torch.float32, device=self.device)
                for bi in range(bs):
                    plen = prompt_lens[bi]
                    padded[bi, :plen] = prompt_list[bi][:plen]
                    attn_mask[bi, :plen] = 1.0

                # Compute max_new_tokens across all prompts
                max_new_tokens = min(max_prompt_len + max_sample_length, max_seq_len) - max_prompt_len
                if max_new_tokens <= 0:
                    for bi in range(bs):
                        trajectories.append({
                            "input_ids": prompt_list[bi].unsqueeze(0),
                            "sampled_logprobs": None,
                            "prompt_length": prompt_lens[bi],
                        })
                    continue

                kv_cache = DynamicCache()

                # First forward: all prompts at once
                outputs = sampling_model(
                    input_ids=padded,
                    attention_mask=attn_mask,
                    use_cache=True,
                    past_key_values=kv_cache,
                )
                next_logits = outputs.logits[:, -1, :].float()
                next_logits = torch.nan_to_num(next_logits, nan=0.0, posinf=1e4, neginf=-1e4)

                if temperature > 0:
                    next_logits = next_logits / temperature

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_mask = cumulative_probs > top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = 0
                    mask = torch.zeros_like(next_logits, dtype=torch.bool)
                    mask.scatter_(1, sorted_indices, sorted_mask)
                    next_logits[mask] = float('-inf')

                next_logprobs = F.log_softmax(next_logits, dim=-1)
                probs = F.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)  # [bs, 1]
                next_token_logprobs = next_logprobs.gather(-1, next_tokens).squeeze(-1)  # [bs]

                # Track per-sequence state
                all_sampled = [next_tokens]  # list of [bs, 1]
                all_logprobs = [next_token_logprobs]  # list of [bs]
                finished = (next_tokens.squeeze(-1) == eos_token_id)  # [bs]

                attn_mask = torch.cat([attn_mask, torch.ones(bs, 1, device=self.device)], dim=-1)

                for step in range(1, max_new_tokens):
                    # Replace finished sequences' input with pad token (but keep in batch)
                    input_tokens = next_tokens.clone()
                    input_tokens[finished] = self.tokenizer.pad_token_id

                    outputs = sampling_model(
                        input_ids=input_tokens,
                        attention_mask=attn_mask,
                        use_cache=True,
                        past_key_values=kv_cache,
                    )

                    next_logits = outputs.logits[:, -1, :].float()
                    next_logits = torch.nan_to_num(next_logits, nan=0.0, posinf=1e4, neginf=-1e4)

                    if temperature > 0:
                        next_logits = next_logits / temperature

                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_mask = cumulative_probs > top_p
                        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                        sorted_mask[..., 0] = 0
                        mask = torch.zeros_like(next_logits, dtype=torch.bool)
                        mask.scatter_(1, sorted_indices, sorted_mask)
                        next_logits[mask] = float('-inf')

                    next_logprobs = F.log_softmax(next_logits, dim=-1)
                    probs = F.softmax(next_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                    next_token_logprobs = next_logprobs.gather(-1, next_tokens).squeeze(-1)

                    # Mask out finished sequences
                    next_tokens[finished] = eos_token_id
                    next_token_logprobs[finished] = 0.0

                    all_sampled.append(next_tokens)
                    all_logprobs.append(next_token_logprobs)

                    newly_finished = (next_tokens.squeeze(-1) == eos_token_id) & ~finished
                    finished = finished | newly_finished

                    attn_mask = torch.cat([attn_mask, torch.ones(bs, 1, device=self.device)], dim=-1)

                    if finished.all():
                        break

                # Unpack per-sequence results
                # all_sampled: list of [bs, 1], all_logprobs: list of [bs]
                stacked_tokens = torch.cat(all_sampled, dim=1)  # [bs, num_steps]
                stacked_logprobs = torch.stack(all_logprobs, dim=1)  # [bs, num_steps]

                for bi in range(bs):
                    plen = prompt_lens[bi]
                    # Find EOS position for this sequence
                    seq_tokens = stacked_tokens[bi]  # [num_steps]
                    eos_positions = (seq_tokens == eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions) > 0:
                        cut = eos_positions[0].item() + 1  # include EOS
                    else:
                        cut = seq_tokens.shape[0]

                    sampled_ids_tensor = seq_tokens[:cut].unsqueeze(0)  # [1, cut]
                    sampled_logprobs_tensor = stacked_logprobs[bi, :cut]  # [cut]

                    full_input_ids = torch.cat([prompt_list[bi].unsqueeze(0), sampled_ids_tensor], dim=-1)

                    trajectories.append({
                        "input_ids": full_input_ids,
                        "sampled_logprobs": sampled_logprobs_tensor,
                        "prompt_length": plen,
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

            if input_ids.shape[1] <= prompt_length:
                teacher_logprobs_list.append(None)
                continue

            sampled_ids = input_ids[:, prompt_length:]

            with torch.no_grad():
                outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=torch.ones(input_ids.shape, dtype=torch.float32, device=input_ids.device),
                    use_cache=False,
                )

                sampled_logits = outputs.logits[:, prompt_length-1:-1, :]

                teacher_logprobs = F.log_softmax(sampled_logits, dim=-1)

                sampled_tokens_reshaped = sampled_ids.unsqueeze(-1)

                if teacher_logprobs.shape[1] != sampled_ids.shape[1]:
                    min_len = min(teacher_logprobs.shape[1], sampled_ids.shape[1])
                    teacher_logprobs = teacher_logprobs[:, :min_len, :]
                    sampled_tokens_reshaped = sampled_tokens_reshaped[:, :min_len, :]

                teacher_logprobs_on_sampled = teacher_logprobs.gather(-1, sampled_tokens_reshaped).squeeze(-1)

                # Pre-compute top-K teacher logits for KL (avoids storing full logits)
                K_KL = 100
                student_vocab = self.student_model.config.vocab_size
                t_logits_for_topk = sampled_logits[0]  # [seq, teacher_vocab]
                if t_logits_for_topk.shape[-1] > student_vocab:
                    t_logits_for_topk = t_logits_for_topk[..., :student_vocab]
                t_topk_vals, t_topk_idx = t_logits_for_topk.float().topk(K_KL, dim=-1)  # [seq, K]

                teacher_logprobs_list.append({
                    "on_sampled": teacher_logprobs_on_sampled,
                    "t_topk_vals": t_topk_vals,   # [seq, K] float32, tiny
                    "t_topk_idx": t_topk_idx,     # [seq, K] int64, tiny
                })

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

            student_logprobs = trajectory["sampled_logprobs"]

            if student_logprobs is None or teacher_logprobs is None:
                continue

            min_len = min(student_logprobs.shape[0], teacher_logprobs.shape[1])
            if min_len == 0:
                continue

            student_logprobs = student_logprobs[:min_len]
            teacher_logprobs = teacher_logprobs[0, :min_len]

            reverse_kl = (student_logprobs.detach() - teacher_logprobs.detach())
            advantages = -reverse_kl

            rho = torch.exp(student_logprobs - student_logprobs.detach())

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
        loss_scale: float = 1.0,
    ) -> float:
        import time as _time
        _t0 = _time.time()
        prompts_batch = {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in prompts_batch.items()
        }

        trajectories = self.sample_from_student(
            prompts=prompts_batch,
            num_trajectories=self.args.opdttt.opd_num_trajectories,
        )
        _t1 = _time.time()

        # 同步trajectory数量：FSDP2 student forward涉及all-gather，
        # 不同rank的trajectory数量必须一致，否则死锁
        num_trajs = torch.tensor([len(trajectories)], device=self.device)
        all_reduce(num_trajs, op="min", group=self.parallel_state.fsdp_group)
        min_trajs = int(num_trajs.item())
        trajectories = trajectories[:min_trajs]

        if not trajectories:
            return 0.0

        # 阶段2: 训练（保留梯度）
        # 筛选valid trajectories（有采样token的），做单次batched FSDP2 forward
        # 所有rank做相同的1次forward+1次backward，避免collective不匹配导致死锁
        valid_trajs = []
        for trajectory in trajectories:
            input_ids = trajectory["input_ids"]
            prompt_length = trajectory["prompt_length"]
            if input_ids.shape[1] > prompt_length:
                valid_trajs.append(trajectory)
            else:
                trajectory["student_logprobs_with_grad"] = None

        # 同步valid数量，确保所有rank要么都做forward，要么都不做
        valid_count_t = torch.tensor([len(valid_trajs) > 0], device=self.device)
        all_reduce(valid_count_t, op="min", group=self.parallel_state.fsdp_group)
        has_valid = bool(valid_count_t.item())

        if has_valid and valid_trajs:
            # Pad all valid trajectories to max length for single batched forward
            max_seq = max(t["input_ids"].shape[1] for t in valid_trajs)
            pad_id = self.tokenizer.pad_token_id
            batch_ids = torch.full((len(valid_trajs), max_seq), pad_id,
                                   dtype=torch.long, device=self.device)
            for bi, t in enumerate(valid_trajs):
                ids = t["input_ids"].to(self.device)
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                seq_len = ids.shape[1]
                batch_ids[bi, :seq_len] = ids[0]

            outputs = self.student_model(
                input_ids=batch_ids,
                attention_mask=torch.ones(batch_ids.shape, dtype=torch.float32, device=self.device),
                use_cache=False,
            )

            for bi, trajectory in enumerate(valid_trajs):
                input_ids = trajectory["input_ids"].to(self.device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                prompt_length = trajectory["prompt_length"]
                seq_len = input_ids.shape[1]

                sampled_logits = outputs.logits[bi, prompt_length-1:seq_len-1, :]
                sampled_logprobs = F.log_softmax(sampled_logits, dim=-1)
                sampled_ids = input_ids[0, prompt_length:]

                min_len = min(sampled_logprobs.shape[0], sampled_ids.shape[0])
                if min_len > 0:
                    student_logprobs_with_grad = sampled_logprobs[:min_len].gather(-1, sampled_ids[:min_len].unsqueeze(-1)).squeeze(-1)
                    trajectory["student_logprobs_with_grad"] = student_logprobs_with_grad
                    trajectory["student_bi"] = bi
                    trajectory["student_logits_start"] = prompt_length - 1
                    trajectory["student_logits_len"] = min_len
                else:
                    trajectory["student_logprobs_with_grad"] = None
                    trajectory["student_bi"] = None
        else:
            for trajectory in valid_trajs:
                trajectory["student_logprobs_with_grad"] = None

        # 2b. 计算教师logprobs（detach）
        teacher_logprobs_list = self.compute_teacher_logprobs_on_sampled(trajectories)
        _t2 = _time.time()


        # 2c & 2d. 计算重要性采样损失 + forward KL（使用有梯度的学生logprobs）
        loss = self.compute_importance_sampling_loss_with_grad(
            trajectories,
            teacher_logprobs_list,
            outputs.logits if (has_valid and valid_trajs) else None,
        )



        # 反向传播
        # 乘以 dp_size 补偿 FSDP2 梯度平均，乘以 loss_scale 用于梯度累积归一化
        loss = loss * self.parallel_state.dp_size * loss_scale
        with self.model_bwd_context:
            loss.backward()
        _t3 = _time.time()
        with open("/h3c/haoxiang/TTT-OPD/timing.log", "a") as _f:
            _f.write(f"rank={self.args.train.local_rank} sampling={_t1-_t0:.1f}s student_fwd={_t2-_t1:.1f}s teacher+loss+bwd={_t3-_t2:.1f}s total={_t3-_t0:.1f}s seq_len={trajectories[0]['input_ids'].shape[1] if trajectories else 0}\n")

        return loss.item()

    def compute_importance_sampling_loss_with_grad(
        self,
        trajectories: List[Dict[str, torch.Tensor]],
        teacher_logprobs_list: List,
        student_logits_batch: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        计算重要性采样损失 + forward KL正则化

        forward KL(teacher||student) 防止模式崩溃：
        - reverse KL (on sampled tokens) 只在学生采样的token上学习 → 模式崩溃
        - forward KL (full vocab) 强制学生完整分布向教师对齐 → 防止崩溃

        Args:
            trajectories: 学生采样轨迹
            teacher_logprobs_list: 教师信息（on_sampled logprobs + full_logits）
            student_logits_batch: 学生的batched logits引用（避免复制）

        Returns:
            组合损失
        """
        total_is_loss = 0.0
        total_kl_loss = 0.0
        total_tokens = 0

        lambda_kl = float(self.args.opdttt.lambda_kl)

        for trajectory, teacher_info in zip(trajectories, teacher_logprobs_list):
            if teacher_info is None:
                continue

            teacher_logprobs = teacher_info["on_sampled"]
            teacher_full_logits = teacher_info.get("full_logits")

            student_logprobs = trajectory.get("student_logprobs_with_grad")

            if student_logprobs is None or teacher_logprobs is None:
                continue

            if student_logprobs.dim() == 2:
                student_logprobs = student_logprobs[0]

            seq_len_s = student_logprobs.shape[0]
            seq_len_t = teacher_logprobs.shape[-1]
            min_len = min(seq_len_s, seq_len_t)
            if min_len == 0:
                continue

            student_logprobs = student_logprobs[:min_len]
            teacher_logprobs = teacher_logprobs[0, :min_len]

            reverse_kl = (student_logprobs.detach() - teacher_logprobs.detach())
            advantages = -reverse_kl

            rho = torch.exp(student_logprobs - student_logprobs.detach())

            loss_per_token = -rho * advantages
            loss_per_token = torch.nan_to_num(loss_per_token, nan=0.0, posinf=0.0, neginf=0.0)

            total_is_loss += loss_per_token.sum()
            total_tokens += min_len

            # Forward KL (top-K近似): KL(teacher || student) ≈ sum over top-K teacher tokens
            # 使用预计算的teacher top-K，只从student logits gather K个值
            if lambda_kl > 0 and student_logits_batch is not None:
                t_topk_vals = teacher_info.get("t_topk_vals")
                t_topk_idx = teacher_info.get("t_topk_idx")
                bi = trajectory.get("student_bi")
                start = trajectory.get("student_logits_start")
                slogits_len = trajectory.get("student_logits_len")
                if all(x is not None for x in [t_topk_vals, t_topk_idx, bi, start, slogits_len]):
                    kl_min_len = min(slogits_len, t_topk_vals.shape[0])
                    if kl_min_len > 0:
                        # Gather student logits at top-K positions (bfloat16 → float32, minimal memory)
                        s_topk = student_logits_batch[bi, start:start+kl_min_len, :].gather(
                            -1, t_topk_idx[:kl_min_len]
                        ).float()  # [kl_min_len, K]

                        t_vals = t_topk_vals[:kl_min_len]  # [kl_min_len, K]

                        t_topk_probs = F.softmax(t_vals, dim=-1)
                        t_topk_log_probs = F.log_softmax(t_vals, dim=-1)
                        s_topk_log_probs = F.log_softmax(s_topk, dim=-1)

                        kl_per_token = (t_topk_probs * (t_topk_log_probs - s_topk_log_probs)).sum(dim=-1)
                        kl_per_token = torch.nan_to_num(kl_per_token, nan=0.0, posinf=0.0, neginf=0.0)

                        total_kl_loss += kl_per_token.sum()

                        del s_topk, t_vals, t_topk_probs, t_topk_log_probs, s_topk_log_probs, kl_per_token

        if total_tokens > 0:
            is_loss = total_is_loss / total_tokens
            kl_loss = total_kl_loss / total_tokens if total_tokens > 0 else torch.tensor(0.0, device=self.device)
            return is_loss + lambda_kl * kl_loss
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def train_step(
        self,
        micro_batch: Dict[str, torch.Tensor],
        length_in_batch: torch.Tensor,
    ) -> Tuple[float, Dict[str, Any]]:
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

        step_loss_dict = getattr(student_outputs, "loss_dict", {})
        return loss.item(), step_loss_dict

    def train(self, train_steps: int, start_step: int = 0):
        """
        主训练循环

        Args:
            train_steps: 总训练步数
            start_step: 起始步数（用于恢复训练）
        """
        global_step = start_step
        last_ckpt_path = None
        best_loss = float('inf')
        best_ckpt_path = None

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
            # 续训时跳过 set_epoch，避免重置已恢复的 dataloader 状态
            if start_step == 0 and hasattr(self.train_dataloader, "set_epoch"):
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
                step_metrics: Dict[str, float] = {}
                if enable_opd:
                    num_micro_batches = len(micro_batches)
                    for micro_batch in micro_batches:
                        loss = self.train_step_opd(
                            prompts_batch=micro_batch,
                            loss_scale=1.0 / max(num_micro_batches, 1),
                        )
                        total_loss += loss
                        del micro_batch
                else:
                    for micro_batch in micro_batches:
                        loss, mb_metrics = self.train_step(micro_batch, length_in_batch)
                        total_loss += loss
                        for k, v in mb_metrics.items():
                            if isinstance(v, (int, float)):
                                step_metrics[k] = step_metrics.get(k, 0.0) + v / max(len(micro_batches), 1)
                        del micro_batch

                # 跨 rank 聚合 step_metrics（FSDP2 数据并行下各 rank 处理不同数据，
                # ttt_relative_contribution 等指标需跨 rank 平均才能与单 GPU 一致）
                if step_metrics:
                    _metric_keys = sorted(step_metrics.keys())
                    _metric_t = torch.tensor(
                        [step_metrics[k] for k in _metric_keys],
                        device=self.device, dtype=torch.float32,
                    )
                    _metric_t = all_reduce(_metric_t, op="mean", group=self.parallel_state.fsdp_group)
                    if isinstance(_metric_t, (int, float)):
                        _metric_t = [_metric_t]
                    for _mk, _mv in zip(_metric_keys, _metric_t):
                        step_metrics[_mk] = _mv

                # 梯度裁剪和优化
                grad_norm = veomni_clip_grad_norm(
                    self.student_model, self.args.train.max_grad_norm
                )

                # 在 zero_grad 之前计算 TTT 参数范数（梯度随后会被清除）
                # FSDP2 下跨 rank 聚合本地分片范数平方，得到完整参数范数（与单 GPU 一致）
                ttt_param_norms = self._compute_ttt_param_norms()

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
                    log_data = {
                        "training/loss": total_loss,
                        "training/perplexity": math.exp(total_loss),
                        "training/grad_norm": grad_norm,
                        "training/lr": lr,
                    }
                    for k, v in step_metrics.items():
                        if k.startswith("ttt_"):
                            log_data[f"ttt/{k}"] = v
                        else:
                            log_data[f"training/{k}"] = v

                    # TTT 模块参数监控指标（完整范数，已跨 rank 聚合）
                    for _pn_k, _pn_v in ttt_param_norms.items():
                        log_data[f"ttt/{_pn_k}"] = _pn_v

                    wandb.log(log_data, step=global_step)

                # 保存检查点
                if (
                    self.args.train.save_steps
                    and global_step % self.args.train.save_steps == 0
                ):
                    last_ckpt_path = self._save_checkpoint(global_step, last_ckpt_path)

                    # Best checkpoint: loss最低时复制为best
                    if self.args.opdttt.save_best and total_loss < best_loss:
                        best_loss = total_loss
                        if self.args.train.global_rank == 0:
                            best_ckpt_path = self._save_best_checkpoint(global_step, last_ckpt_path, best_ckpt_path)

                    # 评估多 context 长度 PPL
                    self._evaluate_ppl(global_step)
                elif global_step == 1 or (start_step > 0 and global_step == start_step + 1):
                    # 首步评估（不保存 checkpoint），及早发现评估 bug
                    # 续训时也对续训后的第一步评估（global_step == start_step + 1）
                    self._evaluate_ppl(global_step)

            start_step = 0

        # 最终检查点：所有rank参与DCP加载（集合操作），仅rank0保存HF权重
        if last_ckpt_path is not None:
            self._save_hf_weights(global_step, last_ckpt_path)
        else:
            logger.info_rank0("跳过HF权重保存：无DCP检查点可用")

    def _broadcast_eval_data(self, eval_ids, max_eval_len):
        """将 rank 0 的评估数据 broadcast 到所有 rank，确保 FSDP2 forward 一致。"""
        device = self.device
        if not dist.is_initialized():
            if eval_ids is None:
                eval_ids = torch.randint(0, self.tokenizer.vocab_size, (1, max_eval_len))
            return eval_ids.to(device)

        shape = torch.tensor(
            list(eval_ids.shape) if eval_ids is not None else [0, 0],
            device=device, dtype=torch.long,
        )
        dist.broadcast(shape, src=0)

        if eval_ids is None:
            eval_ids = torch.zeros(shape[0].item(), shape[1].item(), dtype=torch.long, device=device)
        else:
            eval_ids = eval_ids.to(device)

        dist.broadcast(eval_ids, src=0)
        return eval_ids

    def _compute_ttt_param_norms(self) -> Dict[str, float]:
        """计算 TTT 参数的完整范数（FSDP2 兼容）。

        使用 full_tensor() 获取完整参数（DTensor 时 all_gather，普通 Tensor 时直接使用），
        无需 all_reduce，结果与单 GPU 一致。

        必须在 optimizer.zero_grad() 之前调用（梯度随后会被清除）。
        返回 ttt_conv/ttt_proj 的权重范数与梯度范数（每层均方根）。
        """
        def _full(t):
            if t is None:
                return None
            data = t.data if hasattr(t, "data") else t
            if hasattr(data, "full_tensor"):
                return data.full_tensor()
            return data

        conv_norm_sq = 0.0
        proj_norm_sq = 0.0
        conv_grad_norm_sq = 0.0
        proj_grad_norm_sq = 0.0
        num_layers = 0

        for layer in self.student_model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "ttt_conv"):
                w = _full(mlp.ttt_conv.weight)
                conv_norm_sq += w.norm().item() ** 2
                if mlp.ttt_conv.weight.grad is not None:
                    g = _full(mlp.ttt_conv.weight.grad)
                    conv_grad_norm_sq += g.norm().item() ** 2
                num_layers += 1
            if hasattr(mlp, "ttt_proj") and mlp.ttt_proj is not None:
                w = _full(mlp.ttt_proj.weight)
                proj_norm_sq += w.norm().item() ** 2
                if mlp.ttt_proj.weight.grad is not None:
                    g = _full(mlp.ttt_proj.weight.grad)
                    proj_grad_norm_sq += g.norm().item() ** 2

        results: Dict[str, float] = {}
        if num_layers > 0:
            results["ttt_conv_norm"] = (conv_norm_sq / num_layers) ** 0.5
            results["ttt_proj_norm"] = (proj_norm_sq / num_layers) ** 0.5
            results["ttt_conv_grad_norm"] = (conv_grad_norm_sq / num_layers) ** 0.5
            results["ttt_proj_grad_norm"] = (proj_grad_norm_sq / num_layers) ** 0.5
        return results

    def _evaluate_ppl(self, global_step: int):
        """在多个 context 长度下评估 PPL（固定 target 往前追溯 + TTT on/off 对比）。

        评估方式：
        - 固定 target = 序列最后 target_len 个 token
        - 对每个 ctx_len，用前 ctx_len 个 token 作为 context 预测同一组 target
        - 从训练数据中取多条样本，取平均 PPL 消除单样本波动
        - 同时评估 TTT on（SWA+TTT）和 TTT off（纯 SWA），对比 TTT 贡献
        - 预期：TTT off PPL 基本不随 ctx 变化，TTT on PPL 随 ctx 降低

        FSDP2 下所有 rank 必须参与 forward（集合操作），仅 rank 0 上传 wandb。
        评估数据在首次调用时从训练数据中取固定样本，确保每次评估可比。
        """
        context_lengths = [2048, 4096, 8192, 16384]
        target_len = 2048
        max_eval_len = max(context_lengths) + target_len
        num_eval_samples = int(os.environ.get("OPDTTT_EVAL_SAMPLES", "20"))
        group_size = int(os.environ.get("OPDTTT_EVAL_GROUP_SIZE", str(num_eval_samples)))

        if not hasattr(self, "_eval_input_ids"):
            data_path = self.args.data.train_path
            eval_ids = None
            if self.args.train.global_rank == 0:
                # 从文件末尾取评估样本：训练数据按顺序流式读取 + shuffle buffer，
                # 5000 步内仅读到文件前 ~43%，末尾样本在训练过程中始终样本外，
                # 避免用已训练过的开头样本导致评估失真
                samples = self._load_eval_samples_from_tail(data_path, num_eval_samples, max_eval_len)
                if len(samples) == 0:
                    # 兜底：末尾无足够长样本时，从开头取并重复填充
                    import json as _json
                    with open(data_path, "r") as f:
                        first_line = f.readline()
                    data = _json.loads(first_line)
                    text = data.get("content_split", data.get("content", ""))
                    text = text * ((max_eval_len * 4 // len(text)) + 1)
                    samples.append(self.tokenizer(text, return_tensors="pt")["input_ids"][:, :max_eval_len])
                eval_ids = torch.cat(samples, dim=0)
            self._eval_input_ids = self._broadcast_eval_data(eval_ids, max_eval_len)

        ttt_mlps = [
            layer.mlp for layer in self.student_model.model.layers
            if hasattr(layer.mlp, "enable_opdttt")
        ]

        def _run_eval(ttt_on: bool) -> dict:
            for mlp in ttt_mlps:
                mlp.enable_opdttt = ttt_on
            self.student_model.eval()
            tag = "TTT-on" if ttt_on else "TTT-off"
            all_ppls = {ctx: [] for ctx in context_lengths}
            num_samples = self._eval_input_ids.shape[0]
            with torch.no_grad():
                for s_idx in range(num_samples):
                    for ctx_len in context_lengths:
                        total_len = ctx_len + target_len
                        input_ids = self._eval_input_ids[s_idx:s_idx+1, -total_len:]
                        labels = input_ids.clone()
                        labels[:, :ctx_len] = -100
                        try:
                            outputs = self.student_model(
                                input_ids=input_ids,
                                labels=labels,
                                use_cache=False,
                            )
                            loss = outputs.loss.item()
                            ppl = math.exp(min(loss, 20.0))
                        except torch.cuda.OutOfMemoryError:
                            helper.empty_cache()
                            ppl = float("nan")
                        all_ppls[ctx_len].append(ppl)
                        del outputs
                        helper.empty_cache()
                    # 分组统计
                    if (s_idx + 1) % group_size == 0:
                        g_start = s_idx + 1 - group_size
                        g_end = s_idx + 1
                        parts = []
                        for ctx_len in context_lengths:
                            grp = [p for p in all_ppls[ctx_len][g_start:g_end] if p == p]
                            g_mean = sum(grp) / len(grp) if grp else float("nan")
                            parts.append(f"{ctx_len}:{g_mean:.3f}")
                        logger.info_rank0(
                            f"Step {global_step} Eval PPL {tag} "
                            f"group[{g_start}:{g_end}]: {' | '.join(parts)}"
                        )
            # 整体统计
            results = {}
            for ctx_len in context_lengths:
                valid = [p for p in all_ppls[ctx_len] if p == p]
                if valid:
                    mean = sum(valid) / len(valid)
                    if len(valid) > 1:
                        var = sum((p - mean) ** 2 for p in valid) / (len(valid) - 1)
                        std = var ** 0.5
                    else:
                        std = 0.0
                    results[ctx_len] = mean
                    logger.info_rank0(
                        f"Step {global_step} Eval PPL {tag} @ ctx={ctx_len}: "
                        f"mean={mean:.4f} std={std:.4f} "
                        f"min={min(valid):.4f} max={max(valid):.4f} "
                        f"({len(valid)}/{len(all_ppls[ctx_len])} samples)"
                    )
                else:
                    results[ctx_len] = float("nan")
                    logger.info_rank0(
                        f"Step {global_step} Eval PPL {tag} @ ctx={ctx_len}: "
                        f"mean=nan ({len(valid)}/{len(all_ppls[ctx_len])} samples)"
                    )
            return results

        ppl_on = _run_eval(True)
        ppl_off = _run_eval(False)

        for mlp in ttt_mlps:
            mlp.enable_opdttt = True
        self.student_model.train()

        if self.args.train.global_rank == 0 and self.args.train.use_wandb:
            log_data = {}
            for ctx, ppl in ppl_on.items():
                log_data[f"eval/ppl_ttt_on_{ctx}"] = ppl
            for ctx, ppl in ppl_off.items():
                log_data[f"eval/ppl_ttt_off_{ctx}"] = ppl
            # TTT 贡献：ppl_off - ppl_on（正值表示 TTT 降低 PPL）
            for ctx in ppl_on:
                _on, _off = ppl_on[ctx], ppl_off[ctx]
                if _on == _on and _off == _off:
                    log_data[f"eval/ppl_delta_{ctx}"] = _off - _on
            # 合并折线图：TTT on/off 同图，纵坐标自适应，附差值柱状图
            log_data["eval/ppl_vs_ctx"] = self._plot_ppl_vs_ctx(
                ppl_on, ppl_off, context_lengths, global_step
            )
            wandb.log(log_data, step=global_step)

    def _load_eval_samples_from_tail(self, data_path: str, num_samples: int, min_len: int) -> List[torch.Tensor]:
        """从数据文件末尾向前读取，取最后 num_samples 个 token 长度 >= min_len 的样本。

        训练数据按顺序流式读取（shuffle buffer 仅局部打乱），5000 步内只读到文件前
        ~43%，因此文件末尾的样本在整个训练过程中始终是样本外，用作评估集可避免
        训练-评估数据泄漏。采用从文件尾向前 seek 的方式，避免遍历整个大文件。
        """
        import json as _json
        samples: List[torch.Tensor] = []
        with open(data_path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            chunk_size = 16 * 1024 * 1024  # 16MB
            pos = file_size
            buffer = ""
            while pos > 0 and len(samples) < num_samples:
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos)
                chunk = f.read(read_size).decode("utf-8", errors="ignore")
                buffer = chunk + buffer
                lines = buffer.split("\n")
                if pos > 0:
                    # 第一段可能被 chunk 截断，留到下次拼接
                    buffer = lines[0]
                    complete_lines = lines[1:]
                else:
                    buffer = ""
                    complete_lines = lines
                # 从末尾向前遍历完整行
                for line in reversed(complete_lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                    except Exception:
                        continue
                    text = data.get("content_split", data.get("content", ""))
                    if not text:
                        continue
                    # 粗筛：字符长度至少 min_len*2（保守 token/char 比），减少 tokenize 调用
                    if len(text) < min_len * 2:
                        continue
                    ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
                    if ids.shape[1] >= min_len:
                        samples.append(ids[:, -min_len:])
                        if len(samples) >= num_samples:
                            break
        # samples 是从末尾向前收集的，反转为文件顺序（保证 target 在尾部、context 在前）
        samples.reverse()
        logger.info_rank0(
            f"评估样本: 从文件末尾取 {len(samples)}/{num_samples} 个样本 "
            f"(min_len={min_len}), 读取范围约 [pos={pos}, file_size={file_size}]"
        )
        return samples

    def _plot_ppl_vs_ctx(self, ppl_on: dict, ppl_off: dict, context_lengths: list, global_step: int):
        """生成 PPL vs Context 合并折线图（TTT on/off 同图）。

        - 纵坐标自适应数据范围（不从 0 开始），避免曲线被压扁成一条直线
        - 每点标注数值，便于区分两条极接近的曲线谁在上谁在下
        - 下方附差值柱状图（ppl_off - ppl_on），正值=TTT 降低 PPL
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        ctxs = list(context_lengths)
        ctx_labels = [f"{c // 1024}k" for c in ctxs]
        ys_on = [ppl_on[c] for c in ctxs]
        ys_off = [ppl_off[c] for c in ctxs]

        fig = make_subplots(
            rows=2, cols=1, row_heights=[0.68, 0.32],
            vertical_spacing=0.13,
            subplot_titles=(f"Step {global_step}: PPL vs Context (TTT on vs off)", "Δ PPL (off − on, 正值=TTT 降低)"),
        )
        # 上图：两条折线，标注数值
        fig.add_trace(go.Scatter(
            x=ctx_labels, y=ys_on, name="TTT-on", mode="lines+markers+text",
            text=[f"{v:.3f}" for v in ys_on], textposition="top center",
            line=dict(color="#1f77b4", width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ctx_labels, y=ys_off, name="TTT-off", mode="lines+markers+text",
            text=[f"{v:.3f}" for v in ys_off], textposition="bottom center",
            line=dict(color="#d62728", width=2, dash="dot"),
        ), row=1, col=1)
        # 纵坐标自适应：取所有值的范围并留 15% 边距，不从 0 开始
        all_vals = [v for v in ys_on + ys_off if v == v]
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            pad = max((hi - lo) * 0.15, 1e-3)
            fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
        fig.update_xaxes(title_text="context length", type="category", row=1, col=1)
        fig.update_yaxes(title_text="PPL", row=1, col=1)

        # 下图：差值柱状图（正值绿色=TTT降低，负值红色=TTT升高）
        deltas = [
            (ppl_off[c] - ppl_on[c]) if (ppl_on[c] == ppl_on[c] and ppl_off[c] == ppl_off[c]) else 0.0
            for c in ctxs
        ]
        fig.add_trace(go.Bar(
            x=ctx_labels, y=deltas, name="Δ (off−on)",
            marker_color=["#2ca02c" if d >= 0 else "#d62728" for d in deltas],
            text=[f"{d:+.4f}" for d in deltas], textposition="outside",
        ), row=2, col=1)
        fig.update_yaxes(title_text="Δ PPL", row=2, col=1)
        fig.update_xaxes(title_text="context length", type="category", row=2, col=1)

        fig.update_layout(
            height=580, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5),
            margin=dict(t=90, b=40),
        )
        return fig

    def _save_checkpoint(self, global_step: int, last_ckpt_path: Optional[str]):
        """保存分布式检查点"""
        helper.empty_cache()
        save_path = os.path.join(
            self.args.train.save_checkpoint_path, f"global_step_{global_step}"
        )

        extra_state = {
            "global_step": global_step,
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }

        try:
            extra_state["dataloader_state"] = self.train_dataloader.state_dict()
        except Exception as e:
            logger.warning(f"无法保存 dataloader 状态: {e}")

        try:
            import torch as _torch
            extra_state["rng_state"] = {
                "python": _torch.get_rng_state(),
                "cuda": _torch.cuda.get_rng_state() if _torch.cuda.is_available() else None,
            }
        except Exception as e:
            logger.warning(f"无法保存 RNG 状态: {e}")

        state = {
            "model": self.student_model,
            "optimizer": self.optimizer,
            "extra_state": extra_state,
        }

        self.checkpointer.save(save_path, state, global_steps=global_step)
        dist.barrier()

        # DCP checkpointer 创建嵌套目录结构：save_path/global_step_X
        # 更新 last_ckpt_path 为实际保存的检查点路径
        actual_ckpt_path = os.path.join(save_path, f"global_step_{global_step}")
        logger.info(f"检查点已保存: {actual_ckpt_path}")

        # 轮转旧检查点（仅在 keep_all_checkpoints=False 时）
        if not self.args.opdttt.keep_all_checkpoints:
            if last_ckpt_path is not None and os.path.isdir(last_ckpt_path):
                shutil.rmtree(last_ckpt_path, ignore_errors=True)
                logger.info(f"删除旧检查点: {last_ckpt_path}")

        return actual_ckpt_path

    def _save_best_checkpoint(self, global_step: int, src_ckpt_path: str, prev_best_path: Optional[str]):
        """将当前checkpoint复制为best（仅rank0执行文件操作）"""
        import shutil as _shutil
        best_path = os.path.join(self.args.train.save_checkpoint_path, "best")
        if os.path.isdir(best_path):
            _shutil.rmtree(best_path, ignore_errors=True)
        _shutil.copytree(src_ckpt_path, best_path)
        logger.info(f"Best checkpoint已更新 (step={global_step}): {best_path}")
        return best_path

    def _save_hf_weights(self, global_step: int, last_ckpt_path: Optional[str]):
        """保存 HuggingFace 格式权重

        所有rank参与DCP加载（dcp.load是集合操作），仅rank0保存safetensors。
        """
        save_path = os.path.join(
            self.args.train.save_checkpoint_path, f"global_step_{global_step}", "hf_ckpt"
        )

        # 在保存HuggingFace权重前清空缓存，避免OOM
        # 因为ckpt_to_state_dict会聚合所有分片参数到一起
        helper.empty_cache()
        synchronize()
        dist.barrier()

        # 所有rank参与DCP加载（dcp.load是集合操作）
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=last_ckpt_path,
            ckpt_manager=self.args.train.ckpt_manager,
        )

        # 仅rank0保存safetensors文件
        if self.args.train.global_rank == 0:
            save_model_weights(save_path, model_state_dict, model_assets=[self.model_config] if self.model_config else None)
            logger.info(f"HuggingFace 权重已保存: {save_path}")

        dist.barrier()


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
    teacher_path = teacher_path.strip().strip("'\"") if teacher_path else ""
    if not teacher_path:
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
        teacher_cfg = config.text_config if hasattr(config, 'text_config') and hasattr(config.text_config, 'vocab_size') else config
        teacher_vocab_size = teacher_cfg.vocab_size

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

    logger.info(f"教师模型已加载: {teacher_cfg.num_hidden_layers} 层, "
                f"{teacher_cfg.num_attention_heads} 头, hidden_size={teacher_cfg.hidden_size}")

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
    if args.data.data_type == "conversation":
        chat_template = build_chat_template(args.data.chat_template, tokenizer)
        transform = partial(
            _data_transform.process_conversation_example,
            chat_template=chat_template,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
        logger.info_rank0(
            f"  数据类型: conversation (chat_template={args.data.chat_template}, "
            f"text_keys={args.data.text_keys}, max_seq_len={args.data.max_seq_len})"
        )
    else:
        transform = partial(
            _data_transform.process_plaintext_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
        logger.info_rank0(
            f"  数据类型: plaintext (text_keys={args.data.text_keys}, "
            f"max_seq_len={args.data.max_seq_len})"
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

    # OPD模式：使用pad_to_length=max_seq_len避免crash，在sample_from_student中拆分packed序列
    # 标准模式：使用动态打包填充到max_seq_len
    if args.opdttt.enable_opd_sampling:
        collate_fn_kwargs = {
            "pad_to_length": args.data.max_seq_len,
        }
        use_dyn_bsz = False
        opd_dataloader_batch_size = args.train.global_batch_size // args.train.data_parallel_size
        logger.info_rank0(f"OPD模式: pad_to_length={args.data.max_seq_len}, dataloader_batch_size={opd_dataloader_batch_size}")
    else:
        collate_fn_kwargs = {
            "pad_to_length": args.data.max_seq_len,
        }
        use_dyn_bsz = getattr(args.train, "dyn_bsz", True)
        opd_dataloader_batch_size = args.train.dataloader_batch_size

    train_dataloader = build_dataloader(
        dataloader_type=args.data.dataloader.type,
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=opd_dataloader_batch_size,
        seed=args.train.seed,
        max_seq_len=args.data.max_seq_len,
        train_steps=train_steps,
        dyn_bsz_buffer_size=getattr(args.data, "dyn_bsz_buffer_size", 500),
        dyn_bsz=use_dyn_bsz,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        collate_fn_kwargs=collate_fn_kwargs,
        prefetch_factor=None if args.data.num_workers == 0 else getattr(args.data.dataloader, "prefetch_factor", 2),
    )

    # replay 模式：重放 N 步 dataloader 并保存状态（用于重建缺失的 dataloader checkpoint）
    # 用法: OPDTTT_REPLAY_STEPS=1200 torchrun --nproc_per_node=4 ...
    replay_steps_str = os.environ.get("OPDTTT_REPLAY_STEPS", "").strip()
    if replay_steps_str:
        replay_steps = int(replay_steps_str)
        save_dir = os.environ.get("OPDTTT_REPLAY_SAVE_DIR", "data/output/dataloader_states")
        os.makedirs(save_dir, exist_ok=True)
        logger.info_rank0(f"===== REPLAY 模式: 重放 {replay_steps} 步 dataloader =====")
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(0)
        data_iter = iter(train_dataloader)
        t0 = time.time()
        for i in range(replay_steps):
            _ = next(data_iter)
            if (i + 1) % 100 == 0:
                logger.info_rank0(f"REPLAY: {i+1}/{replay_steps} 步 ({time.time()-t0:.1f}s)")
        dl_state = train_dataloader.state_dict()
        rank = args.train.global_rank
        save_file = os.path.join(save_dir, f"dl_state_step{replay_steps}_rank{rank}.pt")
        torch.save(dl_state, save_file)
        logger.info_rank0(f"REPLAY 完成: 状态保存到 {save_file} (耗时 {time.time()-t0:.1f}s)")
        dist.barrier()
        dist.destroy_process_group()
        return

    # 环境变量覆盖（用于 eval-only 等场景，优先级高于配置文件）
    _eval_ttt_lr = os.environ.get("OPDTTT_EVAL_TTT_LR", "").strip()
    if _eval_ttt_lr:
        args.opdttt.ttt_lr = float(_eval_ttt_lr)
        logger.info_rank0(f"环境变量覆盖: ttt_lr={args.opdttt.ttt_lr} (OPDTTT_EVAL_TTT_LR)")
    _override_load_path = os.environ.get("OPDTTT_LOAD_PATH", "").strip()
    if _override_load_path:
        args.train.checkpoint.load_path = _override_load_path
        logger.info_rank0(f"环境变量覆盖: load_path={args.train.checkpoint.load_path} (OPDTTT_LOAD_PATH)")

    # 构建学生模型
    logger.info_rank0("构建学生模型")
    ModelClass = _get_model_class(args.opdttt.model_type)
    logger.info_rank0(f"模型类型: {args.opdttt.model_type}, 类: {ModelClass.__name__}")

    config_path = args.model.config_path or args.model.model_path
    config = AutoConfig.from_pretrained(config_path)

    # 应用 OPD-TTT 设置
    config.opdttt_mode = True
    config.opdttt_layers = args.opdttt.opdttt_layers
    config.lambda_kl = float(args.opdttt.lambda_kl)
    config.lambda_lm = float(args.opdttt.lambda_lm)
    config.lambda_ntp = float(args.opdttt.lambda_ntp)
    config.lambda_align_rep = float(args.opdttt.lambda_align_rep)
    config.ttt_lr = float(args.opdttt.ttt_lr)
    config.ttt_chunk = int(args.opdttt.ttt_chunk)
    config.ttt_proj = args.opdttt.ttt_proj
    config.ttt_max_norm = float(args.opdttt.ttt_max_norm)

    # 模型类型特定配置
    if args.opdttt.model_type == "qwen3":
        config.ttt_target = "hidden_states"
        config.ttt_mode = True
        config.ttt_layers = args.opdttt.opdttt_layers
        if args.opdttt.sliding_window > 0:
            config.use_sliding_window = True
            config.sliding_window = args.opdttt.sliding_window
            full_every = 7
            config.layer_types = [
                "full_attention" if i % full_every == 3 else "sliding_attention"
                for i in range(config.num_hidden_layers)
            ]
            n_full = sum(1 for t in config.layer_types if t == "full_attention")
            n_sliding = config.num_hidden_layers - n_full
            logger.info_rank0(f"Qwen3 SWA: sliding_window={args.opdttt.sliding_window}, full={n_full}, sliding={n_sliding}")
        else:
            config.layer_types = ["full_attention"] * config.num_hidden_layers
    elif args.opdttt.model_type == "qwen3_5":
        # Qwen3.5 使用 text_config 嵌套结构
        tc = config.text_config
        tc.ttt_target = "hidden_states"
        tc.ttt_mode = True
        tc.ttt_layers = args.opdttt.opdttt_layers
        # 将 OPD-TTT 参数注入 text_config（模型从 text_config 读取）
        tc.opdttt_mode = True
        tc.opdttt_layers = args.opdttt.opdttt_layers
        tc.lambda_kl = float(args.opdttt.lambda_kl)
        tc.lambda_lm = float(args.opdttt.lambda_lm)
        tc.lambda_ntp = float(args.opdttt.lambda_ntp)
        tc.lambda_align_rep = float(args.opdttt.lambda_align_rep)
        tc.ttt_lr = float(args.opdttt.ttt_lr)
        tc.ttt_chunk = int(args.opdttt.ttt_chunk)
        tc.ttt_proj = args.opdttt.ttt_proj
        tc.ttt_max_norm = float(args.opdttt.ttt_max_norm)
        tc.weight_adaptation = args.opdttt.weight_adaptation
        tc.teacher_proj_init = args.opdttt.teacher_proj_init
        # Qwen3.5 已有 layer_types，不需要覆盖
        # SWA: 将 sliding_window 注入 text_config，full_attention 层会使用
        if args.opdttt.sliding_window > 0:
            tc.sliding_window = args.opdttt.sliding_window
            logger.info_rank0(f"Qwen3.5 SWA: sliding_window={args.opdttt.sliding_window} (full_attention layers only)")
        logger.info_rank0(f"Qwen3.5: layers={tc.num_hidden_layers}, layer_types={tc.layer_types}, TTT layers={tc.opdttt_layers}")
    else:
        config.ttt_target = "hidden_states"
        if args.opdttt.sliding_window > 0:
            config.sliding_window = args.opdttt.sliding_window
            logger.info_rank0(f"Llama SWA: sliding_window={args.opdttt.sliding_window}")

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
    teacher_path = teacher_path.strip("'\"")
    if not teacher_path:
        teacher_path = ""

    # Qwen3.5 的属性在 text_config 嵌套结构里，统一用 base_cfg 引用
    base_cfg = config.text_config if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size') else config

    if teacher_path:
        from transformers import AutoConfig as AutoConfigTeacher
        try:
            teacher_config = AutoConfigTeacher.from_pretrained(teacher_path)
            # Qwen3.5 使用 text_config 嵌套结构
            if hasattr(teacher_config, 'text_config') and hasattr(teacher_config.text_config, 'hidden_size'):
                config.teacher_hidden_size = teacher_config.text_config.hidden_size
            else:
                config.teacher_hidden_size = teacher_config.hidden_size
            logger.info_rank0(f"从教师配置读取 teacher_hidden_size: {config.teacher_hidden_size}")
        except Exception as e:
            logger.info_rank0(f"无法从教师配置读取 teacher_hidden_size: {e}，使用学生模型 hidden_size: {base_cfg.hidden_size}")
            config.teacher_hidden_size = base_cfg.hidden_size
    else:
        # 如果没有教师模型，使用学生模型的 hidden_size
        config.teacher_hidden_size = base_cfg.hidden_size
        logger.info_rank0(f"没有配置教师模型，使用学生模型 hidden_size 作为 teacher_hidden_size: {config.teacher_hidden_size}")

    # Qwen3.5: 将 teacher_hidden_size 同步到 text_config
    if args.opdttt.model_type == "qwen3_5" and hasattr(config, 'text_config'):
        config.text_config.teacher_hidden_size = config.teacher_hidden_size

    # 检查是否存在模型权重，如果不存在则从配置初始化
    model_path = args.model.model_path
    has_weights = any(
        f.endswith(('.safetensors', '.bin', '.pt'))
        for f in os.listdir(model_path)
        if os.path.isfile(os.path.join(model_path, f))
    ) if os.path.isdir(model_path) else False

    # FSDP2 会从 weights_path 重新加载权重（不经过 from_pretrained），
    # 需要确保 checkpoint 中不存在形状不匹配的参数。
    # 当 teacher_hidden_size 变化时（如从 SFT 到 OPD），teacher_proj.weight 会不匹配。
    fsdp_weights_path = args.model.model_path if has_weights else None
    if has_weights:
        from safetensors import safe_open
        import tempfile
        from collections import OrderedDict

        # Qwen3.5 权重重映射：model.language_model.* -> model.*，过滤 visual/mtp
        needs_remapping = args.opdttt.model_type == "qwen3_5"
        if needs_remapping:
            logger.info_rank0("Qwen3.5: 检测到多模态权重，执行键名重映射 model.language_model.* -> model.*")
            tmp_dir = tempfile.mkdtemp(prefix="qwen35_ckpt_", dir=os.path.join(_project_root, "data", "output"))
            from safetensors.torch import save_file
            import shutil
            for f in os.listdir(model_path):
                src = os.path.join(model_path, f)
                if not f.endswith('.safetensors'):
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(tmp_dir, f))
                    continue
                remapped = {}
                with safe_open(src, framework="pt") as st:
                    for key in st.keys():
                        if key.startswith("model.language_model."):
                            new_key = "model." + key[len("model.language_model."):]
                            remapped[new_key] = st.get_tensor(key).clone()
                        elif key.startswith("model.visual.") or key.startswith("mtp."):
                            continue
                        elif key == "lm_head.weight":
                            continue  # tied weights
                        else:
                            remapped[key] = st.get_tensor(key).clone()
                save_file(remapped, os.path.join(tmp_dir, f))
            logger.info_rank0(f"Qwen3.5: 键名重映射完成，保存到 {tmp_dir}")
            model_path = tmp_dir
            fsdp_weights_path = tmp_dir

        # 用 meta device 创建模型来获取期望的参数形状
        with torch.device("meta"):
            meta_model = ModelClass(config)

        expected_shapes = {
            name: param.shape for name, param in meta_model.named_parameters()
        }

        # 检查 checkpoint 中的参数形状是否匹配
        mismatched_keys = []
        for f in os.listdir(model_path):
            if not f.endswith('.safetensors'):
                continue
            filepath = os.path.join(model_path, f)
            with safe_open(filepath, framework="pt") as st:
                for key in st.keys():
                    if key in expected_shapes and tuple(st.get_tensor(key).shape) != tuple(expected_shapes[key]):
                        mismatched_keys.append(key)

        del meta_model

        if mismatched_keys:
            logger.info_rank0(f"检测到形状不匹配的参数: {mismatched_keys}")
            logger.info_rank0("创建过滤后的临时 checkpoint（跳过不匹配参数）")

            # 创建临时目录保存过滤后的 checkpoint
            tmp_dir = tempfile.mkdtemp(prefix="opdttt_ckpt_", dir=os.path.join(_project_root, "data", "output"))
            for f in os.listdir(model_path):
                src = os.path.join(model_path, f)
                if not f.endswith('.safetensors'):
                    import shutil
                    shutil.copy2(src, os.path.join(tmp_dir, f))
                    continue
                # 过滤不匹配的 key
                from safetensors.torch import save_file
                filtered = {}
                with safe_open(src, framework="pt") as st:
                    for key in st.keys():
                        if key not in mismatched_keys:
                            filtered[key] = st.get_tensor(key).clone()
                save_file(filtered, os.path.join(tmp_dir, f))
                logger.info_rank0(f"已过滤 {len(mismatched_keys)} 个不匹配参数，保存到 {tmp_dir}")

            fsdp_weights_path = tmp_dir
        else:
            fsdp_weights_path = model_path

    if has_weights:
        if args.opdttt.model_type == "qwen3_5":
            # Qwen3.5: from_pretrained 的 post_init 会覆盖加载的权重，改用手动加载
            logger.info_rank0(f"Qwen3.5: 从配置创建模型 + 手动加载预训练权重: {model_path}")
            student_model = ModelClass._from_config(
                config,
                torch_dtype="bfloat16" if args.train.enable_mixed_precision else "float32",
                attn_implementation=args.model.attn_implementation,
            )
            # 手动加载权重
            from safetensors.torch import load_file
            import glob as _glob
            model_sd = student_model.state_dict()
            loaded = 0
            for sf in sorted(_glob.glob(os.path.join(model_path, "*.safetensors"))):
                ckpt_sd = load_file(sf)
                for k, v in ckpt_sd.items():
                    if k in model_sd and model_sd[k].shape == v.shape:
                        model_sd[k] = v.to(model_sd[k].dtype)
                        loaded += 1
            missing, unexpected = student_model.load_state_dict(model_sd, strict=False)
            logger.info_rank0(f"Qwen3.5: 手动加载 {loaded} 个权重张量, {len(missing)} 个 missing (TTT 层)")
        else:
            logger.info_rank0(f"从预训练权重加载模型: {model_path}")
            student_model = ModelClass.from_pretrained(
                model_path,
                config=config,
                torch_dtype="bfloat16" if args.train.enable_mixed_precision else "float32",
                attn_implementation=args.model.attn_implementation,
                ignore_mismatched_sizes=True,
            )
    else:
        logger.info_rank0(f"从配置初始化模型 (训练从开始): {config_path}")
        student_model = ModelClass._from_config(
            config,
            torch_dtype="bfloat16" if args.train.enable_mixed_precision else "float32",
            attn_implementation=args.model.attn_implementation,
        )

    # 确认模型实际使用的 TTT 配置
    logger.info_rank0(f"模型 TTT 配置确认: ttt_chunk={config.ttt_chunk}, opdttt_layers={config.opdttt_layers}, ttt_lr={config.ttt_lr}")

    # 对于 DDP 模式，需要将模型移动到目标设备（FSDP2 模式由 build_parallelize_model 处理）
    from veomni.distributed.parallel_state import get_parallel_state
    parallel_state = get_parallel_state()
    if parallel_state.dp_mode == "ddp":
        device = torch.device(f"cuda:{parallel_state.local_rank}")
        logger.info_rank0(f"将模型移动到设备: {device}")
        student_model = student_model.to(device)

    # 并行化模型
    # 使用过滤后的 checkpoint 路径（跳过形状不匹配的参数）
    student_model = build_parallelize_model(
        student_model,
        init_device=args.train.init_device,
        weights_path=fsdp_weights_path if has_weights else None,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=student_model._no_split_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
    )

    # DDP 模式下 build_parallelize_model 会 upcast 到 float32，需转回 bfloat16
    # 这与 FSDP2 MixedPrecision（forward 前 cast 到 bf16）等价，PPL 可信
    if parallel_state.dp_mode == "ddp":
        student_model = student_model.bfloat16()
        logger.info_rank0("DDP 模式: 模型已转为 bfloat16 (与 FSDP2 MixedPrecision 等价)")

    helper.print_device_mem_info("构建学生模型后的显存使用")

    # 构建教师模型
    # OPD模式下每个rank都需要教师模型来计算teacher logprobs
    # 非OPD模式下仅rank0需要（用于teacher-student对齐损失）
    teacher_model = None
    teacher_path = args.opdttt.teacher_model_path.strip().strip("'\"") if args.opdttt.teacher_model_path else ""
    if not teacher_path:
        teacher_path = ""
    need_teacher = bool(teacher_path) and (
        args.opdttt.enable_opd_sampling
        or args.opdttt.lambda_kl > 0
        or args.opdttt.lambda_align_rep > 0
    )
    if need_teacher:
        device_str = f"{get_device_type()}:{args.train.local_rank}"
        teacher_model = build_teacher_model(
            teacher_path,
            device_str,
            torch_dtype="bfloat16" if args.train.enable_mixed_precision else "float32",
            attn_implementation=args.model.attn_implementation,
            tokenizer=tokenizer,
        )
        logger.info(f"[rank {args.train.global_rank}] 教师模型已加载: {teacher_path}")
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
        _wb_name = args.train.wandb_name
        if os.environ.get("OPDTTT_EVAL_ONLY", "").strip() == "1":
            _wb_name = f"{_wb_name}-evalonly"
        wandb.init(
            project=args.train.wandb_project,
            name=_wb_name,
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

    # 恢复训练：从DCP检查点加载模型、优化器、学习率调度器
    start_step = 0
    load_path = args.train.checkpoint.load_path.strip() if args.train.checkpoint.load_path else ""
    if load_path:
        logger.info_rank0(f"从检查点恢复训练: {load_path}")
        load_state = {
            "model": student_model,
            "optimizer": optimizer,
            "extra_state": {},
        }
        checkpointer.load(load_path, load_state)
        extra_state = load_state.get("extra_state", {})
        start_step = extra_state.get("global_step", 0)
        if "lr_scheduler" in extra_state:
            lr_scheduler.load_state_dict(extra_state["lr_scheduler"])

        # 恢复 dataloader 状态（断点续训时从中断处继续读取数据）
        if "dataloader_state" in extra_state:
            try:
                train_dataloader.load_state_dict(extra_state["dataloader_state"])
                logger.info_rank0("dataloader 状态已恢复")
            except Exception as e:
                logger.warning(f"无法恢复 dataloader 状态: {e}")
        else:
            logger.warning_rank0("检查点中无 dataloader 状态，数据将从文件开头重新读取")

        # 恢复 RNG 状态
        if "rng_state" in extra_state:
            try:
                import torch as _torch
                _torch.set_rng_state(extra_state["rng_state"]["python"])
                if extra_state["rng_state"]["cuda"] is not None and _torch.cuda.is_available():
                    _torch.cuda.set_rng_state(extra_state["rng_state"]["cuda"])
                logger.info_rank0("RNG 状态已恢复")
            except Exception as e:
                logger.warning(f"无法恢复 RNG 状态: {e}")

        logger.info_rank0(f"恢复成功: start_step={start_step}")
        dist.barrier()

    # eval-only 模式：加载 checkpoint 后只评估一次 PPL，不训练
    # 用于验证评估修复正确性、观察结果与耗时（OPDTTT_EVAL_ONLY=1 触发）
    if os.environ.get("OPDTTT_EVAL_ONLY", "").strip() == "1":
        _eval_step = start_step if start_step > 0 else 0
        logger.info_rank0(f"===== EVAL-ONLY 模式: 评估 step {_eval_step} =====")
        _eval_start = time.time()
        trainer._evaluate_ppl(_eval_step)
        synchronize()
        dist.barrier()
        _eval_elapsed = time.time() - _eval_start
        if args.train.global_rank == 0:
            logger.info(f"===== EVAL-ONLY 完成: 耗时 {_eval_elapsed:.1f}s =====")
        del optimizer, lr_scheduler
        helper.empty_cache()
        dist.barrier()
        dist.destroy_process_group()
        return

    trainer.train(train_steps=train_steps, start_step=start_step)

    # 清理
    synchronize()
    del optimizer, lr_scheduler
    helper.empty_cache()

    dist.barrier()
    dist.destroy_process_group()

    logger.info("训练完成!")


if __name__ == "__main__":
    main()
