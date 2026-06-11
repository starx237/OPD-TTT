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

        self.device = get_torch_device()
        self.parallel_state = get_parallel_state()

        # 设置激活卸载上下文
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            args.train.enable_activation_offload,
            args.train.enable_gradient_checkpointing,
            args.train.activation_gpu_limit,
        )

        # 为 DDP 模式添加 autocast
        if args.train.data_parallel_mode == "ddp" and args.train.enable_mixed_precision:
            from contextlib import contextmanager

            @contextmanager
            def autocast_context_wrapper(inner_context):
                with inner_context:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        yield

            self.model_fwd_context = autocast_context_wrapper(self.model_fwd_context)
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
        with self.model_fwd_context:
            student_outputs = self.student_model(
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch.get("attention_mask"),
                position_ids=micro_batch.get("position_ids"),
                labels=micro_batch["labels"],
                teacher_logits=teacher_outputs["logits"] if teacher_outputs else None,
                teacher_hidden_states=teacher_outputs["hidden_states"] if teacher_outputs else None,
                teacher_embeddings=teacher_outputs["embeddings"] if teacher_outputs else None,
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

                # 计算批次统计
                length_in_batch = torch.tensor(0, dtype=torch.int32, device=self.device)
                for micro_batch in micro_batches:
                    length_in_batch += torch.sum(micro_batch["labels"] != -100)
                length_in_batch = all_reduce(
                    length_in_batch, op="sum", group=self.parallel_state.fsdp_group
                )

                # 处理小批量
                total_loss = 0.0
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
                    self._save_checkpoint(global_step, last_ckpt_path)
                    last_ckpt_path = os.path.join(
                        self.args.train.save_checkpoint_path, f"global_step_{global_step}"
                    )

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
        logger.info(f"检查点已保存: {save_path}")

        # 轮转旧检查点
        if last_ckpt_path is not None and os.path.isdir(last_ckpt_path):
            shutil.rmtree(last_ckpt_path, ignore_errors=True)
            logger.info(f"删除旧检查点: {last_ckpt_path}")

    def _save_hf_weights(self, global_step: int, last_ckpt_path: Optional[str]):
        """保存 HuggingFace 格式权重"""
        save_path = os.path.join(
            self.args.train.save_checkpoint_path, f"global_step_{global_step}", "hf_ckpt"
        )

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
) -> Optional[nn.Module]:
    """
    构建教师模型用于蒸馏

    Args:
        teacher_path: 教师模型路径
        device: 设备
        torch_dtype: 数据类型
        attn_implementation: 注意力实现类型

    Returns:
        教师模型（如果路径有效），否则返回 None
    """
    if not teacher_path:
        return None

    logger.info(f"加载教师模型: {teacher_path}")
    config = AutoConfig.from_pretrained(teacher_path)

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map={"": device},
    )
    teacher_model.eval()

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
    logger_info_rank0("构建分词器")
    tokenizer = build_tokenizer(args.model.tokenizer_path)

    # 准备数据
    logger_info_rank0("准备数据")
    transform = partial(
        _data_transform.process_pretrain_example,
        tokenizer=tokenizer,
        max_seq_len=args.data.max_seq_len,
        text_keys=args.data.text_keys,
    )

    train_dataset = build_dataset(
        dataset_name=args.data.dataset_name,
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

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        max_seq_len=args.data.max_seq_len,
        train_steps=train_steps,
        rmpad=getattr(args.train, "rmpad", True),
        rmpad_with_pos_ids=getattr(args.train, "rmpad_with_pos_ids", False),
        dyn_bsz_margin=getattr(args.train, "dyn_bsz_margin", 0),
        dyn_bsz_buffer_size=getattr(args.data, "dyn_bsz_buffer_size", 500),
        dyn_bsz=getattr(args.train, "dyn_bsz", True),
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
    )

    # 构建学生模型
    logger_info_rank0("构建学生模型")
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
        logger_info_rank0(f"加载教师嵌入用于 PCA 初始化：{args.opdttt.teacher_embeddings_path}")
        teacher_embeddings_for_init = torch.load(args.opdttt.teacher_embeddings_path)
        config.teacher_embeddings_for_init = teacher_embeddings_for_init

    student_model = OPDTTTForCausalLM.from_pretrained(
        args.model.model_path,
        config=config,
        torch_dtype="bfloat16" if args.train.enable_mixed_precision else "float32",
        attn_implementation=args.model.attn_implementation,
    )

    # 并行化模型
    student_model = build_parallelize_model(
        student_model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
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

    # 构建教师模型（仅在 rank 0）
    teacher_model = None
    if args.opdttt.teacher_model_path and args.train.global_rank == 0:
        teacher_model = build_teacher_model(
            args.opdttt.teacher_model_path,
            get_torch_device(),
            torch_dtype="bfloat16" if args.train.enable_mixed_precision else "float32",
            attn_implementation=args.model.attn_implementation,
        )

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
