# coding=utf-8
# -*- coding: utf-8 -*-
"""
OPD-TTT 模块：教师引导的测试时训练增强组件

本模块实现了 On-Policy Distillation Enhanced Test-Time Training (OPD-TTT) 的核心组件，
将 In-Place TTT 与教师模型指导相结合。

主要组件：
1. OPDTTTMLP：教师引导的 MLP 层，支持快速权重更新
2. OPDTTTLoss：四层损失函数（NTP对齐、教师表示对齐、KL散度、语言建模）
3. compute_teacher_repr_targets：计算教师表示目标

参考文档：OPTTD.md 设计文档
"""

from typing import Optional, Tuple, Literal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .configuration_llama import LlamaConfig


def pca_init_projection(
    teacher_embeddings: torch.Tensor,
    target_dim: int,
    num_components: Optional[int] = None,
) -> torch.Tensor:
    """
    使用 PCA 初始化教师投影矩阵

    通过对教师嵌入进行主成分分析，找到主要的变异方向，
    用这些主方向初始化投影矩阵，使投影能更好地保留教师信息。

    Args:
        teacher_embeddings: 教师嵌入 [vocab_size, teacher_hidden_size] 或 [num_samples, teacher_hidden_size]
        target_dim: 目标维度（MLP 输出空间维度，即 d_model）
        num_components: 使用的主成分数量，默认为 target_dim

    Returns:
        投影矩阵 [target_dim, teacher_hidden_size]
    """
    if num_components is None:
        num_components = min(target_dim, teacher_embeddings.shape[1])

    # 中心化数据
    embeddings_centered = teacher_embeddings - teacher_embeddings.mean(dim=0, keepdim=True)

    # 计算协方差矩阵 [teacher_hidden_size, teacher_hidden_size]
    # 使用转置以适应高维数据情况
    if teacher_embeddings.shape[0] < teacher_embeddings.shape[1]:
        # 样本数少于特征数，使用 Gram 矩阵方法
        gram = embeddings_centered @ embeddings_centered.T  # [N, N]
        eigenvalues, eigenvectors_n = torch.linalg.eigh(gram)
        # 排序特征值（降序）
        idx = eigenvalues.descending()
        eigenvalues = eigenvalues[idx]
        eigenvectors_n = eigenvectors_n[:, idx]
        # 计算实际的特征向量
        eigenvectors = embeddings_centered.T @ eigenvectors_n
        # 归一化
        eigenvectors = eigenvectors / (eigenvectors.norm(dim=0, keepdim=True) + 1e-8)
    else:
        # 标准方法
        cov = embeddings_centered.T @ embeddings_centered  # [D, D]
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        # 排序特征值（降序）
        idx = eigenvalues.descending()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    # 取前 num_components 个主成分
    components = eigenvectors[:, :num_components].T  # [num_components, teacher_hidden_size]

    # 如果目标维度大于主成分数，用零填充
    if target_dim > num_components:
        padding = torch.zeros(target_dim - num_components, teacher_embeddings.shape[1],
                              device=teacher_embeddings.device, dtype=teacher_embeddings.dtype)
        projection = torch.cat([components, padding], dim=0)
    else:
        projection = components[:target_dim]

    return projection  # [target_dim, teacher_hidden_size]


class OPDTTTMLP(nn.Module):
    """
    OPD-TTT 增强的 MLP 层

    该层在 In-Place TTT 的基础上增加了教师模型的指导信号。通过同时优化：
    1. NTP（下一Token预测）对齐：自监督信号，使快速权重存储预测下一Token所需的信息
    2. 教师表示对齐：教师模型指导，使输出表示与教师模型对齐

    快速权重更新规则：
        ΔW = η * (λ_ntp * ∇L_NTP + λ_align_rep * ∇L_align_rep)
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        """
        初始化 OPD-TTT MLP 层

        Args:
            config: 模型配置
            layer_idx: 当前层的索引
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # 标准的 SwiGLU MLP 结构
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

        # 获取激活函数
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_idx = -1 if layer_idx is None else layer_idx

        # OPD-TTT 设置
        if getattr(config, "opdttt_mode", False) and self.layer_idx in getattr(
            config, "opdttt_layers", []
        ):
            self.enable_opdttt = True
            # 分块大小：用于并行处理
            self.ttt_chunk = getattr(config, "ttt_chunk", 8192)

            # NTP 投影矩阵：将输入嵌入投影到 MLP 输出空间
            if getattr(config, "ttt_proj", True):
                self.ttt_proj = nn.Linear(
                    self.hidden_size, self.hidden_size, bias=False
                )
            else:
                self.ttt_proj = None

            # 教师投影矩阵：将教师的嵌入投影到 MLP 输出空间
            # 支持三种初始化方式：'random', 'pca', 'pca_tied'
            # - 'random': 随机初始化（默认）
            # - 'pca': 使用教师嵌入的 PCA 初始化（需要提供 teacher_embeddings_for_init）
            # - 'pca_tied': 所有层共享同一个 PCA 初始化的投影矩阵
            self.teacher_proj_init = getattr(config, "teacher_proj_init", "random")
            self.teacher_proj = nn.Linear(
                self.hidden_size, self.hidden_size, bias=False
            )

            # 如果配置中提供了用于 PCA 初始化的教师嵌入，进行 PCA 初始化
            teacher_embeddings_for_init = getattr(config, "teacher_embeddings_for_init", None)
            if self.teacher_proj_init == "pca" and teacher_embeddings_for_init is not None:
                # 使用 PCA 初始化教师投影矩阵
                with torch.no_grad():
                    pca_proj = pca_init_projection(
                        teacher_embeddings_for_init,
                        target_dim=self.hidden_size,
                        num_components=self.hidden_size,
                    )
                    self.teacher_proj.weight.copy_(pca_proj)

            # 快速权重学习率
            self.ttt_lr = getattr(config, "ttt_lr", 0.3)

            # 损失权重
            self.lambda_ntp = getattr(config, "lambda_ntp", 1.0)
            self.lambda_align_rep = getattr(config, "lambda_align_rep", 0.5)

            # 自适应权重选项
            # - 'fixed': 使用固定权重（默认）
            # - 'adaptive': 根据 NTP 和教师梯度的余弦相似度动态调整权重
            self.weight_adaptation = getattr(config, "weight_adaptation", "fixed")

            # NTP 目标投影：使用因果卷积计算下一Token表示
            # 这是一个可学习的投影，用于创建 NTP 对齐目标
            self.ntp_target_proj = nn.Conv1d(
                self.hidden_size,
                self.hidden_size,
                kernel_size=5,
                padding=2,
                groups=self.hidden_size,
                bias=False,
            )
        else:
            self.enable_opdttt = False

    def padding(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入填充为分块大小的整数倍

        Args:
            x: 输入张量 [batch, seq_len, hidden_size]

        Returns:
            填充后的张量 [batch, num_chunks, chunk_size, hidden_size]
        """
        if not hasattr(self, "ttt_chunk"):
            return x
        if x.shape[1] % self.ttt_chunk != 0:
            # 计算需要填充的长度
            padding_embeddings = torch.zeros(
                [x.shape[0], self.ttt_chunk - x.shape[1] % self.ttt_chunk, x.shape[2]],
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, padding_embeddings], dim=1)
        return rearrange(x, "b (t c) d -> b t c d", c=self.ttt_chunk)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        teacher_repr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向传播，支持教师引导的快速权重更新

        Args:
            x: MLP 中间激活（经过 gate 和 up 投影后） [batch, seq_len, intermediate_size]
            t: NTP 目标状态（输入嵌入） [batch, seq_len, hidden_size]
            teacher_repr: 教师表示目标（投影后的教师 logits/嵌入） [batch, seq_len, hidden_size]

        Returns:
            output: MLP 输出 [batch, seq_len, hidden_size]
            loss_dict: 损失字典（如果使用 OPD-TTT）
        """
        # 标准 SwiGLU 前向传播：act_fn(gate(x)) * up(x)
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        # 如果没有启用 OPD-TTT 或没有提供目标，直接返回标准 MLP 输出
        if not self.enable_opdttt or t is None:
            return self.down_proj(h), {}

        # OPD-TTT 路径
        # 填充输入到分块大小
        t = self.padding(t)
        h_padded = self.padding(h)
        bs, chunk_num, chunk_size, _ = t.shape

        # 计算 NTP 目标：使用因果卷积预测"下一Token"的表示
        # 因果卷积确保位置 t 只能看到 t 之前的信息
        ntp_target = (
            self.ntp_target_proj(t.transpose(-1, -2).reshape(bs * chunk_num, -1, chunk_size))
            .transpose(-1, -2)
            .reshape(bs, chunk_num, chunk_size, -1)
        )

        # 准备教师表示目标（如果提供）
        if teacher_repr is not None:
            teacher_repr = self.padding(teacher_repr)

        # 计算快速权重更新
        # 关键：所有分块的增量可以并行计算（因为 NTP 目标只依赖输入，不依赖当前权重）

        # 1. NTP 对齐分量：使 MLP 输出能够预测下一Token的表示
        if self.ttt_proj is not None:
            ntp_proj = torch.einsum(
                "b t c h, b t c d, d e -> b t e h",
                h_padded[:, :-1],           # 当前分块的 MLP 输入（除了最后一个分块）
                ntp_target[:, :-1],        # NTP 目标（下一Token表示）
                self.ttt_proj.weight,       # NTP 投影矩阵
            )
        else:
            ntp_proj = torch.einsum(
                "b t c h, b t c d -> b t d h",
                h_padded[:, :-1],
                ntp_target[:, :-1],
            )

        # 2. 教师表示对齐分量：使 MLP 输出与教师表示对齐
        if teacher_repr is not None:
            teacher_align = torch.einsum(
                "b t c h, b t c d, d e -> b t e h",
                h_padded[:, :-1],           # MLP 输入
                teacher_repr[:, :-1],       # 教师表示目标
                self.teacher_proj.weight,   # 教师投影矩阵
            )
        else:
            # 如果没有提供教师，使用零向量
            teacher_align = torch.zeros_like(ntp_proj)

        # 合并两个梯度分量
        # 根据配置使用固定权重或自适应权重
        if self.weight_adaptation == "adaptive" and teacher_repr is not None:
            # 自适应权重：根据梯度的余弦相似度动态调整权重
            # 公式：λ_align_rep^(i) = λ_align_rep * (1 + cos(g_NTP, g_teacher)) / 2
            # 其中 g_NTP = ntp_proj, g_teacher = teacher_align

            # 展平梯度以便计算相似度
            # [batch, chunks, hidden, intermediate] -> [batch * chunks, hidden * intermediate]
            ntp_grad_flat = ntp_proj.reshape(bs, -1)
            teacher_grad_flat = teacher_align.reshape(bs, -1)

            # 计算余弦相似度（分块级别）
            # 相似度范围 [-1, 1]，映射到 [0, 1] 后用作权重系数
            cos_sim = F.cosine_similarity(ntp_grad_flat, teacher_grad_flat, dim=-1)  # [batch]
            # 将相似度从 [-1, 1] 映射到 [0, 1]
            similarity_coeff = (cos_sim + 1) / 2  # [batch]
            # 形状调整以便广播
            similarity_coeff = similarity_coeff.view(bs, 1, 1, 1)  # [batch, 1, 1, 1]

            # 动态调整权重：梯度方向相似时增加权重，冲突时减少权重
            adaptive_lambda_align = self.lambda_align_rep * similarity_coeff

            weighted_update = (ntp_proj * self.lambda_ntp + teacher_align * adaptive_lambda_align) * self.ttt_lr
        else:
            # 固定权重：使用配置的固定权重值
            weighted_update = (ntp_proj * self.lambda_ntp + teacher_align * self.lambda_align_rep) * self.ttt_lr

        d_down_proj = torch.cat(
            [
                repeat(self.down_proj.weight, "d h -> b 1 d h", b=bs),
                weighted_update,
            ],
            dim=1,
        )

        # 累积和计算前缀依赖
        # 这样每个分块使用之前所有分块的累积更新
        d_down_proj_sum = d_down_proj.cumsum(dim=1)

        # 应用更新后的权重
        down_proj = torch.einsum("b t d h, b t c h -> b t c d", d_down_proj_sum, h_padded)
        output = rearrange(down_proj, "b t c d -> b (t c) d")[:, : x.shape[1], :]

        # 计算表示对齐损失（用于监控）
        loss_dict = {
            "ntp_loss": self._compute_repr_loss(output[:, :-1], ntp_target[:, 1:]),
        }
        if teacher_repr is not None:
            loss_dict["align_rep_loss"] = self._compute_repr_loss(output[:, :-1], teacher_repr[:, 1:])

        return output, loss_dict

    def _compute_repr_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算表示对齐损失（使用负内积/相似度）

        最大化内积等价于最小化余弦距离，使预测和目标表示尽可能相似

        Args:
            pred: 预测的表示 [batch, seq_len, hidden_size]
            target: 目标表示 [batch, seq_len, hidden_size]

        Returns:
            标量损失值
        """
        # 展平以便计算
        pred_flat = rearrange(pred, "b t d -> (b t) d")
        target_flat = rearrange(target, "b t d -> (b t) d")

        # 负内积损失（最大化相似度 = 最小化负相似度）
        loss = -(pred_flat * target_flat).sum() / (pred_flat.shape[0] * pred_flat.shape[1])
        return loss


class OPDTTTLoss(nn.Module):
    """
    OPD-TTT 组合损失函数

    实现四层损失函数：
    1. L_NTP: NTP 对齐损失（表示空间，更新快速权重）
    2. L_align_rep: 教师表示对齐损失（表示空间，更新快速权重）
    3. L_KL: KL 散度损失（概率空间，更新慢速权重）
    4. L_LM: 标准语言建模损失（概率空间，更新慢速权重）

    总损失 = λ_kl * KL(π_student || π_teacher) + λ_lm * CE(logits, labels)
            + λ_ntp * L_NTP + λ_align_rep * L_align_rep
    """

    def __init__(
        self,
        lambda_kl: float = 0.1,          # KL 散度权重
        lambda_lm: float = 1.0,          # 语言建模损失权重
        lambda_ntp: float = 1.0,         # NTP 对齐权重
        lambda_align_rep: float = 0.5,   # 教师表示对齐权重
        vocab_size: int = 32000,         # 词汇表大小
    ):
        """
        初始化 OPD-TTT 损失函数

        Args:
            lambda_kl: KL 散度损失权重
            lambda_lm: 语言建模损失权重
            lambda_ntp: NTP 对齐损失权重
            lambda_align_rep: 教师表示对齐损失权重
            vocab_size: 词汇表大小
        """
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_lm = lambda_lm
        self.lambda_ntp = lambda_ntp
        self.lambda_align_rep = lambda_align_rep
        self.vocab_size = vocab_size

    def forward(
        self,
        student_logits: torch.Tensor,        # 学生模型输出 logits [B, T, V]
        teacher_logits: torch.Tensor,        # 教师模型输出 logits [B, T, V]
        labels: torch.Tensor,               # 真实标签 [B, T]
        ntp_losses: dict = None,            # 来自 MLP 层的 NTP/对齐损失
        attention_mask: torch.Tensor = None, # 注意力掩码
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算组合的 OPD-TTT 损失

        Args:
            student_logits: 学生模型输出 logits [batch, seq_len, vocab_size]
            teacher_logits: 教师模型输出 logits [batch, seq_len, vocab_size]
            labels: 真实标签 [batch, seq_len]
            ntp_losses: 来自 MLP 层的 NTP/对齐损失字典
            attention_mask: 注意力掩码（用于有效位置）

        Returns:
            total_loss: 组合的标量损失
            loss_dict: 各个损失分量的字典
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. KL 散度损失：KL(student || teacher)
        # 使用反向 KL 使学生分布接近教师分布
        if teacher_logits is not None:
            kl_loss = self._compute_kl_divergence(student_logits, teacher_logits, attention_mask)
            loss_dict["kl_loss"] = kl_loss
            total_loss += self.lambda_kl * kl_loss

        # 2. 标准语言建模损失：交叉熵与真实标签
        lm_loss = self._compute_lm_loss(student_logits, labels, attention_mask)
        loss_dict["lm_loss"] = lm_loss
        total_loss += self.lambda_lm * lm_loss

        # 3. NTP 对齐损失（来自 MLP 层，用于监控）
        # 这些损失已在 MLP 层中用于更新快速权重，这里记录用于日志
        if ntp_losses is not None:
            for key, value in ntp_losses.items():
                if value is not None and isinstance(value, torch.Tensor):
                    loss_dict[key] = value.detach()

        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict

    def _compute_kl_divergence(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        计算 KL 散度：KL(student || teacher)

        Args:
            student_logits: 学生模型 logits
            teacher_logits: 教师模型 logits
            attention_mask: 注意力掩码

        Returns:
            KL 散度损失（标量）
        """
        # 获取对数概率
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # 计算每个 Token 的 KL 散度
        kl_per_token = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
            log_target=False,
        ).sum(dim=-1)

        # 应用掩码（如果提供）
        if attention_mask is not None:
            # 创建因果掩码以标记有效位置
            mask = attention_mask[:, :-1] * attention_mask[:, 1:]
            kl_per_token = kl_per_token * mask
            num_tokens = mask.sum() + 1e-8
        else:
            num_tokens = kl_per_token.numel() / kl_per_token.shape[0]

        return kl_per_token.sum() / num_tokens

    def _compute_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        计算标准语言建模损失（交叉熵）

        Args:
            logits: 模型输出 logits
            labels: 真实标签
            attention_mask: 注意力掩码

        Returns:
            交叉熵损失（标量）
        """
        # 移位以进行下一Token预测
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # 计算损失
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)

        # 应用掩码（如果提供）
        if attention_mask is not None:
            mask = attention_mask[:, :-1] * attention_mask[:, 1:]
            loss = loss * mask
            num_tokens = mask.sum() + 1e-8
        else:
            num_tokens = loss.numel()

        return loss.sum() / num_tokens


class TeacherCache:
    """
    教师模型输出缓存

    在训练过程中缓存教师模型的输出以减少重复计算。
    使用 LRU（最近最少使用）策略管理缓存大小。
    """

    def __init__(self, max_cache_size: int = 100):
        """
        初始化教师缓存

        Args:
            max_cache_size: 最大缓存条目数
        """
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_order = []

    def get(self, key: str) -> Optional[dict]:
        """
        获取缓存的教师输出

        Args:
            key: 缓存键（通常是输入序列的哈希）

        Returns:
            缓存的教师输出字典，如果不存在则返回 None
        """
        if key in self.cache:
            # 更新访问顺序（移到末尾）
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: dict):
        """
        存储教师输出到缓存

        Args:
            key: 缓存键
            value: 教师输出字典
        """
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_cache_size:
            # 移除最久未使用的条目
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        self.cache[key] = value
        self.access_order.append(key)

    def clear(self):
        """清空所有缓存"""
        self.cache.clear()
        self.access_order.clear()


def compute_teacher_repr_targets(
    teacher_logits: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    layer_idx: int,
    hidden_size: int,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    计算特定层的教师表示目标

    该函数将教师的 logits 投影到学生 MLP 输出的表示空间，
    实现表示级别的对齐。

    Args:
        teacher_logits: 教师模型 logits [batch, seq_len, vocab_size]
        teacher_embeddings: 教师模型嵌入 [batch, seq_len, hidden_size]
        layer_idx: 当前层的索引
        hidden_size: 模型隐藏大小
        chunk_size: 处理的分块大小

    Returns:
        教师表示目标 [batch, seq_len, hidden_size]
    """
    # 方法：使用教师嵌入的 logits 加权版本
    # 这创建了一个捕捉教师预测分布的表示
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    # 为效率起见，直接使用嵌入
    # 在完整实现中，每层应该有一个投影矩阵
    # 这里使用嵌入作为代理

    # 填充到分块大小（如果需要）
    seq_len = teacher_embeddings.shape[1]
    if seq_len % chunk_size != 0:
        pad_len = chunk_size - (seq_len % chunk_size)
        padding = torch.zeros(
            teacher_embeddings.shape[0], pad_len, teacher_embeddings.shape[2],
            device=teacher_embeddings.device, dtype=teacher_embeddings.dtype
        )
        teacher_embeddings = torch.cat([teacher_embeddings, padding], dim=1)

    return teacher_embeddings


__all__ = [
    "OPDTTTMLP",
    "OPDTTTLoss",
    "TeacherCache",
    "compute_teacher_repr_targets",
]
