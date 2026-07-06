# coding=utf-8
# -*- coding: utf-8 -*-
"""
OPD-TTT 模块：Qwen3.5 版本

基于 OPDQwen3MLP 适配 Qwen3.5 架构。
核心 TTT 逻辑完全复用（conv1d、chunk processing、fast weight update、Frobenius norm 裁剪），
仅适配 config 字段名（Qwen3.5 使用 Qwen3_5TextConfig）。
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def pca_init_projection(
    teacher_embeddings: torch.Tensor,
    target_dim: int,
    num_components: Optional[int] = None,
) -> torch.Tensor:
    if num_components is None:
        num_components = min(target_dim, teacher_embeddings.shape[1])
    embeddings_centered = teacher_embeddings - teacher_embeddings.mean(dim=0, keepdim=True)
    if teacher_embeddings.shape[0] < teacher_embeddings.shape[1]:
        gram = embeddings_centered @ embeddings_centered.T
        eigenvalues, eigenvectors_n = torch.linalg.eigh(gram)
        idx = eigenvalues.descending()
        eigenvalues = eigenvalues[idx]
        eigenvectors_n = eigenvectors_n[:, idx]
        eigenvectors = embeddings_centered.T @ eigenvectors_n
        eigenvectors = eigenvectors / (eigenvectors.norm(dim=0, keepdim=True) + 1e-8)
    else:
        cov = embeddings_centered.T @ embeddings_centered
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        idx = eigenvalues.descending()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    components = eigenvectors[:, :num_components].T
    if target_dim > num_components:
        padding = torch.zeros(target_dim - num_components, teacher_embeddings.shape[1],
                              device=teacher_embeddings.device, dtype=teacher_embeddings.dtype)
        projection = torch.cat([components, padding], dim=0)
    else:
        projection = components[:target_dim]
    return projection


class OPDQwen3_5MLP(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.teacher_hidden_size = getattr(config, "teacher_hidden_size", self.hidden_size)

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_idx = -1 if layer_idx is None else layer_idx

        if getattr(config, "opdttt_mode", False) and self.layer_idx in getattr(
            config, "opdttt_layers", []
        ):
            self.enable_opdttt = True
            self.ttt_chunk = getattr(config, "ttt_chunk", 8192)
            if getattr(config, "ttt_proj", True):
                self.ttt_proj = nn.Linear(
                    self.hidden_size, self.hidden_size, bias=False
                )
            else:
                self.ttt_proj = None
            self.teacher_proj_init = getattr(config, "teacher_proj_init", "random")
            self.teacher_proj = nn.Linear(
                self.teacher_hidden_size, self.hidden_size, bias=False
            )
            teacher_embeddings_for_init = getattr(config, "teacher_embeddings_for_init", None)
            if self.teacher_proj_init == "pca" and teacher_embeddings_for_init is not None:
                with torch.no_grad():
                    pca_proj = pca_init_projection(
                        teacher_embeddings_for_init,
                        target_dim=self.hidden_size,
                        num_components=self.hidden_size,
                    )
                    self.teacher_proj.weight.copy_(pca_proj)
            self.ttt_lr = getattr(config, "ttt_lr", 0.3)
            self.ttt_max_norm = getattr(config, "ttt_max_norm", 0)
            self.lambda_ntp = getattr(config, "lambda_ntp", 1.0)
            self.lambda_align_rep = getattr(config, "lambda_align_rep", 0.5)
            self.weight_adaptation = getattr(config, "weight_adaptation", "fixed")
            self.ttt_conv = nn.Conv1d(
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
        if not hasattr(self, "ttt_chunk"):
            return x
        if x.shape[1] % self.ttt_chunk != 0:
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
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        if not self.enable_opdttt or t is None:
            return self.down_proj(h), {}
        t = self.padding(t)
        h_padded = self.padding(h)
        bs, chunk_num, chunk_size, _ = t.shape
        ntp_target = (
            self.ttt_conv(t.transpose(-1, -2).reshape(bs * chunk_num, -1, chunk_size))
            .transpose(-1, -2)
            .reshape(bs, chunk_num, chunk_size, -1)
        )
        if teacher_repr is not None:
            teacher_repr = self.padding(teacher_repr)
        dtype = h_padded.dtype
        h_float = h_padded[:, :-1].float() if dtype == torch.bfloat16 else h_padded[:, :-1]
        ntp_target_float = ntp_target[:, :-1].float() if dtype == torch.bfloat16 else ntp_target[:, :-1]
        if self.ttt_proj is not None:
            ttt_proj_weight = self.ttt_proj.weight.float() if self.ttt_proj.weight.dtype == torch.bfloat16 else self.ttt_proj.weight
            ntp_proj = torch.einsum(
                "b t c h, b t c d, d e -> b t e h",
                h_float,
                ntp_target_float,
                ttt_proj_weight,
            )
        else:
            ntp_proj = torch.einsum(
                "b t c h, b t c d -> b t d h",
                h_float,
                ntp_target_float,
            )
        ntp_proj = ntp_proj.to(dtype)
        if self.lambda_align_rep > 0 and teacher_repr is not None:
            teacher_repr_float = teacher_repr[:, :-1].float() if teacher_repr.dtype == torch.bfloat16 else teacher_repr[:, :-1]
            teacher_proj_weight = self.teacher_proj.weight.float() if self.teacher_proj.weight.dtype == torch.bfloat16 else self.teacher_proj.weight
            teacher_align = torch.einsum(
                "b t c h, b t c d, e d -> b t e h",
                h_float,
                teacher_repr_float,
                teacher_proj_weight,
            )
            teacher_align = teacher_align.to(dtype)
        else:
            teacher_align = None
        if self.lambda_align_rep > 0 and teacher_align is not None:
            if self.weight_adaptation == "adaptive":
                ntp_grad_flat = ntp_proj.reshape(bs, -1)
                teacher_grad_flat = teacher_align.reshape(bs, -1)
                cos_sim = F.cosine_similarity(ntp_grad_flat, teacher_grad_flat, dim=-1)
                similarity_coeff = (cos_sim + 1) / 2
                similarity_coeff = similarity_coeff.view(bs, 1, 1, 1)
                adaptive_lambda_align = self.lambda_align_rep * similarity_coeff
                weighted_update = (ntp_proj * self.lambda_ntp + teacher_align * adaptive_lambda_align) * self.ttt_lr
            else:
                weighted_update = (ntp_proj * self.lambda_ntp + teacher_align * self.lambda_align_rep) * self.ttt_lr
        else:
            weighted_update = ntp_proj * self.lambda_ntp * self.ttt_lr
        if self.ttt_max_norm > 0:
            weighted_update_norm = torch.norm(weighted_update, p='fro', dim=[2, 3], keepdim=True)
            clip_coef = self.ttt_max_norm / (weighted_update_norm + 1e-8)
            clip_coef = torch.clamp(clip_coef, max=1.0)
            weighted_update = weighted_update * clip_coef
        d_down_proj = torch.cat(
            [
                repeat(self.down_proj.weight, "d h -> b 1 d h", b=bs),
                weighted_update,
            ],
            dim=1,
        )
        d_down_proj_sum = d_down_proj.cumsum(dim=1)
        down_proj = torch.einsum("b t d h, b t c h -> b t c d", d_down_proj_sum, h_padded)
        output = rearrange(down_proj, "b t c d -> b (t c) d")[:, : x.shape[1], :]
        ntp_target_3d = rearrange(ntp_target, "b t c d -> b (t c) d")[:, : x.shape[1], :]
        loss_dict = {
            "ntp_loss": self._compute_repr_loss(output[:, :-1], ntp_target_3d[:, 1:]),
        }
        with torch.no_grad():
            cumulative_update = d_down_proj_sum[:, -1] - d_down_proj_sum[:, 0]
            update_norm = cumulative_update.norm(p='fro')
            total_norm = d_down_proj_sum[:, -1].norm(p='fro')
            loss_dict["ttt_relative_contribution"] = update_norm / (total_norm + 1e-8)
        if self.lambda_align_rep > 0 and teacher_repr is not None:
            teacher_repr_3d = rearrange(teacher_repr, "b t c d -> b (t c) d")[:, : x.shape[1], :]
            dtype = teacher_repr_3d.dtype
            teacher_repr_3d_float = teacher_repr_3d.float() if dtype == torch.bfloat16 else teacher_repr_3d
            teacher_proj_weight = self.teacher_proj.weight.float() if self.teacher_proj.weight.dtype == torch.bfloat16 else self.teacher_proj.weight
            teacher_repr_projected = torch.einsum(
                "b t d, e d -> b t e",
                teacher_repr_3d_float,
                teacher_proj_weight,
            )
            teacher_repr_projected = teacher_repr_projected.to(dtype)
            loss_dict["align_rep_loss"] = self._compute_repr_loss(output[:, :-1], teacher_repr_projected[:, 1:])
        return output, loss_dict

    def _compute_repr_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = rearrange(pred, "b t d -> (b t) d")
        target_flat = rearrange(target, "b t d -> (b t) d")
        loss = -(pred_flat * target_flat).sum() / (pred_flat.shape[0] * pred_flat.shape[1])
        return loss


class OPDTTTLoss(nn.Module):
    def __init__(
        self,
        lambda_kl: float = 0.1,
        lambda_lm: float = 1.0,
        lambda_ntp: float = 1.0,
        lambda_align_rep: float = 0.5,
        vocab_size: int = 248320,
    ):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_lm = lambda_lm
        self.lambda_ntp = lambda_ntp
        self.lambda_align_rep = lambda_align_rep
        self.vocab_size = vocab_size

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        ntp_losses: dict = None,
        attention_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        loss_dict = {}
        total_loss = 0.0
        has_gradient_loss = False
        if teacher_logits is not None:
            kl_loss = self._compute_kl_divergence(student_logits, teacher_logits, attention_mask)
            loss_dict["kl_loss"] = kl_loss
            total_loss += self.lambda_kl * kl_loss
            if self.lambda_kl > 0:
                has_gradient_loss = True
        lm_loss = self._compute_lm_loss(student_logits, labels, attention_mask)
        loss_dict["lm_loss"] = lm_loss
        total_loss += self.lambda_lm * lm_loss
        if self.lambda_lm > 0:
            has_gradient_loss = True
        if ntp_losses is not None:
            for key, value in ntp_losses.items():
                if value is not None and isinstance(value, torch.Tensor):
                    loss_dict[key] = value.detach()
        if not has_gradient_loss and isinstance(total_loss, (int, float)):
            total_loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict

    def _compute_kl_divergence(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        kl_per_token = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
            log_target=False,
        ).sum(dim=-1)
        if attention_mask is not None:
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
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)
        if attention_mask is not None:
            mask = attention_mask[:, :-1] * attention_mask[:, 1:]
            loss = loss * mask
            num_tokens = mask.sum() + 1e-8
        else:
            num_tokens = loss.numel()
        return loss.sum() / num_tokens


class TeacherCache:
    def __init__(self, max_cache_size: int = 100):
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_order = []

    def get(self, key: str) -> Optional[dict]:
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: dict):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_cache_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        self.cache[key] = value
        self.access_order.append(key)

    def clear(self):
        self.cache.clear()
        self.access_order.clear()


def compute_teacher_repr_targets(
    teacher_logits: Optional[torch.Tensor],
    teacher_embeddings: Optional[torch.Tensor],
    layer_idx: int,
    hidden_size: int,
    chunk_size: int = 4096,
) -> Optional[torch.Tensor]:
    if teacher_logits is None and teacher_embeddings is None:
        return None
    if teacher_logits is None and teacher_embeddings is not None:
        seq_len = teacher_embeddings.shape[1]
        if seq_len % chunk_size != 0:
            pad_len = chunk_size - (seq_len % chunk_size)
            padding = torch.zeros(
                teacher_embeddings.shape[0], pad_len, teacher_embeddings.shape[2],
                device=teacher_embeddings.device, dtype=teacher_embeddings.dtype
            )
            teacher_embeddings = torch.cat([teacher_embeddings, padding], dim=1)
        return teacher_embeddings
    if teacher_logits is not None:
        teacher_probs = F.softmax(teacher_logits, dim=-1)
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
    "OPDQwen3_5MLP",
    "OPDTTTLoss",
    "TeacherCache",
    "compute_teacher_repr_targets",
]
