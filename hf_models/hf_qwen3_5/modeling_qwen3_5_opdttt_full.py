# coding=utf-8
"""
OPD-TTT Qwen3.5 完整模型实现

基于 transformers 5.9.0 内置的 Qwen3_5TextModel / Qwen3_5ForCausalLM，
添加 TTT 层支持。仅使用 text_config（纯文本，无视觉）。

关键适配点：
- 继承 transformers 的 Qwen3_5RMSNorm, Qwen3_5Attention, Qwen3_5GatedDeltaNet, Qwen3_5TextRotaryEmbedding
- MLP 替换为 OPDQwen3_5MLP（TTT 在 down_proj 上）
- 4D position_ids (M-RoPE) 处理
- layer_types 支持 linear_attention / full_attention 混合
"""

from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.utils import auto_docstring, can_return_tuple, logging
from transformers.utils.generic import check_model_inputs
from transformers.modeling_layers import GradientCheckpointingLayer

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP as StdQwen3_5MLP,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from .modeling_qwen3_5_opdttt import (
    OPDQwen3_5MLP,
    OPDTTTLoss,
    compute_teacher_repr_targets,
)

logger = logging.get_logger(__name__)


class OPDQwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)

        self.mlp = OPDQwen3_5MLP(config, layer_idx=layer_idx)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.is_opdttt_layer = getattr(config, "opdttt_mode", False) and layer_idx in getattr(
            config, "opdttt_layers", []
        )
        # SWA: sliding window for full_attention layers
        self.sliding_window = getattr(config, "sliding_window", 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        target_states: Optional[torch.Tensor] = None,
        teacher_repr: Optional[Dict[int, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        elif self.layer_type == "full_attention":
            if self.sliding_window > 0:
                kwargs["sliding_window"] = self.sliding_window
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if target_states is None and self.is_opdttt_layer:
            target_states = hidden_states

        layer_teacher_repr = None
        if teacher_repr is not None and self.is_opdttt_layer:
            layer_teacher_repr = teacher_repr.get(self.layer_idx)

        hidden_states, mlp_losses = self.mlp(
            hidden_states,
            t=target_states,
            teacher_repr=layer_teacher_repr,
        )
        hidden_states = residual + hidden_states
        return hidden_states, mlp_losses


class OPDQwen3_5TextModel(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        # 兼容：若传入外层 Qwen3_5Config（含 text_config），自动提取
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            config = config.text_config
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OPDQwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.opdttt_layers = getattr(config, "opdttt_layers", [])
        self.opdttt_mode = getattr(config, "opdttt_mode", False)
        self.ttt_target = getattr(config, "ttt_target", "hidden_states")

        if self.opdttt_mode:
            self.opdttt_loss = OPDTTTLoss(
                lambda_kl=getattr(config, "lambda_kl", 0.1),
                lambda_lm=getattr(config, "lambda_lm", 1.0),
                lambda_ntp=getattr(config, "lambda_ntp", 1.0),
                lambda_align_rep=getattr(config, "lambda_align_rep", 0.5),
                vocab_size=config.vocab_size,
            )

    def _resolve_ttt_target_states(self, inputs_embeds):
        if self.ttt_target == "input_embed":
            return inputs_embeds
        return None

    def compute_ce_loss(self, logits, labels, attention_mask=None):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        if attention_mask is not None:
            shift_mask = attention_mask[..., 1:].contiguous().view(-1)
            loss = loss * shift_mask
            loss = loss.sum() / shift_mask.sum()
        else:
            loss = loss.mean()
        return loss

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=None,
        teacher_outputs=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("必须且只能指定 input_ids 或 inputs_embeds 之一")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None
        if self.config._attn_implementation == "flash_attention_2":
            attention_mask = None
            position_ids = cache_position.unsqueeze(0)
            text_position_ids = cache_position.unsqueeze(0)
        mask_kwargs = dict(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        causal_mask = create_causal_mask(**mask_kwargs)
        sliding_window = getattr(self.config, "sliding_window", 0)
        sw_causal_mask = None
        if sliding_window > 0:
            sw_causal_mask = create_sliding_window_causal_mask(**mask_kwargs)
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        teacher_repr = None
        if teacher_outputs is not None and self.opdttt_mode:
            teacher_repr = self._prepare_teacher_repr(
                teacher_outputs, inputs_embeds.shape[1], inputs_embeds.device
            )
        all_ntp_losses = {}
        target_states = self._resolve_ttt_target_states(inputs_embeds)
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if self.config.layer_types[i] == "linear_attention":
                layer_mask = linear_attn_mask
            else:
                layer_mask = sw_causal_mask if sliding_window > 0 else causal_mask
            hidden_states, layer_losses = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                target_states=target_states,
                teacher_repr=teacher_repr,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            for key, value in layer_losses.items():
                if key not in all_ntp_losses:
                    all_ntp_losses[key] = []
                all_ntp_losses[key].append(value)
        hidden_states = self.norm(hidden_states)
        aggregated_losses = {}
        for key, losses in all_ntp_losses.items():
            if losses:
                aggregated_losses[key] = torch.stack(losses).mean()
        return hidden_states, aggregated_losses

    def _update_linear_attn_mask(self, attention_mask, past_key_values):
        linear_attn_mask = attention_mask
        if (past_key_values is not None and past_key_values.has_previous_state()) or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            linear_attn_mask = None
        return linear_attn_mask

    def _prepare_teacher_repr(self, teacher_outputs, seq_len, device):
        teacher_repr = {}
        teacher_hidden = teacher_outputs.get("hidden_states", [])
        teacher_logits = teacher_outputs.get("logits", None)
        teacher_embeddings = teacher_outputs.get("embeddings", None)
        chunk_size = getattr(self.config, "ttt_chunk", 4096)
        for layer_idx in self.opdttt_layers:
            if layer_idx < len(teacher_hidden):
                layer_repr = compute_teacher_repr_targets(
                    teacher_logits=teacher_logits,
                    teacher_embeddings=teacher_embeddings,
                    layer_idx=layer_idx,
                    hidden_size=self.config.hidden_size,
                    chunk_size=chunk_size,
                )
                if layer_repr is None and layer_idx < len(teacher_hidden):
                    layer_repr = teacher_hidden[layer_idx]
                teacher_repr[layer_idx] = layer_repr
        return teacher_repr


class OPDQwen3_5ForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = Qwen3_5TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPDQwen3_5DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": OPDQwen3_5DecoderLayer}
    # Qwen3.5 权重文件是完整多模态模型，文本权重前缀为 model.language_model.
    # 我们的纯文本模型期望 model. 前缀，需要做键名映射
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Qwen3_5TextConfig):
        # 兼容：若传入外层 Qwen3_5Config（含 text_config），自动提取
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            config = config.text_config
        super().__init__(config)
        self.model = OPDQwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            if module.weight.device.type == "meta":
                return
            if module.weight.shape[0] == module.weight.shape[1]:
                weight_data = module.weight.data
                if hasattr(weight_data, '_local_tensor'):
                    import torch.distributed as dist
                    local_tensor = weight_data._local_tensor
                    local_tensor.zero_()
                    local_rows = local_tensor.shape[0]
                    num_cols = local_tensor.shape[1]
                    rank = dist.get_rank()
                    start_row = rank * local_rows
                    g = torch.Generator(device=local_tensor.device)
                    g.manual_seed(42)
                    all_diag_values = torch.randn(
                        module.weight.shape[0], generator=g,
                        device=local_tensor.device, dtype=local_tensor.dtype
                    ) * std
                    local_row_indices = torch.arange(local_rows, device=local_tensor.device)
                    global_col_indices = start_row + local_row_indices
                    valid_mask = global_col_indices < num_cols
                    local_row_indices = local_row_indices[valid_mask]
                    global_col_indices = global_col_indices[valid_mask]
                    if len(local_row_indices) > 0:
                        local_tensor[local_row_indices, global_col_indices] = all_diag_values[global_col_indices]
                else:
                    weight_data.zero_()
                    diag_size = weight_data.shape[0]
                    diag_values = torch.randn(diag_size, device=weight_data.device, dtype=weight_data.dtype) * std
                    indices = torch.arange(diag_size, device=weight_data.device)
                    weight_data[indices, indices] = diag_values
            else:
                module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            if module.weight.device.type == "meta":
                return
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif "RMSNorm" in module.__class__.__name__:
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)

    def generate(self, *args, **kwargs):
        for p in ['teacher_logits', 'teacher_hidden_states', 'teacher_embeddings']:
            kwargs.pop(p, None)
        return super().generate(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _chunked_ce_loss(self, hidden_states, labels, attention_mask, chunk_size=4096):
        """分块计算 CE loss，避免实例化完整 logits [batch, seq_len, vocab_size]。
        
        vocab_size=248320 时，32K 序列的完整 logits 在 bf16 下约 16GB，
        分块计算每次只需 chunk_size * vocab_size * 2 ≈ 2GB。
        """
        shift_hidden = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:]
        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous()
        else:
            shift_mask = (shift_labels != -100).long()
        
        total_loss = 0.0
        total_tokens = 0
        seq_len = shift_hidden.shape[1]
        lm_weight = self.lm_head.weight  # [vocab_size, hidden_size]
        
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            h_chunk = shift_hidden[:, start:end, :]  # [B, chunk, H]
            l_chunk = shift_labels[:, start:end]      # [B, chunk]
            m_chunk = shift_mask[:, start:end]         # [B, chunk]
            
            logits_chunk = torch.matmul(h_chunk, lm_weight.t())  # [B, chunk, V]
            loss_chunk = F.cross_entropy(
                logits_chunk.view(-1, self.vocab_size),
                l_chunk.view(-1),
                reduction="none",
            ).view(l_chunk.shape)
            
            loss_chunk = loss_chunk * m_chunk
            total_loss = total_loss + loss_chunk.sum()
            total_tokens = total_tokens + m_chunk.sum()
        
        return total_loss / (total_tokens + 1e-8)

    def _chunked_opd_loss(self, hidden_states, teacher_logits, labels, ntp_losses, attention_mask, chunk_size=4096):
        """分块计算 OPD loss（KL + CE），避免实例化完整 student/teacher logits。
        
        OPD 模式下需要 student_logits 和 teacher_logits 计算 KL 散度。
        分块计算：每次只实例化 chunk_size 个 token 的 logits。
        """
        lambda_kl = self.model.opdttt_loss.lambda_kl
        lambda_lm = self.model.opdttt_loss.lambda_lm
        vocab_size = self.vocab_size
        lm_weight = self.lm_head.weight  # [vocab_size, hidden_size]

        shift_hidden = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_teacher = teacher_logits[:, :-1, :] if teacher_logits is not None else None
        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous()
        else:
            shift_mask = (shift_labels != -100).long()

        total_ce = 0.0
        total_kl = 0.0
        total_tokens = 0
        seq_len = shift_hidden.shape[1]

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            h_chunk = shift_hidden[:, start:end, :]
            l_chunk = shift_labels[:, start:end]
            m_chunk = shift_mask[:, start:end]

            # student logits（有梯度）
            s_logits = torch.matmul(h_chunk, lm_weight.t())  # [B, chunk, V]

            # CE loss
            ce_chunk = F.cross_entropy(
                s_logits.view(-1, vocab_size), l_chunk.view(-1), reduction="none"
            ).view(l_chunk.shape)
            ce_chunk = ce_chunk * m_chunk
            total_ce = total_ce + ce_chunk.sum()

            # KL loss
            if shift_teacher is not None and lambda_kl > 0:
                t_chunk = shift_teacher[:, start:end, :]  # [B, chunk, V] 无梯度
                s_log_probs = F.log_softmax(s_logits, dim=-1)
                t_probs = F.softmax(t_chunk, dim=-1)
                kl_chunk = F.kl_div(s_log_probs, t_probs, reduction="none", log_target=False).sum(dim=-1)
                kl_chunk = kl_chunk * m_chunk
                total_kl = total_kl + kl_chunk.sum()

            total_tokens = total_tokens + m_chunk.sum()

        num_tokens = total_tokens + 1e-8
        ce_loss = total_ce / num_tokens
        kl_loss = total_kl / num_tokens if isinstance(total_kl, torch.Tensor) else torch.tensor(0.0, device=hidden_states.device)

        total_loss = lambda_lm * ce_loss + lambda_kl * kl_loss

        loss_dict = {
            "lm_loss": ce_loss.detach(),
            "kl_loss": kl_loss.detach() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            "total_loss": total_loss.detach(),
        }
        for k, v in ntp_losses.items():
            if v is not None and isinstance(v, torch.Tensor):
                loss_dict[k] = v.detach()

        return total_loss, loss_dict

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        teacher_logits=None,
        teacher_hidden_states=None,
        teacher_embeddings=None,
        logits_to_keep=0,
        **kwargs,
    ):
        teacher_outputs = None
        if teacher_logits is not None or teacher_hidden_states is not None:
            teacher_outputs = {
                "logits": teacher_logits,
                "hidden_states": teacher_hidden_states or [],
                "embeddings": teacher_embeddings,
            }
        hidden_states, ntp_losses = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            teacher_outputs=teacher_outputs,
            **kwargs,
        )

        loss = None
        loss_dict = {}

        if labels is not None:
            if self.model.opdttt_mode:
                # 纯 TTT 模式（无教师）：用 chunked CE 避免大 logits OOM
                if teacher_logits is None:
                    loss = self._chunked_ce_loss(hidden_states, labels, attention_mask)
                    loss_dict = {"lm_loss": loss.detach(), **{k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in ntp_losses.items()}}
                else:
                    # OPD 模式（有教师）：chunked KL+CE 避免大 logits OOM
                    loss, loss_dict = self._chunked_opd_loss(
                        hidden_states, teacher_logits, labels, ntp_losses, attention_mask
                    )
            else:
                loss = self._chunked_ce_loss(hidden_states, labels, attention_mask)
                loss_dict = {"lm_loss": loss.detach() if loss is not None else None}

        # 推理/生成时仍需 logits（训练时不计算完整 logits 省显存）
        if labels is None:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        else:
            logits = None

        outputs = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=None,
        )
        if loss_dict:
            outputs.loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        return outputs


__all__ = [
    "OPDQwen3_5ForCausalLM",
    "OPDQwen3_5TextModel",
    "OPDQwen3_5DecoderLayer",
    "OPDQwen3_5MLP",
    "OPDTTTLoss",
]
