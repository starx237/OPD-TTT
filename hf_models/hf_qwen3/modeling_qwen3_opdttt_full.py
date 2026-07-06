# coding=utf-8
"""
OPD-TTT Qwen3 完整模型实现

基于 Llama OPD-TTT 适配 Qwen3 架构（Qwen3Attention 含 q_norm/k_norm）。
训练用 OPDQwen3ForCausalLM，推理用 inference_model/。
"""

from typing import Optional, Tuple, Union, Dict, Any, List
import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.utils import auto_docstring, can_return_tuple, logging
from transformers.utils.generic import check_model_inputs
from transformers.modeling_layers import GradientCheckpointingLayer

from .configuration_qwen3 import Qwen3Config
from .modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Attention,
    repeat_kv,
    rotate_half,
    apply_rotary_pos_emb,
)
from .modeling_qwen3_opdttt import (
    OPDQwen3MLP,
    OPDTTTLoss,
    compute_teacher_repr_targets,
)

logger = logging.get_logger(__name__)


class OPDQwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = OPDQwen3MLP(config, layer_idx=layer_idx)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.is_opdttt_layer = getattr(config, "opdttt_mode", False) and layer_idx in getattr(
            config, "opdttt_layers", []
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        target_states: Optional[torch.Tensor] = None,
        teacher_repr: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
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


class OPDQwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OPDQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
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
            position_ids = cache_position.unsqueeze(0)

        # For flash_attention_2, skip padding mask and rebuild position_ids to avoid
        # the varlen path. flash_attn handles causality and sliding_window natively;
        # padding tokens are ignored in the loss (labels=-100). Without this fix,
        # padding (attention_mask with 0s, position_ids with 0s) triggers the varlen
        # path in flash_attn, whose backward crashes at 32K+ sequences.
        if self.config._attn_implementation == "flash_attention_2":
            attention_mask = None
            position_ids = cache_position.unsqueeze(0)

        mask_kwargs = dict(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        causal_mask = create_causal_mask(**mask_kwargs)
        has_sliding = "sliding_attention" in getattr(self.config, "layer_types", [])
        sw_causal_mask = create_sliding_window_causal_mask(**mask_kwargs) if has_sliding else None

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
            layer_type = self.config.layer_types[i] if has_sliding else "full_attention"
            layer_mask = sw_causal_mask if layer_type == "sliding_attention" else causal_mask
            hidden_states, layer_losses = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
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


class OPDQwen3ForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPDQwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": OPDQwen3DecoderLayer}

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = OPDQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def _init_weights(self, module):
        """初始化权重（论文 §9.3）：Conv1d 零初始化，ttt_proj 稀疏对角"""
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

    def generate(self, **kwargs):
        teacher_params = ['teacher_logits', 'teacher_hidden_states', 'teacher_embeddings']
        for param in teacher_params:
            if param in kwargs:
                del kwargs[param]
        return super().generate(**kwargs)

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

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        loss_dict = {}
        if labels is not None:
            if self.model.opdttt_mode:
                loss, loss_dict = self.model.opdttt_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    ntp_losses=ntp_losses,
                    attention_mask=attention_mask,
                )
                loss_dict.update({k: v.detach() if isinstance(v, torch.Tensor) else v
                                 for k, v in loss_dict.items()})
            else:
                loss = self.model.compute_ce_loss(logits, labels, attention_mask)
                loss_dict = {"lm_loss": loss.detach() if loss is not None else None}

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
    "OPDQwen3ForCausalLM",
    "OPDQwen3Model",
    "OPDQwen3DecoderLayer",
    "OPDQwen3MLP",
    "OPDTTTLoss",
]
