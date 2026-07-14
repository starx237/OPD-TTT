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

"""Qwen3.5 model with In-Place TTT inference support.

Forward path mirrors the training model (hf_models/hf_qwen3_5/) exactly,
with only the MLP using the inference TTT path (chunk-by-chunk weight
adaptation) instead of the training TTT path (batched einsum + loss).
"""

from typing import Optional, Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from opt_einsum import contract

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.utils import can_return_tuple, logging
from transformers.utils.generic import check_model_inputs

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from .configuration_qwen3_5 import Qwen3_5TTTConfig

logger = logging.get_logger(__name__)


# ============================================================================
# TTT-aware cache: extends DynamicCache to persist TTT states across generation
# ============================================================================
class TTTDynamicCache(DynamicCache):
    """DynamicCache extended with per-layer TTT state.

    TTT state per layer: (past_h_tail, past_t_tail, past_w)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ttt_states = [(None, None, None)] * 100

    def TTT_update(self, ttt_state, layer_idx: int) -> None:
        self.ttt_states[layer_idx] = ttt_state


# ============================================================================
# MLP with TTT inference path (identical to official In-Place-TTT)
# ============================================================================
class Qwen3_5TTTMLP(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_idx = -1 if layer_idx is None else layer_idx

        if getattr(config, "opdttt_mode", False) and self.layer_idx in getattr(config, "opdttt_layers", []):
            self.enable_opdttt = True
            self.ttt_chunk = getattr(config, "ttt_chunk", 8192)
            if getattr(config, "ttt_proj", True):
                self.ttt_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            else:
                self.ttt_proj = None
            self.ttt_lr = getattr(config, "ttt_lr", 0.3)
            self.ttt_max_norm = getattr(config, "ttt_max_norm", 0)
            self.lambda_ntp = getattr(config, "lambda_ntp", 1.0)
            self.ttt_conv = nn.Conv1d(
                self.hidden_size, self.hidden_size,
                kernel_size=5, padding=2, groups=self.hidden_size, bias=False,
            )
        else:
            self.enable_opdttt = False

    def padding(self, x):
        if x.shape[1] % self.ttt_chunk != 0:
            pad = torch.zeros(
                [x.shape[0], self.ttt_chunk - x.shape[1] % self.ttt_chunk, x.shape[2]],
                device=x.device, dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)
        return rearrange(x, "b (t c) d -> b t c d", c=self.ttt_chunk)

    def forward(self, x, t=None, past_w=None):
        """Inference forward: if TTT enabled and t is not None, do chunk-by-chunk
        weight adaptation. Otherwise plain down_proj.
        """
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        if not self.enable_opdttt or t is None:
            return self.down_proj(h), None

        present_down_proj_w = self.down_proj.weight.clone() if past_w is None else past_w

        bs, seq_len, _ = x.shape
        if seq_len < self.ttt_chunk:
            return nn.functional.linear(h, present_down_proj_w, self.down_proj.bias), present_down_proj_w

        t_padded = self.padding(t)
        h_padded = self.padding(h)
        bs, chunk_num, chunk_size, _ = t_padded.shape

        t_conv = (
            self.ttt_conv(t_padded.transpose(-1, -2).reshape(bs * chunk_num, -1, chunk_size))
            .transpose(-1, -2)
            .reshape(bs, chunk_num, chunk_size, -1)
        )

        current_w = present_down_proj_w
        y = torch.zeros_like(t_conv)
        for i, current_y, current_t, current_h in zip(range(chunk_num), y[0], t_conv[0], h_padded[0]):
            current_y = contract("d h, c h -> c d", current_w, current_h)
            y[0][i] = current_y
            if seq_len % self.ttt_chunk == 0 or i != chunk_num - 1:
                if self.ttt_proj is not None:
                    # NOTE: lambda_ntp 乘法在官方推理中不存在（官方仅乘 ttt_lr）。
                    # 此处保留是为了与训练代码保持一致（训练中 lambda_ntp=1.0，无数值影响）。
                    # 若后续对齐官方，应移除 * self.lambda_ntp。
                    # NOTE: 训练代码使用 float32 upcast 计算 dw，但测试显示累积差异仅 0.004%
                    # （随机误差部分抵消），此处使用 bfloat16 与官方一致以加速。
                    dw = (
                        contract("c h, c d, d e -> e h", current_h, current_t, self.ttt_proj.weight)
                        * self.ttt_lr * self.lambda_ntp
                    )
                else:
                    dw = contract("c h, c d -> d h", current_h, current_t) * self.ttt_lr * self.lambda_ntp
                if self.ttt_max_norm > 0:
                    dw_norm = dw.norm(p='fro')
                    clip_coef = self.ttt_max_norm / (dw_norm + 1e-8)
                    if clip_coef < 1.0:
                        dw = dw * clip_coef
                current_w = current_w + dw

        out = rearrange(y, "b t c d -> b (t c) d")[:, :seq_len, :]
        return out, current_w


# ============================================================================
# Decoder layer — mirrors OPDQwen3_5DecoderLayer exactly
# ============================================================================
class Qwen3_5TTTDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5TTTConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)

        self.mlp = Qwen3_5TTTMLP(config, layer_idx=layer_idx)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.is_opdttt_layer = getattr(config, "opdttt_mode", False) and layer_idx in getattr(
            config, "opdttt_layers", []
        )
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
        **kwargs,
    ) -> torch.Tensor:
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

        # TTT inference path: manage TTT cache states
        if target_states is not None and self.is_opdttt_layer:
            past_h, past_t, past_w = (
                past_key_values.ttt_states[self.layer_idx] if past_key_values is not None else (None, None, None)
            )
            if past_h is None:
                present_h = hidden_states
                present_t = target_states
            else:
                present_h = torch.cat([past_h, hidden_states], dim=1)
                present_t = torch.cat([past_t, target_states], dim=1)

            if present_h.shape[1] < self.mlp.ttt_chunk:
                hidden_states, present_w = self.mlp(hidden_states, None, past_w)
            else:
                all_hidden_states, present_w = self.mlp(present_h, present_t, past_w)
                hidden_states = all_hidden_states[:, -hidden_states.shape[1]:]

            present_h_tail = present_h[:, -(present_h.shape[1] % self.mlp.ttt_chunk):]
            present_t_tail = present_t[:, -(present_t.shape[1] % self.mlp.ttt_chunk):]
            if present_h_tail.shape[1] % self.mlp.ttt_chunk == 0:
                present_h_tail, present_t_tail = None, None
            if past_key_values is not None:
                past_key_values.TTT_update((present_h_tail, present_t_tail, present_w), self.layer_idx)
        else:
            hidden_states, _ = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


# ============================================================================
# Text model — mirrors OPDQwen3_5TextModel forward exactly
# ============================================================================
class Qwen3_5TTTModel(PreTrainedModel):
    config_class = Qwen3_5TTTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5TTTDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: Qwen3_5TTTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3_5TTTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.opdttt_layers = getattr(config, "opdttt_layers", [])
        self.opdttt_mode = getattr(config, "opdttt_mode", False)
        self.ttt_target = getattr(config, "ttt_target", "hidden_states")

        self.post_init()

    def _init_weights(self, module):
        pass

    def _resolve_ttt_target_states(self, inputs_embeds):
        if self.ttt_target == "input_embed":
            return inputs_embeds
        return None

    def _update_linear_attn_mask(self, attention_mask, past_key_values):
        linear_attn_mask = attention_mask
        if (past_key_values is not None and past_key_values.has_previous_state()) or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            linear_attn_mask = None
        return linear_attn_mask

    @check_model_inputs
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.opdttt_mode and use_cache and past_key_values is None:
            past_key_values = TTTDynamicCache(config=self.config)
        elif use_cache and past_key_values is None:
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

        target_states = self._resolve_ttt_target_states(inputs_embeds)
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if self.config.layer_types[i] == "linear_attention":
                layer_mask = linear_attn_mask
            else:
                layer_mask = sw_causal_mask if sliding_window > 0 else causal_mask
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                target_states=target_states,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ============================================================================
# Causal LM
# ============================================================================
class Qwen3_5TTTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = Qwen3_5TTTConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: Qwen3_5TTTConfig):
        super().__init__(config)
        self.model = Qwen3_5TTTModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        pass

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

    def generate(self, *args, **kwargs):
        if self.model.opdttt_mode:
            if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
                kwargs["past_key_values"] = TTTDynamicCache(config=self.config)
            input_ids = kwargs.get("input_ids", None)
            if input_ids is not None:
                assert len(input_ids) == 1, "only support bs=1 for TTT inference"
        return super().generate(*args, **kwargs)

    @can_return_tuple
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
        logits_to_keep=0,
        **kwargs,
    ):
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


__all__ = [
    "Qwen3_5TTTForCausalLM",
    "Qwen3_5TTTModel",
    "Qwen3_5TTTDecoderLayer",
    "Qwen3_5TTTMLP",
    "TTTDynamicCache",
]
