# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

# 使用正确的导入路径
from transformers.cache_utils import Cache

# RMSNorm不在transformers.modeling_layers中，从llama模型导入或自定义
try:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm
except ImportError:
    # 使用标准RMSNorm实现，优化 FSDP2 兼容性
    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            # 使用 FSDP2 兼容的操作：sum() 代替 mean() 避免 FSDP2 问题
            variance = hidden_states.pow(2).sum(dim=-1, keepdim=True) / hidden_states.shape[-1]
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)

# 导入GatedMLP (如果fla可用，使用fla的实现)
try:
    from fla.modules import GatedMLP as TransformerMLP
except ImportError:
    # 使用标准实现
    class TransformerMLP(nn.Module):
        def __init__(
            self,
            hidden_size: int,
            hidden_ratio: int,
            intermediate_size: Optional[int] = None,
            hidden_act: str = "silu",
            fuse_swiglu: bool = True,
        ):
            super().__init__()
            if intermediate_size is None:
                intermediate_size = hidden_size * hidden_ratio
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.act_fn = nn.SiLU()
            self.fuse_swiglu = fuse_swiglu

        # def forward(self, x, **kwargs):
        #     if self.fuse_swiglu:
        #         return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        #     else:
        #         return self.down_proj(self.act_fn(self.gate_proj(x))) * self.up_proj(x)
        def forward(self, x, **kwargs):
            if self.fuse_swiglu:
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            else:
                # 针对 FSDP2 进行显存生命周期优化的 else 分支：
                left_branch = self.down_proj(self.act_fn(self.gate_proj(x)))
                right_branch = self.up_proj(x)
                output = left_branch.contiguous() * right_branch.contiguous()
                del left_branch, right_branch
                return output

import torch.nn as nn

from .layer_lact_swiglu import LaCTSWIGLULayer
from .configuration_lact_swiglu import LaCTSWIGLUConfig

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


class LaCTBlock(nn.Module):

    def __init__(self, config: LaCTSWIGLUConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps
        )
        self.attn = LaCTSWIGLULayer(
            hidden_size=config.hidden_size,
            num_attn_heads=config.num_attn_heads,
            num_lact_heads=config.num_lact_heads,
            inter_multi=config.inter_multi,
            window_size=config.window_size,
            lact_chunk_size=config.lact_chunk_size,
            qkv_bias=config.qkv_bias,
            attn_qk_norm=config.attn_qk_norm,
            qkv_silu=config.qkv_silu,
            no_v_silu=config.no_v_silu,
            lr_dim=config.lr_dim,
            use_muon=config.use_muon,
            ttt_prenorm=config.ttt_prenorm,
            ttt_nope=config.ttt_nope,
            lr_parameterization=config.lr_parameterization,
            learnable_ttt_scale=config.learnable_ttt_scale,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
            w0_w2_low_rank=config.w0_w2_low_rank,
            use_momentum=config.use_momentum,
            ttt_loss_type=config.ttt_loss_type,
            fw_init_gain=config.fw_init_gain,
            use_fused_kernel=config.use_fused_kernel,
            fp32_states=config.fp32_states,
            attn_implementation=getattr(config, '_attn_implementation', 'flash_attention_2'),
        )

        self.mlp_norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps
        )
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[Any],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class LaCTPreTrainedModel(PreTrainedModel):

    config_class = LaCTSWIGLUConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LaCTBlock"]
    _supports_cache_class = True
    _supports_flash_attn = True  # Support Flash Attention (checked by transformers)
    _supports_flash_attn2 = True  # Support Flash Attention 2.0 (legacy BC)
    _supports_sdpa = True  # Support SDPA (Scaled Dot Product Attention)

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class LaCTModel(LaCTPreTrainedModel):

    _supports_flash_attn2 = True

    def __init__(self, config: LaCTSWIGLUConfig):
        super().__init__(config)

        self.padding_token_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Compatibility: Set _attn_implementation for LaCTSWIGLULayer
        # This allows the layer to use the correct attention implementation
        self.config._attn_implementation = getattr(config, '_attn_implementation', 'flash_attention_2')

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [LaCTBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[Any],
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, (decoder_layer, past_key_values_layer) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # Use torch.utils.checkpoint.checkpoint for gradient checkpointing
                # Note: checkpoint doesn't support kwargs directly, so we pass them as positional args
                def custom_forward(*args):
                    return decoder_layer(
                        hidden_states=args[0],
                        attention_mask=args[1],
                        past_key_values=args[2],
                        output_attentions=args[3],
                        use_cache=args[4],
                        cu_seqlens=cu_seqlens,
                    )

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    past_key_values_layer,
                    output_attentions,
                    use_cache,
                    use_reentrant=True,  # Use reentrant to avoid PyTorch 2.4.0 allocator issue
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values_layer,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cu_seqlens=cu_seqlens,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                past_key_values = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LaCTForCausalLM(LaCTPreTrainedModel, GenerationMixin):

    _supports_flash_attn2 = True

    _torchscript_model_attributes = {
        "model": "LaCTModel",
    }

    def __init__(self, config: LaCTSWIGLUConfig):
        super().__init__(config)
        self.model = LaCTModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[Any],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=return_dict,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]

            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
