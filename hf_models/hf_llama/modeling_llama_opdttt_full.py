# coding=utf-8
# -*- coding: utf-8 -*-
"""
OPD-TTT 完整模型实现

本模块实现了完整的 OPD-TTT 学生模型，支持：
1. 教师模型指导的前向传播
2. 四层损失函数计算
3. 分层表示对齐
4. FSDP2 分布式训练支持

主要组件：
- OPDTTTModel：OPD-TTT 学生模型主体
- OPDTTTDecoderLayer：支持教师引导的解码层
- OPDTTTForCausalLM：完整的因果语言模型
"""

from typing import Optional, Tuple, Union, Dict, Any, List
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.masking_utils import create_causal_mask
from transformers.utils import (
    auto_docstring,
    can_return_tuple,
    logging,
)
from transformers.utils.generic import check_model_inputs

from .configuration_llama import LlamaConfig
from .modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaAttention,
    repeat_kv,
    rotate_half,
    apply_rotary_pos_emb,
)
from .modeling_llama_opdttt import (
    OPDTTTMLP,
    OPDTTTLoss,
    compute_teacher_repr_targets,
)

logger = logging.get_logger(__name__)


class OPDTTTDecoderLayer(nn.Module):
    """
    OPD-TTT 解码层

    该层在标准 LLaMA 解码层的基础上，将 MLP 替换为支持教师引导的 OPDTTTMLP。
    支持：
    1. 标准的注意力计算
    2. 教师引导的 MLP 前向传播
    3. 损失计算和返回
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        """
        初始化 OPD-TTT 解码层

        Args:
            config: 模型配置
            layer_idx: 当前层索引
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # 自注意力层
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        # 使用 OPD-TTT 增强的 MLP
        self.mlp = OPDTTTMLP(config, layer_idx=layer_idx)

        # 层归一化
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 检查是否为 OPD-TTT 层
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
        """
        解码层前向传播，支持教师表示

        Args:
            hidden_states: 输入隐藏状态 [batch, seq_len, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置 ID
            past_key_values: 过去的键值对（用于缓存）
            use_cache: 是否使用缓存
            cache_position: 缓存位置
            position_embeddings: 旋转位置嵌入
            target_states: NTP 目标状态（输入嵌入）
            teacher_repr: 教师表示（用于对齐）

        Returns:
            hidden_states: 输出隐藏状态
            loss_dict: MLP 层的损失字典
        """
        # 残差连接 + 输入层归一化
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力计算
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

        # MLP 前向传播（带 OPD-TTT）
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # 准备 NTP 目标状态
        if target_states is None and self.is_opdttt_layer:
            target_states = kwargs.get("inputs_embeds", hidden_states)

        # 准备教师表示（如果这是 OPD-TTT 层）
        layer_teacher_repr = None
        if teacher_repr is not None and self.is_opdttt_layer:
            # 提取当前层的教师表示
            layer_teacher_repr = teacher_repr.get(self.layer_idx)

        # MLP 前向传播
        hidden_states, mlp_losses = self.mlp(
            hidden_states,
            t=target_states,
            teacher_repr=layer_teacher_repr,
        )
        hidden_states = residual + hidden_states

        return hidden_states, mlp_losses


class OPDTTTModel(nn.Module):
    """
    OPD-TTT 学生模型

    该模型实现了教师学生架构中的学生模型，支持：
    1. 教师模型输出的处理和投影
    2. 分层表示对齐
    3. 完整的前向传播
    """

    def __init__(self, config: LlamaConfig):
        """
        初始化 OPD-TTT 模型

        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token 嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # OPD-TTT 解码层堆叠
        self.layers = nn.ModuleList(
            [OPDTTTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # 最终层归一化
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 旋转位置嵌入
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # OPD-TTT 设置
        self.opdttt_layers = getattr(config, "opdttt_layers", [])
        self.opdttt_mode = getattr(config, "opdttt_mode", False)
        self.ttt_target = getattr(config, "ttt_target", "input_embed")

        # 损失函数
        if self.opdttt_mode:
            self.opdttt_loss = OPDTTTLoss(
                lambda_kl=getattr(config, "lambda_kl", 0.1),
                lambda_lm=getattr(config, "lambda_lm", 1.0),
                lambda_ntp=getattr(config, "lambda_ntp", 1.0),
                lambda_align_rep=getattr(config, "lambda_align_rep", 0.5),
                vocab_size=config.vocab_size,
            )

    def _resolve_ttt_target_states(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        获取 NTP 的目标状态

        Args:
            inputs_embeds: 输入嵌入

        Returns:
            目标状态（通常是输入嵌入本身）
        """
        if self.ttt_target == "input_embed":
            return inputs_embeds
        return None

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        teacher_outputs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """
        模型前向传播，支持教师输出

        Args:
            input_ids: 输入 Token ID [batch, seq_len]
            attention_mask: 注意力掩码
            position_ids: 位置 ID
            past_key_values: 过去的键值对
            inputs_embeds: 输入嵌入
            cache_position: 缓存位置
            use_cache: 是否使用缓存
            teacher_outputs: 教师模型输出字典，包含：
                - logits: 教师 logits [batch, seq_len, vocab_size]
                - hidden_states: 每层的隐藏状态 [num_layers][batch, seq_len, hidden_size]
                - embeddings: 教师输入嵌入 [batch, seq_len, hidden_size]

        Returns:
            last_hidden_state: 最终隐藏状态
            loss_dict: 损失字典
        """
        # 输入验证
        if (input_ids is None) ^ (inputs_embeds is None):
            raise ValueError("必须且只能指定 input_ids 或 inputs_embeds 之一")

        # 获取输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 初始化缓存
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # 计算缓存位置
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # 计算位置 ID
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 创建因果掩码
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 准备教师表示（如果提供）
        teacher_repr = None
        if teacher_outputs is not None and self.opdttt_mode:
            teacher_repr = self._prepare_teacher_repr(
                teacher_outputs,
                inputs_embeds.shape[1],
                inputs_embeds.device,
            )

        # 收集所有层的 NTP/对齐损失
        all_ntp_losses = {}

        # 获取 NTP 目标状态
        target_states = self._resolve_ttt_target_states(inputs_embeds)

        # 逐层前向传播
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, layer_losses = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
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

            # 收集当前层的损失
            for key, value in layer_losses.items():
                if key not in all_ntp_losses:
                    all_ntp_losses[key] = []
                all_ntp_losses[key].append(value)

        # 最终层归一化
        hidden_states = self.norm(hidden_states)

        # 聚合所有层的损失
        aggregated_losses = {}
        for key, losses in all_ntp_losses.items():
            if losses:
                aggregated_losses[key] = torch.stack(losses).mean()

        return hidden_states, aggregated_losses

    def _prepare_teacher_repr(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        seq_len: int,
        device: torch.device,
    ) -> Dict[int, torch.Tensor]:
        """
        准备每层的教师表示

        Args:
            teacher_outputs: 教师模型输出
            seq_len: 序列长度
            device: 设备

        Returns:
            层索引到教师表示的映射
        """
        teacher_repr = {}

        teacher_hidden = teacher_outputs.get("hidden_states", [])
        teacher_logits = teacher_outputs.get("logits", None)
        teacher_embeddings = teacher_outputs.get("embeddings", None)

        chunk_size = getattr(self.config, "ttt_chunk", 4096)

        # 为每个 OPD-TTT 层准备教师表示
        for layer_idx in self.opdttt_layers:
            if layer_idx < len(teacher_hidden):
                # 计算该层的教师表示
                layer_repr = compute_teacher_repr_targets(
                    teacher_logits=teacher_logits,
                    teacher_embeddings=teacher_embeddings,
                    layer_idx=layer_idx,
                    hidden_size=self.config.hidden_size,
                    chunk_size=chunk_size,
                )
                # 如果没有嵌入，使用隐藏状态
                if layer_repr is None and layer_idx < len(teacher_hidden):
                    layer_repr = teacher_hidden[layer_idx]
                teacher_repr[layer_idx] = layer_repr

        return teacher_repr


class OPDTTTForCausalLM(PreTrainedModel):
    """
    OPD-TTT 因果语言模型

    这是完整的 OPD-TTT 学生模型，支持：
    1. 教师模型指导的训练
    2. 四层损失函数
    3. 分布式训练（FSDP2）
    4. 标准的 HuggingFace 接口
    """

    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPDTTTDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": OPDTTTDecoderLayer,
    }

    def __init__(self, config: LlamaConfig):
        """
        初始化 OPD-TTT 模型

        Args:
            config: 模型配置
        """
        super().__init__(config)
        self.model = OPDTTTModel(config)
        self.vocab_size = config.vocab_size

        # LM Head：将隐藏状态投影到词汇表
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重绑定：LM Head 和嵌入层共享权重
        self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """获取输出嵌入层"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """设置输出嵌入层"""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """设置解码器"""
        self.model = decoder

    def get_decoder(self):
        """获取解码器"""
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        teacher_hidden_states: Optional[List] = None,
        teacher_embeddings: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        前向传播，支持教师指导

        Args:
            input_ids: 输入 Token ID
            attention_mask: 注意力掩码
            position_ids: 位置 ID
            past_key_values: 过去的键值对
            inputs_embeds: 输入嵌入
            labels: 真实标签
            use_cache: 是否使用缓存
            cache_position: 缓存位置
            teacher_logits: 教师 logits [batch, seq_len, vocab_size]
            teacher_hidden_states: 教师隐藏状态列表
            teacher_embeddings: 教师输入嵌入
            logits_to_keep: 保留的 logits 数量

        Returns:
            CausalLMOutputWithPast 包含：
                loss: OPD-TTT 组合损失
                logits: 学生模型 logits
                past_key_values: 过去的键值对
                hidden_states: 隐藏状态
        """
        # 准备教师输出字典
        teacher_outputs = None
        if teacher_logits is not None or teacher_hidden_states is not None:
            teacher_outputs = {
                "logits": teacher_logits,
                "hidden_states": teacher_hidden_states or [],
                "embeddings": teacher_embeddings,
            }

        # 模型前向传播
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

        # 计算 logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 计算组合损失
        loss = None
        loss_dict = {}
        if labels is not None:
            loss, loss_dict = self.model.opdttt_loss(
                student_logits=logits,
                teacher_logits=teacher_logits,
                labels=labels,
                ntp_losses=ntp_losses,
                attention_mask=attention_mask,
            )
            # 存储各个损失分量用于日志
            loss_dict.update({k: v.detach() if isinstance(v, torch.Tensor) else v
                             for k, v in loss_dict.items()})

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=self.model.layers[0].self_attn.past_key_values if use_cache else None,
            hidden_states=hidden_states,
            attentions=None,
        )


__all__ = [
    "OPDTTTForCausalLM",
    "OPDTTTModel",
    "OPDTTTDecoderLayer",
    "OPDTTTMLP",
    "OPDTTTLoss",
]
