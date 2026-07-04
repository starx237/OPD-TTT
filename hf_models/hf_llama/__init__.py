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
LLaMA 模型实现，支持 In-Place TTT 和 OPD-TTT
"""

from .configuration_llama import LlamaConfig
from .modeling_llama import LlamaModel, LlamaForCausalLM
from .modeling_llama_opdttt_full import (
    OPDTTTForCausalLM,
    OPDTTTModel,
    OPDTTTDecoderLayer,
    OPDTTTMLP,
    OPDTTTLoss,
)
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

# 注册标准 LLaMA 模型
AutoConfig.register("llama", LlamaConfig, exist_ok=True)
AutoModel.register(LlamaConfig, LlamaModel, exist_ok=True)
AutoModelForCausalLM.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)

# OPDTTTForCausalLM 不注册到 AutoModelForCausalLM，避免覆盖 LlamaForCausalLM
# 需要 OPD-TTT 的脚本应显式 import OPDTTTForCausalLM

# 注意：liger_kernel LCE 前向传播已禁用，因为 Triton/CUDA 驱动不兼容
# 在此系统上（CUDA Driver 470.x）。改用标准前向传播。

__all__ = [
    "LlamaConfig",
    "LlamaModel",
    "LlamaForCausalLM",
    "OPDTTTForCausalLM",
    "OPDTTTModel",
    "OPDTTTDecoderLayer",
    "OPDTTTMLP",
    "OPDTTTLoss",
]
