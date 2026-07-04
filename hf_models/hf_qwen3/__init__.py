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

from .configuration_qwen3 import Qwen3Config
from .modeling_qwen3 import Qwen3Model, Qwen3ForCausalLM
from .modeling_qwen3_opdttt_full import (
    OPDQwen3ForCausalLM,
    OPDQwen3Model,
    OPDQwen3DecoderLayer,
    OPDQwen3MLP,
    OPDTTTLoss,
)
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

# 注册标准 Qwen3 模型
AutoConfig.register("qwen3", Qwen3Config, exist_ok=True)
AutoModel.register(Qwen3Config, Qwen3Model, exist_ok=True)
AutoModelForCausalLM.register(Qwen3Config, Qwen3ForCausalLM, exist_ok=True)

# OPDQwen3ForCausalLM 不注册到 AutoModelForCausalLM，避免覆盖 Qwen3ForCausalLM
# 需要 OPD-TTT 的脚本应显式 import OPDQwen3ForCausalLM

# NOTE: liger_kernel LCE forward disabled due to Triton/CUDA driver incompatibility
# on this system (CUDA Driver 470.x). The standard forward is used instead.

__all__ = [
    "Qwen3Config",
    "Qwen3Model",
    "Qwen3ForCausalLM",
    "OPDQwen3ForCausalLM",
    "OPDQwen3Model",
    "OPDQwen3DecoderLayer",
    "OPDQwen3MLP",
    "OPDTTTLoss",
]
