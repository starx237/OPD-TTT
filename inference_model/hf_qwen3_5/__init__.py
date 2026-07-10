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

from .configuration_qwen3_5 import Qwen3_5TTTConfig
from .modeling_qwen3_5 import (
    Qwen3_5TTTModel,
    Qwen3_5TTTForCausalLM,
)
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register("qwen3_5_opdttt", Qwen3_5TTTConfig, exist_ok=True)
AutoModel.register(Qwen3_5TTTConfig, Qwen3_5TTTModel, exist_ok=True)
AutoModelForCausalLM.register(Qwen3_5TTTConfig, Qwen3_5TTTForCausalLM, exist_ok=True)
