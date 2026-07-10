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

"""Qwen3.5 TTT inference configuration.

Extends the standard Qwen3_5TextConfig with OPD-TTT parameters for inference.
This config is registered as 'qwen3_5_opdttt' model_type.
"""

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen3_5TTTConfig(Qwen3_5TextConfig):
    r"""Configuration for Qwen3.5 with In-Place TTT inference.

    Extends Qwen3_5TextConfig with TTT-specific parameters that control
    test-time training during generation.
    """

    model_type = "qwen3_5_opdttt"

    def __init__(
        self,
        opdttt_mode=False,
        opdttt_layers=None,
        ttt_lr=0.3,
        ttt_chunk=8192,
        ttt_proj=True,
        ttt_max_norm=0,
        ttt_target="hidden_states",
        lambda_ntp=1.0,
        lambda_align_rep=0.0,
        teacher_proj_init="random",
        weight_adaptation="fixed",
        teacher_hidden_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.opdttt_mode = opdttt_mode
        self.opdttt_layers = opdttt_layers if opdttt_layers is not None else []
        self.ttt_lr = ttt_lr
        self.ttt_chunk = ttt_chunk
        self.ttt_proj = ttt_proj
        self.ttt_max_norm = ttt_max_norm
        self.ttt_target = ttt_target
        self.lambda_ntp = lambda_ntp
        self.lambda_align_rep = lambda_align_rep
        self.teacher_proj_init = teacher_proj_init
        self.weight_adaptation = weight_adaptation
        self.teacher_hidden_size = teacher_hidden_size if teacher_hidden_size is not None else self.hidden_size

        if self.ttt_target not in {"hidden_states", "input_embed"}:
            raise ValueError("ttt_target must be one of {'hidden_states', 'input_embed'}")


__all__ = ["Qwen3_5TTTConfig"]
