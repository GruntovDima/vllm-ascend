#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from typing import Any

import torch

from vllm_ascend._310p.ops.quant_batch_matmul import quant_batch_matmul
from vllm_ascend.utils import maybe_trans_zn

from .registry import register_scheme
from .w8a8_base import AscendW8A8Linear310pScheme


@register_scheme("W8A8", "linear")
class AscendW8A8LinearMethod310(AscendW8A8Linear310pScheme):
    """310P-only W8A8 static linear scheme.

    Notes:
      - This scheme is discovered via 310P local registry.
    """

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params: dict[str, Any] = {}
        params["quant_bias"] = torch.empty(output_size, dtype=torch.int32)
        params["deq_scale"] = torch.empty(output_size, dtype=torch.int64)
        params["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if x.dtype != torch.int8:
            x = torch.ops.vllm.quantize(
                x,
                layer.aclnn_input_scale,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
            )

        quant_bias = layer.quant_bias if tp_rank == 0 else None

        # NOTE(310P):
        # quant_batch_matmul_v3 consumes the weight as a FRACTAL_NZ tensor in
        # [K, N] view carrying the ZN byte layout (storage [K1, N1, 16, 32]
        # with a compact K-tail when K % 32 != 0) -- exactly what
        # maybe_trans_zn(weight) produces in process_weights_after_loading.
        # transpose_x2 is therefore False.
        return quant_batch_matmul(
            x,
            layer.weight.data,
            layer.deq_scale,
            bias=quant_bias,
            transpose_x2=False,
            output_dtype=layer.params_dtype,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        expanding_factor = layer.weight.data.shape[1]

        # ---- quant stage tensors ----
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False,
        )
        layer.aclnn_input_scale_reciprocal = torch.nn.Parameter(
            1.0 / layer.aclnn_input_scale.data,
            requires_grad=False,
        )
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor),
            requires_grad=False,
        ).to(layer.aclnn_input_scale.dtype)

        # ---- matmul stage tensor ----
        layer.weight.data = maybe_trans_zn(layer.weight.data)

        # ---- dequant stage tensors ----
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
