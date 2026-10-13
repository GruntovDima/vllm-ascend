# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

import torch
import torch_npu

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import maybe_record_fused_moe_event
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput

_BATCH_SCALE_GATE_UP_X_SHAPE = (640, 2048)
_BATCH_SCALE_GATE_UP_WEIGHT_SHAPE = (256, 512, 2048)
_BATCH_SCALE_DOWN_X_SHAPE = (640, 256)
_BATCH_SCALE_DOWN_WEIGHT_SHAPE = (256, 2048, 256)
_FRACTAL_NZ_FORMAT = 29
_INT8_QUANT_MAX = 127.0


def _supports_named_quant_gmm(x, weight, weight_scale, group_list) -> bool:
    """Layout guard after the validated shape/dtype guard; metadata only."""
    return (
        x.device.type == "npu"
        and tuple(weight_scale.shape) == tuple(weight.shape[:2])
        and tuple(group_list.shape) == (weight.shape[0],)
        and all(t.device == x.device and t.is_contiguous() for t in (x, weight, weight_scale, group_list))
        and all(torch_npu.get_npu_format(t) in (0, 2) for t in (x, weight_scale, group_list))
    )


def _quant_grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    # Only use batch scale calculation for shapes validated on 310P. Keep
    # other workloads on CANN's internal per-token dynamic-quant path.
    supported_shape = (
        tuple(x.shape) == _BATCH_SCALE_GATE_UP_X_SHAPE and tuple(weight.shape) == _BATCH_SCALE_GATE_UP_WEIGHT_SHAPE
    ) or (tuple(x.shape) == _BATCH_SCALE_DOWN_X_SHAPE and tuple(weight.shape) == _BATCH_SCALE_DOWN_WEIGHT_SHAPE)
    x_scale = None
    if (
        supported_shape
        and x.dtype == torch.float16
        and weight.dtype == torch.int8
        and weight_scale.dtype == torch.float32
        and group_list.dtype == torch.int64
        and torch_npu.get_npu_format(weight) == _FRACTAL_NZ_FORMAT
    ):
        # Reduce each row independently before entering the grouped kernel.
        # Keep FP32 scale division and a nonzero scale for all-zero rows.
        absmax = x.abs().amax(dim=-1).float()
        x_scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / _INT8_QUANT_MAX)
        # Custom ops are loaded by worker initialization. An older extension
        # or unsupported tensor layout keeps the original CANN call below.
        # This dispatch uses metadata only, never a device-to-host group read.
        custom_op = getattr(torch.ops._C_ascend, "npu_quant_grouped_matmul_dequant_310", None)
        if custom_op is not None and _supports_named_quant_gmm(x, weight, weight_scale, group_list):
            return custom_op(x, weight, weight_scale, group_list, x_scale)
    return torch_npu.npu_quant_grouped_matmul_dequant(
        x=x,
        quantized_weight=weight,
        weight_scale=weight_scale,
        group_list=group_list,
        quant_mode="pertoken",
        x_scale=x_scale,
    )


def apply_moe_mlp(
    mlp_compute_input: MoEMlpComputeInput,
    quant_method,
) -> tuple[torch.Tensor, torch.npu.Event]:
    """
    Unified MoE MLP entry (310P).

    310P MoE only supports the swiglu (silu) activation, so every method
    implements the fused ``gmm1 + swiglu (+ quant)`` path and the separate
    ``apply_gmm1`` / ``apply_act_quant`` hooks are not used.
    """

    if not quant_method.supports_fused_activation(mlp_compute_input.activation):
        activation = mlp_compute_input.activation
        act_name = getattr(activation, "value", activation)
        raise NotImplementedError(f"310P MoE only supports the swiglu (silu) activation, but got {act_name}.")

    hidden_states, act_out_scale = quant_method.apply_gmm1_act_quant(mlp_compute_input)

    before_gmm2_evt = maybe_record_fused_moe_event(mlp_compute_input.record_events)
    hidden_states = quant_method.apply_gmm2(mlp_compute_input, hidden_states, act_out_scale)
    return hidden_states, before_gmm2_evt
