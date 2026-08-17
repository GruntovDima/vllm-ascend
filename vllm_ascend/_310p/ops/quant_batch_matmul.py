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

import torch

_OP_NAME = "quant_batch_matmul_v3"


def _require_quant_batch_matmul() -> None:
    ascend_ops = getattr(torch.ops, "_C_ascend", None)
    if ascend_ops is None or not hasattr(ascend_ops, _OP_NAME):
        raise RuntimeError(
            "quant_batch_matmul_v3 is unavailable: rebuild the custom ops "
            "(csrc/build_aclnn.sh ascend310p) and ensure "
            "vllm_ascend.vllm_ascend_C is loaded."
        )


def quant_batch_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor | None = None,
    pertoken_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Quantized batch matmul on Ascend 310P.

    Wraps torch.ops._C_ascend.quant_batch_matmul_v3.

    Args:
        x1: int8 activation, shape [..., M, K].
        x2: int8 weight, shape [..., K, N] (ND) or the FRACTAL_NZ tensor in
            [K, N] view with K-major blocks as produced by
            maybe_trans_nz(weight[N, K]).transpose(0, 1).
        scale: per-channel weight dequant scale, shape [N] (int64
            pre-encoded or fp32), or [1], or [Q, N] for group-wise quant.
        offset: optional fp32 dequant offset added after the matmul.
        pertoken_scale: optional fp32 per-token activation scale.
        bias: optional int32 accumulator bias, shape [N].
        transpose_x1: x1 is stored as [..., K, M] if True.
        transpose_x2: x2 is stored as [..., N, K] (ND case) if True. Ignored
            for FRACTAL_NZ weights, whose view is already [K, N].
        output_dtype: cast the fp16 kernel output to this dtype when given
            and different.

    Returns:
        Output tensor with shape [..., M, N].
    """
    _require_quant_batch_matmul()
    if pertoken_scale is not None and pertoken_scale.dtype != torch.float32:
        # The kernel's pertoken_scale slot is DT_FLOAT.
        pertoken_scale = pertoken_scale.to(torch.float32)
    out = torch.ops._C_ascend.quant_batch_matmul_v3(
        x1,
        x2,
        scale,
        offset=offset,
        pertoken_scale=pertoken_scale,
        bias=bias,
        transpose_x1=transpose_x1,
        transpose_x2=transpose_x2,
    )
    if output_dtype is not None and out.dtype != output_dtype:
        out = out.to(output_dtype)
    return out
