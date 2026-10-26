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
"""310P patch: route the MLP activation through the fused swiglu+quant op.

Enabled by VLLM_ASCEND_SWIGLU_QUANT_310P=1 together with the prepared
per-linear quant params (see prepare_swiglu_quant). The wrapper falls back to
the stock forward whenever the fused op's guards reject the call.
"""

from vllm_ascend import envs
from vllm_ascend._310p.ops.swiglu_quant_310p import maybe_swiglu_quant
from vllm_ascend.utils import is_310p

if envs.VLLM_ASCEND_SWIGLU_QUANT_310P and is_310p():
    from vllm_ascend._310p.ops.activation import AscendSiluAndMul310

    _orig_forward = AscendSiluAndMul310.forward

    def _forward_with_fused_swiglu_quant(self, x):
        fused = maybe_swiglu_quant(self, x)
        if fused is not None:
            return fused
        return _orig_forward(self, x)

    AscendSiluAndMul310.forward = _forward_with_fused_swiglu_quant
