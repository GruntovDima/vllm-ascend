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

from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.ops.quant_batch_matmul import quant_batch_matmul


class TestQuantBatchMatmul(TestBase):
    @patch("torch.ops._C_ascend.quant_batch_matmul_v3", create=True)
    def test_call_passthrough_and_dtype_cast(self, mock_op):
        x1 = torch.randint(-128, 127, (32, 128), dtype=torch.int8)
        x2 = torch.randint(-128, 127, (128, 256), dtype=torch.int8)
        scale = torch.randn(256, dtype=torch.int64)
        bias = torch.randint(-128, 127, (256,), dtype=torch.int32)
        pertoken_scale = torch.randn(32, dtype=torch.float16)

        fp16_out = torch.randn(32, 256, dtype=torch.float16)
        mock_op.return_value = fp16_out

        out = quant_batch_matmul(
            x1,
            x2,
            scale,
            pertoken_scale=pertoken_scale,
            bias=bias,
            transpose_x2=False,
            output_dtype=torch.bfloat16,
        )

        mock_op.assert_called_once()
        (args, kwargs) = mock_op.call_args
        self.assertTrue(torch.equal(args[0], x1))
        self.assertTrue(torch.equal(args[1], x2))
        self.assertTrue(torch.equal(args[2], scale))
        self.assertTrue(kwargs["offset"] is None)
        # pertoken_scale is cast to fp32 to match the kernel's DT_FLOAT slot
        self.assertEqual(kwargs["pertoken_scale"].dtype, torch.float32)
        self.assertTrue(torch.equal(kwargs["bias"], bias))
        self.assertFalse(kwargs["transpose_x1"])
        self.assertFalse(kwargs["transpose_x2"])
        # output is cast to the requested dtype
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out, fp16_out.to(torch.bfloat16)))

    @patch("torch.ops._C_ascend", MagicMock(spec=[]), create=True)
    def test_unavailable_op_raises(self):
        with self.assertRaises(RuntimeError):
            quant_batch_matmul(
                torch.randint(-128, 127, (32, 128), dtype=torch.int8),
                torch.randint(-128, 127, (128, 256), dtype=torch.int8),
                torch.randn(256, dtype=torch.int64),
            )
