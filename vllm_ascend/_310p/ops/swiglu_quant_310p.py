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
"""Opt-in fused SwiGlu + per-channel AscendQuant for the 310P eager prefill.

Replaces the `npu_swiglu` -> `npu_quantize` pair (which feeds the static-W8A8
down_proj) with one kernel `swiglu_quant_v310`. The fused kernel is validated
to produce int8 output identical to the split pair at op level (same CANN
source chains: swi_glu fp32 path and AscendQuantV2 fp16 CastOut MODE_ROUND /
M200 branch). Only the uncompiled prefill path is intercepted: decode keeps
the original split path (graph-friendly), and any mismatch of the guard
conditions falls back to the original ops.

The op is invoked directly through the vendor aclnn API (libcust_opapi.so)
via ctypes so that the main vllm_ascend_C extension does not have to be
rebuilt or replaced for deployment.
"""

import ctypes
import os

import torch
import torch_npu

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm_ascend import envs

PREFILL_MIN_TOKENS = 128

_ACL_FLOAT16 = 1
_ACL_INT8 = 2
_ACL_FORMAT_ND = 2

c_i64_p = ctypes.POINTER(ctypes.c_int64)

_LIB = None
_ACL = None


def _load_cann_lib(name):
    try:
        return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    cands = []
    for env in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        base = os.environ.get(env)
        if base:
            cands.append(os.path.join(base, "lib64", name))
            cands.append(os.path.join(base, "runtime", "lib64", name))
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if d:
            cands.append(os.path.join(d, name))
    for c in cands:
        if os.path.isfile(c):
            return ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
    raise OSError(f"{name} not found; source the CANN set_env.sh first")


def _get_lib():
    global _LIB, _ACL
    if _LIB is None:
        _load_cann_lib("libascendcl.so")
        acl = _load_cann_lib("libnnopbase.so")
        acl.aclCreateTensor.restype = ctypes.c_void_p
        acl.aclCreateTensor.argtypes = [
            c_i64_p, ctypes.c_uint64, ctypes.c_int, c_i64_p, ctypes.c_int64,
            ctypes.c_int, c_i64_p, ctypes.c_uint64, ctypes.c_void_p,
        ]
        acl.aclDestroyTensor.restype = None
        acl.aclDestroyTensor.argtypes = [ctypes.c_void_p]
        acl.aclDestroyAclOpExecutor.restype = None
        acl.aclDestroyAclOpExecutor.argtypes = [ctypes.c_void_p]
        for entry in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
            path = os.path.join(entry, "op_api", "lib", "libcust_opapi.so")
            if os.path.isfile(path):
                lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                lib.aclnnSwigluQuantV310GetWorkspaceSize.restype = ctypes.c_int
                lib.aclnnSwigluQuantV310GetWorkspaceSize.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p),
                ]
                lib.aclnnSwigluQuantV310.restype = ctypes.c_int
                lib.aclnnSwigluQuantV310.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p,
                ]
                _LIB = lib
                _ACL = acl
                break
        if _LIB is None:
            raise RuntimeError("libcust_opapi.so not found under ASCEND_CUSTOM_OPP_PATH")
    return _LIB


def _make_tensor(ptr, shape, dtype):
    dims = (ctypes.c_int64 * len(shape))(*shape)
    strides = (ctypes.c_int64 * len(shape))()
    acc = 1
    for i in reversed(range(len(shape))):
        strides[i] = acc
        acc *= shape[i]
    handle = _ACL.aclCreateTensor(
        dims, len(shape), dtype, strides, 0, _ACL_FORMAT_ND, dims, len(shape),
        ctypes.c_void_p(ptr),
    )
    if not handle:
        raise RuntimeError("aclCreateTensor failed")
    return handle


def _swiglu_quant_v310_impl(x: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """aclnn call through ctypes: the runtime extension .so is not modified."""
    lib = _get_lib()
    if not scale.is_contiguous():
        scale = scale.contiguous()
    if not offset.is_contiguous():
        offset = offset.contiguous()
    m, n2 = x.shape
    n = n2 // 2
    y = torch.empty((m, n), dtype=torch.int8, device=x.device)
    tx = _make_tensor(x.data_ptr(), (m, n2), _ACL_FLOAT16)
    ts = _make_tensor(scale.data_ptr(), (n,), _ACL_FLOAT16)
    to = _make_tensor(offset.data_ptr(), (n,), _ACL_FLOAT16)
    ty = _make_tensor(y.data_ptr(), (m, n), _ACL_INT8)
    ws_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p()
    try:
        ret = lib.aclnnSwigluQuantV310GetWorkspaceSize(
            tx, ts, to, ty, ctypes.byref(ws_size), ctypes.byref(executor)
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSwigluQuantV310GetWorkspaceSize failed: {ret}")
        workspace = torch.empty(max(int(ws_size.value), 1), dtype=torch.uint8, device=x.device)
        stream = torch_npu.npu.current_stream().npu_stream
        ret = lib.aclnnSwigluQuantV310(
            ctypes.c_void_p(workspace.data_ptr()), ws_size.value, executor, ctypes.c_void_p(stream)
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSwigluQuantV310 failed: {ret}")
    finally:
        if executor:
            _ACL.aclDestroyAclOpExecutor(executor)
        for handle in (tx, ts, to, ty):
            if handle:
                _ACL.aclDestroyTensor(handle)
    return y


def prepare_swiglu_quant(model) -> int:
    """Attach the down_proj quant parameters to each MLP act module."""
    if not envs.VLLM_ASCEND_SWIGLU_QUANT_310P:
        return 0
    attached = 0
    for _, module in model.named_modules():
        if not hasattr(module, "act_fn") or not hasattr(module, "down_proj"):
            continue
        act = module.act_fn
        linear = module.down_proj
        scheme = getattr(getattr(linear, "quant_method", None), "quant_method", None)
        if not getattr(scheme, "accepts_prequantized_input", False):
            continue
        recip = getattr(linear, "aclnn_input_scale_reciprocal", None)
        offset = getattr(linear, "aclnn_input_offset", None)
        if (
            recip is None
            or offset is None
            or recip.dtype != torch.float16
            or offset.dtype != torch.float16
            or recip.ndim != 1
            or recip.shape != offset.shape
        ):
            continue
        act._swq_recip = recip.detach()
        act._swq_offset = offset.detach()
        attached += 1
    return attached


def maybe_swiglu_quant(act_module, x):
    """Fused path for the eager prefill; None means use the original ops."""
    if not envs.VLLM_ASCEND_SWIGLU_QUANT_310P:
        return None
    recip = getattr(act_module, "_swq_recip", None)
    offset = getattr(act_module, "_swq_offset", None)
    if recip is None or offset is None:
        return None
    if (
        x.dtype != torch.float16
        or x.ndim != 2
        or x.shape[-1] % 64 != 0
        or not x.is_contiguous()
        or x.shape[0] < PREFILL_MIN_TOKENS
    ):
        return None
    if (
        not is_forward_context_available()
        or get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.NONE
    ):
        return None
    return _swiglu_quant_v310_impl(x, recip, offset)
