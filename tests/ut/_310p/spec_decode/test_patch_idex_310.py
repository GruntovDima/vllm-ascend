# SPDX-License-Identifier: Apache-2.0

import importlib.util
import inspect
import sys
import types
from contextlib import contextmanager
from pathlib import Path


class _NativeProposer:
    @staticmethod
    def _scale_block_ids_for_slot_mapping(block_ids, block_size):
        return "generic", block_ids, block_size


class _Proposer310(_NativeProposer):
    def set_inputs_first_pass(self):
        pass

    def _run_merged_draft(self):
        pass

    def _propose(self):
        pass

    def _sample_draft_from_logits(self):
        pass

    @staticmethod
    def _scale_block_ids_for_slot_mapping(block_ids, block_size):
        return "310p", block_ids, block_size


class _GatedDeltaNetAttention:
    @staticmethod
    def _split_ba_for_tp():
        pass

    @staticmethod
    def get_state_shape():
        pass


class _GatedDeltaNetAttention310:
    @staticmethod
    def _forward_core():
        pass

    @staticmethod
    def get_state_dtype():
        pass

    @staticmethod
    def get_attn_backend():
        pass


def _stub_module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


@contextmanager
def _patched_modules(stubs):
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_patch_installs_static_slot_scaler_on_runtime_proposer():
    fla_index = _stub_module("vllm.third_party.flash_linear_attention.ops.index")
    fla_ops = _stub_module("vllm.third_party.flash_linear_attention.ops", index=fla_index)
    stubs = {
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn": _stub_module(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
            QwenGatedDeltaNetAttention=type("QwenGatedDeltaNetAttention", (), {}),
        ),
        "vllm.third_party.flash_linear_attention.ops": fla_ops,
        "vllm.third_party.flash_linear_attention.ops.index": fla_index,
        "vllm_ascend._310p.ops.fla.gdn_310": _stub_module(
            "vllm_ascend._310p.ops.fla.gdn_310",
            AscendGatedDeltaNetAttention310=_GatedDeltaNetAttention310,
        ),
        "vllm_ascend._310p.ops.fla.idex": _stub_module(
            "vllm_ascend._310p.ops.fla.idex",
            prepare_chunk_indices_310=lambda: None,
            prepare_chunk_offsets_310=lambda: None,
        ),
        "vllm_ascend._310p.spec_decode.llm_base_proposer_310": _stub_module(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310",
            AscendSpecDecodeBaseProposer310=_Proposer310,
        ),
        "vllm_ascend.ops.gdn": _stub_module(
            "vllm_ascend.ops.gdn",
            AscendGatedDeltaNetAttention=_GatedDeltaNetAttention,
        ),
        "vllm_ascend.spec_decode.llm_base_proposer": _stub_module(
            "vllm_ascend.spec_decode.llm_base_proposer",
            AscendSpecDecodeBaseProposer=_NativeProposer,
        ),
        "vllm_ascend.utils": _stub_module("vllm_ascend.utils", is_rc_device=lambda: False),
    }
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_idex_310.py"
    spec = importlib.util.spec_from_file_location("_standalone_patch_idex_310", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    with _patched_modules(stubs):
        spec.loader.exec_module(module)

    class RuntimeMTPProposer(_NativeProposer):
        pass

    runtime_proposer = RuntimeMTPProposer()
    assert runtime_proposer._scale_block_ids_for_slot_mapping("block_ids", 128) == (
        "310p",
        "block_ids",
        128,
    )
    assert isinstance(inspect.getattr_static(_NativeProposer, "_scale_block_ids_for_slot_mapping"), staticmethod)
