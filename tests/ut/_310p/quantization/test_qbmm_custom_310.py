from unittest.mock import patch

import pytest

from vllm_ascend._310p.quantization.methods import qbmm_custom


@pytest.fixture(autouse=True)
def _reset_qbmm_registration():
    qbmm_custom._ENABLED = False
    yield
    qbmm_custom._ENABLED = False


def test_requested_qbmm_loads_extension_before_registration():
    with (
        patch.dict("os.environ", {"VLLM_CUSTOM_QBMM": "1"}),
        patch("vllm_ascend.utils.enable_custom_op", return_value=True) as enable,
        patch.object(qbmm_custom, "_raw_qbmm_registered", return_value=True),
        patch.object(qbmm_custom, "direct_register_custom_op") as register,
    ):
        assert qbmm_custom.ensure_registered()

    enable.assert_called_once_with()
    register.assert_called_once_with(
        op_name="qbmm_v3x",
        op_func=qbmm_custom._qbmm_v3x,
        mutates_args=[],
        fake_impl=qbmm_custom._qbmm_v3x_fake,
    )


def test_requested_qbmm_fails_if_extension_cannot_load():
    with (
        patch.dict("os.environ", {"VLLM_CUSTOM_QBMM": "1"}),
        patch("vllm_ascend.utils.enable_custom_op", return_value=False),
        pytest.raises(RuntimeError, match="custom-op extension could not be loaded"),
    ):
        qbmm_custom.ensure_registered()


def test_requested_qbmm_fails_if_raw_schema_is_missing():
    with (
        patch.dict("os.environ", {"VLLM_CUSTOM_QBMM": "1"}),
        patch("vllm_ascend.utils.enable_custom_op", return_value=True),
        patch.object(qbmm_custom, "_raw_qbmm_registered", return_value=False),
        pytest.raises(RuntimeError, match="quant_batch_matmul_v3_x is not registered"),
    ):
        qbmm_custom.ensure_registered()
