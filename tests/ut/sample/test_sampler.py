from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import (
    AscendSampler,
    AscendTopKTopPSampler,
    _apply_top_k_top_p_custom,
)


class TestAscendSampler(TestBase):
    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, "topk_topp_sampler"))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)


def test_custom_topk_topp_caches_unavailable_extension_result():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    k = torch.tensor([2])
    p = torch.tensor([0.9])
    fallback_result = torch.tensor([[4.0, 3.0, -float("inf"), -float("inf")]])

    with (
        patch("vllm_ascend.sample.sampler._CUSTOM_TOP_K_TOP_P_OP", None),
        patch("vllm_ascend.sample.sampler._CUSTOM_TOP_K_TOP_P_RESOLVED", False),
        patch("vllm_ascend.sample.sampler.enable_custom_op", return_value=False) as enable,
        patch(
            "vllm_ascend.sample.sampler._apply_top_k_top_p_pytorch",
            return_value=fallback_result,
        ) as fallback,
    ):
        first_result = _apply_top_k_top_p_custom(logits, k, p, top_k=2)
        second_result = _apply_top_k_top_p_custom(logits, k, p, top_k=2)

    assert first_result is fallback_result
    assert second_result is fallback_result
    enable.assert_called_once_with()
    assert fallback.call_count == 2
    fallback.assert_called_with(logits, k, p, 2)
