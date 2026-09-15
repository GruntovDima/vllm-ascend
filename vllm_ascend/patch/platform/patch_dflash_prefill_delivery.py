"""Opt-in BS1 first-token delivery before queueing another decode batch.

Only the final prompt batch is drained early. Normal decode retains upstream
queue filling and overlap. This does not skip model work or alter sampling.
"""

from functools import wraps

from vllm.logger import init_logger
from vllm.v1.engine.core import EngineCore

from vllm_ascend import envs

logger = init_logger(__name__)


def _is_final_prompt_batch(core, output) -> bool:
    config = core.vllm_config
    speculative = config.speculative_config
    if (
        not core.async_scheduling
        or core.is_pooling_model
        or speculative is None
        or speculative.method != "dflash"
        or config.parallel_config.data_parallel_size != 1
        or config.parallel_config.pipeline_parallel_size != 1
        or config.scheduler_config.max_num_seqs != 1
        or output.pending_structured_output_tokens
        or len(output.num_scheduled_tokens) != 1
        or output.scheduled_spec_decode_tokens
    ):
        return False
    req_id, count = next(iter(output.num_scheduled_tokens.items()))
    request = core.scheduler.requests.get(req_id)
    if request is None or request.num_output_tokens != 0 or request.use_structured_output:
        return False
    computed = None
    for new_request in output.scheduled_new_reqs:
        if new_request.req_id == req_id:
            computed = new_request.num_computed_tokens
            break
    if computed is None:
        cached = output.scheduled_cached_reqs
        if req_id not in cached.req_ids:
            return False
        computed = cached.num_computed_tokens[cached.req_ids.index(req_id)]
    # Use this batch's immutable scheduling snapshot, not the scheduler's
    # already advanced num_computed_tokens/async output placeholders.
    return computed < request.num_prompt_tokens <= computed + count


def _drain_first_prompt_output(core):
    future, output, exec_future = core.batch_queue.pop()
    with core.log_error_detail(output), core.log_iteration_details(output):
        model_output = future.result()
        if model_output is None:
            exec_future.result()
            raise RuntimeError("unexpected error")
    core._process_aborts_queue()
    result = core.scheduler.update_from_output(output, model_output)
    logger.info_once("DFlash final-prefill output delivered before next batch enqueue")
    # No new model invocation in this step. Existing async post_step needs
    # no draft-token RPC, exactly as when upstream only drains its queue.
    return result, False


def _patch_engine_core() -> None:
    original = EngineCore.step_with_batch_queue
    if getattr(original, "_ascend_dflash_prefill_delivery", False):
        return

    @wraps(original)
    def step_with_prefill_delivery(self):
        queue = self.batch_queue
        if queue is not None and len(queue) == 1 and _is_final_prompt_batch(self, queue[-1][1]):
            return _drain_first_prompt_output(self)
        return original(self)

    step_with_prefill_delivery._ascend_dflash_prefill_delivery = True
    EngineCore.step_with_batch_queue = step_with_prefill_delivery


if envs.VLLM_ASCEND_DFLASH_PREFILL_DELIVERY:
    _patch_engine_core()
