# mypy: ignore-errors
"""Coalesce client requests that arrive together (idle -> busy transition).

Why: when the engine core is idle it waits for the first ADD message, handles
it, drains whatever else is already in its input queue, and schedules a step.
A second ADD sent microseconds later can still be behind the first in the
input thread; it is then handled only after that step. Two simultaneous
requests therefore run as two sequential prefills instead of one batched
prefill (BS2 TEST2: one request 491 ms TTFT, the other 1186 ms; the batched
baseline had both at 1040 ms).

How: after an idle->busy drain, keep handling input messages until the queue
has been empty for VLLM_ASCEND_INPUT_BURST_QUIET_MS or
VLLM_ASCEND_INPUT_BURST_GRACE_MS has elapsed since the transition. Steps while
the engine is already running are untouched, so steady-state decode scheduling
does not change.

Numerically neutral: it only decides which requests share the first step; the
same kernels run on the same inputs.
"""

import os
import queue
import time

from vllm.v1.engine.core import EngineCoreProc


def _ms(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default) / 1000.0
    except ValueError:
        return default / 1000.0


_GRACE_S = _ms("VLLM_ASCEND_INPUT_BURST_GRACE_MS", 0.0)
_QUIET_S = _ms("VLLM_ASCEND_INPUT_BURST_QUIET_MS", 2.0)

if _GRACE_S > 0 and not getattr(EngineCoreProc, "_ascend_input_burst_patched", False):
    _orig_process_input_queue = EngineCoreProc._process_input_queue

    def _process_input_queue(self):
        was_idle = not self.has_work()
        _orig_process_input_queue(self)
        if not was_idle or not self.has_work():
            return
        now = time.monotonic()
        hard_deadline = now + _GRACE_S
        quiet_until = now + _QUIET_S
        while True:
            try:
                item = self.input_queue.get_nowait()
            except queue.Empty:
                now = time.monotonic()
                if now >= hard_deadline or now >= quiet_until:
                    return
                time.sleep(0.0001)
                continue
            self._handle_client_request(*item)
            quiet_until = min(hard_deadline, time.monotonic() + _QUIET_S)

    EngineCoreProc._process_input_queue = _process_input_queue
    EngineCoreProc._ascend_input_burst_patched = True
