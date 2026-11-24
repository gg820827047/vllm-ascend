# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in, bounded scheduler observations and actual waiting-branch events.

The tracker belongs to one EngineCore scheduler. schedule() and its synchronous
helpers run on that EngineCore thread; there is no cross-process/global request
registry. Wrappers forward original arguments, results and exceptions unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)

_TRACKER_ATTR = "_ascend_admission_diagnostics"
_WRAPPED_ATTR = "_ascend_admission_init_wrapped"
_MAX_SAMPLES = 4
_MAX_REQUEST_ID_LENGTH = 256
_MILLISECONDS_PER_SECOND = 1000.0


def _name(value: Any) -> str:
    return str(getattr(value, "name", value))


def _request(args: tuple, kwargs: dict) -> Any:
    return args[0] if args else kwargs.get("request")


class SchedulerAdmissionDiagnostics:
    """Keep actual branch stops, helper outcomes and return constraints separate."""

    def __init__(self, scheduler: Any, interval_s: float, clock: Callable[[], float] = time.perf_counter):
        if interval_s <= 0:
            raise ValueError("admission diagnostic interval must be positive")
        self.scheduler = scheduler
        self.interval_s = interval_s
        self.clock = clock
        self.original_schedule = scheduler.schedule
        self.active = False
        self.helper_owners: tuple | None = None
        self.last_emit: float | None = None
        self.window_start = clock()
        self.window_steps = 0
        self.schedule_total_ms = 0.0
        self.schedule_max_ms = 0.0
        self.observations: Counter[str] = Counter()
        self.step_limits: Counter[str] = Counter()
        self.branch_reasons: Counter[str] = Counter()
        self.branch_sources: set[str] = set()
        self.branch_samples: list[dict] = []
        self.omitted_branch_samples = 0
        self.timings: dict[str, list[float]] = {}
        self.samples: list[dict] = []
        self.omitted_samples = 0
        self.warned = False

    def _safe(self, operation: Callable, *args: Any) -> None:
        # An observational failure must not fail an inference request or mask an
        # original scheduler exception. Warn once, without per-step traceback I/O.
        try:
            operation(*args)
        except Exception as error:
            if not self.warned:
                self.warned = True
                logger.warning("SchedulerAdmission diagnostic error: %s", error)

    def _observe(self, reason: str, request: Any) -> None:
        self.observations[reason] += 1
        if len(self.samples) >= _MAX_SAMPLES:
            self.omitted_samples += 1
            return
        status = _name(getattr(request, "status", "unknown"))
        self.samples.append(
            {
                "reason": reason,
                "request_id": str(getattr(request, "request_id", "unknown"))[:_MAX_REQUEST_ID_LENGTH],
                "status": status,
                "phase": "running" if status == "RUNNING" else "waiting",
            }
        )

    def enter_schedule_source(self, source: str) -> None:
        """Called inside an instrumented schedule body before its waiting loop."""
        if self.active:
            self.branch_sources.add(source)

    def record_branch(
        self,
        reason: str,
        request: Any,
        action: str,
        token_budget: int,
    ) -> None:
        """Called immediately before the real waiting-path break/continue."""
        if self.active:
            location = None
            if len(self.branch_samples) < _MAX_SAMPLES:
                # Resolve only bounded samples, without walking a Python stack
                # or retaining frame objects in the diagnostic window.
                try:
                    frame = sys._getframe(1)
                    location = f"{frame.f_code.co_filename}:{frame.f_lineno}"
                    del frame
                except Exception:
                    # A restricted frame-introspection policy must not affect
                    # the scheduler's decision or suppress the event counter.
                    pass
            self._safe(self._record_branch, reason, request, action, location, token_budget)

    def _record_branch(self, reason, request, action, location, token_budget) -> None:
        # A natural end of an empty queue is not an admission failure. This is
        # O(1); do not scan or peek queues solely for logging a RequestID.
        if request is None and not (self.scheduler.waiting or getattr(self.scheduler, "kv_holding_waiting", ())):
            return
        self.branch_reasons[reason] += 1
        if len(self.branch_samples) >= _MAX_SAMPLES:
            self.omitted_branch_samples += 1
            return
        self.branch_samples.append(
            {
                "reason": reason,
                "scope": "request" if request is not None else "waiting_queue",
                "request_id": (str(request.request_id)[:_MAX_REQUEST_ID_LENGTH] if request is not None else None),
                "status": _name(request.status) if request is not None else None,
                "phase": "waiting",
                "action": action,
                "location": location,
                "token_budget": token_budget,
                "timestamp": time.time(),
            }
        )

    def _record_helper(self, name: str, elapsed_ms: float, args: tuple, kwargs: dict, result: Any) -> None:
        timing = self.timings.setdefault(name, [0, 0.0, 0.0])
        timing[0] += 1
        timing[1] += elapsed_ms
        timing[2] = max(timing[2], elapsed_ms)
        request = _request(args, kwargs)
        if name == "allocate_slots" and result is None:
            self._observe("kv_allocation_failed", request)
        elif name == "cache_match" and result[0] is None:
            self._observe("cache_lookup_pending", request)
        elif name == "blocked_state" and not result:
            self._observe("request_state_not_ready", request)

    def _wrap_helper(self, owner: Any, method_name: str, stage: str) -> None:
        original = getattr(owner, method_name, None)
        if not callable(original):
            return
        # Only instance methods of this scheduler / its owned manager and root
        # connector are wrapped. No global KV manager or connector class changes.
        if getattr(original, _TRACKER_ATTR, None) is self:
            return

        @wraps(original)
        def observed(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            start = self.clock()
            result = original(*args, **kwargs)
            elapsed_ms = (self.clock() - start) * _MILLISECONDS_PER_SECOND
            self._safe(self._record_helper, stage, elapsed_ms, args, kwargs, result)
            return result

        setattr(observed, _TRACKER_ATTR, self)
        setattr(owner, method_name, observed)

    def _ensure_helpers(self) -> None:
        # Subclasses may replace these objects after the base constructor. Wait
        # until the first schedule, and refresh when an owned object is replaced.
        owners = (self.scheduler, self.scheduler.kv_cache_manager, self.scheduler.connector)
        if self.helper_owners is not None and all(a is b for a, b in zip(owners, self.helper_owners)):
            return
        self._wrap_helper(owners[0], "_handle_blocked_waiting_request", "blocked_state")
        self._wrap_helper(owners[1], "allocate_slots", "allocate_slots")
        self._wrap_helper(owners[1], "get_computed_blocks", "local_cache_lookup")
        if owners[2] is not None:
            self._wrap_helper(owners[2], "get_num_new_matched_tokens", "cache_match")
        self.helper_owners = owners

    def _snapshot(self, output: Any) -> dict:
        scheduler = self.scheduler
        pool = scheduler.kv_cache_manager.block_pool
        running = len(scheduler.running)
        waiting = len(scheduler.waiting)
        holding = len(getattr(scheduler, "kv_holding_waiting", ()))
        stream_waiting = getattr(scheduler, "num_waiting_for_streaming_input", 0)
        balance_enabled = bool(getattr(scheduler, "_balance_enabled", False))
        balance_counts = []
        if balance_enabled:
            for value in scheduler.balance_queue:
                # The CANN balance gate uses CPU tensors. Never create a device
                # synchronization solely to obtain diagnostic fields.
                if isinstance(value, int):
                    balance_counts.append(value)
                elif getattr(getattr(value, "device", None), "type", None) == "cpu":
                    balance_counts.append(value.item())
        return {
            "running": running,
            "waiting": waiting,
            "kv_holding_waiting": holding,
            "streaming_input_waiting": stream_waiting,
            "active_limit": getattr(scheduler, "max_num_active_reqs", scheduler.max_num_running_reqs),
            "running_limit": scheduler.max_num_running_reqs,
            "token_budget": scheduler.max_num_scheduled_tokens,
            "scheduled_tokens": getattr(output, "total_num_scheduled_tokens", 0),
            "scheduled_new_requests": len(getattr(output, "scheduled_new_reqs", ())),
            "free_kv_blocks": pool.get_num_free_blocks(),
            "pause_state": _name(getattr(scheduler, "_pause_state", "unknown")),
            "balance_enabled": balance_enabled,
            "balance_running_counts": balance_counts,
        }

    def _finish(self, elapsed_ms: float, output: Any) -> None:
        self.window_steps += 1
        self.schedule_total_ms += elapsed_ms
        self.schedule_max_ms = max(self.schedule_max_ms, elapsed_ms)
        state = self._snapshot(output)
        if state["waiting"] + state["kv_holding_waiting"] > 0:
            # These are observed limits at schedule return, NOT a claim that
            # any particular unexamined request hit this branch in schedule().
            if state["running"] + state["streaming_input_waiting"] >= state["active_limit"]:
                self.step_limits["active_limit_reached"] += 1
            if state["scheduled_tokens"] >= state["token_budget"]:
                self.step_limits["token_budget_exhausted"] += 1
            if state["pause_state"] not in ("UNPAUSED", "unknown"):
                self.step_limits["scheduler_paused"] += 1
            if state["balance_enabled"] and state["running_limit"] in state["balance_running_counts"]:
                self.step_limits["dp_balance_running_limit_reached"] += 1
        now = self.clock()
        if self.last_emit is not None and now - self.last_emit < self.interval_s:
            return
        # Emit even for an idle queue: host-heavy schedules without backlog are
        # useful evidence, and this remains at most one line per interval.
        parallel = self.scheduler.vllm_config.parallel_config
        payload = {
            "pid": os.getpid(),
            "dp_rank": parallel.data_parallel_rank,
            "scheduler": type(self.scheduler).__name__,
            "timestamp": time.time(),
            "window_ms": (now - self.window_start) * _MILLISECONDS_PER_SECOND,
            "steps": self.window_steps,
            "schedule_total_ms": self.schedule_total_ms,
            "schedule_max_ms": self.schedule_max_ms,
            "latest": state,
            "observations": dict(self.observations),
            "step_limits": dict(self.step_limits),
            "branch_reasons": dict(self.branch_reasons),
            "branch_sources": sorted(self.branch_sources),
            "branch_samples": self.branch_samples,
            "omitted_branch_samples": self.omitted_branch_samples,
            "helper_timings": {
                name: {"calls": values[0], "total_ms": values[1], "max_ms": values[2]}
                for name, values in self.timings.items()
            },
            "samples": self.samples,
            "omitted_samples": self.omitted_samples,
        }
        try:
            logger.info("[SchedulerAdmission] %s", json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
        finally:
            # Preserve the output rate limit even when the log sink fails.
            self.last_emit = now
            self.window_start = now
            self.window_steps = 0
            self.schedule_total_ms = 0.0
            self.schedule_max_ms = 0.0
            self.observations.clear()
            self.step_limits.clear()
            self.branch_reasons.clear()
            self.branch_sources.clear()
            self.branch_samples.clear()
            self.omitted_branch_samples = 0
            self.timings.clear()
            self.samples.clear()
            self.omitted_samples = 0

    def schedule(self, *args, **kwargs):
        if self.active:
            return self.original_schedule(*args, **kwargs)
        self._safe(self._ensure_helpers)
        self.active = True
        start = self.clock()
        try:
            output = self.original_schedule(*args, **kwargs)
        finally:
            self.active = False
        elapsed_ms = (self.clock() - start) * _MILLISECONDS_PER_SECOND
        self._safe(self._finish, elapsed_ms, output)
        return output


def instrument_scheduler(scheduler: Any, interval_s: float) -> SchedulerAdmissionDiagnostics:
    existing = getattr(scheduler, _TRACKER_ATTR, None)
    if existing is not None:
        return existing
    tracker = SchedulerAdmissionDiagnostics(scheduler, interval_s)

    @wraps(tracker.original_schedule)
    def observed_schedule(*args, **kwargs):
        return tracker.schedule(*args, **kwargs)

    setattr(scheduler, _TRACKER_ATTR, tracker)
    scheduler.schedule = observed_schedule
    return tracker


def install_scheduler_admission_diagnostics(base_scheduler: type, interval_s: float) -> None:
    """Install once on the upstream base, covering existing Ascend subclasses."""
    original = base_scheduler.__init__
    if getattr(original, _WRAPPED_ATTR, False):
        return

    @wraps(original)
    def observed_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        instrument_scheduler(self, interval_s)

    setattr(observed_init, _WRAPPED_ATTR, True)
    base_scheduler.__init__ = observed_init
