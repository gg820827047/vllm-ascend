# SPDX-License-Identifier: Apache-2.0
"""CPU-only observations and scheduling-equivalence tests; no NPU imports."""

import importlib.util
import inspect
import json
import os
import sys
import types
import unittest
from collections import deque
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_source(name, relative_path):
    path = Path(__file__).resolve().parents[3] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnostics = load_source("admission_diagnostics_cpu", "vllm_ascend/core/scheduler_admission_diagnostics.py")
envs = load_source("admission_envs_cpu", "vllm_ascend/envs.py")


class Status(Enum):
    WAITING = auto()
    RUNNING = auto()
    WAITING_FOR_REMOTE_KVS = auto()


class CacheManager:
    def __init__(self):
        self.block_pool = SimpleNamespace(get_num_free_blocks=lambda: 16)
        self.allocation = []
        self.calls = []

    def allocate_slots(self, request, num_tokens):
        self.calls.append((request.request_id, num_tokens))
        return self.allocation

    def get_computed_blocks(self, request):
        return (), 0


class Connector:
    def __init__(self, result):
        self.result = result

    def get_num_new_matched_tokens(self, request, local_tokens):
        return self.result


class Scheduler:
    def __init__(self, count=1, connector=None):
        self.waiting = deque(SimpleNamespace(request_id=f"request-{i}", status=Status.WAITING) for i in range(count))
        self.kv_holding_waiting = deque()
        self.running = []
        self.kv_cache_manager = CacheManager()
        self.connector = connector
        self.max_num_active_reqs = 8
        self.max_num_running_reqs = 8
        self.max_num_scheduled_tokens = 32
        self._pause_state = SimpleNamespace(name="UNPAUSED")
        self.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=2))
        self.failure = None

    def _handle_blocked_waiting_request(self, request):
        return request.status == Status.WAITING

    def schedule(self, throttle_prefills=False):
        if self.failure is not None:
            raise self.failure
        output = SimpleNamespace(total_num_scheduled_tokens=0, scheduled_new_reqs=[])
        if throttle_prefills:
            return output
        for request in tuple(self.waiting):
            if len(self.running) >= self.max_num_active_reqs:
                break
            if output.total_num_scheduled_tokens >= self.max_num_scheduled_tokens:
                break
            if not self._handle_blocked_waiting_request(request):
                continue
            _, local_tokens = self.kv_cache_manager.get_computed_blocks(request)
            if self.connector is not None:
                matched, _ = self.connector.get_num_new_matched_tokens(request, local_tokens)
                if matched is None:
                    continue
            if self.kv_cache_manager.allocate_slots(request, num_tokens=1) is None:
                continue
            self.waiting.remove(request)
            self.running.append(request)
            request.status = Status.RUNNING
            output.scheduled_new_reqs.append(request)
            output.total_num_scheduled_tokens += 1
        return output


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class AdmissionDiagnosticsTests(unittest.TestCase):
    def attach(self, scheduler, interval=1.0):
        clock = Clock()
        tracker = diagnostics.SchedulerAdmissionDiagnostics(scheduler, interval, clock)
        scheduler.schedule = tracker.schedule
        return tracker, clock

    def read_log(self, mocked):
        return json.loads(mocked.call_args.args[1])

    def test_pending_lookup_has_request_id_and_does_not_claim_kv_failure(self):
        scheduler = Scheduler(connector=Connector((None, False)))
        self.attach(scheduler)
        with patch.object(diagnostics.logger, "info") as output:
            scheduler.schedule()
        payload = self.read_log(output)
        self.assertEqual(payload["observations"], {"cache_lookup_pending": 1})
        self.assertEqual(payload["samples"][0]["request_id"], "request-0")
        self.assertEqual(payload["samples"][0]["phase"], "waiting")
        self.assertEqual(payload["dp_rank"], 2)
        self.assertEqual(payload["latest"]["waiting"], 1)

    def test_real_allocation_failure_and_kwargs_forwarding(self):
        scheduler = Scheduler()
        scheduler.kv_cache_manager.allocation = None
        self.attach(scheduler)
        with patch.object(diagnostics.logger, "info") as output:
            scheduler.schedule()
        self.assertEqual(self.read_log(output)["observations"], {"kv_allocation_failed": 1})
        self.assertEqual(scheduler.kv_cache_manager.calls, [("request-0", 1)])
        self.assertFalse(scheduler.running)

    def test_request_not_ready_retains_its_actual_status(self):
        scheduler = Scheduler()
        scheduler.waiting[0].status = Status.WAITING_FOR_REMOTE_KVS
        self.attach(scheduler)
        with patch.object(diagnostics.logger, "info") as output:
            scheduler.schedule()
        payload = self.read_log(output)
        self.assertEqual(payload["observations"], {"request_state_not_ready": 1})
        self.assertEqual(payload["samples"][0]["status"], "WAITING_FOR_REMOTE_KVS")

    def test_running_allocation_retry_is_not_reported_as_waiting_denial(self):
        class Retrying(Scheduler):
            def schedule(self, throttle_prefills=False):
                request = SimpleNamespace(request_id="running-request", status=Status.RUNNING)
                self.kv_cache_manager.allocation = None
                self.kv_cache_manager.allocate_slots(request, num_tokens=1)
                self.kv_cache_manager.allocation = []
                self.kv_cache_manager.allocate_slots(request, num_tokens=1)
                return SimpleNamespace(total_num_scheduled_tokens=1, scheduled_new_reqs=[])

        scheduler = Retrying()
        self.attach(scheduler)
        with patch.object(diagnostics.logger, "info") as output:
            scheduler.schedule()
        payload = self.read_log(output)
        self.assertEqual(payload["observations"]["kv_allocation_failed"], 1)
        self.assertEqual(payload["samples"][0]["phase"], "running")
        self.assertEqual(payload["helper_timings"]["allocate_slots"]["calls"], 2)

    def test_token_limit_is_step_constraint_not_per_request_observation(self):
        scheduler = Scheduler(count=3)
        scheduler.max_num_scheduled_tokens = 1
        self.attach(scheduler)
        with patch.object(diagnostics.logger, "info") as output:
            scheduler.schedule()
        payload = self.read_log(output)
        self.assertEqual(payload["step_limits"], {"token_budget_exhausted": 1})
        self.assertEqual(payload["observations"], {})
        self.assertEqual(payload["samples"], [])
        self.assertEqual(payload["latest"]["waiting"], 2)

    def test_active_limit_and_balance_gate_are_distinct(self):
        scheduler = Scheduler()
        scheduler.max_num_active_reqs = 0
        scheduler.max_num_running_reqs = 4
        scheduler._balance_enabled = True
        scheduler.balance_queue = [1, 4, 0, 0]
        self.attach(scheduler)
        with patch.object(diagnostics.logger, "info") as output:
            scheduler.schedule()
        payload = self.read_log(output)
        self.assertEqual(
            payload["step_limits"],
            {
                "active_limit_reached": 1,
                "dp_balance_running_limit_reached": 1,
            },
        )
        self.assertEqual(payload["latest"]["balance_running_counts"], [1, 4, 0, 0])
        self.assertFalse(payload["observations"])

    def test_no_device_item_or_waiting_scan_added(self):
        class NoIteration:
            def __len__(self):
                return 5

            def __iter__(self):
                raise AssertionError("diagnostics must not scan waiting requests")

        class DeviceValue:
            device = SimpleNamespace(type="npu")

            def item(self):
                raise AssertionError("diagnostics must not synchronize a device tensor")

        scheduler = Scheduler()
        scheduler.waiting = NoIteration()
        scheduler._balance_enabled = True
        scheduler.balance_queue = [DeviceValue()]
        self.attach(scheduler)
        with patch.object(diagnostics.logger, "info") as output:
            scheduler.schedule(throttle_prefills=True)
        self.assertEqual(self.read_log(output)["latest"]["waiting"], 5)

    def test_rate_limit_retains_window_counters_and_bounds_request_samples(self):
        scheduler = Scheduler(count=20, connector=Connector((None, False)))
        tracker, clock = self.attach(scheduler)
        with patch.object(diagnostics.logger, "info") as output:
            scheduler.schedule()
            first = self.read_log(output)
            self.assertEqual(len(first["samples"]), 4)
            self.assertEqual(first["omitted_samples"], 16)
            clock.now = 0.5
            scheduler.schedule()
            self.assertEqual(output.call_count, 1)
            clock.now = 1.0
            scheduler.schedule()
            second = self.read_log(output)
            self.assertEqual(output.call_count, 2)
        self.assertEqual(second["steps"], 2)
        self.assertEqual(second["observations"]["cache_lookup_pending"], 40)
        self.assertEqual(len(second["samples"]), 4)
        self.assertEqual(second["omitted_samples"], 36)
        self.assertFalse(tracker.observations)

    def test_original_results_arguments_and_decisions_are_preserved(self):
        for count in (0, 1, 12):
            for result in ((0, False), (None, False), (3, True)):
                for throttle in (False, True):
                    with self.subTest(count=count, result=result, throttle=throttle):
                        plain = Scheduler(count=count, connector=Connector(result))
                        observed = Scheduler(count=count, connector=Connector(result))
                        self.attach(observed)
                        expected = plain.schedule(throttle_prefills=throttle)
                        with patch.object(diagnostics.logger, "info"):
                            actual = observed.schedule(throttle_prefills=throttle)
                        self.assertEqual(actual.total_num_scheduled_tokens, expected.total_num_scheduled_tokens)
                        self.assertEqual(
                            [r.request_id for r in actual.scheduled_new_reqs],
                            [r.request_id for r in expected.scheduled_new_reqs],
                        )
                        self.assertEqual(
                            [r.request_id for r in observed.waiting], [r.request_id for r in plain.waiting]
                        )
                        self.assertEqual(observed.kv_cache_manager.calls, plain.kv_cache_manager.calls)

    def test_original_exception_is_not_swallowed_and_active_flag_is_cleared(self):
        scheduler = Scheduler()
        scheduler.failure = ValueError("original failure")
        tracker, _ = self.attach(scheduler)
        with self.assertRaises(ValueError) as caught:
            scheduler.schedule()
        self.assertIs(caught.exception, scheduler.failure)
        self.assertFalse(tracker.active)

    def test_logging_failure_does_not_change_schedule_return(self):
        scheduler = Scheduler()
        _, clock = self.attach(scheduler)
        with (
            patch.object(diagnostics.logger, "info", side_effect=OSError("disk full")),
            patch.object(diagnostics.logger, "warning") as warning,
        ):
            result = scheduler.schedule()
            self.assertEqual(result.total_num_scheduled_tokens, 1)
            clock.now = 0.1
            scheduler.schedule()
            warning.assert_called_once()

    def test_owned_connector_replacement_after_constructor(self):
        scheduler = Scheduler()
        self.attach(scheduler)
        scheduler.connector = Connector((None, False))
        with patch.object(diagnostics.logger, "info") as output:
            scheduler.schedule()
        self.assertEqual(self.read_log(output)["observations"], {"cache_lookup_pending": 1})

    def test_helper_calls_outside_schedule_are_not_observed(self):
        scheduler = Scheduler(connector=Connector((None, False)))
        tracker, _ = self.attach(scheduler)
        with patch.object(diagnostics.logger, "info"):
            scheduler.schedule()
        scheduler.connector.get_num_new_matched_tokens(scheduler.waiting[0], 0)
        self.assertFalse(tracker.observations)

    def test_installation_preserves_signature_and_instruments_subclasses_once(self):
        class Base(Scheduler):
            pass

        class Derived(Base):
            def __init__(self):
                super().__init__(connector=Connector((None, False)))
                self.ready = True

            def schedule(self, throttle_prefills=False):
                self.assert_ready = self.ready
                return super().schedule(throttle_prefills=throttle_prefills)

        original_signature = inspect.signature(Derived.schedule)
        diagnostics.install_scheduler_admission_diagnostics(Base, 1)
        init = Base.__init__
        diagnostics.install_scheduler_admission_diagnostics(Base, 1)
        self.assertIs(Base.__init__, init)
        scheduler = Derived()
        tracker = getattr(scheduler, diagnostics._TRACKER_ATTR)
        self.assertIs(diagnostics.instrument_scheduler(scheduler, 1), tracker)
        self.assertEqual(str(inspect.signature(scheduler.schedule)), "(throttle_prefills=False)")
        self.assertEqual(inspect.signature(Derived.schedule), original_signature)
        with patch.object(diagnostics.logger, "info") as output:
            scheduler.schedule()
        self.assertTrue(scheduler.assert_ready)
        self.assertEqual(self.read_log(output)["observations"]["cache_lookup_pending"], 1)

    def test_environment_defaults_and_validation(self):
        flag = "VLLM_ASCEND_SCHEDULER_ADMISSION_DEBUG"
        interval = "VLLM_ASCEND_SCHEDULER_ADMISSION_LOG_INTERVAL_MS"
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(envs.env_variables[flag]())
            self.assertEqual(envs.env_variables[interval](), 1000)
        with patch.dict(os.environ, {flag: "1", interval: "250"}):
            self.assertTrue(envs.env_variables[flag]())
            self.assertEqual(envs.env_variables[interval](), 250)
        for value in ("0", "-1", "nan", "oops"):
            with patch.dict(os.environ, {interval: value}), self.assertRaises(ValueError):
                envs.env_variables[interval]()
        with patch.dict(os.environ, {flag: "yes"}), self.assertRaises(ValueError):
            envs.env_variables[flag]()

    def test_platform_patch_finds_original_base_under_replaced_scheduler_name(self):
        class Base(Scheduler):
            pass

        Base.__module__ = "vllm.v1.core.sched.scheduler"
        Base.__name__ = "Scheduler"

        class Balance(Base):
            pass

        sched = types.ModuleType("vllm.v1.core.sched")
        sched.scheduler = SimpleNamespace(Scheduler=Balance)
        ascend = types.ModuleType("vllm_ascend")
        ascend.envs = SimpleNamespace(VLLM_ASCEND_SCHEDULER_ADMISSION_LOG_INTERVAL_MS=1000)
        replacements = {
            "vllm.v1.core.sched": sched,
            "vllm_ascend": ascend,
            "vllm_ascend.core.scheduler_admission_diagnostics": diagnostics,
        }
        with patch.dict(sys.modules, replacements), patch.object(diagnostics.logger, "info"):
            load_source(
                "admission_platform_cpu",
                "vllm_ascend/patch/platform/patch_scheduler_admission_diagnostics.py",
            )
            scheduler = Balance()
            self.assertIsNotNone(getattr(scheduler, diagnostics._TRACKER_ATTR, None))
            result = scheduler.schedule()
        self.assertEqual(result.total_num_scheduled_tokens, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
