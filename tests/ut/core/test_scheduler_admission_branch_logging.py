# SPDX-License-Identifier: Apache-2.0
"""Execute actual scheduler guards and verify diagnostic-only source changes."""

import ast
import copy
import difflib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[3]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = load_source("branch_test_harness", Path(__file__).with_name("test_scheduler_admission_diagnostics.py"))
diagnostics = harness.diagnostics
installer = load_source("branch_patch_installer", _ROOT / "tools/diagnostics/apply_scheduler_admission_core_patch.py")


class StripDiagnosticNodes(ast.NodeTransformer):
    def visit_Assign(self, node):  # noqa: N802
        if any(isinstance(t, ast.Name) and t.id == "_admission_debug" for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):  # noqa: N802
        if any(isinstance(n, ast.Name) and n.id == "_admission_debug" for n in ast.walk(node.test)):
            return None
        return self.generic_visit(node)


def waiting_loop(path):
    tree = ast.parse(path.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "schedule")
    return next(
        n
        for n in ast.walk(function)
        if isinstance(n, ast.While) and ("waiting" in ast.unparse(n.test) or ast.unparse(n.test) == "token_budget > 0")
    )


def execute_guard(node, namespace, filename):
    # Use a one-iteration for-loop so the production guard's break/continue
    # execute unchanged, including continue paths, without an infinite loop.
    loop = ast.For(
        target=ast.Name(id="_probe", ctx=ast.Store()),
        iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
        body=[copy.deepcopy(node)],
        orelse=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
    exec(compile(module, str(filename), "exec"), namespace)


class BranchLoggingTests(unittest.TestCase):
    def setUp(self):
        self.records = json.loads((_ROOT / "tests/ut/core/data/scheduler_admission_original_ast.json").read_text())
        self.paths = [_ROOT / p for p in self.records if p.startswith("vllm_ascend/")]

    def context(self):
        scheduler = harness.Scheduler()
        tracker = diagnostics.SchedulerAdmissionDiagnostics(scheduler, 1)
        tracker.active = True
        request = scheduler.waiting[0]
        request.has_encoder_inputs = False
        return tracker, {
            "self": scheduler,
            "_admission_debug": tracker,
            "request": request,
            "num_running": 0,
            "token_budget": 8,
            "new_blocks": None,
            "request_queue": scheduler.waiting,
            "ready_to_schedule": False,
            "_skip_request": lambda q: None,
            "skip_request": lambda q: None,
            "RequestStatus": harness.Status,
            "request_id": request.request_id,
            "logger": SimpleNamespace(debug=lambda *args: None),
        }

    def test_removing_probes_restores_all_four_original_schedule_asts(self):
        for path in self.paths:
            with self.subTest(path=path.name):
                record = self.records[str(path.relative_to(_ROOT))]
                tree = ast.parse(path.read_text())
                cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == record["class"])
                method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")
                stripped = StripDiagnosticNodes().visit(method)
                digest = installer.ast_fingerprint(stripped)
                self.assertEqual(digest, record["ast_sha256"])

    def test_real_active_limit_guards_report_queue_scope_without_peeking(self):
        for path in self.paths:
            with self.subTest(path=path.name):
                node = next(
                    n
                    for n in ast.walk(waiting_loop(path))
                    if isinstance(n, ast.If) and "max_num_active_reqs" in ast.unparse(n.test)
                )
                tracker, context = self.context()
                context["num_running"] = context["self"].max_num_active_reqs
                context["self"].running = [object()] * context["self"].max_num_active_reqs
                execute_guard(node, context, path)
                self.assertEqual(tracker.branch_reasons, {"active_request_limit": 1})
                self.assertEqual(tracker.branch_samples[0]["scope"], "waiting_queue")
                self.assertIsNone(tracker.branch_samples[0]["request_id"])
                self.assertEqual(tracker.branch_samples[0]["action"], "break")
                self.assertTrue(tracker.branch_samples[0]["location"].startswith(str(path)))

    def test_real_kv_failure_and_blocked_state_guards_have_request_ids(self):
        for path in self.paths:
            for kind, reason in (
                ("new_blocks is None", "kv_allocation_failed"),
                ("_handle_blocked_waiting_request", "request_state_not_ready"),
            ):
                with self.subTest(path=path.name, reason=reason):
                    node = next(
                        n for n in ast.walk(waiting_loop(path)) if isinstance(n, ast.If) and kind in ast.unparse(n.test)
                    )
                    tracker, context = self.context()
                    context["request"].status = harness.Status.WAITING_FOR_REMOTE_KVS
                    execute_guard(node, context, path)
                    self.assertEqual(tracker.branch_reasons, {reason: 1})
                    self.assertEqual(tracker.branch_samples[0]["request_id"], "request-0")
                    self.assertEqual(tracker.branch_samples[0]["scope"], "request")

    def test_non_taken_guard_does_not_report_a_denial(self):
        for path in self.paths:
            node = next(
                n
                for n in ast.walk(waiting_loop(path))
                if isinstance(n, ast.If) and "max_num_active_reqs" in ast.unparse(n.test)
            )
            tracker, context = self.context()
            execute_guard(node, context, path)
            self.assertFalse(tracker.branch_reasons)

    def test_actual_while_else_only_reports_natural_token_exhaustion(self):
        for path in self.paths:
            original = waiting_loop(path)
            for should_break, pending, expected in ((False, True, 1), (True, True, 0), (False, False, 0)):
                with self.subTest(path=path.name, should_break=should_break, pending=pending):
                    tracker, context = self.context()
                    if not pending:
                        context["self"].waiting.clear()
                    context["token_budget"] = 1 if should_break else 0
                    # A break after budget becomes zero must NOT execute else.
                    body = ast.parse("token_budget = 0\nbreak").body
                    loop = ast.While(
                        test=ast.parse("token_budget > 0", mode="eval").body,
                        body=body,
                        orelse=copy.deepcopy(original.orelse),
                    )
                    module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
                    exec(compile(module, str(path), "exec"), context)
                    self.assertEqual(tracker.branch_reasons.get("token_budget_exhausted", 0), expected)

    def test_branch_samples_are_bounded_and_disabled_outside_schedule(self):
        tracker, context = self.context()
        for _ in range(20):
            tracker.record_branch("test", context["request"], "continue", 8)
        self.assertEqual(tracker.branch_reasons["test"], 20)
        self.assertEqual(len(tracker.branch_samples), 4)
        self.assertEqual(tracker.omitted_branch_samples, 16)
        tracker.active = False
        tracker.record_branch("outside", context["request"], "continue", 8)
        self.assertNotIn("outside", tracker.branch_reasons)

    def test_every_waiting_break_continue_has_a_direct_probe(self):
        for path in self.paths:
            loop = waiting_loop(path)
            parents = {child: node for node in ast.walk(loop) for child in ast.iter_child_nodes(node)}
            for jump in ast.walk(loop):
                if not isinstance(jump, (ast.Break, ast.Continue)):
                    continue
                parent = parents[jump]
                nearest = parent
                while not isinstance(nearest, (ast.For, ast.While)):
                    nearest = parents[nearest]
                if nearest is not loop:
                    continue
                siblings = next(
                    value for _, value in ast.iter_fields(parent) if isinstance(value, list) and jump in value
                )
                probe = siblings[siblings.index(jump) - 1]
                self.assertIsInstance(probe, ast.If, (path.name, jump.lineno))
                self.assertIn("_admission_debug", ast.unparse(probe.test))
                self.assertIn("record_branch", ast.unparse(probe))

    def test_branch_counters_sources_and_samples_reset_at_log_boundary(self):
        scheduler = harness.Scheduler()
        clock = harness.Clock()
        tracker = diagnostics.SchedulerAdmissionDiagnostics(scheduler, 1, clock)
        tracker.active = True
        tracker.enter_schedule_source("test.schedule")
        tracker.record_branch("test_stop", scheduler.waiting[0], "break", 8)
        output = SimpleNamespace(total_num_scheduled_tokens=0, scheduled_new_reqs=[])
        with patch.object(diagnostics.logger, "info") as log:
            tracker._finish(2, output)
            payload = json.loads(log.call_args.args[1])
        self.assertEqual(payload["branch_reasons"], {"test_stop": 1})
        self.assertEqual(payload["branch_sources"], ["test.schedule"])
        self.assertEqual(payload["branch_samples"][0]["action"], "break")
        self.assertFalse(tracker.branch_reasons)
        self.assertFalse(tracker.branch_sources)
        self.assertFalse(tracker.branch_samples)

    def test_core_patch_metadata_matches_original_fingerprint(self):
        metadata = json.loads((_ROOT / "tools/diagnostics/vllm-v0.32-scheduler-admission.json").read_text())
        self.assertEqual(
            metadata["original_schedule_ast_sha256"], self.records["vllm/v1/core/sched/scheduler.py"]["ast_sha256"]
        )


class InstallerTests(unittest.TestCase):
    def fixture(self, root, alter_outside=False):
        before = "class Scheduler:\n    label = 1\n    def schedule(self):\n        return 1\n"
        after = before.replace("        return 1", "        value = 2\n        return value")
        if alter_outside:
            after = after.replace("label = 1", "label = 2")
        target = root / installer._TARGET
        target.parent.mkdir(parents=True)
        target.write_text(before)
        relative = str(installer._TARGET)
        patch_text = f"diff --git a/{relative} b/{relative}\n" + "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="a/" + relative,
                tofile="b/" + relative,
            )
        )
        (root / "core.patch").write_text(patch_text)
        metadata = root / "meta.json"
        metadata.write_text(
            json.dumps(
                {
                    "target": relative,
                    "patch": "core.patch",
                    "original_schedule_ast_sha256": installer.schedule_parts(before)[0],
                    "patched_schedule_ast_sha256": installer.schedule_parts(after)[0],
                }
            )
        )
        return target, metadata, before, after

    def test_check_apply_idempotence_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, meta, before, after = self.fixture(root)
            installer.operate(root, "check", meta)
            self.assertEqual(target.read_text(), before)
            self.assertFalse(target.with_name("scheduler.py.admission-debug.backup").exists())
            installer.operate(root, "apply", meta)
            self.assertEqual(target.read_text(), after)
            installer.operate(root, "apply", meta)
            installer.operate(root, "restore", meta)
            self.assertEqual(target.read_text(), before)

    def test_algorithm_mismatch_is_rejected_without_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, meta, before, _ = self.fixture(root)
            changed = before.replace("return 1", "return 999")
            target.write_text(changed)
            with self.assertRaisesRegex(ValueError, "differs"):
                installer.operate(root, "apply", meta)
            self.assertEqual(target.read_text(), changed)

    def test_restore_rejects_later_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, meta, _, _ = self.fixture(root)
            installer.operate(root, "apply", meta)
            changed = target.read_text() + "# later user edit\n"
            target.write_text(changed)
            with self.assertRaisesRegex(ValueError, "changed since"):
                installer.operate(root, "restore", meta)
            self.assertEqual(target.read_text(), changed)

    def test_patch_touching_code_outside_schedule_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, meta, before, _ = self.fixture(root, alter_outside=True)
            with self.assertRaisesRegex(ValueError, "confined"):
                installer.operate(root, "apply", meta)
            self.assertEqual(target.read_text(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
