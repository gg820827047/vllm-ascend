"""Safe tests: mocked host settings plus one disposable child process only."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bind_threads_v3 as bind


def arguments(**kwargs):
    values = dict(
        role="P",
        dry_run=False,
        workers_only=False,
        verify_only=False,
        root_pid=[],
        pid_map=None,
        port_base=7100,
        irq=False,
        irq_map=None,
        verbose=False,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def nodes():
    return {n: list(range(n * 24, (n + 1) * 24)) for n in range(8)}


def processes(role="P"):
    plan = bind.layout(role, nodes())
    result = {}
    for dp, tp in plan["workers"]:
        pid = 1000 + dp * 100 + tp
        title = f"VLLM::Worker_DP{dp}_TP{tp}_EP0"
        result[pid] = bind.Process(pid, 3000 + dp, "1", [title], title)
    for dp in plan["engines"]:
        pid = 2000 + dp
        title = f"VLLM::EngineCore_DP{dp}"
        result[pid] = bind.Process(pid, 3000 + dp, "1", [title], title)
        pid = 3000 + dp
        argv = ["/usr/bin/python3", "/bin/vllm", "serve", "model", "--port", str(7100 + dp)]
        result[pid] = bind.Process(pid, 1, "1", argv, " ".join(argv))
        pid = 4000 + dp
        title = "VLLM::ApiServer_0"
        result[pid] = bind.Process(pid, 3000 + dp, "1", [title], title)
    return result


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)

    def test_cpu_ranges_and_masks(self):
        self.assertEqual(bind.cpu_list("2-10,32,32"), list(range(2, 11)) + [32])
        self.assertEqual(bind.format_cpus([32, 3, 2, 4]), "2-4,32")
        self.assertEqual(bind.hex_mask([0, 31, 32]), "00000001,80000001")
        self.assertEqual(int(bind.hex_mask([191]).replace(",", ""), 16), 1 << 191)
        for value in ("3-2", "1--2", "foo"):
            with self.assertRaises(bind.BindingError):
                bind.cpu_list(value)

    def test_layout_preserves_roles_and_real_cpu_lists(self):
        for role, worker_count, pool_size in (("P", 16, 9), ("D", 16, 8)):
            plan = bind.layout(role, nodes())
            self.assertEqual(len(plan["workers"]), worker_count)
            for cpus in plan["workers"].values():
                self.assertEqual(len(cpus), pool_size)
                self.assertEqual(sum(map(len, bind.worker_pools(cpus).values())), pool_size)
                self.assertEqual(len({c // 24 for c in cpus}), 1)
        odd_nodes = {n * 2: [c * 2 for c in cpus] for n, cpus in nodes().items()}
        plan = bind.layout("P", odd_nodes)
        self.assertEqual(plan["workers"][(0, 0)], list(range(4, 22, 2)))
        self.assertEqual(plan["workers"][(0, 2)], list(range(52, 70, 2)))

    def test_main_thread_uses_tid_not_truncated_comm(self):
        cpus = list(range(2, 11))
        target = bind.Target(100, "worker", "worker", cpus, bind.worker_pools(cpus))
        self.assertEqual(bind.thread_role(target, 100, "VLLM::Worker_DP"), ("main", list(range(2, 7))))
        self.assertEqual(bind.thread_role(target, 101, "acl_thread_0"), ("acl", [9]))
        self.assertEqual(bind.thread_role(target, 102, "release_thread"), ("release", [10]))
        self.assertEqual(bind.thread_role(target, 103, "RT_RECYCLE_0"), ("aux", [7, 8]))
        engine = bind.Target(100, "engine", "engine", [20, 21, 22, 23])
        self.assertEqual(bind.thread_role(engine, 100, "VLLM::EngineCor"), ("main", [20]))

    def test_topology_checks_smt(self):
        with tempfile.TemporaryDirectory() as directory:
            sysroot = Path(directory)
            cpu = sysroot / "devices/system/cpu"
            cpu.mkdir(parents=True)
            (cpu / "online").write_text("0-191")
            for n, cpus in nodes().items():
                path = sysroot / f"devices/system/node/node{n}"
                path.mkdir(parents=True)
                (path / "cpulist").write_text(bind.format_cpus(cpus))
            for c in range(192):
                path = cpu / f"cpu{c}/topology"
                path.mkdir(parents=True)
                (path / "thread_siblings_list").write_text(str(c))
            with patch.object(bind, "SYS", sysroot):
                self.assertEqual(bind.topology(), nodes())
                (cpu / "cpu0/topology/thread_siblings_list").write_text("0,1")
                with self.assertRaisesRegex(bind.BindingError, "SMT"):
                    bind.topology()

    def test_discovery_includes_api_children_and_avoids_tp10(self):
        procs = processes()
        title = "VLLM::Worker_DP0_TP10_EP0"
        procs[9999] = bind.Process(9999, 1, "1", [title], title)
        targets = bind.validate_targets(bind.discover(arguments(), bind.layout("P", nodes()), procs, {}))
        self.assertEqual(len(targets), 28)  # 16 workers + 4 engines + 4 launchers + 4 API children.
        self.assertIn(4000, [t.pid for t in targets])
        self.assertNotIn(9999, [t.pid for t in targets])
        self.assertEqual(next(t.pid for t in targets if t.tag == "DP0/WorkerTP1"), 1001)

    def test_duplicate_worker_fails_instead_of_picking_first(self):
        procs = processes()
        procs[9999] = bind.Process(9999, 1, "1", ["VLLM::Worker_DP0_TP1"], "VLLM::Worker_DP0_TP1")
        with self.assertRaisesRegex(bind.BindingError, "唯一PID"):
            bind.discover(arguments(), bind.layout("P", nodes()), procs, {})

    def test_scope_and_pid_map_validation(self):
        procs = processes()
        title = "VLLM::Worker_DP0_TP1"
        procs[9999] = bind.Process(9999, 1, "1", [title], title)
        args = arguments(root_pid=[3000, 3001, 3002, 3003])
        self.assertEqual(len(bind.discover(args, bind.layout("P", nodes()), procs, {})), 28)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pid.json"
            path.write_text('{"workers":{"0/0":true}}')
            with self.assertRaisesRegex(bind.BindingError, "正整数"):
                bind.discover(arguments(pid_map=str(path)), bind.layout("P", nodes()), procs, {})

    def test_conflicting_roles_fail(self):
        with self.assertRaisesRegex(bind.BindingError, "多个不同角色"):
            bind.validate_targets([bind.Target(10, "a", "api", [1]), bind.Target(10, "b", "worker", [2])])

    def fake_item(self):
        return {
            "pid": 100,
            "tid": 100,
            "pid_start": "1",
            "tid_start": "1",
            "tag": "worker",
            "thread_role": "main",
            "comm": "VLLM::Worker_DP",
            "before": [0, 1, 2],
            "after": [1],
        }

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            with (
                patch.object(bind, "snapshots", return_value=[self.fake_item()]),
                patch.object(bind.os, "sched_setaffinity") as affinity,
                patch.object(bind, "write_setting") as setting,
                patch.object(bind, "service_change") as service,
            ):
                bind.apply(arguments(dry_run=True), [], state, [])
                affinity.assert_not_called()
                setting.assert_not_called()
                service.assert_not_called()
            self.assertFalse(state.exists())

    def test_apply_verify_restore_and_original_numa_value(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            balancing = Path(directory) / "numa_balancing"
            balancing.write_text("3\n")
            masks = {100: {0, 1, 2}}
            item = self.fake_item()

            def snapshot(_):
                return [dict(item, before=sorted(masks[100]))]

            with (
                patch.object(bind, "NUMA_BALANCING", balancing),
                patch.object(bind, "BOOT_ID", balancing.parent / "boot_id"),
                patch.object(bind, "snapshots", side_effect=snapshot),
                patch.object(bind, "same_thread", return_value=True),
                patch.object(bind.os, "sched_getaffinity", side_effect=lambda tid: masks[tid]),
                patch.object(
                    bind.os, "sched_setaffinity", side_effect=lambda tid, cpus: masks.update({tid: set(cpus)})
                ),
            ):
                (balancing.parent / "boot_id").write_text("boot")
                bind.apply(arguments(), [], state_path, [bind.setting(balancing, "0", "text")])
                self.assertEqual(masks[100], {1})
                self.assertEqual(bind.text(balancing), "0")
                self.assertEqual(state_path.stat().st_mode & 0o777, 0o600)
                state = json.loads(state_path.read_text())
                self.assertEqual(state["status"], "applied")
                bind.restore_state(state)
                self.assertEqual(masks[100], {0, 1, 2})
                self.assertEqual(bind.text(balancing), "3")

    def test_cpuset_truncation_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            boot = Path(directory) / "boot_id"
            boot.write_text("boot")
            state_path = Path(directory) / "state.json"
            item = self.fake_item()
            item["after"] = [1, 2]
            masks = {100: {0, 1, 2}}

            def set_mask(tid, cpus):
                masks[tid] = {1} if cpus == {1, 2} else set(cpus)

            with (
                patch.object(bind, "BOOT_ID", boot),
                patch.object(bind, "snapshots", return_value=[item]),
                patch.object(bind, "same_thread", return_value=True),
                patch.object(bind.os, "sched_getaffinity", side_effect=lambda tid: masks[tid]),
                patch.object(bind.os, "sched_setaffinity", side_effect=set_mask),
            ):
                with self.assertRaisesRegex(bind.BindingError, "cpuset"):
                    bind.apply(arguments(), [], state_path, [])
                self.assertEqual(masks[100], {0, 1, 2})
                self.assertEqual(json.loads(state_path.read_text())["status"], "rolled_back")

    def test_restore_preserves_external_changes_and_checks_boot(self):
        with tempfile.TemporaryDirectory() as directory:
            boot = Path(directory) / "boot_id"
            boot.write_text("boot")
            state = dict(schema=1, boot_id="boot", threads=[self.fake_item()], settings=[])
            with (
                patch.object(bind, "BOOT_ID", boot),
                patch.object(bind, "same_thread", return_value=True),
                patch.object(bind.os, "sched_getaffinity", return_value={2}),
                patch.object(bind.os, "sched_setaffinity") as setter,
            ):
                with self.assertRaisesRegex(bind.BindingError, "其他操作"):
                    bind.restore_state(state)
                setter.assert_not_called()
                state["boot_id"] = "old-boot"
                with self.assertRaisesRegex(bind.BindingError, "本次主机启动"):
                    bind.restore_state(state)

    def test_restore_skips_reused_tid(self):
        with tempfile.TemporaryDirectory() as directory:
            boot = Path(directory) / "boot_id"
            boot.write_text("boot")
            state = dict(schema=1, boot_id="boot", threads=[self.fake_item()], settings=[])
            with (
                patch.object(bind, "BOOT_ID", boot),
                patch.object(bind, "same_thread", return_value=False),
                patch.object(bind.os, "sched_setaffinity") as setter,
            ):
                bind.restore_state(state)
                setter.assert_not_called()

    def test_cli_rejects_invalid_combinations(self):
        for argv in (
            [],
            ["--role", "P", "--irq"],
            ["--role", "P", "--verify-only", "--disable-numa-balancing"],
            ["--restore", "state.json", "--workers-only"],
            ["--role", "P", "--root-pid", "0"],
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bind.parse_args(argv)

    def test_real_child_thread_affinity_and_restore(self):
        # Only our disposable child is touched; no host sysctl, IRQ, or service.
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < 2:
            self.skipTest("need two allowed CPUs for the disposable child test")
        code = """import json,os,threading,time
t=threading.Thread(target=lambda: time.sleep(60));t.start()
print(json.dumps([os.getpid(),t.native_id]),flush=True)
time.sleep(60)
"""
        with subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True) as child:
            try:
                pid, tid = json.loads(child.stdout.readline())
                target = bind.Target(pid, "disposable-test", "worker", cpus, {})
                with patch.object(
                    bind,
                    "thread_role",
                    side_effect=lambda t, i, c: ("main" if i == pid else "aux", [cpus[0] if i == pid else cpus[1]]),
                ):
                    items = bind.snapshots([target])
                    state = dict(schema=1, boot_id=bind.text(bind.BOOT_ID), threads=items, settings=[])
                    try:
                        self.assertEqual(bind.apply_affinities(items), 2)
                        self.assertEqual(bind.verify([target]), 2)
                        self.assertEqual(os.sched_getaffinity(pid), {cpus[0]})
                        self.assertEqual(os.sched_getaffinity(tid), {cpus[1]})
                    finally:
                        bind.restore_state(state)
                    self.assertEqual(os.sched_getaffinity(pid), set(cpus))
                    self.assertEqual(os.sched_getaffinity(tid), set(cpus))
            finally:
                child.terminate()
                child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
