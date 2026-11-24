#!/usr/bin/env bash
# A3 192 CPUs / 8 NUMA; Python 3 standard library only.
# This file is self-contained; bind_threads_v3.py is its editable source.
set -euo pipefail
if ! command -v python3 >/dev/null 2>&1; then
    echo 'ERROR: python3 is required' >&2
    exit 1
fi
exec python3 - "$@" <<'PYTHON_BIND_CORE_V3'
#!/usr/bin/env python3
"""Audited, reversible thread affinity for the user's 192-CPU A3 P/D layout.

Uses the same per-TID sched_setaffinity syscall as taskset. No vLLM import,
third-party Python package, persistent sysctl file, or vLLM restart is needed.
"""

import argparse
import datetime
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

PROC = Path("/proc")
SYS = Path("/sys")
NUMA_BALANCING = PROC / "sys/kernel/numa_balancing"
BOOT_ID = PROC / "sys/kernel/random/boot_id"
EXPECTED_NODES = 8
CPUS_PER_NODE = 24


class BindingError(RuntimeError):
    pass


def log(message):
    print(message, flush=True)


def cpu_list(text):
    result = set()
    for part in text.strip().split(","):
        if not part:
            continue
        if not re.fullmatch(r"\d+(?:-\d+)?", part):
            raise BindingError(f"非法CPU列表: {text!r}")
        ends = list(map(int, part.split("-")))
        low, high = ends[0], ends[-1]
        if high < low:
            raise BindingError(f"倒序CPU范围: {part}")
        result.update(range(low, high + 1))
    return sorted(result)


def format_cpus(cpus):
    values = sorted(set(cpus))
    groups = []
    for value in values:
        if groups and value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return ",".join(str(g[0]) if len(g) == 1 else f"{g[0]}-{g[-1]}" for g in groups)


def hex_mask(cpus):
    mask = sum(1 << cpu for cpu in set(cpus))
    parts = []
    while mask:
        parts.append(f"{mask & 0xFFFFFFFF:08x}")
        mask >>= 32
    return ",".join(reversed(parts)) or "00000000"


def text(path):
    return path.read_text().strip()


def start_time(pid, tid=None):
    path = PROC / str(pid)
    if tid is not None:
        path = path / "task" / str(tid)
    # comm may contain spaces and parentheses; fields after the last ')' start at 3.
    tail = text(path / "stat").rsplit(")", 1)[1].split()
    return tail[19]


@dataclass
class Process:
    pid: int
    ppid: int
    start: str
    argv: list
    title: str


@dataclass
class Target:
    pid: int
    tag: str
    kind: str
    cpus: list
    pools: dict = field(default_factory=dict)


def read_processes():
    result = {}
    for path in PROC.iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = text(path / "stat").rsplit(")", 1)[1].split()
            argv = [s.decode(errors="replace") for s in (path / "cmdline").read_bytes().split(b"\0") if s]
            if argv:
                result[int(path.name)] = Process(int(path.name), int(stat[1]), stat[19], argv, " ".join(argv))
        except (OSError, IndexError):
            continue
    return result


def below(pid, root, processes):
    seen = set()
    while pid in processes and pid not in seen:
        if pid == root:
            return True
        seen.add(pid)
        pid = processes[pid].ppid
    return False


def topology():
    online = set(cpu_list(text(SYS / "devices/system/cpu/online")))
    nodes = {}
    for path in (SYS / "devices/system/node").glob("node[0-9]*"):
        cpus = sorted(set(cpu_list(text(path / "cpulist"))) & online)
        if cpus:
            nodes[int(path.name[4:])] = cpus
    nodes = dict(sorted(nodes.items()))
    if len(nodes) != EXPECTED_NODES or any(len(c) != CPUS_PER_NODE for c in nodes.values()):
        raise BindingError(f"需要8个各24在线CPU的NUMA节点，实际: { {n: len(c) for n, c in nodes.items()} }")
    if len(set().union(*map(set, nodes.values()))) != EXPECTED_NODES * CPUS_PER_NODE:
        raise BindingError("NUMA CPU集合重叠或总数不是192")
    for cpu in set().union(*map(set, nodes.values())):
        siblings = SYS / f"devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        if not siblings.exists():
            raise BindingError(f"无法核对SMT: {siblings}")
        if len(set(cpu_list(text(siblings))) & online) != 1:
            raise BindingError("本分配方案要求SMT关闭；检测到同一物理核有多个在线线程")
    return nodes


def layout(role, nodes):
    ordered = list(nodes.values())
    engines = 4 if role == "P" else 8
    workers = 4 if role == "P" else 2
    plan = {"workers": {}, "engines": {}, "apis": {}, "irq": {}}
    for dp in range(engines):
        first = ordered[2 * dp if role == "P" else dp]
        if role == "P":
            second = ordered[2 * dp + 1]
            spans = [first[2:11], first[11:20], second[2:11], second[11:20]]
            plan["engines"][dp], plan["apis"][dp] = first[20:24], second[22:24]
            if dp == 0:
                plan["coordinators"] = second[20:22]
        else:
            spans = [first[2:10], first[10:18]]
            plan["engines"][dp], plan["apis"][dp] = first[18:22], first[23:24]
            if dp == 0:
                plan["coordinators"] = first[22:23]
        for tp in range(workers):
            pool = spans[tp]
            plan["workers"][(dp, tp)] = pool
            plan["irq"][(dp, tp)] = (ordered[2 * dp + tp // 2] if role == "P" else first)[:2]
    # Role pools may share within one process, but distinct processes must not overlap.
    used = set()
    pools = list(plan["workers"].values()) + list(plan["engines"].values()) + list(plan["apis"].values())
    for pool in pools + [plan["coordinators"]]:
        if used & set(pool):
            raise BindingError("内部错误: 角色CPU池重叠")
        used.update(pool)
    return plan


def worker_pools(cpus):
    if len(cpus) not in (8, 9):
        raise BindingError(f"worker需要8或9个CPU，实际{cpus}")
    return {"main": cpus[:5], "aux": cpus[5:-2], "acl": cpus[-2:-1], "release": cpus[-1:]}


def thread_role(target, tid, comm):
    if target.kind == "worker":
        if tid == target.pid:
            return "main", target.pools["main"]
        if comm.startswith("acl_thread"):
            return "acl", target.pools["acl"]
        if comm.startswith("release_thread"):
            return "release", target.pools["release"]
        return "aux", target.pools["aux"]
    if target.kind == "engine":
        return ("main", target.cpus[:1]) if tid == target.pid else ("aux", target.cpus[1:])
    return "all", target.cpus


def listener_pids():
    try:
        result = subprocess.run(["ss", "-H", "-ltnp"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return {}
    ports = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        port = fields[3].rsplit(":", 1)[-1]
        if port.isdigit():
            ports.setdefault(int(port), set()).update(map(int, re.findall(r"pid=(\d+)", line)))
    return ports


def launcher_port(proc):
    is_vllm = any(Path(arg).name == "vllm" for arg in proc.argv) and "serve" in proc.argv
    is_vllm |= any("vllm.entrypoints.openai.api_server" in arg for arg in proc.argv)
    if not is_vllm:
        return None
    for index, arg in enumerate(proc.argv):
        if arg == "--port" and index + 1 < len(proc.argv) and proc.argv[index + 1].isdigit():
            return int(proc.argv[index + 1])
        if arg.startswith("--port=") and arg[7:].isdigit():
            return int(arg[7:])
    return None


def is_api(proc):
    return bool(re.search(r"(?:^|\s)(?:VLLM::)?ApiServer(?:[_\s]|$)", proc.title, re.I))


def discover(args, plan, processes, ports):
    scoped = {
        pid: proc
        for pid, proc in processes.items()
        if not args.root_pid or any(below(pid, root, processes) for root in args.root_pid)
    }
    for root in args.root_pid:
        if root not in processes:
            raise BindingError(f"--root-pid {root} 不存在或不可读取")
    overrides = json.loads(Path(args.pid_map).read_text()) if args.pid_map else {}
    if not isinstance(overrides, dict) or set(overrides) - {"workers", "engines", "apis", "coordinators"}:
        raise BindingError("pid-map须为JSON对象，仅支持workers/engines/apis/coordinators")
    for section in ("workers", "engines", "apis"):
        values = overrides.get(section, {})
        if not isinstance(values, dict):
            raise BindingError(f"pid-map.{section}须为JSON对象")
        for key, value in values.items():
            pids = value if section == "apis" else [value]
            if not isinstance(pids, list) or not pids or any(type(p) is not int or p <= 0 for p in pids):
                raise BindingError(f"pid-map.{section}.{key}: 需要正整数PID；apis的值须为非空PID数组")
            valid = (
                {f"{d}/{t}" for d, t in plan["workers"]} if section == "workers" else {str(d) for d in plan[section]}
            )
            if key not in valid:
                raise BindingError(f"pid-map.{section}存在布局外的键: {key}")
    coordinators_override = overrides.get("coordinators", [])
    if not isinstance(coordinators_override, list) or any(type(p) is not int or p <= 0 for p in coordinators_override):
        raise BindingError("pid-map.coordinators须为正整数PID数组")
    targets = []

    def unique(candidates, tag, explicit=None):
        selected = [explicit] if explicit is not None else sorted(candidates)
        if len(selected) != 1 or selected[0] not in scoped:
            raise BindingError(f"{tag}: 需要唯一PID，候选={selected}；用--root-pid限定服务或--pid-map明确指定")
        return selected[0]

    for (dp, tp), pool in plan["workers"].items():
        pattern = re.compile(rf"(?:^|\s)VLLM::Worker_DP{dp}_TP{tp}(?=[_\s]|$)")
        explicit = overrides.get("workers", {}).get(f"{dp}/{tp}")
        pid = unique([p for p, v in scoped.items() if pattern.search(v.title)], f"DP{dp}/TP{tp}", explicit)
        targets.append(Target(pid, f"DP{dp}/WorkerTP{tp}", "worker", pool, worker_pools(pool)))
    if args.workers_only:
        return targets
    worker_pids = {target.pid for target in targets}
    for dp, pool in plan["engines"].items():
        pattern = re.compile(rf"(?:^|\s)VLLM::EngineCore_DP{dp}(?=[_\s]|$)")
        pid = unique(
            [p for p, v in scoped.items() if pattern.search(v.title)],
            f"DP{dp}/EngineCore",
            overrides.get("engines", {}).get(str(dp)),
        )
        targets.append(Target(pid, f"DP{dp}/EngineCore", "engine", pool))
        api_override = overrides.get("apis", {}).get(str(dp))
        api_pids = set(api_override or [])
        if api_override is None:
            port = args.port_base + dp
            launchers = [p for p, v in scoped.items() if launcher_port(v) == port]
            if len(launchers) > 1:
                raise BindingError(f"API端口{port}有多个launcher候选{launchers}，请使用--root-pid或--pid-map")
            owners = set(ports.get(port, set())) & set(scoped)
            api_pids |= {p for p in owners if is_api(scoped[p]) or launcher_port(scoped[p]) == port}
            for parent in launchers:
                api_pids.add(parent)
                api_pids.update(p for p, v in scoped.items() if is_api(v) and below(p, parent, processes))
        if not api_pids or any(p not in scoped or p in worker_pids for p in api_pids):
            raise BindingError(
                f"DP{dp}/API无法可靠发现。默认查端口{args.port_base + dp}；"
                "共享单API端口请用--pid-map分配API进程，或用--workers-only只测试worker。"
            )
        for pid in sorted(api_pids):
            targets.append(Target(pid, f"DP{dp}/API", "api", plan["apis"][dp]))
    coordinators = overrides.get("coordinators")
    if coordinators is None:
        coordinators = [p for p, v in scoped.items() if re.search(r"(?:^|\s)VLLM::DPCoordinator(?=[_\s]|$)", v.title)]
    for pid in coordinators:
        if pid not in scoped:
            raise BindingError(f"DPCoordinator PID不可见: {pid}")
        targets.append(Target(pid, "DPCoordinator", "coordinator", plan["coordinators"]))
    if not coordinators:
        log("NOTE: 未找到DPCoordinator；该角色可能不在此部署模式中。")
    return targets


def validate_targets(targets):
    seen = {}
    result = []
    for target in targets:
        if target.pid in seen:
            previous = seen[target.pid]
            if previous.tag != target.tag or previous.cpus != target.cpus:
                raise BindingError(f"同一PID {target.pid}分配到多个不同角色: {previous.tag}, {target.tag}")
            continue
        seen[target.pid] = target
        result.append(target)
    return result


def snapshots(targets):
    items = []
    for target in targets:
        pid_start = start_time(target.pid)
        count = 0
        for path in sorted((PROC / str(target.pid) / "task").iterdir(), key=lambda p: int(p.name)):
            tid = int(path.name)
            try:
                comm = text(path / "comm")
                tid_start = start_time(target.pid, tid)
                before = sorted(os.sched_getaffinity(tid))
            except (FileNotFoundError, ProcessLookupError):
                continue
            role, cpus = thread_role(target, tid, comm)
            items.append(
                {
                    "pid": target.pid,
                    "tid": tid,
                    "pid_start": pid_start,
                    "tid_start": tid_start,
                    "tag": target.tag,
                    "thread_role": role,
                    "comm": comm,
                    "before": before,
                    "after": list(cpus),
                }
            )
            count += 1
        if not count or not any(item["tid"] == target.pid and item["pid"] == target.pid for item in items):
            raise BindingError(f"{target.tag}: 主线程已退出或不可见")
    return items


def same_thread(item):
    try:
        return (
            start_time(item["pid"]) == item["pid_start"] and start_time(item["pid"], item["tid"]) == item["tid_start"]
        )
    except (OSError, IndexError):
        return False


def print_plan(items, verbose):
    groups = {}
    for item in items:
        key = (item["tag"], item["pid"], item["thread_role"], tuple(item["after"]))
        groups.setdefault(key, []).append(item["tid"])
    for (tag, pid, role, cpus), tids in groups.items():
        log(f"PLAN {tag} pid={pid} role={role} cpus={format_cpus(cpus)} tids={','.join(map(str, tids))}")
    if verbose:
        for item in items:
            log(
                f"  TID={item['tid']} comm={item['comm']} {format_cpus(item['before'])} -> {format_cpus(item['after'])}"
            )


def setting(path, after, kind):
    if not path.exists():
        raise BindingError(f"设置不存在: {path}")
    before = text(path)
    return {"path": str(path), "before": before, "after": after, "kind": kind}


def irq_settings(args, plan):
    if not args.irq:
        return []
    if not args.irq_map:
        raise BindingError("--irq需要--irq-map真实设备映射；脚本不猜测CQ=SQ+1，也不假定logical NPU/card/chip映射")
    mapping = json.loads(Path(args.irq_map).read_text())
    if not mapping.get("devices"):
        raise BindingError("IRQ映射devices不能为空")
    result, used, queues, workers = [], set(), set(), set()
    for device in mapping["devices"]:
        key = (device.get("dp"), device.get("tp"))
        if key not in plan["workers"]:
            raise BindingError(f"IRQ映射的dp/tp不在当前布局: {key}")
        if key in workers:
            raise BindingError(f"IRQ映射重复dp/tp: {key}")
        workers.add(key)
        bdf = device.get("pci_bdf", "")
        if not re.fullmatch(r"[\da-fA-F]{4}:[\da-fA-F]{2}:[\da-fA-F]{2}\.[0-7]", bdf):
            raise BindingError(f"PCI BDF需填写真实完整地址: {bdf!r}")
        msi = SYS / "bus/pci/devices" / bdf.lower() / "msi_irqs"
        for name, cpu in zip(["sq_irq", "cq_irq"], plan["irq"][key]):
            irq = device.get(name)
            if type(irq) is not int or irq < 0 or irq in used or not (msi / str(irq)).exists():
                raise BindingError(f"{bdf}/{name}={irq!r}无效、重复或不属于该PCI设备")
            used.add(irq)
            result.append(setting(PROC / f"irq/{irq}/smp_affinity_list", str(cpu), "cpu-list"))
        for field_name, cpu in zip(["send_workqueue", "recv_workqueue"], plan["irq"][key]):
            name = device.get(field_name)
            if name:
                if not re.fullmatch(r"dev\d+_sq_(?:send|recv)_wq", name):
                    raise BindingError(f"非法workqueue名称: {name}")
                if name in queues:
                    raise BindingError(f"IRQ映射重复workqueue: {name}")
                queues.add(name)
                result.append(setting(SYS / f"devices/virtual/workqueue/{name}/cpumask", hex_mask([cpu]), "hex"))
    return result


def equal_setting(actual, expected, kind):
    if kind == "cpu-list":
        return cpu_list(actual) == cpu_list(expected)
    if kind == "hex":
        return int(actual.replace(",", ""), 16) == int(expected.replace(",", ""), 16)
    return actual.strip() == expected.strip()


def write_setting(item, value):
    path = Path(item["path"])
    path.write_text(value + "\n")
    actual = text(path)
    if not equal_setting(actual, value, item["kind"]):
        if value == item["after"]:
            item["observed_after"] = actual
        raise BindingError(f"设置回读不一致: {path}")
    if item["kind"] == "cpu-list":
        effective = path.with_name("effective_affinity_list")
        if effective.exists() and cpu_list(text(effective)) != cpu_list(value):
            raise BindingError(f"IRQ有效亲和不一致: {effective}")


def service_active():
    try:
        return subprocess.run(["systemctl", "is-active", "--quiet", "irqbalance"], capture_output=True).returncode == 0
    except OSError:
        return False


def service_change(action):
    subprocess.run(["systemctl", action, "irqbalance"], check=True)


def save_state(path, state, create=False):
    payload = json.dumps(state, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if create:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as output:
            output.write(payload)
        return
    fd, temporary = tempfile.mkstemp(prefix=".bind-state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as output:
            output.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def restore_state(state, dry_run=False):
    if state.get("schema") != 1 or state.get("boot_id") != text(BOOT_ID):
        raise BindingError("状态文件格式不支持或不是本次主机启动；拒绝恢复")
    failures = []
    for item in reversed(state["threads"]):
        if not same_thread(item):
            log(f"SKIP 原线程已退出或PID/TID复用: {item['pid']}/{item['tid']}")
            continue
        try:
            current = sorted(os.sched_getaffinity(item["tid"]))
        except ProcessLookupError:
            log(f"SKIP 恢复期间退出TID={item['tid']}")
            continue
        except OSError as error:
            failures.append(f"TID{item['tid']}: {error}")
            continue
        if current == item["before"]:
            continue
        if current != item["after"] and current != item.get("observed_after"):
            failures.append(f"TID{item['tid']}亲和被其他操作更改，保留当前值")
            continue
        if not dry_run:
            try:
                os.sched_setaffinity(item["tid"], set(item["before"]))
                if sorted(os.sched_getaffinity(item["tid"])) != item["before"]:
                    raise BindingError("恢复回读不一致")
            except (OSError, BindingError) as error:
                failures.append(f"TID{item['tid']}: {error}")
                continue
        log(f"{'DRY restore' if dry_run else 'RESTORE'} TID={item['tid']} -> {format_cpus(item['before'])}")
    for item in reversed(state["settings"]):
        path = Path(item["path"])
        valid = path == NUMA_BALANCING or bool(re.fullmatch(r"/proc/irq/\d+/smp_affinity_list", str(path)))
        valid |= bool(re.fullmatch(r"/sys/devices/virtual/workqueue/dev\d+_sq_(?:send|recv)_wq/cpumask", str(path)))
        if not valid:
            failures.append(f"状态文件包含不支持恢复的路径: {path}")
            continue
        try:
            current = text(path)
            if equal_setting(current, item["before"], item["kind"]):
                continue
            if not equal_setting(current, item["after"], item["kind"]) and not (
                "observed_after" in item and equal_setting(current, item["observed_after"], item["kind"])
            ):
                failures.append(f"{path}被其他操作更改，保留当前值")
                continue
            if not dry_run:
                write_setting(item, item["before"])
            log(f"{'DRY restore' if dry_run else 'RESTORE'} {path} -> {item['before']}")
        except (OSError, BindingError) as error:
            failures.append(str(error))
    if state.get("irqbalance_stopped") and not service_active():
        if not dry_run:
            try:
                service_change("start")
            except (OSError, subprocess.CalledProcessError) as error:
                failures.append(f"恢复irqbalance失败: {error}")
            else:
                log("RESTORE irqbalance started")
        else:
            log("DRY start irqbalance")
    if failures:
        raise BindingError("恢复未全部完成: " + "; ".join(failures))


def apply_affinities(items):
    count = 0
    for item in items:
        if not same_thread(item):
            if item["tid"] == item["pid"]:
                raise BindingError(f"{item['tag']}主线程已退出或复用")
            log(f"SKIP 已退出TID={item['tid']}")
            continue
        os.sched_setaffinity(item["tid"], set(item["after"]))
        actual = sorted(os.sched_getaffinity(item["tid"]))
        if actual != item["after"]:
            # Kernel cpusets can silently intersect the requested mask. Remember
            # that actual write so rollback can restore it without guessing.
            item["observed_after"] = actual
            raise BindingError(
                f"TID={item['tid']}亲和回读不一致: expected={item['after']}, actual={actual}；检查cpuset"
            )
        count += 1
    return count


def verify(targets):
    items = snapshots(targets)
    wrong = [item for item in items if item["before"] != item["after"]]
    for item in wrong:
        log(
            f"MISMATCH {item['tag']} TID={item['tid']} comm={item['comm']} "
            f"actual={format_cpus(item['before'])} expected={format_cpus(item['after'])}"
        )
    if wrong:
        raise BindingError(f"{len(wrong)}/{len(items)}个现存线程亲和不匹配；新线程可能刚创建，请在warmup完成后重跑")
    return len(items)


def apply(args, targets, state_path, settings):
    items = snapshots(targets)
    print_plan(items, args.verbose)
    for item in settings:
        log(f"{'DRY' if args.dry_run else 'PLAN'} {item['path']}: {item['before']} -> {item['after']}")
    active = args.irq and service_active()
    if active:
        log("DRY stop irqbalance" if args.dry_run else "PLAN stop irqbalance (恢复时按原状态启动)")
    if args.dry_run:
        log(f"DRY完成: {len(targets)}个进程、{len(items)}个线程；未改变亲和/NUMA/IRQ/服务，未写状态文件。")
        return
    state = {
        "schema": 1,
        "boot_id": text(BOOT_ID),
        "hostname": socket.gethostname(),
        "created": datetime.datetime.now().astimezone().isoformat(),
        "role": args.role,
        "status": "prepared",
        "threads": items,
        "settings": settings,
        "irqbalance_stopped": False,
    }
    save_state(state_path, state, create=True)
    log(f"STATE {state_path} (mode 0600)")
    try:
        count = apply_affinities(items)
        # One post-pass catches threads created during the initial assignment.
        known = {(item["tid"], item["tid_start"]) for item in items}
        fresh = [item for item in snapshots(targets) if (item["tid"], item["tid_start"]) not in known]
        if fresh:
            state["threads"].extend(fresh)
            save_state(state_path, state)
            count += apply_affinities(fresh)
        for item in settings:
            if item["path"] == str(NUMA_BALANCING):
                write_setting(item, item["after"])
        if active:
            # Journal before the action, including a possible interrupt while
            # systemctl is running. Starting an originally active service is safe.
            state["irqbalance_stopped"] = True
            save_state(state_path, state)
            service_change("stop")
        for item in settings:
            if item["path"] != str(NUMA_BALANCING):
                write_setting(item, item["after"])
        verified = verify(targets)
        state["status"] = "applied"
        state["verified_threads"] = verified
        save_state(state_path, state)
        log(f"SUCCESS: 已设置{count}次线程亲和，逐线程验证{verified}个现存线程。")
        log(f"恢复命令: bash bind_threads_v3.sh --restore {state_path}")
        log("后续新建线程会继承创建者亲和；本脚本不是持续运行的绑核代理。")
    except (OSError, BindingError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
        state["status"] = "failed"
        state["error"] = str(error)
        try:
            restore_state(state)
            state["status"] = "rolled_back"
            log("FAIL: 本次修改已回滚到记录的原值。")
        except (OSError, BindingError) as rollback_error:
            state["rollback_error"] = str(rollback_error)
            log(f"FAIL: 自动回滚未完成，请查看状态文件: {rollback_error}")
        save_state(state_path, state)
        raise BindingError(str(error)) from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="bind_threads_v3.sh", description=__doc__)
    parser.add_argument("--role", choices=["P", "D"])
    parser.add_argument("--dry-run", action="store_true", help="完整发现和预览；不改变任何系统设置")
    parser.add_argument("--workers-only", action="store_true", help="仅worker线程，适合先做单变量实验")
    parser.add_argument("--verify-only", action="store_true", help="只验证当前线程亲和是否符合布局")
    parser.add_argument("--port-base", type=int, default=7100, help="独立API端口起点，默认7100")
    parser.add_argument("--root-pid", type=int, action="append", default=[], help="仅匹配这些PID及后代，可多次指定")
    parser.add_argument("--pid-map", help="进程标题/共享API端口无法唯一发现时使用JSON显式映射")
    parser.add_argument("--state-file", help="原值备份JSON；不得覆盖既有文件")
    parser.add_argument("--restore", metavar="STATE_JSON", help="恢复该文件记录的同一主机启动中的设置")
    parser.add_argument(
        "--disable-numa-balancing", action="store_true", help="本次将主机级numa_balancing设0；默认不修改"
    )
    parser.add_argument("--irq", action="store_true", help="独立可选IRQ/workqueue实验，需要--irq-map")
    parser.add_argument("--irq-map", help="明确dp/tp、PCI BDF、SQ/CQ IRQ及可选workqueue名称的JSON")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.restore:
        if (
            args.role
            or args.disable_numa_balancing
            or args.irq
            or args.verify_only
            or args.workers_only
            or args.root_pid
            or args.pid_map
            or args.state_file
            or args.irq_map
            or args.port_base != 7100
        ):
            parser.error("--restore仅与--dry-run/--verbose组合")
    elif not args.role:
        parser.error("必须显式指定--role P或--role D")
    if not 1 <= args.port_base <= 65528:
        parser.error("--port-base须在1..65528")
    if args.verify_only and (args.disable_numa_balancing or args.irq):
        parser.error("--verify-only不允许同时修改NUMA/IRQ")
    if args.irq_map and not args.irq:
        parser.error("--irq-map须与--irq组合")
    if args.irq and not args.irq_map:
        parser.error("--irq需要--irq-map；不会推测CQ IRQ编号")
    if any(pid <= 0 for pid in args.root_pid):
        parser.error("--root-pid须为正整数")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.restore:
        path = Path(args.restore).resolve()
        state = json.loads(path.read_text())
        restore_state(state, args.dry_run)
        if not args.dry_run:
            state["status"] = "restored"
            save_state(path, state)
        log("恢复检查完成；仅恢复已记录且身份未变化的线程，不推测未记录新线程的原亲和。")
        return
    nodes = topology()
    for node, cpus in nodes.items():
        log(f"NUMA{node}: {format_cpus(cpus)}")
    plan = layout(args.role, nodes)
    processes = read_processes()
    targets = validate_targets(discover(args, plan, processes, listener_pids()))
    log(
        "NOTE: 自动绑核开启状态应以vLLM最终配置/日志确认；不根据cmdline关键词猜测。"
        "此方案将覆盖目标线程亲和，不迁移内存页。"
    )
    if args.verify_only:
        log(f"VERIFY通过: {verify(targets)}个现存线程")
        return
    settings = []
    if args.disable_numa_balancing:
        settings.append(setting(NUMA_BALANCING, "0", "text"))
        log("NOTE: --disable-numa-balancing作用于整台主机；原值将保存，恢复时保留其他操作者后续更改。")
    settings.extend(irq_settings(args, plan))
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    state_path = Path(args.state_file or f"bind-state-{args.role}-{stamp}-{os.getpid()}.json").resolve()
    apply(args, targets, state_path, settings)


if __name__ == "__main__":
    try:
        main()
    except (BindingError, OSError, ValueError, KeyError, IndexError) as failure:
        print(f"ERROR: {failure}", file=sys.stderr)
        sys.exit(1)

PYTHON_BIND_CORE_V3
