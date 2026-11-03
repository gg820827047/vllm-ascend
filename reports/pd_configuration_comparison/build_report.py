#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze official CI/doc sources and render the A3 P/D migration supplement."""

import csv
import difflib
import hashlib
import html
import io
import json
import re
import shlex
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = "8fde6d4e335fe300e2b949aab0e31659cce0811d"
NEW = "38fb7a95f30cd7d715890cf89e3e640d7992b6d9"
DOC = "docs/source/tutorials/models/DeepSeek-V4.1-Flash.md"
OLD_CI = "tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V4.1-Flash-w8a8-PD-prefix.yaml"
NEW_CI = "tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V4.1-Flash-w8a8-PD-MRV2.yaml"
GITHUB = "https://github.com/vllm-project/vllm-ascend/blob/"


def git_text(ref, path):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=REPO, text=True)


def options(argv):
    out = {}
    i = 0
    while i < len(argv):
        key = argv[i]
        assert key.startswith("--"), (key, argv)
        i += 1
        value = True
        if i < len(argv) and not argv[i].startswith("--"):
            value = argv[i]
            i += 1
        if isinstance(value, str) and value.startswith("{"):
            value = json.loads(value)
        assert key not in out, key
        out[key] = value
    return out


def ci_normalized(data):
    result = {}
    for role, config, template in zip(("P", "D"), data["config"], data["templates"], strict=True):
        values = {
            "DP_SIZE": str(config["dp_size"]),
            "TP_SIZE": str(config["tp_size"]),
            "CP_SIZE": str(config.get("cp_size", 1)),
        }
        argv = [
            values.get(str(arg).removeprefix("${").removesuffix("}"), str(arg))
            for arg in template["server_cmd_template"]
        ]
        result[role] = {"topology": config, "env": template["envs"], "options": options(argv)}
    return result


def doc_normalized(raw):
    section = raw[raw.index("### 5.2 ") :]
    for end in ('\n=== "A2 series"', "\n### 5.3 "):
        if end in section:
            section = section[: section.index(end)]
    result = {}
    scripts = []
    launchers = []
    for block in re.findall(r"```(?:shell|bash)\n(.*?)```", section, re.S):
        if "vllm serve" in block:
            argv = shlex.split(block[block.index("vllm serve") :].replace("\\\n", " "))[3:]
            scripts.append(
                {"options": options(argv), "env": dict(re.findall(r"^\s*export ([A-Z_0-9]+)=(.*)$", block, re.M))}
            )
        if "python launch_online_dp.py" in block:
            argv = shlex.split(block[block.index("python launch_online_dp.py") :].replace("\\\n", " "))[2:]
            launchers.append(options(argv))
    assert len(scripts) == len(launchers) == 2
    for role, script, launcher in zip(("P", "D"), scripts, launchers, strict=True):
        script["launcher"] = launcher
        result[role] = script
    return result


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {
            k: v
            for key, item in sorted(value.items())
            for k, v in flatten(item, f"{prefix}.{key}" if prefix else key).items()
        }
    return {prefix: value}


def display(value):
    if value is None:
        return "未显式指定"
    if isinstance(value, bool):
        return "开启" if value else "关闭"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def note(path):
    if "VLLM_USE_V2_MODEL_RUNNER" in path:
        return "两端均显式迁移至 V2；PCP4 组合不能只把环境变量切回 V1。"
    if any(key in path for key in ("dp_size", "tp_size", "cp_size", "parallel-size")):
        return "P 端迁移为 DP2/TP2/PCP4；设备分配、代理 endpoint 和 connector 两端拓扑需同步更新。"
    if "max-num-seqs" in path:
        return "P 每个 DP 上限 16→32，但 DP4→DP2，节点名义请求上限仍为 64。"
    if "gpu-memory-utilization" in path:
        return "P 的预算比例 0.90→0.92；可用 KV 容量仍取决于权重、图和运行时内存。"
    if "enable_dsa_cp" in path:
        return "删除旧配置，改用 --prefill-context-parallel-size 4；新代码已移除旧字段。"
    if "kv_connector" in path:
        return "connector 由 Hybrid 换为 V2；注册到不同 Python 模块，不能仅视为改名。"
    return "显式参数一致；这不证明两个版本的底层实现和最终默认行为相同。"


LAUNCHER = '''#!/usr/bin/env python3
"""Report-derived launcher for the documented two-node A3 1P1D topology.

Uses one process per external DP rank; PP=DCP=1 in this example.
No extra harness sp_size device dimension is configured; MoE SP is unchanged.
Dry-run checks allocation and prints commands without starting vLLM.
"""

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--dp-size-local", type=int, default=-1)
    parser.add_argument("--dp-rank-start", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pcp-size", type=int, default=1)
    parser.add_argument("--dp-address", required=True)
    parser.add_argument("--dp-rpc-port", default="12321")
    parser.add_argument("--vllm-start-port", type=int, default=7100)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--devices", help="Optional comma-separated logical device IDs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    local = args.dp_size if args.dp_size_local == -1 else args.dp_size_local
    if min(args.dp_size, local, args.tp_size, args.pcp_size) < 1:
        parser.error("Parallel sizes must be positive")
    if args.dp_rank_start < 0 or args.dp_rank_start + local > args.dp_size:
        parser.error("Local DP ranks must fit in the global DP size")
    if not args.template.is_file():
        parser.error("Template file does not exist")
    width = args.tp_size * args.pcp_size
    count = local * width
    devices = args.devices.split(",") if args.devices else list(map(str, range(count)))
    if len(devices) != count or len(set(devices)) != count or any(not device.strip() for device in devices):
        parser.error("Device IDs must be unique and match local_DP * TP * PCP")
    commands = []
    for i in range(local):
        visible = ",".join(devices[i * width : (i + 1) * width])
        command = [
            "bash",
            str(args.template.resolve()),
            visible,
            str(args.vllm_start_port + i),
            str(args.dp_size),
            str(args.dp_rank_start + i),
            args.dp_address,
            str(args.dp_rpc_port),
            str(args.tp_size),
            str(args.pcp_size),
        ]
        commands.append(command)
        print(
            json.dumps(
                {
                    "dp_rank": args.dp_rank_start + i,
                    "visible_devices": visible,
                    "api_port": args.vllm_start_port + i,
                    "command": shlex.join(command),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.dry_run:
        return
    processes = []
    try:
        for command in commands:
            processes.append(subprocess.Popen(command))
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise SystemExit("A DP process failed; stopping the remaining processes")
            time.sleep(0.25)
        if any(process.returncode for process in processes):
            raise SystemExit(1)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
'''


def render_script(data, role):
    template = data["templates"][0 if role == "P" else 1]
    env = template["envs"]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Derived from the pinned official 0.32 CI config by this report.",
        "# Statically checked; not executed on A3 in this workspace.",
        ': "${MODEL_PATH:?Set MODEL_PATH to the W8A8 checkpoint directory}"',
        ': "${LOCAL_IP:?Set LOCAL_IP to this node service IP}"',
        ': "${NIC_NAME:?Set NIC_NAME to this node network interface}"',
        '[[ $# -eq 8 ]] || { echo "Expected: devices port dp rank address rpc tp pcp" >&2; exit 2; }',
        'export ASCEND_RT_VISIBLE_DEVICES="$1"',
        'export SERVER_PORT="$2"',
        'export DP_SIZE="$3"',
        'export DP_RANK="$4"',
        'export DP_ADDRESS="$5"',
        'export DP_RPC_PORT="$6"',
        'export TP_SIZE="$7"',
        'export CP_SIZE="$8"',
        'export HCCL_IF_IP="$LOCAL_IP"',
        'export GLOO_SOCKET_IFNAME="$NIC_NAME"',
        'export TP_SOCKET_IFNAME="$NIC_NAME"',
        'export HCCL_SOCKET_IFNAME="$NIC_NAME"',
    ]
    dynamic = {"ASCEND_RT_VISIBLE_DEVICES", "SERVER_PORT", "no_proxy", "NO_PROXY"}
    for key, value in env.items():
        if key not in dynamic:
            lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.extend(
        [
            'export no_proxy="127.0.0.1,localhost,${LOCAL_IP},${PREFILL_IP:-},${DECODE_IP:-}"',
            'export NO_PROXY="$no_proxy"',
            'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/lib/"',
            "",
            'exec vllm serve "$MODEL_PATH" \\',
            "    --served-model-name dsv41 \\",
        ]
    )
    opts = options(template["server_cmd_template"])
    for i, (key, value) in enumerate(opts.items()):
        suffix = " \\" if i < len(opts) - 1 else ""
        if value is True:
            arg = ""
        else:
            raw = (
                json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, dict) else str(value)
            )
            arg = " " + (f'"{raw}"' if raw.startswith("$") else shlex.quote(raw))
        lines.append(f"    {key}{arg}{suffix}")
    return "\n".join(lines) + "\n"


def e(value):
    return html.escape(str(value), quote=True)


def main():
    # Freeze source bytes before normalizing configuration or generating examples.
    evidence = HERE / "evidence"
    examples = HERE / "examples"
    evidence.mkdir(exist_ok=True)
    examples.mkdir(exist_ok=True)
    sources = {}
    for name, ref, path in (
        ("ci_030", OLD, OLD_CI),
        ("ci_032", NEW, NEW_CI),
        ("doc_030", OLD, DOC),
        ("doc_032", NEW, DOC),
    ):
        raw = git_text(ref, path)
        filename = f"{name}.txt"
        (evidence / filename).write_text(raw)
        sources[name] = {
            "ref": ref,
            "path": path,
            "url": GITHUB + ref + "/" + path,
            "snapshot": "evidence/" + filename,
            "sha256": hashlib.sha256(raw.encode()).hexdigest(),
        }
    old_raw = (evidence / "ci_030.txt").read_text()
    new_raw = (evidence / "ci_032.txt").read_text()
    old_data, new_data = yaml.safe_load(old_raw), yaml.safe_load(new_raw)
    old, new = ci_normalized(old_data), ci_normalized(new_data)
    docs = {key: doc_normalized((evidence / f"{key}.txt").read_text()) for key in ("doc_030", "doc_032")}
    rows = []
    for role in ("P", "D"):
        a, b = flatten(old[role]), flatten(new[role])
        for key in sorted(a.keys() | b.keys()):
            before, after = a.get(key), b.get(key)
            rows.append(
                {
                    "role": role,
                    "path": key,
                    "old": before,
                    "new": after,
                    "changed": before != after,
                    "interpretation": note(key) if before != after else "显式参数一致；最终执行行为仍需核对日志。",
                }
            )
    benchmarks = []
    for name in old_data["benchmarks"].keys() | new_data["benchmarks"].keys():
        a, b = old_data["benchmarks"].get(name, {}), new_data["benchmarks"].get(name, {})
        for key in sorted(a.keys() | b.keys()):
            if a.get(key) != b.get(key):
                benchmarks.append({"case": name, "key": key, "old": a.get(key), "new": b.get(key)})
    benchmarks.sort(key=lambda row: (row["case"], row["key"]))
    document = {
        "scope": "DeepSeek V4.1 Flash W8A8 / two A3 nodes / 16 logical devices per node / external DP / 1P1D",
        "date": "2026-11-03",
        "sources": sources,
        "ci": {"old": old, "new": new},
        "docs": docs,
        "rows": rows,
        "benchmark_config_changes": benchmarks,
        "validation_scope": "Static source/config comparison and launcher dry-run; no NPU execution or performance measurement",
    }
    (HERE / "comparison.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    (HERE / "ci.diff").write_text(
        "".join(
            difflib.unified_diff(
                old_raw.splitlines(True), new_raw.splitlines(True), fromfile=OLD_CI, tofile=NEW_CI, n=0
            )
        )
    )
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer, lineterminator="\n")
    writer.writerow(["role", "path", "old", "new", "changed", "interpretation"])
    writer.writerows(
        [row["role"], row["path"], display(row["old"]), display(row["new"]), row["changed"], row["interpretation"]]
        for row in rows
    )
    (HERE / "ci_comparison.csv").write_text(csv_buffer.getvalue(), encoding="utf-8-sig")
    (examples / "launch_online_dp_032.py").write_text(LAUNCHER)
    for role, name in (("P", "prefill"), ("D", "decode")):
        (examples / f"run_{name}_032.sh").write_text(render_script(new_data, role))

    migration_rows = [
        ("P/D Runner", "VLLM_USE_V2_MODEL_RUNNER=0", "两端改为 1", "PCP4 使用 V2；不能只切环境变量而保留不兼容布局。"),
        ("P 启动拓扑", "DP4 / TP4", "DP2 / TP2 / PCP4", "2 个 P endpoint；每个 DP 实例分配 8 个设备。"),
        (
            "P 模板参数",
            "没有 PCP 参数",
            "增加 --prefill-context-parallel-size 4",
            "--tensor-parallel-size 改为 2；launcher 增加第 8 个参数 PCP。",
        ),
        (
            "P 设备切分",
            "每 DP 实例分配 TP=4 个设备",
            "每 DP 实例分配 TP×PCP=8 个设备",
            "DP0：0–7；DP1：8–15。仅修改 CLI 不能修正旧 launcher。",
        ),
        ("D 启动拓扑", "DP8 / TP2", "保持 DP8 / TP2，PCP=1", "8 个 D endpoint；每个实例仍分配 2 个设备。"),
        ("P max-num-seqs", "16", "32", "节点名义请求上限仍为 4×16=2×32=64。"),
        ("P 内存预算", "gpu-memory-utilization=0.90", "改为 0.92", "D 保持 0.95；比例不代表实际剩余 HBM 或 KV 容量。"),
        (
            "D 批处理",
            "原始 tag 文档：400 tokens / 32 seqs",
            "保持 400 / 32",
            "若你用的是当前网页文档的 512/72，需改回 CI 的 400/32。",
        ),
        (
            "旧 DSA-CP",
            "P additional-config 中 enable_dsa_cp=true",
            "删除该键，使用 PCP4",
            "保留 enable_fused_mc2=1、enable_flashcomm1=true、enable_shared_expert_dp=true、enable_cpu_binding=true。",
        ),
        (
            "KV connector",
            "两端 MooncakeHybridConnector",
            "两端改为 MooncakeConnectorV2",
            "角色仍为 P producer / D consumer；不同 connector 对应不同模块。",
        ),
        (
            "connector 拓扑",
            "prefill dp_size=4,tp_size=4",
            "两端同时改为 prefill dp_size=2,tp_size=2",
            "decode dp_size=8,tp_size=2 不变；按 CI 配置填写，不能把 PCP4 乘进 tp_size。",
        ),
        (
            "代理 endpoint",
            "P：7100–7103；D：7100–7107",
            "P：7100、7101；D 保持 7100–7107",
            "PCP worker 不各自暴露一个 API endpoint，不能保留不存在的 P 地址。",
        ),
        (
            "加载线程",
            "tag 文档 P/D 都是 num_threads=128",
            "P 改为 32，D 改为 16",
            "两版 CI 原本即 32/16；这是文档→CI 调整，主要影响启动，不是 CI 版本性能增量。",
        ),
        (
            "环境变量",
            "文档部分变量未显式设置",
            "按附带模板补齐 allocator、OMP、超时等 CI env",
            "HCCL_BUFFSIZE P=1024、D=1800；AIV 只在 P 设置；网络接口/IP 仍按实际环境。",
        ),
        (
            "图/投机/Engram",
            "P eager；D FULL_DECODE_ONLY；DSpark=5；CPU offload＋共享内存",
            "均保持；D static kernel=false、recompute_scheduler_enable=true",
            "不加入 CI 未显式添加的 replay/静态 kernel 开关。",
        ),
        (
            "服务别名",
            "文档使用 --served-model-name dsv41，CI 未指定别名",
            "本报告模板保留文档的 dsv41 别名",
            "保留原文档客户端请求中的 model=dsv41；这是迁移便利项，不是 CI 性能参数变化。",
        ),
    ]
    launch_p = 'python launch_online_dp_032.py --template ./run_prefill_032.sh --dp-size 2 --dp-size-local 2 --tp-size 2 --pcp-size 4 --dp-address "$LOCAL_IP" --dp-rpc-port 12321 --vllm-start-port 7100'
    launch_d = 'python launch_online_dp_032.py --template ./run_decode_032.sh --dp-size 8 --dp-size-local 8 --tp-size 2 --pcp-size 1 --dp-address "$LOCAL_IP" --dp-rpc-port 12321 --vllm-start-port 7100'
    usage = f"""# 将 examples/ 的三个文件复制到对应 A3 容器的同一目录。
# 两端使用同一份 W8A8 checkpoint，容器设备/驱动挂载、--ipc=host 沿用文档。
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-w8a8
export PREFILL_IP=192.0.2.10   # 替换为实际 P 节点 IP
export DECODE_IP=192.0.2.11    # 替换为实际 D 节点 IP
export NIC_NAME=YOUR_NETWORK_INTERFACE

# P 节点
export LOCAL_IP="$PREFILL_IP"
{launch_p} --dry-run
# 核对可见设备 0–7 / 8–15 和 API 7100 / 7101 后，去掉 --dry-run 启动。

# D 节点（在 D 容器执行，先设置同样的 MODEL_PATH/PREFILL_IP/DECODE_IP/NIC_NAME）
export LOCAL_IP="$DECODE_IP"
{launch_d} --dry-run
# 核对 8 个 DP 实例各使用 2 个设备、API 7100–7107，再去掉 --dry-run。

# 代理：P endpoint 列表改为 P_IP:7100、P_IP:7101；D 仍是 D_IP:7100–7107。
# 其他代理参数、请求格式和网络连通配置沿用原文档。
"""
    (examples / "usage.txt").write_text(usage)
    table_rows = "".join(
        f'<tr data-role="{r["role"]}" data-changed="{int(r["changed"])}"><td>{r["role"]}</td><td><code>{e(r["path"])}</code></td><td>{e(display(r["old"]))}</td><td>{e(display(r["new"]))}</td><td>{e(r["interpretation"])}</td></tr>'
        for r in rows
    )
    migration_html = "".join(
        "<tr>" + "".join(f"<td>{e(cell)}</td>" for cell in row) + "</tr>" for row in migration_rows
    )
    source_html = "".join(
        f'<li><a href="{e(s["url"])}">{e(name)} · {s["ref"][:12]}</a> · <a href="{s["snapshot"]}">离线原文</a><br><code>{e(s["path"])}</code></li>'
        for name, s in sources.items()
    )
    benchmark_html = "".join(
        f"<tr><td>{e(r['case'])}</td><td>{e(r['key'])}</td><td>{e(r['old'])}</td><td>{e(r['new'])}</td></tr>"
        for r in benchmarks
    )
    code_links = [
        ("旧文档启动器（按 TP 切设备）", OLD, DOC, 283),
        ("0.32 CI 启动器（TP×CP×SP×PP）", NEW, "tests/e2e/common/multi_node/external_dp/external_dp_config.py", 66),
        (
            "0.32 decoder replay 的 SP/PP/DCP/图限制",
            NEW,
            "vllm_ascend/models/deepseek_v41/decoder_replay_layers.py",
            96,
        ),
        ("0.32 重计算调度会关闭 replay", NEW, "vllm_ascend/platform.py", 1527),
        ("0.32 connector 注册到不同模块", NEW, "vllm_ascend/distributed/kv_transfer/__init__.py", 90),
        ("CI baseline 是输出吞吐验收门槛", NEW, "tests/e2e/common/multi_node/benchmark_results.py", 45),
        ("0.32 配置校验；旧 enable_dsa_cp 已移除", NEW, "vllm_ascend/ascend_config.py", 1831),
    ]
    code_html = "".join(
        f'<li><a href="{GITHUB}{ref}/{path}#L{line}">{e(label)}</a></li>' for label, ref, path, line in code_links
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeepSeek V4.1 · A3 P/D 配置对照与迁移</title>
<style>
:root{{--ink:#19314a;--muted:#586d80;--line:#dce5ed;--blue:#125ea3;--paper:#f4f7fb;--warn:#fff2dc}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.7 system-ui,-apple-system,"Noto Sans SC",sans-serif}}a{{color:var(--blue)}}nav{{position:fixed;inset:0 auto 0 0;width:230px;background:#132c45;color:white;padding:34px 24px}}nav a{{display:block;color:#d9e9f8;text-decoration:none;padding:9px 0}}nav small{{display:block;margin-top:24px;color:#b6c9dc}}main{{margin-left:230px;max-width:1390px;padding:40px}}header{{background:#193d60;color:white;padding:34px;border-radius:18px}}header p{{color:#dce9f5}}h1{{font-size:32px;line-height:1.3;margin:8px 0 18px}}h2{{font-size:23px;margin:0 0 16px}}h3{{margin-top:28px}}section{{margin-top:26px;background:white;border:1px solid var(--line);border-radius:16px;padding:28px;scroll-margin-top:15px}}.eyebrow{{font-size:12px;letter-spacing:2px}}.cards{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:20px 0}}.card{{padding:22px;background:#edf4fc;border-radius:12px}}.card b{{font-size:21px;display:block}}.callout{{padding:18px 20px;border-left:4px solid #3682bd;background:#eef5fc;margin:18px 0}}.warn{{background:var(--warn);border-color:#c48320}}.muted{{color:var(--muted)}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid var(--line);text-align:left;vertical-align:top;padding:12px}}th{{background:#eaf1f8;white-space:nowrap}}td{{min-width:100px;overflow-wrap:anywhere}}#ci-table td:nth-child(2){{min-width:220px}}#ci-table td:last-child{{min-width:260px}}tr[data-changed="1"]{{background:#fff9ec}}code{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.9em;overflow-wrap:anywhere}}pre{{background:#112b43;color:#e2eef8;border-radius:12px;padding:20px;overflow:auto;font-size:13px;line-height:1.6}}.controls,.downloads{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:16px 0}}input[type=search]{{padding:10px;border:1px solid #b7cbdc;border-radius:6px;min-width:220px}}select,button,.button{{border:1px solid #b7cbdc;border-radius:6px;background:white;color:var(--ink);padding:9px 12px;text-decoration:none;cursor:pointer}}.downloads a{{font-size:13px}}li{{margin:8px 0}}footer{{color:var(--muted);margin:26px 0}}[hidden]{{display:none!important}}@media(max-width:900px){{nav{{position:static;width:auto;padding:20px}}nav a{{display:inline-block;margin-right:15px}}nav small{{display:none}}main{{margin:0;padding:18px}}section{{padding:18px}}.cards{{grid-template-columns:1fr}}h1{{font-size:27px}}}}@media print{{nav,.controls,.downloads,button{{display:none}}main{{margin:0;padding:0;max-width:none}}header{{background:white;color:var(--ink);padding:12px}}header p{{color:var(--ink)}}section{{break-inside:auto;border:0;padding:10px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;color:black;background:#f3f3f3}}.table-wrap{{overflow:visible}}td,th{{min-width:0!important;font-size:10px;padding:6px}}}}
</style></head><body>
<nav><b>vLLM Ascend</b><p>A3 · DeepSeek V4.1<br>30rc2 → 32rc</p><a href="#conclusion">结论与范围</a><a href="#ci">两版 CI 差异</a><a href="#migration">从文档如何调整</a><a href="#examples">启动模板与命令</a><a href="#performance">性能与 replay</a><a href="#sources">来源与验证</a><a href="../index.html">返回三版本报告</a><small>2026-11-03<br>固定 SHA · 离线 HTML<br>未执行 A3 推理</small></nav>
<main><header><div class="eyebrow">OFFICIAL CI COMPARISON · MIGRATION GUIDE</div><h1>DeepSeek V4.1<br>A3 P/D 配置对照与迁移</h1><p>比较两版官方 CI；以 0.30rc2 文档的启动方法为起点，给出迁移到 0.32 CI 拓扑的具体调整。</p><button onclick="window.print()">打印 / 保存 PDF</button></header>
<section id="conclusion"><h2>先确认这不是“同配置只升级版本”</h2><p>范围：同名 Eco-Tech/DeepSeek-V4.1-Flash-w8a8 模型、两台 A3、每台 16 个逻辑设备、外部 DP、1P1D。没有比较其他 DeepSeek 型号，也没有核验两次下载的 checkpoint 内容是否相同。</p><div class="cards"><div class="card"><span>0.30rc2 官方 CI · V1</span><b>P：DP4 × TP4 = 16</b><span>D：DP8 × TP2 = 16</span></div><div class="card"><span>0.32rc 官方 CI · V2</span><b>P：DP2 × TP2 × PCP4 = 16</b><span>D：DP8 × TP2 = 16</span></div></div><div class="callout"><b>迁移必须同时调整：</b>两端 Runner、P 的 PCP/TP/DP、launcher 设备切分、connector 两端拓扑和代理的 P endpoint 数。只把 Runner 改为 V2 或只加 PCP 参数不完整。</div><p>0.30 选用正式 tag <code>{OLD}</code>；0.32 选用 release 分支固定快照 <code>{NEW}</code>。这与原三阶段报告的 30 分支快照/32 历史快照不同，本页是独立部署补充，不更改原报告基线。</p><div class="callout warn">0.32 分支文档仍声明以 0.30rc2 为验证基线，且保留已移除的 enable_dsa_cp；它和 0.32 CI 不是同一套配置。当前网页文档的 D=512/72，也不是本页目标 CI 的 D=400/32。</div></section>
<section id="ci"><h2>两版 CI 的显式配置差异</h2><p>展开 YAML 锚点、解析 CLI 内嵌 JSON 后比较。黄色行为差异；未指定不等于关闭。相同的显式参数不能证明底层实现相同。</p><div class="controls"><label><input id="changed-only" type="checkbox" checked> 只看差异</label><label>角色 <select id="role"><option value="">P＋D</option><option>P</option><option>D</option></select></label><input id="search" type="search" placeholder="搜索字段或含义" aria-label="搜索配置"><span id="count" aria-live="polite"></span></div><div class="table-wrap"><table id="ci-table"><thead><tr><th>角色</th><th>字段</th><th>0.30rc2 CI</th><th>0.32rc CI</th><th>含义</th></tr></thead><tbody>{table_rows}</tbody></table></div><div class="downloads"><a class="button" href="ci_comparison.csv" download>下载完整 CSV</a><a class="button" href="comparison.json" download>规范化 JSON</a><a class="button" href="ci.diff" download>原始 CI diff</a></div><h3>CI 性能验收数值也有调整</h3><div class="table-wrap"><table><thead><tr><th>用例</th><th>字段</th><th>30</th><th>32</th></tr></thead><tbody>{benchmark_html}</tbody></table></div><p>baseline 是配置中的输出吞吐验收基准，结合 threshold=0.95 使用，不是本次实际跑出的结果。16K 的 1900→1800 与 128K prefix90 的 1700→1800 不能作为整体版本变快或变慢的实测证据。输入数据集、请求数 256、并发 64、输出上限 1024 以及 request_rate 2.7/2.5 均保持一致；投机接受率基线 0.726、容差 0.05 也未改。</p></section>
<section id="migration"><h2>基于 0.30rc2 文档，具体改什么</h2><div class="table-wrap"><table><thead><tr><th>位置</th><th>文档起点</th><th>0.32 CI 目标</th><th>说明</th></tr></thead><tbody>{migration_html}</tbody></table></div><h3>保持的执行方式</h3><p>P 继续 eager，8192 token budget，prefix caching 开启；D 继续 FULL_DECODE_ONLY、异步调度、prefix caching 关闭。两端 DSpark=5、block-size=128、quantization=ascend、Engram CPU offload＋DP shared memory；D 继续 static kernel=false、共享专家 overlap 与重计算调度。</p><p>原始 tag 文档 P/D 的加载线程是 128/128，当前 0.32 分支文档则未指定加载线程；目标 CI 明确为 P=32、D=16。附带脚本按 CI 目标生成，不把这个文档偏差算作两版 CI 的新增优化。</p></section>
<section id="examples"><h2>迁移后的启动模板与 dry-run</h2><p>以下文件由本报告从固定 CI 参数生成，<b>不是官方原始脚本</b>。保留文档的“一 DP 一进程”启动方式、网络接口设置和 dsv41 服务别名，并补上 PCP 设备分配。CI 原文件未指定 served-model-name，模板额外保留该别名以兼容原文档请求；仅做静态核对和 dry-run，尚未在 A3 执行。</p><div class="downloads"><a class="button" href="examples/launch_online_dp_032.py" download>下载启动器</a><a class="button" href="examples/run_prefill_032.sh" download>P 模板</a><a class="button" href="examples/run_decode_032.sh" download>D 模板</a><a class="button" href="examples/usage.txt" download>操作命令</a></div><pre>{e(usage)}</pre><p>本示例按设备 ID 0–15 分配。若容器可见 ID 不同，用启动器的 <code>--devices</code> 显式给出完整、不重复的 16 个逻辑设备 ID。PP、DCP 和 FlashComm 相关设置按本 CI 配置保留，不把此启动器用于任意多机并行拓扑。</p><details><summary>查看启动器完整代码</summary><pre>{e(LAUNCHER)}</pre></details><details><summary>查看 P 模板完整命令</summary><pre>{e((examples / "run_prefill_032.sh").read_text())}</pre></details><details><summary>查看 D 模板完整命令</summary><pre>{e((examples / "run_decode_032.sh").read_text())}</pre></details></section>
<section id="performance"><h2>如何解释性能差异</h2><ul><li><b>P 的节点名义请求上限不变：</b>4×16=2×32=64。各 DP 调度器预算求和则从 4×8192=32768 变为 2×8192=16384；这是预算口径，不是实际吞吐或每张卡计算量。PCP 的分布、padding 与通信另行决定执行开销。</li><li><b>PCP 是同一请求的上下文并行：</b>长输入可能受益，短输入或低负载的通信收益比需要实测。不能仅因开 PCP4 就推断更快。</li><li><b>D 的 CI 拓扑和批处理参数未改：</b>若 TPOT 下降，优先对照 Runner、实际图命中/回退、DSpark 接受率、请求批次及 connector 等待；不能只归因于 P 新增 PCP。</li><li><b>两端 connector 必须同步迁移：</b>Hybrid 与 V2 指向不同 Python 实现；元数据中的 P dp/tp 必须与真实拓扑一致，代理不能继续路由到已经不存在的 7102/7103。</li><li><b>重现 CI 与隔离版本回归是两种实验：</b>先复现各自受支持配置；再设计受支持的控制组，分别改变 Runner、拓扑和 connector。PCP4 不可直接强制 V1，不能把不支持的组合当成控制组。</li></ul><h3>SWA replay 不能视为本配置必然获得的收益</h3><p>两份 CI 均没有显式传入 replay 开关。官方 0.30rc2 tag 的 Ascend 代码未提供本次回移的 replay 接口，不能把“回移分支默认 OFF”套用于原始官方 tag。0.32 上游默认值为 ON，但该 CI 的 D 端 <code>recompute_scheduler_enable=true</code> 会触发 Ascend 平台降级关闭 replay；P 端若实际启用 SP，则 decoder 裁剪也受源码 gate 限制。应查看最终配置和 warning，不将开关默认值等同于实际裁剪。</p><p>保留模型、权重校验值、采样参数、输入/输出长度、请求数、并发、请求率、预热和 prefix hit 状态；记录 TTFT、TPOT/ITL、输出吞吐、成功率、实际输出长度、接受率、KV 容量及抢占。本文没有新的性能数字。</p></section>
<section id="sources"><h2>可追溯来源与验证边界</h2><ul>{source_html}</ul><h3>核对过的关键代码</h3><ul>{code_html}</ul><p>构建命令：<code>python reports/pd_configuration_comparison/build_report.py</code>，依赖 PyYAML。原文 SHA-256、规范化配置、全部行级差异见 comparison.json 和 manifest.json；本地静态/浏览器验证见 validation.json。本报告没有运行官方 CI、没有拉取大体积镜像、没有执行 NPU/HCCL 或模型推理。</p></section><footer>2026-11-03 · 固定来源快照 · 全文及筛选离线可用，外部源码链接需要联网。</footer></main>
<script>
'use strict';const controls=['changed-only','role','search'].map(id=>document.getElementById(id));const rows=[...document.querySelectorAll('#ci-table tbody tr')];function filter(){{const q=controls[2].value.trim().toLowerCase();let visible=0;for(const row of rows){{const show=(!controls[0].checked||row.dataset.changed==='1')&&(!controls[1].value||row.dataset.role===controls[1].value)&&(!q||row.textContent.toLowerCase().includes(q));row.hidden=!show;if(show)visible++;}}document.getElementById('count').textContent=`显示 ${{visible}} / ${{rows.length}} 项`;}}for(const control of controls)control.addEventListener('input',filter);filter();
</script></body></html>"""
    (HERE / "report.html").write_text(html_text)
    manifest = {
        str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(HERE.rglob("*"))
        if p.is_file()
        and p.name not in {"manifest.json", "validation.json", "preview.png"}
        and "__pycache__" not in p.parts
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"rows": len(rows), "changed": sum(row["changed"] for row in rows), "benchmark_changes": benchmarks},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
