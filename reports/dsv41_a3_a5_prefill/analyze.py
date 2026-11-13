#!/usr/bin/env python3
"""Analyze local Ascend exports, keeping step attribution and evidence auditable.

Requires only the Python standard library. Inputs remain outside the report.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import re
import sqlite3
import statistics
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = Path("/tmp/dsv41-a3-a5-prefill-inputs")


def num(x):
    try:
        v = float(str(x).strip().strip('"'))
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def merge(intervals):
    result = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if result and s <= result[-1][1]:
            result[-1][1] = max(result[-1][1], e)
        else:
            result.append([s, e])
    return result


def union(intervals):
    return sum(e - s for s, e in merge(intervals))


def summary(values):
    if not values:
        return {}
    vals = sorted(values)

    def pct(q):
        p = (len(vals) - 1) * q
        i = int(p)
        return vals[i] + (vals[min(i + 1, len(vals) - 1)] - vals[i]) * (p - i)

    return dict(
        n=len(vals),
        total=sum(vals),
        mean=statistics.mean(vals),
        median=statistics.median(vals),
        p95=pct(0.95),
        minimum=vals[0],
        maximum=vals[-1],
        cv=statistics.pstdev(vals) / statistics.mean(vals) if statistics.mean(vals) else 0,
    )


def token(shape):
    match = re.match(r"\D*(\d+)", shape)
    return int(match[1]) if match else None


def family(typ):
    if any(
        x in typ.lower()
        for x in ["megamoe", "dispatchffn", "groupedmatmul", "moeinit", "moedispatch", "moetoken", "moecombine"]
    ):
        return "MoE"
    if any(x in typ.lower() for x in ["lightning", "mla", "flashattention", "indexer", "sparseattention"]):
        return "Attention/Indexer"
    if "HcPre" in typ or "HcPost" in typ or "MhcPost" in typ:
        return "HC Pre/Post"
    if typ.startswith("hcom_"):
        return "通信"
    if any(x in typ.lower() for x in ["matmul", "gemm", "quant"]):
        return "矩阵乘/量化"
    return "其他小算子"


def aggregate(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["type"]].append(r)
    return [
        dict(
            type=k,
            count=len(rs),
            sum_us=sum(r["dur_us"] for r in rs),
            median_us=statistics.median(r["dur_us"] for r in rs),
            max_us=max(r["dur_us"] for r in rs),
            family=family(k),
            shapes=collections.Counter(r["shape"] for r in rs).most_common(4),
        )
        for k, rs in sorted(groups.items(), key=lambda kv: -sum(r["dur_us"] for r in kv[1]))
    ]


def window_metrics(rows):
    if not rows:
        return {}
    s, e = min(r["s"] for r in rows), max(r["e"] for r in rows)
    comp = union((r["s"], r["e"]) for r in rows if not r["comm"])
    comm = union((r["s"], r["e"]) for r in rows if r["comm"])
    busy = union((r["s"], r["e"]) for r in rows)
    aicore = union((r["s"], r["e"]) for r in rows if r["core"] in ("AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV"))
    aicpu = union((r["s"], r["e"]) for r in rows if r["core"] == "AI_CPU")
    active_without_aicpu = union((r["s"], r["e"]) for r in rows if r["core"] != "AI_CPU")
    return dict(
        start_us=s,
        end_us=e,
        span_us=e - s,
        compute_us=comp,
        comm_us=comm,
        comm_only_us=busy - comp,
        overlap_us=comp + comm - busy,
        free_us=e - s - busy,
        free_pct=100 * (e - s - busy) / (e - s),
        count=len(rows),
        kernel_count=sum(not r["comm"] for r in rows),
        aicore_busy_us=aicore,
        aicpu_busy_us=aicpu,
        gap_excluding_aicpu_us=e - s - active_without_aicpu,
    )


def host_summary(ops, s, e):
    selected = [o for o in ops if s <= o["s"] < e]
    groups = collections.defaultdict(list)
    for o in selected:
        groups[o["name"]].append(o["e"] - o["s"])
    return dict(
        count=len(selected),
        top=[dict(name=k, **summary(v)) for k, v in sorted(groups.items(), key=lambda kv: -sum(kv[1]))[:25]],
    )


def analyze(path, dataset):
    match = re.search(r"dp(\d+)_pp(\d+)_tp(\d+)_dcp(\d+)_ep(\d+)_rank(\d+)", str(path))
    rank = dict(zip(["dp", "pp", "tp", "dcp", "ep", "rank"], map(int, match.groups())))
    key = f"{dataset}-dp{rank['dp']}-tp{rank['tp']}"
    db = next(path.glob("ascend_pytorch_profiler_*.db"))
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    strings = dict(conn.execute("SELECT id,value FROM STRING_IDS"))
    meta = dict(conn.execute("SELECT * FROM META_DATA"))
    session = conn.execute("SELECT * FROM SESSION_TIME_INFO").fetchone()
    origin = int(session[0])

    # Integer nanoseconds first; absolute epoch floats would lose sub-us precision.
    def rel(ns):
        return (int(ns) - origin) / 1000

    ops = [
        dict(s=rel(s), e=rel(e), tid=tid, connection=connection, name=strings.get(name, str(name)))
        for s, e, tid, connection, name in conn.execute(
            "SELECT startNs,endNs,globalTid,connectionId,name FROM PYTORCH_API WHERE type=50001"
        )
    ]
    ops.sort(key=lambda o: o["s"])
    groups = collections.defaultdict(list)
    for o in ops:
        groups[o["name"]].append(o)
    hcpre = groups.get("_C_ascend::npu_hc_pre_v3", []) or groups.get("custom::npu_hc_pre_v2", [])
    hcpost = groups.get("_C_ascend::npu_hc_post", []) or groups.get("cann_ops_transformer::mhc_post", [])
    dsa = groups.get("vllm::dsa_v41_forward", [])
    if dataset == "a3_030":
        # This dataset is context only; different workload and no MegaMoE.
        conn.close()
        return dict(
            id=key, dataset=dataset, rank=rank, path=str(path), metadata=meta, session_ns=session, context_only=True
        )
    assert len(dsa) % 43 == 0, (key, len(dsa))
    assert len(hcpre) == len(hcpost) == len(dsa) * 2, (key, len(hcpre), len(hcpost), len(dsa))
    # A PYTORCH_API connectionId indexes CONNECTION_IDS; TASK uses its raw value.
    raw_to_host = {}
    host_connections = {o["connection"]: o for o in ops if o["connection"] is not None}
    for group_id, raw_id in conn.execute("SELECT id,connectionId FROM CONNECTION_IDS"):
        if group_id in host_connections:
            raw_to_host[raw_id] = {**host_connections[group_id], "link_method": "torch_connection"}
    # C++/custom kernels can bypass PyTorch op recording. A same-main-thread CANN
    # launch still gives a causal host launch timestamp, unlike nearest-time guessing.
    main_tids = {o["tid"] for o in hcpre}
    for start, end, tid, raw_id, name_id in conn.execute(
        "SELECT startNs,endNs,globalTid,connectionId,name FROM CANN_API"
    ):
        if raw_id not in raw_to_host and tid in main_tids:
            raw_to_host[raw_id] = dict(
                s=rel(start), e=rel(end), tid=tid, name=strings.get(name_id, ""), link_method="cann_main_thread"
            )
    task_by_start = collections.defaultdict(list)
    for start, end, raw_id, gid in conn.execute("SELECT startNs,endNs,connectionId,globalTaskId FROM TASK"):
        task_by_start[int(start)].append((raw_id, gid))
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "COMMUNICATION_OP" in tables:
        for start, end, raw_id in conn.execute("SELECT startNs,endNs,connectionId FROM COMMUNICATION_OP"):
            task_by_start[int(start)].append((raw_id, None))
    # Only selected Python scopes are read. P.rar also contains ~1.8M Python frames/rank.
    scope_ids = [
        i
        for i, n in strings.items()
        if (
            (("model_runner" in n) and n.endswith(": execute_model"))
            or n.endswith(": _dummy_run")
            or n.endswith(": _prepare_inputs_impl")
            or n.endswith(": prepare_inputs")
            or n.endswith(": update_requests")
        )
    ]
    scopes = []
    for part in range(0, len(scope_ids), 100):
        ids = scope_ids[part : part + 100]
        scopes.extend(
            dict(s=rel(s), e=rel(e), name=strings[n])
            for s, e, n in conn.execute(
                "SELECT startNs,endNs,name FROM PYTORCH_API WHERE type=50003 AND name IN ("
                + ",".join("?" for _ in ids)
                + ")",
                ids,
            )
        )
    npu = conn.execute("SELECT * FROM NPU_INFO").fetchall()
    trace_count = conn.execute("SELECT count(*) FROM PYTORCH_API WHERE type=50003").fetchone()[0]
    conn.close()
    rows = []
    for lineno, row in enumerate(csv.DictReader((path / "kernel_details.csv").open()), 2):
        if row["Accelerator Core"] == "COMMUNICATION" and not row["Name"].startswith("hcom_"):
            continue  # Child communication tasks are nested under communication OPs.
        ns = int(Decimal(row["Start Time(us)"].strip()) * 1000)
        s = rel(ns)
        dur = float(row["Duration(us)"])
        point = None
        gid = None
        link_method = "unmatched"
        candidates = sorted(
            task_by_start.get(ns, []),
            key=lambda item: raw_to_host.get(item[0], {}).get("link_method") != "torch_connection",
        )
        for raw_id, gid0 in candidates:
            if raw_id in raw_to_host:
                point = raw_to_host[raw_id]["s"]
                gid = gid0
                link_method = raw_to_host[raw_id]["link_method"]
                break
        rows.append(
            dict(
                s=s,
                e=s + dur,
                dur_us=dur,
                type=row["Type"],
                name=row["Name"],
                shape=row["Input Shapes"],
                core=row["Accelerator Core"],
                dtype=row.get("Input Data Types", ""),
                stream=row["Stream ID"],
                comm=row["Accelerator Core"] == "COMMUNICATION",
                line=lineno,
                host_s=point,
                gid=gid,
                link_method=link_method,
                pmu={
                    k: num(row.get(k))
                    for k in [
                        "aicore_time(us)",
                        "aic_mac_ratio",
                        "aic_mte2_ratio",
                        "aiv_vec_ratio",
                        "aiv_mte2_ratio",
                        "cube_utilization(%)",
                    ]
                },
            )
        )
    rows.sort(key=lambda r: r["s"])
    steps = []
    for i in range(len(dsa) // 43):
        pre = hcpre[i * 86 : (i + 1) * 86]
        post = hcpost[i * 86 : (i + 1) * 86]
        start, end = pre[0]["s"], post[-1]["e"]
        main_end = post[79]["e"]
        sr = [r for r in rows if r["host_s"] is not None and start <= r["host_s"] < end]
        main = [r for r in sr if r["host_s"] < main_end]
        hc = [r for r in main if r["type"] == "HcPre"]
        # Multiple host wrapper names can refer to one launch; all durations use hardware rows once.
        counts = collections.Counter(token(r["shape"]) for r in hc)
        real_pre = [
            o
            for o in groups.get("_prepare_prefill_inputs_kernel", [])
            if (steps[-1]["host_end_us"] if steps else -1e20) < o["s"] < start
        ]
        execute = [o for o in scopes if o["name"].endswith(": execute_model") and o["s"] <= start < o["e"]]
        dummy = [o for o in scopes if o["name"].endswith(": _dummy_run") and o["s"] <= start < o["e"]]
        # The exact execute_model scope is absent from the older ZIP. Gloo is a reconstruction anchor only.
        previous = steps[-1]["host_end_us"] if steps else -1e20
        sync = [o for o in groups.get("gloo:all_reduce", []) if previous < o["s"] < start]
        prep_start = min(o["s"] for o in execute) if execute else min([o["s"] for o in sync + real_pre] + [start])
        prep_rows = [r for r in rows if r["host_s"] is not None and prep_start <= r["host_s"] < start]
        layers = []
        for j in range(43):
            ls, le = pre[j * 2]["s"], post[j * 2 + 1]["e"]
            lr = [r for r in sr if ls <= r["host_s"] < le]
            lhc = [r for r in lr if r["type"] == "HcPre"]
            mega = [r for r in lr if r["type"] == "MegaMoe"]
            layers.append(
                dict(
                    layer=j,
                    role="target" if j < 40 else "draft",
                    tokens=token(lhc[0]["shape"]) if lhc else None,
                    metrics=window_metrics(lr),
                    mega=mega,
                    operators=aggregate(lr),
                )
            )
        metric = window_metrics(main)
        # Device-gap attribution is observational: overlapping a CPU op does not establish a dependency.
        gaps = []
        merged = merge((r["s"], r["e"]) for r in main)
        gap_intervals = sorted(
            ((left[1], right[0]) for left, right in zip(merged, merged[1:]) if right[0] - left[1] >= 10),
            key=lambda ab: ab[0] - ab[1],
        )[:20]
        for a, b in gap_intervals:
            overlaps = collections.defaultdict(float)
            for o in ops:
                ov = min(b, o["e"]) - max(a, o["s"])
                if ov > 0:
                    overlaps[o["name"]] += ov
            gaps.append(
                dict(
                    start_us=a,
                    end_us=b,
                    duration_us=b - a,
                    host_overlap=sorted(overlaps.items(), key=lambda kv: -kv[1])[:5],
                )
            )
        is_prefill = bool(real_pre) or bool(execute and max([k or 0 for k in counts], default=0) > 6)
        steps.append(
            dict(
                index=i,
                kind="dummy" if dummy else ("prefill" if is_prefill else "unclassified"),
                host_start_us=start,
                host_end_us=end,
                host_main_end_us=main_end,
                execute_scopes=execute,
                dummy_scopes=dummy,
                prefill_prepares=real_pre,
                main_tokens=dict(counts),
                metrics=metric,
                all_forward_metrics=window_metrics(sr),
                operators=aggregate(main),
                host=host_summary(ops, start, main_end),
                layers=layers,
                prep=dict(
                    start_us=prep_start,
                    end_us=start,
                    elapsed_us=start - prep_start,
                    scope="exact execute_model entry" if execute else "reconstructed sync/input anchor",
                    gloo=sync,
                    metrics=window_metrics(prep_rows),
                    operators=aggregate(prep_rows),
                    host=host_summary(ops, prep_start, start),
                ),
                gaps=sorted(gaps, key=lambda g: -g["duration_us"])[:20],
                correlation_coverage=dict(rows=len(sr), hcpre_main=len(hc)),
            )
        )
    step_file = list(csv.DictReader((path / "step_trace_time.csv").open()))
    pm = path.parent / "profiler_info_0.json"
    if not pm.exists():
        pm = next(path.parent.glob("profiler_info_*.json"), pm)
    info = json.loads(pm.read_text()) if pm.exists() else {}
    # Compact timeline retains every attributed hardware row, not just a downsample.
    timeline = [
        dict(
            s=round(r["s"], 3),
            d=r["dur_us"],
            type=r["type"],
            name=r["name"],
            shape=r["shape"],
            stream=r["stream"],
            line=r["line"],
            host_s=r["host_s"],
            pmu=r["pmu"],
            core=r["core"],
            link_method=r["link_method"],
        )
        for r in rows
    ]
    result = dict(
        id=key,
        dataset=dataset,
        rank=rank,
        path=str(path),
        metadata=meta,
        npu=npu,
        profiler_info=info,
        session_ns=session,
        python_trace_count=trace_count,
        whole_metrics=window_metrics(rows),
        exported_step_summary=step_file,
        operators=aggregate(rows),
        steps=steps,
        matched_hardware_rows=sum(r["host_s"] is not None for r in rows),
        total_hardware_rows=len(rows),
        attribution_methods=dict(collections.Counter(r["link_method"] for r in rows)),
        timeline=timeline,
    )
    print(
        key,
        "steps",
        [
            (
                s["kind"],
                s["main_tokens"],
                round(s["metrics"].get("span_us", 0) / 1000, 2),
                round(s["metrics"].get("free_pct", 0), 2),
            )
            for s in steps
        ],
        flush=True,
    )
    return result


def main():
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    ranks = []
    sources = [
        ("a3_032", sorted((RAW / "a3").glob("*0.32*/**/ASCEND_PROFILER_OUTPUT"))),
        ("a3_recent", sorted((RAW / "a3_recent/P").glob("dp*/ASCEND_PROFILER_OUTPUT"))),
        ("a5_bs12", sorted((RAW / "a5_bs12").glob("*/dp*/ASCEND_PROFILER_OUTPUT"))),
    ]
    for label, paths in sources:
        for path in paths:
            r = analyze(path, label)
            (out / f"{r['id']}.json").write_text(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
            ranks.append({k: v for k, v in r.items() if k != "timeline"})
    manifest = []
    for file in [
        "/home/gengc/Downloads/20261026-DSv4.1-1P1D-profiler.zip",
        "/home/gengc/Downloads/P.rar",
        "/home/gengc/Downloads/vllm_profile_prefill-c192-bs12.zip",
    ]:
        p = Path(file)
        h = hashlib.sha256()
        with p.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                h.update(chunk)
        manifest.append(dict(path=file, bytes=p.stat().st_size, sha256=h.hexdigest()))
    (out / "analysis.json").write_text(json.dumps(dict(archives=manifest, ranks=ranks), ensure_ascii=False, indent=2))
    print("saved", out / "analysis.json", flush=True)


if __name__ == "__main__":
    main()
