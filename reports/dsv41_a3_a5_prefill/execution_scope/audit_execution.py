#!/usr/bin/env python3
"""Audit archived Worker-local execution using causal Host/device correlations.

Inputs: selectively extracted DP0/TP0 database and kernel_details.csv under
/tmp/dsv41-execution-boundaries; the original archives are never modified.
All arithmetic starts with integer epoch nanoseconds. Python/module intervals
identify ownership; their Host return is never a device completion marker.
"""

from __future__ import annotations

import collections
import csv
import gzip
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = Path("/tmp/dsv41-execution-boundaries")


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged) / 1e6


def audit(db, baseline, *, raw_root=RAW):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    strings = dict(conn.execute("SELECT id,value FROM STRING_IDS"))
    origin = int(conn.execute("SELECT * FROM SESSION_TIME_INFO").fetchone()[0])
    base = origin + round(baseline["prep"]["start_us"] * 1000)
    ops = [
        dict(s=int(s), e=int(e), tid=tid, connection=cid, name=strings[n])
        for s, e, tid, cid, n in conn.execute(
            "SELECT startNs,endNs,globalTid,connectionId,name FROM PYTORCH_API WHERE type=50001"
        )
    ]
    chosen = {
        i: n
        for i, n in strings.items()
        if (n.startswith("nn.Module:") and any(x in n for x in ["DeepseekV41", "Sampler"]))
        or any(
            n.endswith(": " + x)
            for x in [
                "execute_model",
                "sample_tokens",
                "sample",
                "_sample",
                "compute_logits",
                "propose",
                "propose_draft_token_ids",
                "_propose",
                "prepare_inputs",
                "_prepare_inputs",
                "prepare_attn",
                "_build_metadata",
                "_build_attention_metadata",
                "_build_attn_group_metadata",
                "dispatch_cg_and_sync_dp",
                "_update_states",
                "_prepare_inputs_impl",
                "hc_collapse",
                "prepare",
                "_prepare_compact_replay",
                "build_attn_metadata",
                "_precompute_context_kv",
                "precompute_and_store_context_kv",
                "_build_uniform_attn_metadata",
                "_sample_sequential",
                "build_draft_attn_metadata",
                "_run_merged_draft",
                "_sample_from_logits",
                "post_forward",
            ]
        )
    }
    scopes = [
        dict(s=int(s), e=int(e), tid=tid, name=strings[n])
        for s, e, tid, n in conn.execute(
            "SELECT startNs,endNs,globalTid,name FROM PYTORCH_API WHERE type=50003 AND name IN ("
            + ",".join("?" for _ in chosen)
            + ")",
            list(chosen),
        )
    ]
    scopes.sort(key=lambda x: (x["s"], -x["e"]))
    hc_start = origin + round(baseline["host_start_us"] * 1000)
    runner = min(
        (
            s
            for s in scopes
            if s["name"].endswith(": execute_model") and "model_runner" in s["name"] and s["s"] <= hc_start < s["e"]
        ),
        key=lambda x: x["s"],
    )
    assert runner["s"] == base
    tid = runner["tid"]
    target = next(
        s for s in scopes if s["name"] == "nn.Module: AscendDeepseekV41ForCausalLM_0" and base <= s["s"] < runner["e"]
    )
    sample = next(
        s
        for s in scopes
        if s["name"].endswith(": sample_tokens")
        and "model_runner" in s["name"]
        and runner["e"] <= s["s"] < runner["e"] + 100_000_000
    )
    draft = next(
        s for s in scopes if s["name"] == "nn.Module: DeepseekV41DSparkModel_0" and sample["s"] <= s["s"] < sample["e"]
    )
    layers = [
        s
        for s in scopes
        if s["name"].startswith("nn.Module: DeepseekV41DecoderLayer_")
        and s["name"].removeprefix("nn.Module: DeepseekV41DecoderLayer_").isdecimal()
        and target["s"] <= s["s"] < target["e"]
    ]
    draft_layers = [
        s
        for s in scopes
        if s["name"].startswith("nn.Module: DeepseekV41DSparkDecoderLayer_")
        and s["name"].removeprefix("nn.Module: DeepseekV41DSparkDecoderLayer_").isdecimal()
        and draft["s"] <= s["s"] < draft["e"]
    ]
    assert len(layers) == 40 and len(draft_layers) == 3
    proposers = [
        s
        for s in scopes
        if sample["s"] <= s["s"] < draft["s"]
        and (
            (s["name"].endswith(": propose") and "dspark/speculator" in s["name"])
            or s["name"].endswith(": propose_draft_token_ids")
        )
    ]
    propose = min(proposers, key=lambda x: x["s"])
    bounds = [
        ("worker_prepare", base, target["s"]),
        ("target_input", target["s"], layers[0]["s"]),
        ("target_layers", layers[0]["s"], layers[-1]["e"]),
        ("target_tail", layers[-1]["e"], target["e"]),
        ("runner_after_forward", target["e"], sample["s"]),
        ("target_sampling", sample["s"], propose["s"]),
        ("draft_prepare", propose["s"], draft_layers[0]["s"]),
        ("draft_layers", draft_layers[0]["s"], draft_layers[-1]["e"]),
        ("draft_head_and_output", draft_layers[-1]["e"], sample["e"]),
    ]
    # CONNECTION_IDS links the PyTorch group ID to runtime raw connection IDs.
    by_group = {o["connection"]: o for o in ops if o["connection"] is not None}
    raw_host = {}
    for group, raw in conn.execute("SELECT id,connectionId FROM CONNECTION_IDS"):
        if group in by_group:
            raw_host[raw] = {**by_group[group], "link": "torch_connection"}
    cann = [
        dict(s=int(s), e=int(e), tid=t, connection=cid, name=strings[n])
        for s, e, t, cid, n in conn.execute("SELECT startNs,endNs,globalTid,connectionId,name FROM CANN_API")
    ]
    for c in cann:
        if c["connection"] not in raw_host and c["tid"] == tid:
            raw_host[c["connection"]] = {**c, "link": "cann_main_thread"}
    tasks = collections.defaultdict(list)
    for s, e, cid, gid, stream, task in conn.execute(
        "SELECT startNs,endNs,connectionId,globalTaskId,streamId,taskId FROM TASK"
    ):
        tasks[int(s)].append(dict(connection=cid, gid=gid, stream=stream, task=task))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "COMMUNICATION_OP" in tables:
        for s, e, cid in conn.execute("SELECT startNs,endNs,connectionId FROM COMMUNICATION_OP"):
            tasks[int(s)].append(dict(connection=cid, gid=None, stream=None, task=None))

    def at(ns):
        return (ns - base) / 1e6

    rows = []
    for line, row in enumerate(csv.DictReader((db.parent / "kernel_details.csv").open()), 2):
        if row["Accelerator Core"] == "COMMUNICATION" and not row["Name"].startswith("hcom_"):
            continue
        s = int(Decimal(row["Start Time(us)"]) * 1000)
        e = s + round(Decimal(row["Duration(us)"]) * 1000)
        host = None
        task = {}
        for t in sorted(tasks[s], key=lambda x: raw_host.get(x["connection"], {}).get("link") != "torch_connection"):
            if t["connection"] in raw_host:
                host, task = raw_host[t["connection"]], t
                break
        phase = next((p for p, a, b in bounds if host and a <= host["s"] < b), None)
        rows.append(
            dict(
                s=s,
                e=e,
                line=line,
                name=row["Name"],
                type=row["Type"],
                stream=row["Stream ID"],
                task=row["Task ID"],
                core=row["Accelerator Core"],
                shape=row["Input Shapes"],
                dtype=row["Input Data Types"],
                output=row["Output Shapes"],
                phase=phase,
                raw_connection=task.get("connection"),
                host_name=host["name"] if host else None,
                host_s=host["s"] if host else None,
                link=host["link"] if host else None,
            )
        )

    def event(row):
        return dict(
            type=row["type"],
            name=row["name"],
            csv_line=row["line"],
            stream=row["stream"],
            task=row["task"],
            connection=row["raw_connection"],
            start_ms=at(row["s"]),
            end_ms=at(row["e"]),
            host_start_ms=at(row["host_s"]) if row["host_s"] is not None else None,
            host_name=row["host_name"],
            link=row["link"],
            shape=row["shape"],
            dtype=row["dtype"],
            core=row["core"],
            output=row["output"],
        )

    def metrics(selected):
        if not selected:
            return dict(count=0)
        comp = [r for r in selected if r["core"] != "COMMUNICATION"]
        core = [r for r in selected if r["core"] in ("AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV")]
        busy = union((r["s"], r["e"]) for r in selected)
        cbusy = union((r["s"], r["e"]) for r in comp)
        span = (max(r["e"] for r in selected) - min(r["s"] for r in selected)) / 1e6
        return dict(
            count=len(selected),
            start_ms=min(at(r["s"]) for r in selected),
            end_ms=max(at(r["e"]) for r in selected),
            span_ms=span,
            kernel_union_ms=cbusy,
            comm_only_ms=busy - cbusy,
            idle_ms=span - busy,
            aicore_union_ms=union((r["s"], r["e"]) for r in core),
            aicpu_union_ms=union((r["s"], r["e"]) for r in selected if r["core"] == "AI_CPU"),
            kernel_sum_ms=sum((r["e"] - r["s"]) / 1e6 for r in comp),
            first=event(min(selected, key=lambda r: r["s"])),
            last=event(max(selected, key=lambda r: r["e"])),
        )

    phases = []
    for p, a, b in bounds:
        rs = [r for r in rows if r["phase"] == p]
        types = collections.defaultdict(list)
        for r in rs:
            types[r["type"]].append(r)
        phases.append(
            dict(
                id=p,
                host_start_ms=at(a),
                host_end_ms=at(b),
                host_wall_ms=(b - a) / 1e6,
                device=metrics(rs),
                operators=[
                    dict(type=t, count=len(v), sum_ms=sum((r["e"] - r["s"]) / 1e6 for r in v))
                    for t, v in sorted(types.items(), key=lambda kv: -sum(r["e"] - r["s"] for r in kv[1]))
                ],
            )
        )
    # Per-layer ownership is determined by captured module intervals, never by
    # taking the next device task in timestamp order.
    layer_rows = []
    for idx, layer in enumerate(layers + draft_layers):
        rs = [r for r in rows if r["host_s"] is not None and layer["s"] <= r["host_s"] < layer["e"]]
        layer_rows.append(
            dict(
                layer=idx,
                role="target" if idx < 40 else "draft",
                scope=layer["name"],
                host_start_ms=at(layer["s"]),
                host_end_ms=at(layer["e"]),
                device=metrics(rs),
            )
        )
    # Recorded Python scope hierarchy for boundary task attribution.
    boundary_rows = []
    for p in phases:
        for edge in ("first", "last"):
            ev = p["device"].get(edge)
            if not ev:
                continue
            h = base + round(ev["host_start_ms"] * 1e6)
            parents = [
                dict(name=strings[n], start_ms=at(int(s)), end_ms=at(int(e)))
                for s, e, n in conn.execute(
                    "SELECT startNs,endNs,name FROM PYTORCH_API WHERE type=50003 AND globalTid=? "
                    "AND startNs<=? AND endNs>=?",
                    (tid, str(h), str(h)),
                )
            ]
            parents.sort(key=lambda x: (x["start_ms"], -x["end_ms"]))
            boundary_rows.append(dict(phase=p["id"], edge=edge, task=ev, python_parents=parents))

    anchors = []
    for phase, typ in [
        ("worker_prepare", "RepeatInterleave"),
        ("worker_prepare", "SparseFlashMlaMetadata"),
        ("target_layers", "HcPre"),
        ("target_layers", "MegaMoe"),
        ("target_layers", "SparseFlashMla"),
        ("draft_layers", "DispatchFFNCombine"),
        ("draft_layers", "GroupedMatmul"),
    ]:
        candidates = [r for r in rows if r["phase"] == phase and r["type"] == typ]
        if not candidates:
            continue
        r = min(candidates, key=lambda r: r["s"])
        h = r["host_s"]
        parents = [
            strings[n]
            for s, e, n in sorted(
                conn.execute(
                    "SELECT startNs,endNs,name FROM PYTORCH_API WHERE type=50003 AND globalTid=? "
                    "AND startNs<=? AND endNs>=?",
                    (tid, str(h), str(h)),
                ),
                key=lambda x: (int(x[0]), -int(x[1])),
            )
        ]
        anchors.append(dict(phase=phase, task=event(r), python_parents=parents))

    compact_scopes = [
        dict(name=s["name"], start_ms=at(s["s"]), end_ms=at(s["e"]), tid=s["tid"])
        for s in scopes
        if base <= s["s"] < sample["e"] and s["tid"] == tid
    ]
    runtime = [
        dict(name=c["name"], start_ms=at(c["s"]), end_ms=at(c["e"]), tid=c["tid"])
        for c in cann
        if base <= c["s"] < sample["e"] and any(x in c["name"].lower() for x in ["synchroniz", "memcpy"])
    ]
    selected = [r for r in rows if r["phase"]]
    unmapped = [r for r in rows if r["host_s"] is None and r["e"] > base and r["s"] < max(r["e"] for r in selected)]
    data = dict(
        database_archive_member=str(db.relative_to(raw_root)) if db.is_relative_to(raw_root) else str(db),
        origin_ns=origin,
        worker_entry_epoch_ns=base,
        worker_entry_session_ms=(base - origin) / 1e6,
        runner=runner["name"],
        runner_return_ms=at(runner["e"]),
        sample_return_ms=at(sample["e"]),
        phases=phases,
        layers=layer_rows,
        model_path_device=metrics(selected),
        scopes=compact_scopes,
        boundary_stacks=boundary_rows,
        operator_anchors=anchors,
        runtime_sync_and_copy=runtime,
        correlation=dict(
            total_rows=len(rows),
            matched_rows=sum(r["host_s"] is not None for r in rows),
            selected_rows=len(selected),
            unmapped_in_device_window=len(unmapped),
            unmapped_types=dict(collections.Counter(r["type"] for r in unmapped)),
            unmapped_tasks=[event(r) for r in unmapped],
        ),
        previous_work_before_first_target=metrics(
            [
                r
                for r in rows
                if r["host_s"] is not None
                and r["host_s"] < base
                and r["e"] > base
                and r["s"] < next(p for p in phases if p["id"] == "target_layers")["device"]["start_ms"] * 1e6 + base
            ]
        ),
    )
    conn.close()
    return data, [dict(phase=r["phase"], **event(r)) for r in selected]


def main():
    previous = json.loads(gzip.decompress((HERE.parent / "data/report_data.json.gz").read_bytes()))
    output = {
        "method": (
            "Worker-local ownership by captured Python/module intervals and connection IDs; "
            "milliseconds relative to each selected Worker entry. Kernel sums are not latency savings."
        ),
        "samples": {},
    }
    for label, rid, idx, folder in [("A3", "a3_recent-dp0-tp0", 1, "P"), ("A5", "a5_bs12-dp0-tp0", 2, "vllm_profile")]:
        rank = next(r for r in previous["ranks"] if r["id"] == rid)
        db = next((RAW / folder).rglob("ascend_pytorch_profiler_0.db"))
        evidence, rows = audit(db, rank["steps"][idx])
        evidence.update(
            rank_id=rid,
            step_index=idx,
            global_round=rank["steps"][idx]["global_round"],
            baseline_metrics=rank["steps"][idx]["metrics"],
        )
        output["samples"][label] = evidence
        with (HERE / f"{label.lower()}_tasks.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(
            label,
            "phases",
            [
                (
                    p["id"],
                    round(p["host_wall_ms"], 3),
                    round(p["device"].get("span_ms", 0), 3),
                    p["device"].get("count"),
                )
                for p in evidence["phases"]
            ],
        )
        print(label, "end", evidence["model_path_device"]["end_ms"], "coverage", evidence["correlation"])
    output["ep_first_mega_alignment"] = {}
    for dataset, rd in [("a3_recent", 1), ("a5_bs12", 2)]:
        points = []
        for rank in previous["ranks"]:
            if rank["dataset"] != dataset:
                continue
            step = next((s for s in rank["steps"] if s.get("global_round") == rd), None)
            if not step:
                continue
            mega = step["layers"][0]["mega"][0]
            points.append(
                dict(
                    rank=rank["id"],
                    kind=step["kind"],
                    start_ns=rank["session_ns"][0] + round(mega["s"] * 1000),
                    end_ns=rank["session_ns"][0] + round(mega["e"] * 1000),
                )
            )
        start = min(p["start_ns"] for p in points)
        output["ep_first_mega_alignment"][dataset] = dict(
            arrival_spread_ms=(max(p["start_ns"] for p in points) - start) / 1e6,
            completion_spread_ms=(max(p["end_ns"] for p in points) - min(p["end_ns"] for p in points)) / 1e6,
            ranks=[
                dict(
                    rank=p["rank"],
                    kind=p["kind"],
                    start_ms=(p["start_ns"] - start) / 1e6,
                    end_ms=(p["end_ns"] - start) / 1e6,
                )
                for p in points
            ],
        )
    (HERE / "evidence.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
