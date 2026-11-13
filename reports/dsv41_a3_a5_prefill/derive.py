#!/usr/bin/env python3
"""Build A3/A5 comparison evidence, using confirmed archive identities."""

import csv
import gzip
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

from analyze import HERE, summary, union

PRIMARY_A3 = "a3_recent-dp0-tp0"
PRIMARY_A5 = "a5_bs12-dp0-tp0"
A3_STEP = 1
A5_STEP = 2


def display_type(name):
    if "MqsmlaKernel" in name:
        return "MQ-SMLA (custom kernel)"
    if "mqsmla_metadata" in name.lower():
        return "MQ-SMLA Metadata (custom)"
    if "QuantSparseLightningIndexer" in name:
        return "QLI Sparse (custom kernel)"
    if "Quant_lightning_indexer" in name:
        return "QLI Dense (custom kernel)"
    if "metadata_kernel" in name:
        return "Custom Metadata kernel"
    return name


def write_csv(name, rows):
    if not rows:
        return
    with (HERE / "data" / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def operator_map(operators):
    groups = {}
    for op in operators:
        key = display_type(op["type"])
        out = groups.setdefault(key, dict(count=0, sum_us=0, raw_types=[], shapes=[]))
        out["count"] += op["count"]
        out["sum_us"] += op["sum_us"]
        out["raw_types"].append(op["type"])
        out["shapes"].extend(op["shapes"][:1])
    return groups


def shape_groups(rank, step):
    s = rank["steps"][step]
    lo = s["layers"][21]["metrics"]["start_us"]
    hi = s["layers"][39]["metrics"]["end_us"]
    origin = rank["session_ns"][0]
    groups = {}
    for line, row in enumerate(csv.DictReader((Path(rank["path"]) / "kernel_details.csv").open()), 2):
        t = (int(Decimal(row["Start Time(us)"].strip()) * 1000) - origin) / 1000
        if lo <= t <= hi and row["Accelerator Core"] != "AI_CPU" and not row["Type"].startswith("hcom_"):
            key = (row["Type"], row["Input Shapes"], row["Input Data Types"])
            groups.setdefault(key, []).append((float(row["Duration(us)"]), line))
    return groups


def call_parents(rank, step, pattern, in_prep=False):
    scope = rank["steps"][step]
    origin = rank["session_ns"][0]
    low = scope["prep"]["start_us"] if in_prep else scope["host_start_us"]
    high = scope["host_start_us"] if in_prep else scope["host_main_end_us"]
    db = next(Path(rank["path"]).glob("ascend_pytorch_profiler_*.db"))
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    op = conn.execute(
        "SELECT a.startNs,a.globalTid FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name "
        "WHERE a.type=50001 AND s.value LIKE ? AND CAST(a.startNs AS INTEGER)>=? "
        "AND CAST(a.startNs AS INTEGER)<? ORDER BY CAST(a.startNs AS INTEGER) LIMIT 1",
        (pattern, int(origin + low * 1000), int(origin + high * 1000)),
    ).fetchone()
    parents = []
    if op:
        t, tid = int(op[0]), op[1]
        parents = [
            dict(name=n, duration_us=d / 1000)
            for n, d in conn.execute(
                "SELECT s.value,CAST(a.endNs AS INTEGER)-CAST(a.startNs AS INTEGER) AS duration "
                "FROM PYTORCH_API a JOIN STRING_IDS s ON a.name=s.id "
                "WHERE a.type=50003 AND a.globalTid=? AND CAST(a.startNs AS INTEGER)<=? "
                "AND CAST(a.endNs AS INTEGER)>=? ORDER BY duration LIMIT 15",
                (tid, t, t),
            )
        ]
    conn.close()
    return parents


def main():
    raw = json.loads((HERE / "data/analysis.json").read_text())
    ranks = raw["ranks"]
    byid = {r["id"]: r for r in ranks}
    for rank in ranks:
        events = json.loads((HERE / "data" / f"{rank['id']}.json").read_text())["timeline"]
        for i, step in enumerate(rank["steps"]):
            total = sum(o["e"] - o["s"] for o in step["prep"]["gloo"])
            overlap = 0
            if i:
                prev = rank["steps"][i - 1]
                selected = [
                    e
                    for e in events
                    if e["host_s"] is not None and prev["host_start_us"] <= e["host_s"] < prev["host_end_us"]
                ]
                overlap = union(
                    (max(e["s"], o["s"]), min(e["s"] + e["d"], o["e"])) for e in selected for o in step["prep"]["gloo"]
                )
            step["prep"]["gloo_prev_forward_overlap_us"] = overlap
            step["prep"]["gloo_outside_prev_forward_us"] = total - overlap
    a, p = byid[PRIMARY_A3], byid[PRIMARY_A5]
    sa, sp = a["steps"][A3_STEP], p["steps"][A5_STEP]
    oa, op = operator_map(sa["operators"]), operator_map(sp["operators"])
    comparisons = []
    for typ in sorted(
        set(oa) | set(op), key=lambda t: -max(oa.get(t, {}).get("sum_us", 0), op.get(t, {}).get("sum_us", 0))
    ):
        va, vp = oa.get(typ, {}), op.get(typ, {})
        comparisons.append(
            dict(
                type=typ,
                a_count=va.get("count", 0),
                p_count=vp.get("count", 0),
                a_us=va.get("sum_us", 0),
                p_us=vp.get("sum_us", 0),
                a_shapes=va.get("shapes", []),
                p_shapes=vp.get("shapes", []),
                a_raw=va.get("raw_types", []),
                p_raw=vp.get("raw_types", []),
            )
        )
    write_csv(
        "operator_comparison.csv",
        [{k: v for k, v in c.items() if k in ["type", "a_count", "p_count", "a_us", "p_us"]} for c in comparisons],
    )
    # Align EP rounds by actual first-layer completion clusters, never local index.
    mega_wait = []
    rounds = {}
    for dataset in ["a3_recent", "a5_bs12"]:
        selected = [r for r in ranks if r["dataset"] == dataset]
        base = min(r["session_ns"][0] for r in selected)
        anchors = []
        for rank in selected:
            offset = (rank["session_ns"][0] - base) / 1000
            for step in rank["steps"]:
                m = step["layers"][0]["mega"][0]
                anchors.append(dict(rank=rank, step=step, end=(offset + m["e"]) / 1000))
        clusters = []
        for anchor in sorted(anchors, key=lambda a: a["end"]):
            if not clusters or anchor["end"] - clusters[-1][0]["end"] > 2.0:
                clusters.append([])
            clusters[-1].append(anchor)
        rounds[dataset] = []
        for number, cluster in enumerate(clusters):
            rounds[dataset].append(
                dict(
                    round=number,
                    rank_count=len(cluster),
                    end_min_ms=cluster[0]["end"],
                    end_max_ms=cluster[-1]["end"],
                    expected_ranks=len(selected),
                )
            )
            for anchor in cluster:
                rank, step = anchor["rank"], anchor["step"]
                step["global_round"] = number
                step["round_rank_count"] = len(cluster)
                offset = (rank["session_ns"][0] - base) / 1000
                for layer in step["layers"][:40]:
                    m = layer["mega"][0]
                    mega_wait.append(
                        dict(
                            dataset=dataset,
                            rank=rank["id"],
                            dp=rank["rank"]["dp"],
                            tp=rank["rank"]["tp"],
                            round=number,
                            local_step=step["index"],
                            layer=layer["layer"],
                            kind=step["kind"],
                            start_ms=(offset + m["s"]) / 1000,
                            end_ms=(offset + m["e"]) / 1000,
                            duration_ms=m["dur_us"] / 1000,
                            aic_mac_ratio=m["pmu"].get("aic_mac_ratio"),
                            csv_line=m["line"],
                            source=str(Path(rank["path"]) / "kernel_details.csv"),
                        )
                    )
    write_csv("mega_cross_rank.csv", mega_wait)
    ranksteps = []
    for rank in ranks:
        for step in rank["steps"]:
            m = step["metrics"]
            mega = next(o for o in step["operators"] if o["type"] == "MegaMoe")
            ranksteps.append(
                dict(
                    rank=rank["id"],
                    step=step["index"],
                    global_round=step.get("global_round", ""),
                    kind=step["kind"],
                    main_shapes=str(step["main_tokens"]),
                    span_ms=m["span_us"] / 1000,
                    free_pct=m["free_pct"],
                    kernel_count=m["kernel_count"],
                    aicore_busy_ms=m["aicore_busy_us"] / 1000,
                    aicpu_busy_ms=m["aicpu_busy_us"] / 1000,
                    mega_sum_ms=mega["sum_us"] / 1000,
                    prep_ms=step["prep"]["elapsed_us"] / 1000,
                    gloo_sum_ms=sum(o["e"] - o["s"] for o in step["prep"]["gloo"]) / 1000,
                    gloo_overlap_prev_forward_ms=step["prep"]["gloo_prev_forward_overlap_us"] / 1000,
                    gloo_outside_prev_forward_ms=step["prep"]["gloo_outside_prev_forward_us"] / 1000,
                    source=str(Path(rank["path"]) / "kernel_details.csv"),
                )
            )
    write_csv("rank_steps.csv", ranksteps)
    stability = []
    for dp in range(4):
        selected = [r for r in ranks if r["dataset"] == "a3_recent" and r["rank"]["dp"] == dp]
        for typ in ["HcPre", "HcPost", "SparseFlashMla", "MatMulV2", "QuantBatchMatmulV3", "RmsNorm", "MegaMoe"]:
            values = []
            for rank in selected:
                o = next(o for o in rank["steps"][1]["operators"] if o["type"] == typ)
                values.append(dict(tp=rank["rank"]["tp"], sum_us=o["sum_us"], median_us=o["median_us"]))
            stability.append(
                dict(
                    dataset="a3_recent",
                    dp=dp,
                    step=1,
                    type=typ,
                    values=values,
                    median_spread_pct=100
                    * (max(v["median_us"] for v in values) / min(v["median_us"] for v in values) - 1),
                )
            )
    ea, ep = byid["a3_recent-dp1-tp0"], byid["a5_bs12-dp2-tp0"]
    ga, gp = shape_groups(ea, 0), shape_groups(ep, 0)
    exact = []
    for key in ga.keys() & gp.keys():
        if len(ga[key]) < 3 or len(gp[key]) < 3:
            continue
        va, vp = summary([v[0] for v in ga[key]]), summary([v[0] for v in gp[key]])
        exact.append(
            dict(
                type=key[0],
                shape=key[1],
                dtype=key[2],
                a_count=va["n"],
                p_count=vp["n"],
                a_median_us=va["median"],
                p_median_us=vp["median"],
                a_csv_line=ga[key][0][1],
                p_csv_line=gp[key][0][1],
                a_rank=ea["id"],
                p_rank=ep["id"],
            )
        )
    exact.sort(key=lambda r: -r["a_median_us"] * r["a_count"])
    write_csv("exact_shape_tail.csv", exact)
    repeat_parents = call_parents(a, A3_STEP, "aten::repeat_interleave", True)
    fusion_parents = call_parents(p, A5_STEP, "%rms_norm_dynamic_mx_quant%")
    sample_rows = {}
    for rank, step in [(a, sa), (p, sp)]:
        candidates = {}
        for event in json.loads((HERE / "data" / f"{rank['id']}.json").read_text())["timeline"]:
            if event["host_s"] is not None and step["host_start_us"] <= event["host_s"] < step["host_main_end_us"]:
                candidates.setdefault(event["type"], event["line"])
        records = list(csv.DictReader((Path(rank["path"]) / "kernel_details.csv").open()))
        sample_rows[rank["dataset"]] = {typ: dict(line=line, **records[line - 2]) for typ, line in candidates.items()}
    selected = [
        "a3_recent-dp0-tp0",
        "a3_recent-dp0-tp1",
        "a3_recent-dp0-tp2",
        "a3_recent-dp0-tp3",
        "a3_recent-dp1-tp0",
        "a3_recent-dp1-tp1",
        "a3_recent-dp2-tp2",
        "a3_recent-dp3-tp0",
        *[f"a5_bs12-dp{i}-tp0" for i in range(8)],
        "a3_032-dp0-tp0",
    ]
    timelines = []
    for rid in selected:
        rank = json.loads((HERE / "data" / f"{rid}.json").read_text())
        names = []
        shapes = []
        cores = []

        def intern(xs, value):
            if value not in xs:
                xs.append(value)
            return xs.index(value)

        events = [
            [
                e["s"],
                e["d"],
                intern(names, e["type"]),
                intern(shapes, e["shape"]),
                intern(cores, e["core"]),
                e["line"],
                e["host_s"],
            ]
            for e in rank["timeline"]
        ]
        timelines.append(dict(id=rid, names=names, shapes=shapes, cores=cores, events=events))
    compact = json.loads(json.dumps(ranks))
    for rank in compact:
        rank.pop("operators", None)
        for step in rank["steps"]:
            step["gaps"] = step["gaps"][:8]
            for layer in step["layers"]:
                layer.pop("operators", None)
    report = dict(
        archives=raw["archives"],
        ranks=compact,
        comparisons=comparisons,
        ranksteps=ranksteps,
        mega_wait=mega_wait,
        global_rounds=rounds,
        stability=stability,
        exact_shapes=exact,
        repeat_parents=repeat_parents,
        fusion_parents=fusion_parents,
        sample_rows=sample_rows,
        timelines=timelines,
        preferred=dict(a_id=PRIMARY_A3, a_step=A3_STEP, p_id=PRIMARY_A5, p_step=A5_STEP),
        hardware_status="Confirmed by user: P.rar=A3; vllm_profile_prefill-c192-bs12.zip=A5 (Ascend950, DP8/TP1/EP8).",
    )
    text = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    (HERE / "data/report_data.json").write_text(text)
    with gzip.open(HERE / "data/report_data.json.gz", "wt", encoding="utf-8", compresslevel=6) as stream:
        stream.write(text)
    (HERE / "data/manifest.json").write_text(json.dumps(raw["archives"], ensure_ascii=False, indent=2) + "\n")
    print(
        "ranks",
        len(ranks),
        "windows",
        len(ranksteps),
        "exact shapes",
        len(exact),
        "report bytes",
        len(text.encode()),
        "rounds",
        rounds,
        flush=True,
    )


if __name__ == "__main__":
    main()
