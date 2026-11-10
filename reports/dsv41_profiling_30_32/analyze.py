#!/usr/bin/env python3
"""Read Ascend profiler ZIPs without extraction; emit auditable summaries."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
DEFAULT_ARCHIVES = [
    Path("/home/gengc/Downloads/20261026-DSv4.1-1P1D-profiler.zip"),
    Path("/home/gengc/Downloads/20261109-032.zip"),
]


def number(value):
    try:
        result = float(str(value).strip())
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def union_duration(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] * (hi - position) + values[hi] * (position - lo) if lo != hi else values[lo]


def stats(values):
    return (
        {
            "count": len(values),
            "sum_us": sum(values),
            "mean_us": statistics.mean(values),
            "p50_us": percentile(values, 0.5),
            "p95_us": percentile(values, 0.95),
            "min_us": min(values),
            "max_us": max(values),
        }
        if values
        else {}
    )


def classify(name):
    text = name.lower()
    if any(
        x in text
        for x in [
            "hcom_",
            "hccl",
            "allreduce",
            "allgather",
            "reducescatter",
            "alltoall",
            "broadcast",
        ]
    ):
        return "communication"
    if any(x in text for x in ["memcpy", "hixl", "batchput", "batchget"]):
        return "transfer"
    return "other"


def record_kind(row):
    if row.get("Accelerator Core") == "COMMUNICATION":
        return "communication_op" if row["Name"].startswith("hcom_") else "communication_task"
    return "kernel"


def csv_rows(archive, member):
    with archive.open(member) as binary:
        yield from csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig", newline=""))


def dataset_id(archive_index, path):
    if archive_index == 0:
        if "0.30profiler/" in path:
            return "v030_unlabelled"
        return "v032_p_tp4" if "-P0/" in path else "v032_d_tp2"
    return "v032_other_host11" if path.startswith("7.150.0.11/") else "v032_other_host35"


def compact_kernel(row):
    return {
        k: row.get(k, "")
        for k in [
            "Name",
            "Type",
            "Stream ID",
            "Start Time(us)",
            "Duration(us)",
            "Input Shapes",
            "Input Data Types",
            "OP State",
            "Accelerator Core",
        ]
    }


def analyze_rank(z, prefix, sample_id, archive_index):
    match = re.search(r"dp(\d+)_pp(\d+)_tp(\d+)_dcp(\d+)_ep(\d+)_rank(\d+)", prefix)
    rank = dict(zip(["dp", "pp", "tp", "dcp", "ep", "rank"], map(int, match.groups()))) if match else {}
    files = {
        p.rsplit("/", 1)[-1]: p
        for p in z.namelist()
        if p.startswith(prefix + "/") and p.count("/") == prefix.count("/") + 1
    }
    rows = list(csv_rows(z, files["kernel_details.csv"]))
    type_values, shape_values = defaultdict(list), defaultdict(list)
    by_state, by_core, by_category, by_kind = Counter(), Counter(), Counter(), Counter()
    intervals = []
    for row in rows:
        duration = number(row["Duration(us)"])
        start = number(row["Start Time(us)"])
        intervals.append((start, start + duration))
        typ = row["Type"]
        type_values[typ].append(duration)
        shape_key = (
            typ,
            row.get("Input Shapes", ""),
            row.get("Input Data Types", ""),
            row.get("Input Formats", ""),
            row.get("Output Shapes", ""),
            row.get("Output Data Types", ""),
            row.get("Accelerator Core", ""),
            row.get("OP State", ""),
        )
        shape_values[shape_key].append(duration)
        by_state[row.get("OP State")] += 1
        by_core[row.get("Accelerator Core")] += duration
        by_category[classify(row["Name"])] += duration
        by_kind[record_kind(row)] += 1
    merged = merge_intervals(intervals)
    origin = merged[0][0] if merged else 0
    gaps = sorted([(b[0] - a[1], a[1] - origin) for a, b in zip(merged, merged[1:])], reverse=True)[:12]
    step = list(csv_rows(z, files["step_trace_time.csv"]))
    op = list(csv_rows(z, files["operator_details.csv"]))
    host_values = defaultdict(float)
    for row in op:
        host_values[row["Name"]] += number(row.get("Host Self Duration(us)"))
    comm = json.loads(z.read(files["communication.json"]))
    comm_types = defaultdict(
        lambda: {
            "count": 0,
            "elapsed_ms": 0,
            "wait_ms": 0,
            "transit_ms": 0,
            "idle_ms": 0,
        }
    )
    all_bandwidths_zero = True
    for step_name, step_data in comm.items():
        if not isinstance(step_data, dict):
            continue
        for family, operations in step_data.items():
            if not isinstance(operations, dict):
                continue
            for name, event in operations.items():
                if not isinstance(event, dict):
                    continue
                info = event.get("Communication Time Info", {})
                typ = name.split("__")[0]
                target = comm_types[typ]
                target["count"] += 1
                for key, source in [
                    ("elapsed_ms", "Elapse Time(ms)"),
                    ("wait_ms", "Wait Time(ms)"),
                    ("transit_ms", "Transit Time(ms)"),
                    ("idle_ms", "Idle Time(ms)"),
                ]:
                    target[key] += number(info.get(source))
                for data in event.get("Communication Bandwidth Info", {}).values():
                    if number(data.get("Transit Size(MB)")) > 0:
                        all_bandwidths_zero = False
    return {
        "id": sample_id,
        "archive_index": archive_index,
        "prefix": prefix,
        "rank": rank,
        "files": files,
        "kernel_count": len(rows),
        "kernel_sum_us": sum(sum(v) for v in type_values.values()),
        "kernel_union_us": union_duration(intervals),
        "kernel_span_us": merged[-1][1] - origin if merged else 0,
        "kernel_origin_us": origin,
        "kernel_types": {k: stats(v) for k, v in type_values.items()},
        "kernel_shapes": [{"key": list(k), **stats(v)} for k, v in shape_values.items()],
        "states": dict(by_state),
        "core_sum_us": dict(by_core),
        "category_sum_us": dict(by_category),
        "record_kinds": dict(by_kind),
        "step_summary": step,
        "host_self_top": sorted(host_values.items(), key=lambda p: p[1], reverse=True)[:25],
        "communication": dict(comm_types),
        "communication_all_bandwidths_zero": all_bandwidths_zero,
        "largest_kernel_gaps": [{"gap_us": g, "start_relative_us": s} for g, s in gaps],
        "longest_kernels": [
            compact_kernel(r) for r in sorted(rows, key=lambda r: number(r["Duration(us)"]), reverse=True)[:12]
        ],
    }


def analyze(archives, output):
    output.mkdir(parents=True, exist_ok=True)
    db_metadata = []
    result = {
        "archives": [],
        "ranks": [],
        "notes": [
            "Nested ZIPs duplicate top-level exported rank data; they are not counted a second time.",
            "Version/role labels derive from archive paths unless confirmed by configuration.",
            "Kernel duration sums are work sums, not elapsed latency; interval union is also exported.",
            "Fused kernels may contain communication internally; generic type matching cannot split it.",
            "COMMUNICATION logical ops and their AIV/AICPU tasks both occur in kernel_details.csv; "
            "do not sum them as independent elapsed time.",
        ],
    }
    for archive_index, path in enumerate(archives):
        with ZipFile(path) as z:
            result["archives"].append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "uncompressed_bytes": sum(i.file_size for i in z.infolist()),
                    "nested_duplicates": [i.filename for i in z.infolist() if i.filename.endswith(".zip")],
                }
            )
            prefixes = sorted({name.rsplit("/", 1)[0] for name in z.namelist() if name.endswith("/kernel_details.csv")})
            for prefix in prefixes:
                sample_id = dataset_id(archive_index, prefix)
                rank = analyze_rank(z, prefix, sample_id, archive_index)
                result["ranks"].append(rank)
                if rank["rank"].get("tp") == 0:
                    db_file = next(
                        p
                        for n, p in rank["files"].items()
                        if n.startswith("ascend_pytorch_profiler_") and n.endswith(".db")
                    )
                    connection = sqlite3.connect(":memory:")
                    try:
                        connection.deserialize(z.read(db_file))
                        db_metadata.append(
                            {
                                "dataset": sample_id,
                                "rank": rank["rank"],
                                "meta": dict(connection.execute("select * from META_DATA")),
                                "npu": connection.execute("select * from NPU_INFO").fetchall(),
                                "session": connection.execute("select * from SESSION_TIME_INFO").fetchall(),
                            }
                        )
                    finally:
                        connection.close()
                print(
                    sample_id,
                    rank["rank"],
                    "kernels",
                    rank["kernel_count"],
                    "span_s",
                    round(rank["kernel_span_us"] / 1e6, 3),
                    flush=True,
                )
    (output / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "db_metadata.json").write_text(json.dumps(db_metadata, ensure_ascii=False, indent=2) + "\n")
    fields = [
        "dataset",
        "dp",
        "tp",
        "type",
        "count",
        "sum_us",
        "mean_us",
        "p50_us",
        "p95_us",
        "max_us",
    ]
    with (output / "kernel_types.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for rank in result["ranks"]:
            for name, data in rank["kernel_types"].items():
                writer.writerow(
                    {
                        k: v
                        for k, v in {
                            "dataset": rank["id"],
                            **rank["rank"],
                            "type": name,
                            **data,
                        }.items()
                        if k in fields
                    }
                )
    print("Saved", output / "analysis.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", nargs="+", type=Path, default=DEFAULT_ARCHIVES)
    parser.add_argument("--output", type=Path, default=HERE / "data")
    options = parser.parse_args()
    analyze(options.archives, options.output)
