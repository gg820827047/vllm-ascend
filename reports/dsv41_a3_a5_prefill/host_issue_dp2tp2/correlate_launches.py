#!/usr/bin/env python3
"""Correlate same-kernel CPU enqueue, CANN launch and hardware start without nearest-time guessing."""

import csv
import gzip
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = json.loads(gzip.decompress((ROOT.parent / "data" / "report_data.json.gz").read_bytes()))
all_results = []
for rid in ["a3_recent-dp2-tp0", "a3_recent-dp2-tp2"]:
    rank = next(r for r in D["ranks"] if r["id"] == rid)
    meta = json.loads((ROOT / (rid + ".json")).read_text())
    spans = sorted(
        [s for s in meta["selected_scopes"] if s["name"].endswith(": multistream_preprocess")],
        key=lambda s: s["start_session_ms"],
    )
    rec = json.loads(gzip.decompress((ROOT / (rid + "-records.json.gz")).read_bytes()))
    origin = rec["origin_ns"]
    names = {int(k): v for k, v in rec["strings"].items()}
    conn = sqlite3.connect(f"file:{next(Path(rank['path']).glob('*.db'))}?mode=ro", uri=True)
    with (Path(rank["path"]) / "kernel_details.csv").open() as f:
        csvrows = {
            (int(Decimal(row["Start Time(us)"].strip()) * 1000), row["Stream ID"], row["Task ID"]): (line, row)
            for line, row in enumerate(csv.DictReader(f), 2)
        }
    selected = []
    for layer in [0, 10, 21]:
        span = spans[layer]
        low = origin + round(span["start_session_ms"] * 1e6)
        high = origin + round(span["end_session_ms"] * 1e6)
        for s, e, t, n, group in rec["python"]:
            if (
                t != 50001
                or not low <= s < high
                or names[n]
                not in [
                    "aclnnMatmul",
                    "aclnnRmsNorm",
                    "aclnnDynamicQuantV2",
                    "aclnnScatterNdUpdateSk",
                    "aclnnQuantMatmulWeightNz",
                ]
            ):
                continue
            rawids = [r[0] for r in conn.execute("SELECT connectionId FROM CONNECTION_IDS WHERE id=?", (group,))]
            for rawid in rawids:
                for task_start, task_end, stream, taskid in conn.execute(
                    "SELECT startNs,endNs,streamId,taskId FROM TASK WHERE connectionId=?", (rawid,)
                ):
                    key = (int(task_start), str(stream), str(taskid))
                    if key not in csvrows:
                        continue
                    line, row = csvrows[key]
                    launches = [
                        {
                            "start_ns": int(cs),
                            "end_ns": int(ce),
                            "tid": int(tid) & 0xFFFFFFFF,
                            "name": names.get(int(cn), str(cn)),
                        }
                        for cs, ce, tid, cn in conn.execute(
                            "SELECT startNs,endNs,globalTid,name FROM CANN_API WHERE connectionId=?", (rawid,)
                        )
                        if names.get(int(cn)) == "launch"
                    ]
                    selected.append(
                        {
                            "rank": rid,
                            "layer": layer,
                            "kernel": row["Type"],
                            "stream": stream,
                            "task_id": taskid,
                            "csv_line": line,
                            "shape": row["Input Shapes"],
                            "enqueue_name": names[n],
                            "enqueue_start_ns": s,
                            "enqueue_end_ns": e,
                            "kernel_start_ns": int(task_start),
                            "kernel_end_ns": int(task_end),
                            "launches": launches,
                        }
                    )
    conn.close()
    all_results.extend(selected)
(ROOT / "kernel_connections.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2) + "\n")
for x in all_results:
    if x["layer"] != 0:
        continue
    meta = json.loads((ROOT / (x["rank"] + ".json")).read_text())
    origin = meta["origin_ns"]
    print(
        x["rank"],
        x["kernel"],
        "S" + str(x["stream"]),
        "CSV" + str(x["csv_line"]),
        "enqueue_session_ms",
        round((x["enqueue_start_ns"] - origin) / 1e6, 6),
        "launch_session_ms",
        [
            (round((launch["start_ns"] - origin) / 1e6, 6), round((launch["end_ns"] - origin) / 1e6, 6), launch["tid"])
            for launch in x["launches"]
        ],
        "device_session_ms",
        round((x["kernel_start_ns"] - origin) / 1e6, 6),
        round((x["kernel_end_ns"] - origin) / 1e6, 6),
    )
