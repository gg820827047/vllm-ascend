#!/usr/bin/env python3
"""Inspect recorded main-thread wall time; do not equate self wall with CPU runtime."""

import collections
import concurrent.futures
import gzip
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "data" / "report_data.json.gz"
D = json.loads(gzip.decompress(SOURCE.read_bytes()))
RIDS = ["a3_recent-dp0-tp0", "a3_recent-dp2-tp0", "a3_recent-dp2-tp2"]
COMMON = min(r["session_ns"][0] for r in D["ranks"] if r["id"] in RIDS)


def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(e, out[-1][1])
        else:
            out.append([s, e])
    return out


def analyze(rid):
    r = next(x for x in D["ranks"] if x["id"] == rid)
    origin = int(r["session_ns"][0])
    step = r["steps"][1]
    low = origin + round(step["prep"]["start_us"] * 1000)
    main_start = origin + round(step["host_start_us"] * 1000)
    high = origin + round(step["host_main_end_us"] * 1000)
    c = sqlite3.connect(f"file:{next(Path(r['path']).glob('*.db'))}?mode=ro", uri=True)
    strings = dict(c.execute("SELECT id,value FROM STRING_IDS"))
    preids = [i for i, n in strings.items() if n in ["_C_ascend::npu_hc_pre_v3", "custom::npu_hc_pre_v2"]]
    tid = c.execute(
        "SELECT globalTid FROM PYTORCH_API WHERE type=50001 AND name IN ("
        + ",".join("?" for _ in preids)
        + ") LIMIT 1",
        preids,
    ).fetchone()[0]
    rows = [
        (max(int(s), low), min(int(e), high), int(typ), int(n), int(conn) if conn is not None else None)
        for s, e, typ, n, conn in c.execute(
            "SELECT startNs,endNs,type,name,connectionId FROM PYTORCH_API "
            "WHERE globalTid=? AND endNs>? AND startNs<? AND type IN (50001,50003)",
            (tid, low, high),
        )
    ]
    cann = [
        (int(s), int(e), int(typ), int(n), int(t), int(conn) if conn is not None else None, int(depth))
        for s, e, typ, n, t, conn, depth in c.execute(
            "SELECT startNs,endNs,type,name,globalTid,connectionId,depth FROM CANN_API WHERE endNs>? AND startNs<?",
            (low, high),
        )
    ]
    c.close()
    print(rid, "main_tid", tid, "python", len(rows), "cann", len(cann), flush=True)

    def top_self(lo, hi):
        events = sorted(
            [(max(s, lo), min(e, hi), n) for s, e, t, n, _ in rows if t == 50003 and s < hi and e > lo],
            key=lambda x: (x[0], -x[1]),
        )
        stack = []
        totals = collections.defaultdict(lambda: [0, 0, 0])

        def finish():
            s, e, n, children = stack.pop()
            child_busy = sum(b - a for a, b in merge(children))
            totals[n][0] += 1
            totals[n][1] += e - s
            totals[n][2] += max(0, e - s - child_busy)
            if stack:
                stack[-1][3].append((s, e))

        for s, e, n in events:
            while stack and not (stack[-1][0] <= s and e <= stack[-1][1]):
                finish()
            stack.append([s, e, n, []])
        while stack:
            finish()
        return [
            {"name": strings[n], "count": v[0], "total_ms": v[1] / 1e6, "self_wall_ms": v[2] / 1e6}
            for n, v in sorted(totals.items(), key=lambda kv: -kv[1][2])[:40]
        ]

    selected_suffix = [
        ": dispatch_cg_and_sync_dp",
        ": prepare_attn",
        ": build_attn_metadata",
        ": _build_metadata",
        ": _prepare_inputs_impl",
        ": _prepare_compact_replay",
        ": _coordinate_replay_across_dp",
        ": multistream_preprocess",
        ": npu_stream_switch",
    ]
    scopes = [
        {
            "name": strings[n],
            "start_session_ms": (s - origin) / 1e6,
            "end_session_ms": (e - origin) / 1e6,
            "duration_ms": (e - s) / 1e6,
            "start_common_ms": (s - COMMON) / 1e6,
            "end_common_ms": (e - COMMON) / 1e6,
        }
        for s, e, t, n, _ in rows
        if t == 50003 and any(strings[n].endswith(suf) for suf in selected_suffix)
    ]
    cg = collections.defaultdict(lambda: [0, 0, 0])
    for s, e, t, n, thread, _, _ in cann:
        if thread == tid:
            v = cg[(t, n)]
            v[0] += 1
            v[1] += e - s
            v[2] = max(v[2], e - s)
    cann_top = [
        {"name": strings.get(n, str(n)), "type": t, "count": v[0], "sum_ms": v[1] / 1e6, "max_ms": v[2] / 1e6}
        for (t, n), v in sorted(cg.items(), key=lambda kv: -kv[1][1])[:35]
    ]
    result = {
        "rank": rid,
        "origin_ns": origin,
        "common_origin_ns": COMMON,
        "main_global_tid": tid,
        "main_native_tid": tid & 0xFFFFFFFF,
        "worker_enter_session_ms": (low - origin) / 1e6,
        "target_host_enter_session_ms": (main_start - origin) / 1e6,
        "target_host_end_session_ms": (high - origin) / 1e6,
        "worker_enter_common_ms": (low - COMMON) / 1e6,
        "target_host_enter_common_ms": (main_start - COMMON) / 1e6,
        "target_host_end_common_ms": (high - COMMON) / 1e6,
        "prep_self": top_self(low, main_start),
        "target_self": top_self(main_start, high),
        "selected_scopes": scopes,
        "cann_main_top": cann_top,
    }
    (ROOT / (rid + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    # Source-bound records retain integer ns for subsequent short-window attribution.
    with gzip.open(ROOT / (rid + "-records.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(
            {"origin_ns": origin, "main_tid": tid, "strings": strings, "python": rows, "cann": cann},
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return result


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(analyze, RIDS))
    for r in results:
        print(
            "\n",
            r["rank"],
            "prep main boundaries",
            r["worker_enter_common_ms"],
            r["target_host_enter_common_ms"],
            r["target_host_end_common_ms"],
        )
        print("PREP TOP", json.dumps(r["prep_self"][:10], ensure_ascii=False))
        print("TARGET TOP", json.dumps(r["target_self"][:12], ensure_ascii=False))
        print("CANN TOP", json.dumps(r["cann_main_top"][:8], ensure_ascii=False))
