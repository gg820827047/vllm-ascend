#!/usr/bin/env python3
"""Stream representative Chrome traces; preserve original event evidence."""

import argparse
import collections
import io
import json
from pathlib import Path
from zipfile import ZipFile

import ijson
from analyze import HERE, stats


def stream_events(archive, path):
    with archive.open(path) as stream:
        buffered = io.BufferedReader(stream)
        prefix = "item" if buffered.peek(64).lstrip().startswith(b"[") else "traceEvents.item"
        yield from ijson.items(buffered, prefix)


def analyze_trace(z, rank):
    path = rank["files"]["trace_view.json"]
    counts = collections.Counter()
    durations = collections.defaultdict(list)
    categories = collections.Counter()
    threads = []
    evidence = []
    long_events = []
    high_level = []
    origin = rank["kernel_origin_us"]
    for event in stream_events(z, path):
        categories[str(event.get("cat"))] += 1
        name = event.get("name", "")
        if event.get("ph") == "M" and name in ["process_name", "thread_name"]:
            threads.append(event)
        if event.get("ph") != "X":
            continue
        duration = float(event.get("dur", 0))
        if event.get("cat") == "cpu_op":
            counts[name] += 1
            durations[name].append(duration)
        if name.startswith("vllm::") or any(
            s in name.lower() for s in ["profilerstep", "compiled", "graph_replay", "cudagraph"]
        ):
            high_level.append(event)
        if duration >= 10000:
            long_events.append(event)
        selected = event.get("cat") == "cpu_op" and (
            name.startswith("vllm::")
            or any(
                s in name.lower()
                for s in [
                    "replay",
                    "engram",
                    "embedding",
                    "lightning",
                    "bmm",
                    "transpose",
                    "contiguous",
                    "zeros",
                ]
            )
        )
        if selected:
            evidence.append(event)
    long_events = sorted(long_events, key=lambda e: float(e.get("dur", 0)), reverse=True)[:60]
    groups = collections.defaultdict(list)
    for event in high_level:
        groups[event["name"]].append(event)
    sequence = sorted(
        [e for e in high_level if e["name"] == "vllm::dsa_v41_forward"],
        key=lambda e: e["ts"],
    )
    # Forty main layers followed by three draft invocations is verified against
    # the source model and 129-event sequence; do not assume this for other models.
    high_stats = {name: stats([float(e.get("dur", 0)) for e in values]) for name, values in groups.items()}
    return {
        "dataset": rank["id"],
        "rank": rank["rank"],
        "path": path,
        "categories": dict(categories),
        "threads": threads,
        "cpu_counts": dict(counts),
        "cpu_durations": {name: stats(values) for name, values in durations.items()},
        "long_events": long_events,
        "high_level_stats": high_stats,
        "attention_sequence": [
            {
                "index": i,
                "relative_us": float(e["ts"]) - origin,
                "duration_us": float(e["dur"]),
                "pid": e.get("pid"),
                "tid": e.get("tid"),
                "args": e.get("args"),
            }
            for i, e in enumerate(sequence)
        ],
        "evidence_events": evidence,
    }


def main(data_dir):
    analysis = json.loads((data_dir / "analysis.json").read_text())
    representative = [r for r in analysis["ranks"] if r["rank"]["tp"] == 0]
    output = []
    for archive_index, archive in enumerate(analysis["archives"]):
        with ZipFile(archive["path"]) as z:
            for rank in representative:
                if rank["archive_index"] != archive_index:
                    continue
                print("Reading", rank["id"], rank["rank"], flush=True)
                result = analyze_trace(z, rank)
                output.append(result)
                print(
                    "events:",
                    sum(result["categories"].values()),
                    "attention:",
                    len(result["attention_sequence"]),
                    "longest:",
                    [(e["name"], round(float(e["dur"]) / 1000, 1)) for e in result["long_events"][:4]],
                    flush=True,
                )
    (data_dir / "traces.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, default=float) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    main(parser.parse_args().data_dir)
