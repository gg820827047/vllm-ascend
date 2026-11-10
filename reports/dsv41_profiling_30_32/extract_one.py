#!/usr/bin/env python3
"""Extract one rank/file for MindStudio Insight, without expanding the ZIP."""

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

from analyze import HERE

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True)
parser.add_argument("--dp", type=int, default=0)
parser.add_argument("--tp", type=int, default=0)
parser.add_argument(
    "--file",
    default="trace_view.json",
    choices=[
        "trace_view.json",
        "kernel_details.csv",
        "operator_details.csv",
        "step_trace_time.csv",
    ],
)
parser.add_argument("--output", type=Path, required=True)
options = parser.parse_args()
data = json.loads((HERE / "data/analysis.json").read_text())
rank = next(
    r
    for r in data["ranks"]
    if r["id"] == options.dataset and r["rank"]["dp"] == options.dp and r["rank"]["tp"] == options.tp
)
with ZipFile(data["archives"][rank["archive_index"]]["path"]) as archive:
    member = rank["files"][options.file]
    print("Bytes:", archive.getinfo(member).file_size)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, options.output.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
print(options.output)
