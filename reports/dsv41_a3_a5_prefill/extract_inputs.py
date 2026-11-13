#!/usr/bin/env python3
"""Extract only profiler exports and metadata; never change source archives."""

import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

from analyze import RAW


def main():
    source = Path("/home/gengc/Downloads")
    out = RAW / "a3"
    out.mkdir(parents=True, exist_ok=True)
    with ZipFile(source / "20261026-DSv4.1-1P1D-profiler.zip") as archive:
        for info in archive.infolist():
            path = Path(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {path}")
            if info.is_dir() or "/ASCEND_PROFILER_OUTPUT/" not in info.filename:
                continue
            if "0.30profiler/" not in info.filename and "-P0/" not in info.filename:
                continue
            keep = path.name in (
                "kernel_details.csv",
                "operator_details.csv",
                "step_trace_time.csv",
                "communication.json",
                "communication_matrix.json",
                "api_statistic.csv",
                "op_statistic.csv",
                "analysis.db",
            )
            if keep or path.name.startswith("ascend_pytorch_profiler"):
                archive.extract(info, out)
    tar = shutil.which("bsdtar")
    if not tar:
        raise SystemExit("P.rar requires bsdtar/libarchive with RAR5 support; the old unrar-free 0.0.2 cannot read it.")
    rar = source / "P.rar"
    names = subprocess.check_output([tar, "-tf", str(rar)], text=True).splitlines()
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Unsafe archive path: {name}")
    out = RAW / "a3_recent"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            tar,
            "-xf",
            str(rar),
            "-C",
            str(out),
            "--include",
            "*/ASCEND_PROFILER_OUTPUT/*",
            "--include",
            "*/profiler_info*.json",
            "--include",
            "*/profiler_metadata.json",
            "--include",
            "*/device_*/info.json.*",
            "--include",
            "P/cluster_analysis*",
            "--include",
            "P/*.profiler_out.txt",
        ],
        check=True,
    )
    out = RAW / "a5_bs12"
    out.mkdir(parents=True, exist_ok=True)
    with ZipFile(source / "vllm_profile_prefill-c192-bs12.zip") as archive:
        for info in archive.infolist():
            path = Path(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {path}")
            if info.is_dir():
                continue
            export = (
                "/ASCEND_PROFILER_OUTPUT/" in info.filename
                and path.suffix in (".csv", ".db", ".json")
                and path.name != "trace_view.json"
            )
            metadata = (
                "profiler_info" in path.name or path.name == "profiler_metadata.json" or "/info.json." in info.filename
            )
            if export or metadata:
                archive.extract(info, out)
    print("Inputs:", RAW)


if __name__ == "__main__":
    main()
