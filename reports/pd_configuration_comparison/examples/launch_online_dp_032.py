#!/usr/bin/env python3
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
