#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply/restore the paired v0.32 core scheduler source-only diagnostic patch.

No import of vLLM, torch or Ascend is needed. Compatibility is checked against
the schedule AST, allowing unrelated core-file changes but never guessing when
the scheduling algorithm differs. Stop the service before apply or restore.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

_TARGET = Path("vllm/v1/core/sched/scheduler.py")
_METADATA = Path(__file__).with_name("vllm-v0.32-scheduler-admission.json")


def ast_fingerprint(node):
    # Python 3.13 changed ast.dump's default rendering of empty lists. Keep
    # fingerprints compatible with the deployed Python 3.12 runtime.
    kwargs = {"include_attributes": False}
    if sys.version_info >= (3, 13):
        kwargs["show_empty"] = True
    return hashlib.sha256(ast.dump(node, **kwargs).encode()).hexdigest()


def schedule_parts(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")
    lines = source.splitlines(keepends=True)
    fingerprint = ast_fingerprint(method)
    return fingerprint, "".join(lines[: method.lineno - 1]), "".join(lines[method.end_lineno :])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def locate_root(value):
    if value:
        root = Path(value).resolve()
        if (root / _TARGET).is_file():
            return root
        if (root / "v1/core/sched/scheduler.py").is_file() and root.name == "vllm":
            return root.parent
        raise ValueError(f"Cannot locate {_TARGET} under {root}")
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise ValueError("Specify --vllm-root; the vllm package could not be located")
    return locate_root(str(Path(spec.origin).parent))


def operate(root, action, metadata_path=_METADATA):
    metadata = json.loads(metadata_path.read_text())
    if metadata["target"] != str(_TARGET):
        raise ValueError("Unexpected patch target")
    target = root / _TARGET
    backup = target.with_name("scheduler.py.admission-debug.backup")
    state_path = target.with_name("scheduler.py.admission-debug.json")
    current = target.read_bytes()
    if action == "restore":
        state = json.loads(state_path.read_text())
        original = backup.read_bytes()
        if digest(original) != state["before_sha256"]:
            raise ValueError("Backup checksum mismatch; refusing restoration")
        if digest(current) == state["before_sha256"]:
            print(f"Already restored: {target}")
            return
        if digest(current) != state["after_sha256"]:
            raise ValueError("Core file changed since application; refusing to overwrite later edits")
        target.write_bytes(original)
        print(f"RESTORED {target}; backup retained at {backup}")
        return

    source = current.decode()
    fingerprint, prefix, suffix = schedule_parts(source)
    if fingerprint == metadata["patched_schedule_ast_sha256"]:
        print(f"ALREADY APPLIED {target}")
        return
    if fingerprint != metadata["original_schedule_ast_sha256"]:
        raise ValueError(
            "Scheduler.schedule differs from the paired v0.32 implementation. "
            "No file was changed; provide the runtime core SHA/source to adapt this patch."
        )
    patch_path = metadata_path.parent / metadata["patch"]
    patch_bytes = patch_path.read_bytes()
    headers = [line for line in patch_bytes.splitlines() if line.startswith(b"diff --git ")]
    expected = f"diff --git a/{_TARGET} b/{_TARGET}".encode()
    if headers != [expected]:
        raise ValueError("Patch must contain exactly the expected scheduler file")
    command = ["git", "apply", "--no-index"]
    subprocess.run([*command, "--check", str(patch_path.resolve())], cwd=root, check=True)
    if action == "check":
        print(f"COMPATIBLE {target}; run with --apply after stopping the service")
        return
    if backup.exists() and backup.read_bytes() != current:
        raise ValueError(f"Different backup already exists: {backup}")
    if not backup.exists():
        shutil.copy2(target, backup)
    try:
        subprocess.run([*command, str(patch_path.resolve())], cwd=root, check=True)
        patched = target.read_bytes()
        after_ast, after_prefix, after_suffix = schedule_parts(patched.decode())
        if after_ast != metadata["patched_schedule_ast_sha256"] or (prefix, suffix) != (after_prefix, after_suffix):
            raise ValueError("Verification failed: modifications must be confined to the expected schedule method")
        state_path.write_text(
            json.dumps({"before_sha256": digest(current), "after_sha256": digest(patched)}, indent=2) + "\n"
        )
    except BaseException:
        target.write_bytes(current)
        raise
    print(f"APPLIED {target}; original saved at {backup}. Restart the service.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", help="vLLM repository root, package directory, or site-packages root")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true")
    group.add_argument("--restore", action="store_true")
    group.add_argument("--check", action="store_true", help="Read-only compatibility check (default)")
    args = parser.parse_args()
    action = "restore" if args.restore else "apply" if args.apply else "check"
    operate(locate_root(args.vllm_root), action)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, StopIteration, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
