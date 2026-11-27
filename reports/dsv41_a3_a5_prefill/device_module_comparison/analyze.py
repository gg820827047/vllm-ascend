#!/usr/bin/env python3
"""Attribute Device tasks to model features using captured causal call paths.

This tool reads task ledgers from execution_scope and the matching profiler DB.
Host intervals are used only to recover ownership; no Host cost comparison is
performed. A kernel belongs to exactly one module, including fused kernels.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import heapq
import importlib.util
import json
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = (
    "Embedding",
    "Engram",
    "Attn-Proj",
    "Attn-Core",
    "Indexer/DSA",
    "KV/RoPE",
    "Norm/Quant",
    "MHC/HC",
    "MoE-Gate",
    "MoE-SharedExpert",
    "MoE-RoutedExpert",
    "MoE-Dispatch/Combine",
    "MoE-Fused",
    "MoE-Boundary/TP",
    "Device-Metadata",
    "Draft-Context",
    "Sampling/Head",
    "Layout/Boundary",
    "Unresolved",
)
NORM_TYPES = frozenset(
    ("RmsNorm", "RmsNormCast", "RmsNormDynamicMxQuant", "DynamicQuant", "DynamicMxQuant", "AddRmsNorm")
)
PHASE_STAGES = {
    "worker_prepare": "device_prepare",
    "target_input": "target_model",
    "target_layers": "target_model",
    "target_tail": "target_model",
    "runner_after_forward": "target_model",
    "target_sampling": "sampling",
    "draft_prepare": "draft",
    "draft_layers": "draft",
    "draft_head_and_output": "draft",
}


def load_json(path):
    return json.loads(Path(path).read_text())


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def get_inputs(spec, config_dir):
    """Use a preserved ledger or audit a newly selected Worker execution."""
    if spec.get("evidence_json") and spec.get("tasks_csv"):
        evidence = load_json(resolve(config_dir, spec["evidence_json"]))["samples"][spec["label"]]
        task_path = resolve(config_dir, spec["tasks_csv"])
        with task_path.open() as f:
            rows = list(csv.DictReader(f))
        return evidence, rows, hashlib.sha256(task_path.read_bytes()).hexdigest()
    if not isinstance(spec.get("worker_entry_ns"), int):
        raise ValueError("A new trace needs an integer worker_entry_ns; use --list-workers first")
    module_spec = importlib.util.spec_from_file_location(
        "execution_audit", HERE.parent / "execution_scope/audit_execution.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    db = resolve(config_dir, spec["database"])
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    origin = int(conn.execute("SELECT * FROM SESSION_TIME_INFO").fetchone()[0])
    conn.close()
    offset_us = (spec["worker_entry_ns"] - origin) / 1000
    evidence, rows = module.audit(
        db,
        {"prep": {"start_us": offset_us}, "host_start_us": offset_us + 0.001},
        raw_root=db.parent.parent,
    )
    evidence.update(rank_id=spec.get("rank_id", spec["label"]), step_index=spec.get("step_index"))
    return evidence, rows, hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def list_workers(db):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    strings = dict(conn.execute("SELECT id,value FROM STRING_IDS"))
    names = [
        n
        for n, value in strings.items()
        if "vllm_ascend/worker/" in value and value.endswith(": execute_model") and "model_runner" in value
    ]
    if not names:
        raise ValueError("No Ascend model_runner.execute_model scope was captured")
    sql = (
        "SELECT startNs,endNs,name FROM PYTORCH_API WHERE type=50003 AND name IN ("
        + ",".join("?" for _ in names)
        + ") ORDER BY startNs"
    )
    out = [
        dict(worker_entry_ns=int(s), host_scope_ms=(int(e) - int(s)) / 1e6, scope=strings[n])
        for s, e, n in conn.execute(sql, names)
    ]
    conn.close()
    print(json.dumps(out, ensure_ascii=False, indent=2))


def union_ns(intervals):
    total = 0
    start = end = None
    for a, b in sorted(intervals):
        if start is None:
            start, end = a, b
        elif a <= end:
            end = max(end, b)
        else:
            total += end - start
            start, end = a, b
    return total + (end - start if start is not None else 0)


def trace_context(rows, db, evidence):
    """Recover the exact submitting thread, then its active Python scopes."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    strings = dict(conn.execute("SELECT id,value FROM STRING_IDS"))
    base = evidence["worker_entry_epoch_ns"]
    stop = base + round(evidence["sample_return_ms"] * 1e6)
    groups = collections.defaultdict(list)
    sql = (
        "SELECT startNs,endNs,globalTid,connectionId,name FROM PYTORCH_API "
        "WHERE type=50001 AND startNs>=? AND startNs<=?"
    )
    for s, e, tid, cid, n in conn.execute(sql, (str(base), str(stop))):
        groups[cid].append((int(s), int(e), tid, strings[n]))
    raw = collections.defaultdict(list)
    needed = {int(r["connection"]) for r in rows if r["connection"]}
    for group, cid in conn.execute("SELECT id,connectionId FROM CONNECTION_IDS"):
        if cid in needed and group in groups:
            raw[cid].extend(groups[group])
    sql = "SELECT startNs,endNs,globalTid,connectionId,name FROM CANN_API WHERE startNs>=? AND startNs<=?"
    for s, e, tid, cid, n in conn.execute(sql, (str(base), str(stop))):
        if cid in needed:
            raw[cid].append((int(s), int(e), tid, strings[n]))
    hosts = collections.defaultdict(list)
    for row in rows:
        ns = base + round(float(row["host_start_ms"]) * 1e6)
        candidates = [v for v in raw[int(row["connection"])] if v[0] == ns and v[3] == row["host_name"]]
        if not candidates:
            raise ValueError(f"No exact Host correlation: {row['csv_line']} / {row['connection']}")
        row["host_tid"] = candidates[0][2]
        row["host_ns"] = ns
        row["host_end_ns"] = candidates[0][1]
        hosts[row["host_tid"]].append(row)
    # A sweep over intervals recovers ancestry without nearest-time guesses.
    # Scope expiry is half-open, so a returning sibling is not a parent.
    for tid, selected in hosts.items():
        sql = "SELECT startNs,endNs,name FROM PYTORCH_API WHERE type=50003 AND globalTid=? AND startNs<=? AND endNs>=?"
        scopes = sorted((int(s), -int(e), strings[n]) for s, e, n in conn.execute(sql, (tid, str(stop), str(base))))
        active = {}
        expiry = []
        i = 0
        for row in sorted(selected, key=lambda r: r["host_ns"]):
            point = row["host_ns"]
            while i < len(scopes) and scopes[i][0] <= point:
                s, ne, name = scopes[i]
                active[i] = (s, ne, name)
                heapq.heappush(expiry, (-ne, i))
                i += 1
            while expiry and expiry[0][0] <= point:
                _, ix = heapq.heappop(expiry)
                active.pop(ix, None)
            row["parents"] = [
                v[2]
                for v in sorted(active.values())
                if "vllm_ascend/" in v[2] or v[2].startswith("nn.Module:") or "/vllm/" in v[2]
            ]
    conn.close()
    return rows


def scope_owner(name):
    """Recognize a functional boundary; generic Linear wrappers are ignored."""
    function = name.rsplit(": ", 1)[-1]
    if "VocabParallelEmbedding" in name or function == "embed_input_ids":
        return "Embedding"
    if ("/engram/" in name and "/model_state.py" not in name and "/hash_state.py" not in name) or "engram" in function:
        return "Engram"
    if function in ("hc_pre", "hc_post", "hc_collapse"):
        return "MHC/HC"
    if "/shared_experts.py" in name or "SharedExperts" in name:
        return "MoE-SharedExpert"
    if function in ("_apply_router_linear", "_gate_router_logits", "_prepare_router_input", "compute_gate_logits"):
        return "MoE-Gate"
    if "/router/" in name or function in ("_select_experts", "_compute_routing"):
        return "MoE-Gate"
    if "/indexer.py" in name or function in (
        "_select_sparse_indices",
        "_prepare_indexer_inputs",
        "_quantize_indexer_query",
        "_indexer_hidden_states",
    ):
        return "Indexer/DSA"
    if "/compressor.py" in name or function in (
        "_write_compressed_source",
        "_write_swa_cache",
        "_write_attention_cache",
        "_scatter_swa_kv",
        "scatter_cache_sk",
        "_apply_rotary",
        "_commit_private_circle_prefill",
    ):
        return "KV/RoPE"
    if function in (
        "_forward_attention",
        "_attention",
        "_sparse_mla_attention",
        "_private_circle_attention",
        "_native_attention",
    ):
        return "Attn-Core"
    if function in ("_project_output", "_forward_o_proj", "_prepare_cmp_attention"):
        return "Attn-Proj"
    if function in ("combine_hidden_states", "precompute_and_store_context_kv", "_precompute_context_kv"):
        return "Draft-Context"
    if "/moe_mlp.py" in name or function in ("_apply_mlp", "npu_grouped_matmul_swiglu_quant"):
        return "MoE-RoutedExpert"
    if function in ("multistream_preprocess", "_preprocess") and "/attention/dsa_v41.py" in name:
        return "Attn-Proj"
    if "/prepare_finalize.py" in name or "/moe_comm_method.py" in name:
        return "MoE-Boundary/TP"
    if "/routed_experts.py" in name:
        return "MoE-RoutedExpert"
    if "/fused_moe/fused_moe.py" in name or "AscendMoERunner_" in name or "DeepseekV41MoE_" in name:
        return "MoE-Boundary/TP"
    if function in ("compute_logits", "_sample_from_logits", "_sample_sequential", "_sample", "sample"):
        return "Sampling/Head"
    if "Sampler" in name or "ParallelLMHead" in name:
        return "Sampling/Head"
    return None


def classify(row):
    """Keep fused work together; physical task ownership is exclusive."""
    typ = row["type"]
    parents = row["parents"]
    if typ in ("MegaMoe", "DispatchFFNCombine"):
        return "MoE-Fused", "fused_type", typ
    if typ.startswith("MoeDistribute") or typ.startswith("MoeInitRouting"):
        return "MoE-Dispatch/Combine", "dispatch_combine_type", typ
    if typ in ("HcPre", "HcPost", "MhcPost"):
        return "MHC/HC", "hc_type", typ
    if typ in NORM_TYPES:
        return "Norm/Quant", "standalone_norm_quant_type", typ
    if (
        typ in ("KvCompressEpilogV2", "ScatterNdUpdateSk", "InplacePartialRotaryMul", "RotaryPositionEmbedding")
        or "FusedPostScatter" in typ
    ):
        return "KV/RoPE", "kv_rope_type", typ
    if typ == "SparseFlashMla" or "MqsmlaKernel" in typ:
        return "Attn-Core", "attention_core_type", typ
    # Device preparation is a separate functional bucket, even when driven by
    # different Runner implementations. Engram kernels retain their owner.
    owners = [(scope_owner(p), p) for p in reversed(parents)]
    owners = [(m, p) for m, p in owners if m]
    if row["phase"] == "worker_prepare":
        if owners and owners[0][0] in ("Embedding", "Engram"):
            return owners[0][0], "captured_scope", owners[0][1]
        return "Device-Metadata", "device_prepare_phase", row["phase"]
    if owners:
        if owners[0][0] == "MoE-RoutedExpert":
            shared = next(((m, p) for m, p in owners if m == "MoE-SharedExpert"), None)
            if shared is not None:
                return shared[0], "captured_shared_mlp_scope", shared[1]
        return owners[0][0], "captured_scope", owners[0][1]
    if row["phase"] == "target_sampling":
        return "Sampling/Head", "sampling_phase", row["phase"]
    if row["phase"] in ("draft_prepare",):
        return "Device-Metadata", "draft_prepare_phase", row["phase"]
    if parents:
        return "Layout/Boundary", "outside_named_feature", parents[-1]
    return "Unresolved", "missing_semantic_scope", ""


def prepare_rows(rows):
    for row in rows:
        row["module"], row["rule"], row["basis"] = classify(row)
        row["stage"] = PHASE_STAGES[row["phase"]]
        row["start_ns"] = round(float(row["start_ms"]) * 1e6)
        row["end_ns"] = round(float(row["end_ms"]) * 1e6)
        row["duration_us"] = (row["end_ns"] - row["start_ns"]) / 1000
        row["kind"] = (
            "hcom"
            if row["core"] == "COMMUNICATION"
            else "comm_helper"
            if row["type"] in ("allgatherAicpuKernel", "allreduceAicpuKernel")
            else "kernel"
        )
        row["layer"] = None
        for parent in reversed(row["parents"]):
            for prefix in ("nn.Module: DeepseekV41DecoderLayer_", "nn.Module: DeepseekV41DSparkDecoderLayer_"):
                if parent.startswith(prefix) and parent.removeprefix(prefix).isdecimal():
                    row["layer"] = int(parent.removeprefix(prefix))
                    break
            if row["layer"] is not None:
                break
    return rows


def metrics(rows):
    if not rows:
        return dict(count=0, resident_us=0, union_us=0, hcom_us=0, comm_helper_us=0, types=[])
    types = collections.defaultdict(list)
    for row in rows:
        types[row["type"]].append(row)
    return dict(
        count=len(rows),
        resident_us=sum(r["duration_us"] for r in rows),
        union_us=union_ns((r["start_ns"], r["end_ns"]) for r in rows) / 1000,
        hcom_us=sum(r["duration_us"] for r in rows if r["kind"] == "hcom"),
        comm_helper_us=sum(r["duration_us"] for r in rows if r["kind"] == "comm_helper"),
        types=[
            dict(type=t, count=len(rr), resident_us=sum(r["duration_us"] for r in rr))
            for t, rr in sorted(types.items(), key=lambda item: -sum(r["duration_us"] for r in item[1]))
        ],
    )


def summarize(rows):
    groups = {
        "primary": [r for r in rows if r["stage"] != "draft"],
        "target_layers": [r for r in rows if r["phase"] == "target_layers"],
        "sampling": [r for r in rows if r["stage"] == "sampling"],
        "draft": [r for r in rows if r["stage"] == "draft"],
    }
    out = {}
    for scope, rr in groups.items():
        modules = {module: metrics([r for r in rr if r["module"] == module]) for module in MODULES}
        total = metrics(rr)
        assert sum(v["count"] for v in modules.values()) == len(rr)
        assert abs(sum(v["resident_us"] for v in modules.values()) - total["resident_us"]) < 0.00001
        out[scope] = dict(total=total, modules=modules)
    assert sum(out[k]["total"]["count"] for k in ("primary", "draft")) == len(rows)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "inputs.json")
    parser.add_argument(
        "--context-cache", type=Path, help="Optional previously recovered contexts for this exact input"
    )
    parser.add_argument("--list-workers", type=Path, help="List selectable Worker scopes in a new profiler DB")
    args = parser.parse_args()
    if args.list_workers:
        list_workers(args.list_workers)
        return
    config = load_json(args.config)
    samples = {}
    for spec in config["samples"]:
        label = spec["label"]
        evidence, rows, ledger_hash = get_inputs(spec, args.config.parent)
        if args.context_cache:
            cached = json.loads(
                gzip.decompress((args.context_cache / f"{label.lower()}_contexts.json.gz").read_bytes())
            )
            identities = lambda rr: [(r["csv_line"], r["connection"], r["start_ms"], r["shape"]) for r in rr]
            if identities(rows) != identities(cached):
                raise ValueError("Context cache does not match the selected task ledger")
            rows = cached
        else:
            rows = trace_context(rows, resolve(args.config.parent, spec["database"]), evidence)
        rows = prepare_rows(rows)
        samples[label] = dict(
            metadata=spec["metadata"],
            timing=evidence,
            source_ledger_sha256=ledger_hash,
            summary=summarize(rows),
            tasks=rows,
        )
        print(label, "tasks", len(rows), "unresolved", sum(r["module"] == "Unresolved" for r in rows), flush=True)
    output = dict(
        method=(
            "Exact Connection ID / submitting thread / captured functional scopes. "
            "No nearest-time or name-only MatMul attribution."
        ),
        modules=MODULES,
        comparability=config["comparability"],
        samples=samples,
    )
    (HERE / "analysis.json.gz").write_bytes(gzip.compress(json.dumps(output, ensure_ascii=False).encode(), mtime=0))
    for label, sample in samples.items():
        fields = [
            "stage",
            "phase",
            "layer",
            "module",
            "rule",
            "basis",
            "type",
            "name",
            "duration_us",
            "start_ms",
            "end_ms",
            "kind",
            "core",
            "stream",
            "task",
            "connection",
            "csv_line",
            "shape",
            "dtype",
            "output",
        ]
        with (HERE / f"{label.lower()}_module_tasks.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(sample["tasks"])


if __name__ == "__main__":
    main()
