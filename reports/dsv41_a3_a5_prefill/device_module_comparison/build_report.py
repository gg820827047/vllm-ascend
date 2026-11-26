#!/usr/bin/env python3
"""Render the Device feature ledger without requiring the raw profiler DBs."""
# Keep narrative paragraphs and HTML templates intact, as in nearby reports.
# ruff: noqa: E501

from __future__ import annotations

import collections
import csv
import gzip
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEATURES = {
    "Embedding/Engram": (
        "Token embedding、Engram查表/门控及其通信。",
        "A3有独立Engram与TP恢复；A5窗口只观察到词表Gather/Cast。特性配置或执行范围仍需核对，不能写成融合收益。",
        ["embedding", "embed_tp", "engram_tp", "engram_gate"],
    ),
    "Attn-Proj": (
        "Attention Q_a/Q_b/KV线性投影和输出投影；不含Indexer自己的投影。",
        "A3的40次输出投影AllReduce及辅助任务计入本行。A5 TP1无这条显式TP通信；普通/量化矩阵乘路径也不同。",
        ["attn_proj", "oproj"],
    ),
    "Attn-Core": (
        "SWA/global attention核心及该核心内部准备。",
        "A3 SparseFlashMla；A5 MQ-SMLA DSL及其输入准备。算法/缓存ABI、query和上下文未对齐，不计算耗时差额。",
        ["attn_core"],
    ),
    "Indexer/DSA": (
        "Indexer的Q/权重投影、稀疏索引选择和专门前处理。",
        "A3 QLI V2；A5 QLI/稀疏QLI和IndexerPrologueQw等专门内核。Prologue准确融合边界待A5源码，不能从名字推断。",
        ["indexer", "indexer_core"],
    ),
    "KV/RoPE": (
        "独立RoPE、压缩投影/池化、KV写入和量化写回epilogue。",
        "A3 ScatterNdUpdateSk/Triton池化等；A5 KvCompressEpilogV2、CompressorV2、FusedPostScatter等。保留各自KV格式，先核对功能边界。",
        ["kv_write", "compressor", "compressed_source"],
    ),
    "Norm/Quant": (
        "独立Norm和动态量化统一统计；融合Norm+Quant只计一次。",
        "A5每主干40次RmsNormDynamicMxQuant；A3 Q路径分开Norm与INT8量化。可评估INT8双输出融合，需保留Indexer的浮点qr。",
        ["attn_proj", "norm_fusion_ref"],
    ),
    "MHC/HC": (
        "HC Pre/Post，以及该参考实现中HC collapse的独立任务。",
        "主干两边均80次Pre、80次Post；HcPost/MhcPost为不同接口。不能由调用次数直接推定性能一致。",
        ["hc", "tail"],
    ),
    "MoE-Gate": (
        "独立router logits、Top-K/hash及其相关准备。",
        "A3 MatMulV2，A5 FusedMatMul；两边均有MoeGatingTopKHash。Gate没有因为MegaMoE而全部消失。",
        ["gate"],
    ),
    "MoE-SharedExpert": (
        "独立共享专家MLP及该分支的结果恢复通信。",
        "A3独立Quant MatMul/激活及40次AllGather；A5共享专家已随权重传入MegaMoE，单列空白不代表零计算。",
        ["shared", "shared_gather", "shared_fusion"],
    ),
    "MoE-RoutedExpert": (
        "可独立观察的路由专家GMM/激活；融合kernel不拆回此行。",
        "当前主干两边FFN主体都在MegaMoE内。草稿中可出现独立GroupedMatmul；不能把融合FFN同时计入两行。",
        ["mega_dispatch"],
    ),
    "MoE-Dispatch/Combine": (
        "独立专家dispatch/combine和routing布局变换。",
        "当前主干内部dispatch/combine随MegaMoE保留；草稿可见独立MoeDistributeDispatch/Combine。外部TP恢复另列。",
        ["mega_dispatch"],
    ),
    "MoE-Fused": (
        "MegaMoe / DispatchFFNCombine完整驻留；包含内部计算、通信和等待。",
        "主干各40次MegaMoE：A3 INT8 routed，A5 MXFP4 routed并融合shared。草稿A3的DispatchFFNCombine也保留完整融合边界。",
        ["mega_args", "mega_dispatch", "shared_fusion"],
    ),
    "MoE-Boundary/TP": (
        "MoE外围prepare/finalize、输出合并及路由分支TP恢复。",
        "A3另40次AllGather来自routed finalize；共享分支40次在SharedExpert行。A5仍有外部buffer/加法准备，不能全部说成被融合。",
        ["routed_gather", "shared_gather"],
    ),
    "Device-Metadata": (
        "本轮设备侧窗口、block/slot、Attention/indexer metadata及草稿准备。",
        "只统计Device任务，不比较Host耗时。Runner也会改变设备准备路径；可识别冗余/融合候选，不能当纯硬件差异。",
        ["metadata", "replay_metadata"],
    ),
    "Sampling/Head": (
        "目标输出头、采样；不同Runner的执行函数位置不影响此功能归属。",
        "A5 LM head出现在execute_model尾部，A3主要在sample_tokens内。按功能合并，避免把函数边界差误当缺失计算。",
        ["lm_head"],
    ),
    "Draft-Context": (
        "草稿aux hidden合并、context KV投影等实际计算；不把这些GEMM算成纯metadata。",
        "草稿上下文准备独立列示；Norm/Quant、RoPE与写回仍按其物理边界归属。",
        ["draft"],
    ),
    "Layout/Boundary": (
        "模型外层与replay边界的复制、重排和尚未细分到上述功能的任务。",
        "显式保留未细分成本，不把未知工作强塞给Attention/MoE。可按任务与源码继续核对。",
        ["tail"],
    ),
    "Unresolved": (
        "缺少可用语义调用范围的任务。",
        "若出现必须保留并补证据；当前样本的精确连接已恢复，结构性未细分项仍单独列示。",
        [],
    ),
}


def esc(value):
    return html.escape(str(value), quote=True)


def number(value):
    return f"{value:,.2f}"


def canonical(typ):
    aliases = (
        ("MqsmlaKernel", "MQ-SMLA (DSL)"),
        ("QuantSparseLightningIndexer", "Sparse QLI (DSL)"),
        ("Quant_lightning_indexer", "QLI (DSL)"),
        ("IndexerPrologueQwKernel", "IndexerPrologueQw (DSL)"),
        ("FusedPostScatterKernel", "FusedPostScatter (DSL)"),
        ("mqsmla_metadata_kernel", "MQ-SMLA Metadata (DSL)"),
    )
    for key, alias in aliases:
        if key in typ:
            return alias
    if typ.startswith("_Z15metadata_kernel"):
        return "Metadata kernel (custom)"
    return typ


def difference_note(module, data, scope):
    descriptions = []
    for label, sample in data["samples"].items():
        value = sample["summary"][scope]["modules"][module]
        types = {v["type"]: v["count"] for v in value["types"]}
        if module == "Embedding/Engram":
            count = sum(
                r["module"] == module and r["stage"] != "draft" and "engram" in r["basis"].lower()
                for r in sample["tasks"]
            )
            descriptions.append(f"{label} 有 {count} 条独立Engram相关记录")
        elif module == "MHC/HC":
            descriptions.append(
                f"{label} Pre {types.get('HcPre', 0)} / Post {types.get('HcPost', 0) + types.get('MhcPost', 0)} 次"
            )
        elif module == "Norm/Quant":
            descriptions.append(f"{label} RmsNormDynamicMxQuant {types.get('RmsNormDynamicMxQuant', 0)} 次")
        elif module == "MoE-Fused":
            descriptions.append(
                f"{label} MegaMoe {types.get('MegaMoe', 0)} / DispatchFFNCombine {types.get('DispatchFFNCombine', 0)} 次"
            )
    if module == "Embedding/Engram":
        return "；".join(descriptions) + "。未观察到不自动等于融合或功能关闭；需核对配置及范围。"
    if module == "MHC/HC":
        return "；".join(descriptions) + "。还含独立collapse任务；接口/shape仍需对齐。"
    if module == "Norm/Quant":
        return "；".join(descriptions) + "。跨Norm/Quant融合只计一次；INT8融合需保留Indexer使用的浮点qr。"
    if module == "MoE-Fused":
        return (
            "；".join(descriptions) + "。内部FFN/通信/等待不可由外层trace拆分；共享专家是否融合看各样本配置与实际调用。"
        )
    generic = {
        "Attn-Proj": "输出投影显式TP通信随本行归属；TP1没有这条路径时不虚构成本。普通/量化GEMM和布局以实际执行节点为准。",
        "MoE-Gate": "对照实际Gate GEMM和Top-K/hash实现。独立Gate保留本行，不能因存在MegaMoE就认定Gate已全部融合。",
        "MoE-SharedExpert": "独立共享分支含其结果恢复通信；若共享权重已进入MegaMoE，本行不虚构独立成本。",
        "MoE-RoutedExpert": "独立GMM/激活才计本行；融合FFN保留MoE-Fused。没有独立任务不等于没有专家计算。",
        "MoE-Dispatch/Combine": "独立专家交换与布局变换计本行；融合内部通信保留MoE-Fused。外部TP恢复另列。",
        "MoE-Boundary/TP": "routed finalize的TP恢复计本行；shared恢复计共享分支，不能重复统计。外围buffer/合并也保留。",
        "Sampling/Head": "不同Runner可能把LM head放在execute_model尾部或sample_tokens内；本表按功能合并，避免把函数位置变化当成缺失计算。",
    }
    return generic.get(module, FEATURES[module][1])


def communication_table(data):
    labels = list(data["samples"])
    paths = (
        ("Attention输出投影", "Attn-Proj", "target_layers", "hcom_allReduce_", "oproj"),
        ("routed finalize结果恢复", "MoE-Boundary/TP", "target_layers", "hcom_allGather_", "routed_gather"),
        ("shared finalize结果恢复", "MoE-SharedExpert", "target_layers", "hcom_allGather_", "shared_gather"),
        ("词表Embedding恢复", "Embedding/Engram", "primary", "hcom_allReduce_", "embed_tp"),
        ("Engram输入gather", "Embedding/Engram", "primary", "hcom_allGather_", "engram_tp"),
    )
    out = (
        "<table><tr><th>实际功能路径 / 归属</th>"
        + "".join(f"<th>{esc(label)}</th>" for label in labels)
        + "<th>源码</th></tr>"
    )
    for title, module, phase, typ, ref in paths:
        out += f"<tr><td>{esc(title)} → {esc(module)}</td>"
        for sample in data["samples"].values():
            rr = [
                r
                for r in sample["tasks"]
                if r["module"] == module
                and r["type"] == typ
                and (r["phase"] == phase or (phase == "primary" and r["stage"] != "draft"))
            ]
            out += f"<td>{esc(typ)} × {len(rr)} / {number(sum(r['duration_us'] for r in rr))} μs</td>"
        out += f'<td><a href="#src-{ref}">参考源码与捕获调用链</a></td></tr>'
    return out + "</table>"


def moe_composite(data):
    """Compare the same feature group despite differing fusion boundaries."""
    out = "<table><tr><th>派生组合视图（不再加入模块合计）</th>"
    out += "".join(f"<th>{esc(label)}</th>" for label in data["samples"]) + "</tr>"
    export = []
    for stage, title in (("primary", "目标段MoE子项整体"), ("draft", "草稿MoE子项整体")):
        out += f"<tr><th>{title}</th>"
        for label, sample in data["samples"].items():
            rows = [
                r
                for r in sample["tasks"]
                if r["module"].startswith("MoE-") and ((r["stage"] == "draft") == (stage == "draft"))
            ]
            merged = []
            for start, end in sorted((r["start_ns"], r["end_ns"]) for r in rows):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            resident = sum(r["duration_us"] for r in rows)
            union = sum(e - s for s, e in merged) / 1000
            out += f"<td>{len(rows)} 记录；Σ {number(resident)} μs<br>区间并集 {number(union)} μs</td>"
            export.append(
                dict(
                    scope=stage,
                    sample=label,
                    group="MoE(all physical submodules)",
                    count=len(rows),
                    resident_us=resident,
                    union_us=union,
                    derived=True,
                )
            )
        out += "</tr>"
    with (HERE / "logical_group_comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(export[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(export)
    return out + "</table>"


def task_cell(summary, module, sample):
    value = summary["modules"][module]
    if not value["count"]:
        if (
            module == "MoE-SharedExpert"
            and sample["metadata"].get("fused_shared_experts")
            and summary["modules"]["MoE-Fused"]["count"]
        ):
            return '<td class="fused">无独立任务<br>包含在 MoE-Fused<br><small>不可拆分，不是零计算</small></td>'
        if module in ("MoE-RoutedExpert", "MoE-Dispatch/Combine") and summary["modules"]["MoE-Fused"]["count"]:
            return '<td class="fused">主体包含在 MoE-Fused<br><small>此范围无独立任务</small></td>'
        return "<td>未观察到独立任务</td>"
    types = collections.Counter()
    for item in value["types"]:
        types[canonical(item["type"])] += item["count"]
    names = [f"<code>{esc(t)}</code> × {n}" for t, n in types.items()]
    visible = "<br>".join(names[:4])
    extra = "<details><summary>全部类型</summary>" + "<br>".join(names) + "</details>" if len(names) > 4 else ""
    costs = (
        f"<b>Σ {number(value['resident_us'])} μs</b> · {value['count']} 任务"
        f"<br><small>区间并集 {number(value['union_us'])} μs"
        f"<br>hcom {number(value['hcom_us'])} μs / 辅助 {number(value['comm_helper_us'])} μs</small>"
    )
    return f"<td>{costs}<div class=types>{visible}{extra}</div></td>"


def feature_table(data, scope, sources):
    labels = list(data["samples"])
    rows = []
    for module in data["modules"]:
        description, _, refs = FEATURES[module]
        difference = difference_note(module, data, scope)
        links = " / ".join(f'<a href="#src-{esc(ref)}">{esc(sources[ref]["title"])}</a>' for ref in refs)
        row = f"<tr><th>{esc(module)}<small>{esc(description)}</small></th>"
        for label in labels:
            sample = data["samples"][label]
            row += task_cell(sample["summary"][scope], module, sample)
        row += f"<td>{esc(difference)}<div class=sourcelinks>{links}</div></td></tr>"
        rows.append(row)
    header = (
        "<tr><th>模型功能模块</th>"
        + "".join(f"<th>{esc(label)}：Device观察成本与执行节点</th>" for label in labels)
        + "<th>归属规则 / 实现差异 / 源码依据</th></tr>"
    )
    return '<div class="scroll"><table class="feature">' + header + "".join(rows) + "</table></div>"


def example_block(data, module, label):
    rows = [r for r in data["samples"][label]["tasks"] if r["module"] == module]
    if not rows:
        return f"<p>{esc(label)}：没有独立任务，见融合归属说明。</p>"
    selected = [max(rows, key=lambda r: r["duration_us"])]
    communication = [r for r in rows if r["kind"] == "hcom"]
    if communication and selected[0] not in communication:
        selected.append(communication[0])
    out = []
    for r in selected:
        frames = [p for p in r["parents"] if "vllm_ascend/" in p]
        session_base = data["samples"][label]["timing"]["worker_entry_session_ms"]
        body = (
            f"<p>阶段：{esc(r['phase'])}；层：{esc(r['layer'])}；module rule：{esc(r['rule'])}；依据：{esc(r['basis'])}</p>"
            f"<p>本地 +{float(r['start_ms']):.6f} → +{float(r['end_ms']):.6f} ms；"
            f"Stream {esc(r['stream'])} / Task {esc(r['task'])} / Connection {esc(r['connection'])} / CSV {esc(r['csv_line'])}。</p>"
            f"<p>SESSION坐标：{session_base + float(r['start_ms']):.6f} → {session_base + float(r['end_ms']):.6f} ms；Insight显示零点不同则先用任务ID校准。</p>"
            f"<p>Input：<code>{esc(r['shape'])}</code><br>dtype：<code>{esc(r['dtype'])}</code><br>Output：<code>{esc(r['output'])}</code></p>"
            f"<pre>{esc(chr(10).join(frames))}</pre>"
        )
        out.append(
            f"<details><summary>{esc(label)} / {esc(canonical(r['type']))} / {number(r['duration_us'])} μs</summary>{body}</details>"
        )
    return "".join(out)


def observed_timing(data):
    keys = (
        ("Worker入口 → 最后关联NPU任务完成", lambda s: s["timing"]["model_path_device"]["end_ms"]),
        ("本轮Device首任务 → 末任务（实际跨度）", lambda s: s["timing"]["model_path_device"]["span_ms"]),
        (
            "目标40层：首HcPre → 末HcPost",
            lambda s: (
                max(
                    r["end_ns"]
                    for r in s["tasks"]
                    if r["phase"] == "target_layers" and r["type"] in ("HcPost", "MhcPost")
                )
                / 1e6
                - min(r["start_ns"] for r in s["tasks"] if r["phase"] == "target_layers" and r["type"] == "HcPre") / 1e6
            ),
        ),
        (
            "草稿3层设备包络（独立列示）",
            lambda s: next(p["device"]["span_ms"] for p in s["timing"]["phases"] if p["id"] == "draft_layers"),
        ),
    )
    out = (
        "<table><tr><th>本轮实测时间口径（ms）</th>" + "".join(f"<th>{esc(k)}</th>" for k in data["samples"]) + "</tr>"
    )
    for name, get in keys:
        out += (
            f"<tr><td>{esc(name)}</td>" + "".join(f"<td>{get(s):.3f}</td>" for s in data["samples"].values()) + "</tr>"
        )
    return out + "</table>"


def operator_table(data):
    labels = list(data["samples"])
    html_rows = []
    output_rows = []
    for module in data["modules"]:
        aliases = set()
        by_sample = {}
        for label, sample in data["samples"].items():
            dd = collections.defaultdict(list)
            for task in sample["tasks"]:
                if task["module"] == module and task["stage"] != "draft":
                    dd[canonical(task["type"])].append(task)
                    aliases.add(canonical(task["type"]))
            by_sample[label] = dd
        rows = []
        for typ in sorted(aliases):
            row = {"module": module, "type_alias": typ}
            cells = [f"<td>{esc(typ)}</td>"]
            for label in labels:
                rr = by_sample[label].get(typ, [])
                total = sum(r["duration_us"] for r in rr)
                shapes = list(dict.fromkeys(r["shape"] for r in rr))
                dtypes = list(dict.fromkeys(r["dtype"] for r in rr))
                row.update(
                    {
                        f"{label}_count": len(rr),
                        f"{label}_resident_us": total,
                        f"{label}_input_shapes": " | ".join(shapes),
                        f"{label}_dtypes": " | ".join(dtypes),
                    }
                )
                cells.append(
                    f"<td>{len(rr)} 次 / {number(total)} μs<details><summary>shape / dtype</summary><pre>{esc(chr(10).join(shapes))}\n{esc(chr(10).join(dtypes))}</pre></details></td>"
                )
            output_rows.append(row)
            rows.append("<tr>" + "".join(cells) + "</tr>")
        header = "<tr><th>算子类别（别名）</th>" + "".join(f"<th>{esc(k)}</th>" for k in labels) + "</tr>"
        html_rows.append(
            f"<details><summary>{esc(module)} · {len(rows)} 类节点</summary><div class=scroll><table>{header}{''.join(rows)}</table></div></details>"
        )
    with (HERE / "operator_type_comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    return "".join(html_rows)


def main():
    data = json.loads(gzip.decompress((HERE / "analysis.json.gz").read_bytes()))
    source_data = json.loads((HERE / "sources.json").read_text())
    sources = {s["id"]: s for s in source_data["sources"]}
    metadata = ""
    for label, sample in data["samples"].items():
        m = sample["metadata"]
        balance = "已启用专家强制均衡（受控基准）" if m.get("forced_expert_balance") else "未启用专家强制均衡"
        version = (
            f"<br>版本线索：{esc(m['user_reported_base_version'])}（用户说明；镜像源码SHA待明确）"
            if m.get("user_reported_base_version")
            else ""
        )
        metadata += f"<div class=card><b>{esc(label)} · {esc(m['hardware'])}</b><p>DP{m['dp']} × TP{m['tp']} / EP{m['ep']} / {esc(m['runner'])}<br>{esc(m['native_quantization'])}<br>CANN {esc(m['cann'])}；{balance}{version}</p><small>{esc(m['archive'])} / {esc(sample['timing']['rank_id'])} / local step {sample['timing']['step_index']}</small></div>"
    primary = feature_table(data, "primary", sources)
    core = feature_table(data, "target_layers", sources)
    draft = feature_table(data, "draft", sources)
    examples = "".join(
        f"<details><summary>{esc(module)}</summary>{''.join(example_block(data, module, label) for label in data['samples'])}</details>"
        for module in data["modules"]
    )
    code = "".join(
        f'<details id="src-{esc(s["id"])}"><summary>{esc(s["title"])}</summary><p><a href="{esc(s["url"])}">{esc(s["path"])}:{s["line"]}</a> · 固定参考版本 <code>{esc(s["commit"][:12])}</code></p><pre>{esc(s["snippet"])}</pre></details>'
        for s in source_data["sources"]
    )
    operators = operator_table(data)
    rows_csv = []
    for scope in ("primary", "target_layers", "draft"):
        for module in data["modules"]:
            row = dict(scope=scope, module=module)
            for label, s in data["samples"].items():
                v = s["summary"][scope]["modules"][module]
                row.update(
                    {
                        f"{label}_count": v["count"],
                        f"{label}_resident_us": v["resident_us"],
                        f"{label}_union_us": v["union_us"],
                        f"{label}_hcom_us": v["hcom_us"],
                        f"{label}_comm_helper_us": v["comm_helper_us"],
                    }
                )
            rows_csv.append(row)
    with (HERE / "module_comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_csv[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_csv)
    coverage = " / ".join(
        f"{label} {len(s['tasks'])} 任务，主段 {s['summary']['primary']['total']['count']}，草稿 {s['summary']['draft']['total']['count']}"
        for label, s in data["samples"].items()
    )
    prior = "；".join(
        f"{label} 前轮关联工作到入口后 +{s['timing']['previous_work_before_first_target'].get('end_ms', 0):.3f} ms"
        for label, s in data["samples"].items()
    )
    eligibility = (
        "已标记为受控，仍须审查特性与融合归属"
        if data["comparability"].get("controlled")
        else "未受控，暂不计算Δ或加速比"
    )
    downloads = " · ".join(
        f'<a href="{esc(label.lower())}_module_tasks.csv">{esc(label)}逐任务归属</a>' for label in data["samples"]
    )
    # The report separates observed costs, causal ownership, and comparison
    # eligibility. No module subtraction is emitted for uncontrolled workloads.
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DSv4.1 P侧Device模块与算子差异</title>
<style>
:root{{font-family:system-ui,"Noto Sans CJK SC",sans-serif;color:#192d40;color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f9;line-height:1.6}}main{{max-width:1540px;margin:22px auto;background:white;padding:28px 34px;border:1px solid #dbe5ef;border-radius:12px}}h1{{font-size:28px;margin:12px 0}}h2{{font-size:20px;margin-top:30px}}h3{{font-size:17px}}a{{color:#1e5ca4}}nav,.note,small{{font-size:13px;color:#51677b}}p{{margin:8px 0}}.notice{{padding:12px 16px;background:#edf4ff;border-left:4px solid #3672b7}}.cards{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:14px 0}}.card{{padding:12px 16px;border:1px solid #dbe5ef;border-radius:8px}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}}th,td{{text-align:left;vertical-align:top;border:1px solid #dbe5ef;padding:10px 12px}}th{{background:#f0f5fa}}.feature{{min-width:1140px;table-layout:fixed}}.feature th:first-child{{width:15%}}.feature th:last-child{{width:26%}}.feature th small{{display:block;font-weight:normal;margin-top:5px}}.types{{font-size:12px;margin-top:8px;overflow-wrap:anywhere}}.fused{{background:#fff8e8}}.sourcelinks{{margin-top:7px;font-size:12px}}code,pre{{font-family:ui-monospace,SFMono-Regular,monospace}}code{{overflow-wrap:anywhere}}pre{{padding:10px;background:#f5f8fb;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}details{{margin:8px 0}}summary{{cursor:pointer;color:#1b527e}}.legend{{padding:12px 16px;border:1px dashed #9eb9d2}}.numbers{{font-variant-numeric:tabular-nums}}li{{margin:5px 0}}@media(max-width:850px){{main{{margin:0;padding:18px 14px}}.cards{{grid-template-columns:1fr}}h1{{font-size:23px}}}}@media print{{nav{{display:none}}main{{border:0;margin:0;padding:0}}details[open]{{break-inside:avoid}}table{{font-size:9px}}th,td{{padding:5px}}.feature{{min-width:0}}}}@page{{size:A3 landscape;margin:12mm}}
</style></head><body><main>
<nav><a href="../../llm_service_optimization/index.html">优化总览</a> · <a href="../stage_brief.html">全局简版</a> · <a href="../execution_scope/report.html">旧执行边界审计</a> · 2026-11-26</nav>
<h1>DSv4.1 P侧：Device模块与算子差异</h1>
<p class=notice><b>本轮打法：</b>沿模型功能归属设备任务，先列清“算了什么、用了什么算子、哪些已融合、TP通信落在哪里”。<b>不做Host耗时比较，不把当前A3/A5耗时差额当性能结论。</b>Runner作为实现背景保留；它仍可能影响Device准备和算子路径。</p>
<div class=cards>{metadata}</div>
<p class=note>输入配置标记：<b>{eligibility}</b>。query/历史KV、TP/EP、每专家负载和特性配置逐项审查；卡片中的均衡状态来自输入配置，并非脚本从shape自动确认。即使TP1、同shape并强制均衡，也仍需核对上下文与EP负载；量化不同则比较原生方案，不能解释为纯硬件差异。</p>
<h2 id=timing>1. 当前一个有效step的真实观测时长</h2>
{observed_timing(data)}
<p class=note>第1行从Worker.execute_model入口计时，以本轮最后一个精确关联NPU任务完成为终点；不是Host函数返回时间，也不是完整128k请求TTFT。EngineCore完整step墙钟与最终CPU输出完成不在本包的完整捕获范围。Device跨度直接取实际首尾，并非模块累加；它包含空泡和融合内等待。{prior}，故第1行也不能直接比较净算术能力。</p>
<h2 id=modules>2. 按模型功能归属：目标段及设备准备</h2>
<div class=legend><b>主表范围：</b>本轮Device准备、Embedding/Engram、40层主干、目标收尾和目标采样；草稿另表。包含采样是为了把不同Runner中位置不同的同类输出头工作归在一起。<br><b>指标：</b>Σ为该模块所有已归属任务的驻留累计，包含显式hcom及通信辅助；“区间并集”消除该模块内部重叠。模块之间仍可重叠，<b>Σ或各模块并集都不能相加当step耗时</b>。MegaMoE内等待也属于驻留，不等于纯FFN计算。</div>
{primary}
<p class=note>{esc(coverage)}。每条设备记录只计一个模块，hcom聚合事件与辅助kernel分别列示；融合不能拆成虚构的shared/routed/communication耗时。Layout/Boundary保留结构性未细分项，归属覆盖不代表每个小任务的语义都已细拆。</p>
<h3>跨融合边界的组合视图</h3>
<p>仅对比两边的MegaMoE还不够：A5含共享专家，A3有独立共享分支。下表将Gate、Shared、Routed、Dispatch/Combine、Fused及MoE外围重新组合，保留整体功能成本；独立Norm/Quant仍按主表定义排除。当前仍不计算速度差，后续受控数据应先看组合，再看子项。</p>
{moe_composite(data)}
<details><summary>仅看40层主干：排除设备准备、Embedding前置、收尾与采样</summary>{core}</details>
<h2 id=communication>3. A3的TP通信具体算到哪里</h2>
{communication_table(data)}
<p class=note>A5 TP1没有上述显式TP恢复任务；EP8的专家交换仍在MegaMoE内，不能写成“A5无通信”。共享专家行的A3通信已计入，不能再在MoE-Boundary重复统计。</p>
<h2 id=draft>4. 草稿模型单列：相同规则，不混入40层</h2>
<p>草稿准备、3层与草稿头/候选整理独立归属。A3的DispatchFFNCombine保留完整融合边界；A5有可独立观察的GMM与dispatch/combine。query、量化和负载不同，目前不能用36.82 ms与1.00 ms的差额宣称收益。</p>
<details><summary>展开草稿Device模块表</summary>{draft}</details>
<h2 id=types>5. 模块内算子类型、shape与dtype对照</h2>
<p>同名不自动等于同规格，DSL长名字仅在展示时使用别名；原始名称、shape、dtype、时间和任务ID全部保留在CSV。Norm/Quant按独立物理节点统一计数；压缩epilogue等跨功能融合保留KV/RoPE归属。</p>
{operators}
<h2 id=attribution>6. 归属方法与可在Insight核对的任务</h2>
<ol><li>沿PYTORCH_API.connectionId组 → CONNECTION_IDS原始Connection → TASK → 精确startNs匹配CSV，恢复任务的提交线程与操作。</li><li>在该线程同一操作时刻恢复捕获的Python/nn.Module祖先，识别Indexer、共享专家、输出投影等功能边界。MatMul不靠名字或最近时间猜归属。</li><li>HC、完整MoE融合、独立Norm/Quant和RoPE等采用明确物理边界规则；其余优先采用最内层可识别功能调用。结构性边界显式保留。</li><li>通信使用同一调用祖先归属；一个任务只计一次。源码用于解释功能与分支条件，运行路径以捕获栈和设备任务为准。</li></ol>
{examples}
<h2 id=source>7. 固定参考源码与采集源码的边界</h2>
<p class=notice>以下是CANN参考快照 <code>36b73a9c431d</code> 的真实源码片段与网页链接。A3采集函数路径与其对应，但采集镜像精确SHA未提供；A5来自 <code>/opt/vllm-ascend-dsnext</code>。用户说明A5为0.27版本背景；已检查的官方 <a href="https://github.com/vllm-project/vllm-ascend/tree/0ce124446a0f59c431da76e31ffcc53489622513">0.27.1rc快照</a> 没有采集中的dsa_v41.py，尚不能据版本号替代镜像源码。<b>A5列使用采集栈、输入/输出与融合接口证据；不把参考源码冒充A5镜像原文。</b>精确源码或分支明确后可补逐行差异。</p>
{code}
<h2 id=next>8. 下一批受控数据：沿同一台账继续比较</h2>
<table><tr><th>对齐项</th><th>必须记录的内容</th></tr><tr><td>输入工作量</td><td>相同query长度、历史KV长度、批内请求数；同层/同轮比较。KV容量与max-model-len不能代替实际上下文。</td></tr><tr><td>并行与路由</td><td>A3 TP1，记录DP/EP、总/本地专家数、top-k、每专家token分布；强制均衡用于受控基准，结果单独标记，不代表真实路由吞吐。</td></tr><tr><td>模型特性</td><td>Engram、SWA replay/尾窗、缓存命中与窗口、DSpark、KV dtype及原生量化。A3存在而A5未观察到的Engram路径必须核对。</td></tr><tr><td>采集与范围</td><td>同样排除启动/捕获/warmup；选真实Prefill；记录镜像、core/Ascend SHA、CANN、时间坐标，覆盖整组EP并保留前序工作情况。</td></tr></table>
<p>先用受控数据建立模块成本差，再从Norm+Quant、KV epilogue、Indexer专门前处理、MoE融合边界中选择A3实际支持的实现。算子微基准固定shape/dtype/布局，完整MegaMoE用整组EP与固定专家负载；最终仍回到A3模型执行A/B。</p>
<details><summary>重新生成：保留现在的统计方法，替换输入</summary><p>使用 <a href="new_inputs.template.json">新输入配置模板</a>，填入真实采集信息：</p><pre>python analyze.py --list-workers /path/to/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db
# 在新的配置中填写 database、worker_entry_ns 和 metadata；
# 去掉旧 evidence_json / tasks_csv，即可从新DB和同目录kernel_details.csv抽取。
python analyze.py --config new_inputs.json
python build_report.py</pre><p>当前归属规则针对DSv4.1的40层与3层DSpark捕获。新版本出现新的融合或调用边界时，先审查未细分项，再更新scope_owner规则；不自动声称新数据已可比。仅渲染已归档结果不需要原始DB。</p></details>
<p>下载：<a href="module_comparison.csv">模块汇总CSV</a> · <a href="logical_group_comparison.csv">MoE组合CSV</a> · <a href="operator_type_comparison.csv">类型/shape对照CSV</a> · {downloads} · <a href="analysis.json.gz">完整关联与归属数据</a> · <a href="sources.json">源码索引</a> · <a href="inputs.json">当前输入配置</a></p>
</main><script>
function revealSource() {{
  const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (!target) return;
  for (let node = target; node; node = node.parentElement) {{
    if (node.tagName === 'DETAILS') node.open = true;
  }}
}}
window.addEventListener('hashchange', revealSource);
revealSource();
</script></body></html>"""
    (HERE / "report.html").write_text(page.rstrip() + "\n")
    print("Built Device feature report and CSV ledgers")


if __name__ == "__main__":
    main()
