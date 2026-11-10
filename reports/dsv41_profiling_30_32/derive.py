#!/usr/bin/env python3
"""Build source/trace evidence, layer reconstruction and compact timelines."""

import argparse
import json
import re
import subprocess
from pathlib import Path
from zipfile import ZipFile

from analyze import HERE, csv_rows, number, record_kind

WORKSPACE = HERE.parent.parent
ASCEND_OLD = "7d825229c77025cfb185a010c25819c1214b9450"
ASCEND_NEW = "d08fa9d4380684b64fd7a45a876fc080ecbd1d22"


def discover_path(relative_paths, marker):
    """Find inputs in either a repository checkout or the original workspace."""
    for parent in (WORKSPACE, *WORKSPACE.parents):
        for relative in relative_paths:
            candidate = parent / relative
            if (candidate / marker).exists():
                return candidate
    return None


def load(name):
    return json.loads((HERE / "data" / name).read_text())


def first_tokens(shape):
    match = re.match(r"^\D*(\d+)", shape)
    return int(match[1]) if match else None


def excerpt(repo, revision, path, term, before=4, after=10):
    content = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=repo, text=True)
    lines = content.splitlines()
    index = next(i for i, line in enumerate(lines) if term in line)
    lo, hi = max(0, index - before), min(len(lines), index + after)
    repository_name = "vllm-ascend" if path.startswith("vllm_ascend/") else "vllm"
    return {
        "revision": revision,
        "path": path,
        "line": lo + 1,
        "text": "\n".join(f"{i + 1}: {lines[i]}".rstrip() for i in range(lo, hi)),
        "url": f"https://github.com/vllm-project/{repository_name}/blob/{revision}/{path}#L{lo + 1}",
    }


def main(ascend_repo, vllm_repo, pr_reports):
    analysis, traces, db = (
        load("analysis.json"),
        load("traces.json"),
        load("db_metadata.json"),
    )
    results = {
        "layers": [],
        "engram_windows": [],
        "topology": [],
        "sources": [],
        "timelines": [],
        "pr_verification": [],
    }
    for item in db:
        groups = json.loads(item["meta"].get("parallel_group_info", "{}"))
        sizes = {g["group_name"]: len(g["global_ranks"]) for g in groups.values()}
        results["topology"].append(
            {
                "dataset": item["dataset"],
                "rank": item["rank"],
                "groups": groups,
                "tp_size": sizes.get("tp"),
                "ep_size": sizes.get("ep"),
                "dp_if_no_cp": sizes.get("ep", 0) // max(sizes.get("tp", 1), 1),
            }
        )
    for trace in traces[:2]:
        events = sorted(trace["evidence_events"], key=lambda e: float(e["ts"]))
        lookups = [e for e in events if e["name"] == "_engram_host_uva_gather_dequant_kernel"]
        for i, event in enumerate(lookups):
            start = max(
                float(e["ts"])
                for e in events
                if e["name"] == "aten::contiguous" and float(e["ts"]) < float(event["ts"])
            )
            preceding = [e for e in events if start <= float(e["ts"]) < float(event["ts"])]
            results["engram_windows"].append(
                {
                    "dataset": trace["dataset"],
                    "lookup": i,
                    "timestamp_us": event["ts"],
                    "preceding_names": [e["name"] for e in preceding],
                    "zero_calls": sum(e["name"] == "aten::zeros" for e in preceding),
                    "events": preceding + [event],
                }
            )
    for ai, archive in enumerate(analysis["archives"]):
        with ZipFile(archive["path"]) as z:
            for rank in analysis["ranks"]:
                if rank["archive_index"] != ai or rank["rank"]["tp"] != 0:
                    continue
                rows = list(csv_rows(z, rank["files"]["kernel_details.csv"]))
                if rank["id"] in ["v030_unlabelled", "v032_p_tp4"]:
                    sizes = {1023, 1022, 828} if rank["id"] == "v030_unlabelled" else {8188, 128}
                    hc = sorted(
                        [r for r in rows if r["Type"] == "HcPre" and first_tokens(r["Input Shapes"]) in sizes],
                        key=lambda r: number(r["Start Time(us)"]),
                    )
                    assert len(hc) == 240
                    layer_pairs = [hc[i : i + 2] for i in range(0, 240, 2)]
                    for i, pair in enumerate(layer_pairs):
                        assert first_tokens(pair[0]["Input Shapes"]) == first_tokens(pair[1]["Input Shapes"])
                        token_count = first_tokens(pair[0]["Input Shapes"])
                        if rank["id"] == "v032_p_tp4":
                            assert token_count == (8188 if i % 40 < 21 else 128)
                        results["layers"].append(
                            {
                                "dataset": rank["id"],
                                "step": i // 40,
                                "layer": i % 40,
                                "tokens": token_count,
                                "hcpre_duration_us": sum(number(r["Duration(us)"]) for r in pair),
                                "first_relative_us": number(pair[0]["Start Time(us)"]) - rank["kernel_origin_us"],
                            }
                        )
                names, shapes, intern_names, intern_shapes = [], [], {}, {}
                compact = []
                for row in rows:
                    name, shape = row["Type"], row.get("Input Shapes", "")
                    if name not in intern_names:
                        intern_names[name] = len(names)
                        names.append(name)
                    if shape not in intern_shapes:
                        intern_shapes[shape] = len(shapes)
                        shapes.append(shape)
                    lane = (
                        "通信op"
                        if record_kind(row) == "communication_op"
                        else "通信task"
                        if record_kind(row) == "communication_task"
                        else (
                            "MoE融合"
                            if any(s in name for s in ["Moe", "MoE", "DispatchFFN", "GroupedMatmul"])
                            else "Attention/索引"
                            if any(s in name for s in ["Mla", "Lightning", "Indexer"])
                            else "HIXL/拷贝"
                            if any(s in name for s in ["Hixl", "Memcpy"])
                            else "其他NPU"
                        )
                    )
                    compact.append(
                        [
                            round(
                                number(row["Start Time(us)"]) - rank["kernel_origin_us"],
                                3,
                            ),
                            number(row["Duration(us)"]),
                            intern_names[name],
                            intern_shapes[shape],
                            lane,
                        ]
                    )
                trace = next(t for t in traces if t["dataset"] == rank["id"] and t["rank"] == rank["rank"])
                host_events = trace["long_events"] + [
                    e
                    for e in trace["evidence_events"]
                    if e["name"]
                    in [
                        "aten::zeros",
                        "aten::contiguous",
                        "_engram_host_uva_gather_dequant_kernel",
                    ]
                ]
                host = [
                    {
                        "start": float(e["ts"]) - rank["kernel_origin_us"],
                        "duration": float(e["dur"]),
                        "name": e["name"],
                        "pid": e.get("pid"),
                        "tid": e.get("tid"),
                    }
                    for e in host_events
                    if e.get("cat") == "cpu_op" or e["name"] == "Free"
                ]
                results["timelines"].append(
                    {
                        "dataset": rank["id"],
                        "rank": rank["rank"],
                        "origin_us": rank["kernel_origin_us"],
                        "span_us": rank["kernel_span_us"],
                        "names": names,
                        "shapes": shapes,
                        "events": compact,
                        "host": host,
                        "archive": archive["path"],
                        "source": rank["files"]["kernel_details.csv"],
                    }
                )
    selections = json.loads((pr_reports / "deepseek_summary/summary.json").read_text())["records"]
    # These labels describe the evidence available in these captures, not a
    # universal judgment on the PR or a replacement for an isolated benchmark.
    special = {
        17496: (
            "机制已验证",
            "主干21–39层从8188缩为128；3轮×19层×2个HcPre=114条小窗口记录。",
            "同配置关闭replay再采集；对齐batch后测端到端TTFT。",
        ),
        19657: (
            "机制已验证",
            "0.30的6次Engram lookup前各有一次aten::zeros；0.32对应6个窗口为0次。",
            "同shape、同TP/DP对照zeros/empty；再测整个请求延迟。",
        ),
        17742: (
            "未隔离前缀命中",
            "decoder尾窗裁剪已见，但这批导出没有请求级prefix-hit/窗口重建标记，不能由此证明cache-hit恢复路径。",
            "带请求ID采集冷/热前缀、Store来源、replay-start和重建区间。",
        ),
        18936: (
            "路径相容",
            "官方32的replay执行与新runner API相容；未单独隔离API迁移的收益。",
            "记录实际runner并保留启动日志。",
        ),
        19321: (
            "CP条件未证实",
            "未获得PCP-size元数据；SWA裁剪不等价于PCP支持分支实际执行。",
            "PCP>1拓扑下记录group及相同请求的开关对照。",
        ),
        20210: (
            "CP条件未证实",
            "看到多DP拓扑，但未证明PCP+DP组合；不能仅凭DP文件名验证此PR。",
            "记录PCP组与replay上下文跨rank一致性。",
        ),
        18081: (
            "编译路径有证据",
            "11月9日D侧有一次npu_fx_compiler inference；不能据此证明全模型静态kernel或量化提速。",
            "采集编译配置、图段、命中bucket和eager回退。",
        ),
        18191: (
            "编译路径有证据",
            "同上：编译执行出现；inputs_embeds动态维收益没有单独对照。",
            "固定输入嵌入路径，对比编译图/重编译次数。",
        ),
        19409: (
            "尚未验证",
            "未获得embedding缓冲容量/有效行数及该拷贝的调用栈，不能由全局copy_减少归因。",
            "加embedding scope，核对固定容量copy消失与实际字节数。",
        ),
        18293: (
            "算子路径已见",
            "两版均执行HcPre，但token形状与并行布局不同；同名kernel不能证明加载了哪一版实现。",
            "记录自定义算子包hash，按完全相同shape隔离测试。",
        ),
        20441: (
            "算子路径已见",
            "两版均有VllmAscendQuantLightningIndexerV2/v3包装；名称不变，内部vector/TopK优化无法凭名称确认。",
            "核对算子二进制和sparse_count；同shape与相同TopK比较。",
        ),
        16731: (
            "本V4.1路径不对应",
            "PR改的是device_op的通用QLI路径；本次V4.1实际为自定义npu_quant_lightning_indexer_v3。",
            "不要把V4.1 QLI同名现象归因给通用SFA替换。",
        ),
        18786: (
            "本V4.1路径不对应",
            "PR只改attention/sfa_v1.py的V up-proj；V4.1的TransposeBatchMatMul主要是DSA O-proj。",
            "先沿模型调用链确认入口，不能由Transpose减少直接归因。",
        ),
        18256: (
            "本V4.1路径未证实",
            "改动核心是通用MLA prolog/cache布局，未证明本次V4.1进入该路线。",
            "补调用栈或MLA prolog标记再核对隐式contiguous。",
        ),
        17105: (
            "采集不覆盖显存",
            "无memory_record/operator_memory；稳态kernel trace不能证明权重scale释放。",
            "在模型加载前后采集allocated/reserved及权重张量引用。",
        ),
        17344: (
            "采集不覆盖显存",
            "本次没有模型加载阶段的内存与引用记录。",
            "加载阶段快照+热重载对照。",
        ),
        20503: (
            "配置条件未证实",
            "OP State=static在30也存在；它不是enable_static_kernel已开启的证明。",
            "记录enable_static_kernel、MRV2/MTP和实际捕获尺寸。",
        ),
    }
    kv_ids = {17739, 18907, 18899, 19755, 19370, 18647, 14455, 17817, 19472}
    cp_ids = {18506, 19190, 19007, 17995, 18112, 19578}
    compile_ids = {19756, 19778, 18644, 18074, 19304}
    for record in selections:
        pr = record["pr"]
        verdict = special.get(pr)
        if verdict is None and pr in kv_ids:
            verdict = (
                "Store证据不足",
                "当前导出没有可关联的Store lookup/hit/save请求scope；HIXL事件本身不能证明来自KV Pool。",
                "添加Store命中来源、请求ID与异步load/save scope，冷热缓存分别验证。",
            )
        elif verdict is None and pr in cp_ids:
            verdict = (
                "CP条件未证实",
                "TP/EP组可恢复，但PCP/DCP-size及具体元数据路径没有完整记录。",
                "保留PCP/DCP实际配置及逐组metadata/通信scope。",
            )
        elif verdict is None and pr in compile_ids:
            verdict = (
                "配置条件未证实",
                "融合/快路径要求具体FX pattern、dtype或运行模式；现有统计不足以单独定位。",
                "加FX图与融合命中日志，固定shape逐项A/B。",
            )
        elif verdict is None:
            verdict = (
                "尚未验证",
                "未找到能够唯一对应该PR的调用scope、二进制版本和同条件对照。",
                "按适用条件补scope与单变量A/B，不从总耗时倒推PR收益。",
            )
        results["pr_verification"].append(
            {
                **record,
                "trace_verdict": verdict[0],
                "trace_evidence": verdict[1],
                "next_validation": verdict[2],
            }
        )
    source_specs = [
        (
            ascend_repo,
            ASCEND_NEW,
            "vllm_ascend/models/deepseek_v41/engram/embedding.py",
            "out = torch.empty(",
        ),
        (
            ascend_repo,
            ASCEND_OLD,
            "vllm_ascend/models/deepseek_v41/engram/embedding.py",
            "out = torch.zeros(",
        ),
        (
            ascend_repo,
            ASCEND_NEW,
            "vllm_ascend/models/deepseek_v41/decoder_replay_layers.py",
            "def make_decoder_replay",
        ),
        (vllm_repo, "v0.32.0", "vllm/config/cache.py", "swa_bounded_replay: bool"),
        (
            vllm_repo,
            "v0.32.0",
            "vllm/config/vllm.py",
            "use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER",
        ),
    ]
    results["sources"] = [excerpt(*spec) for spec in source_specs]
    (HERE / "data/evidence.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    # Portable PR matrix without raw trace payloads.
    (HERE / "data/pr_verification.json").write_text(
        json.dumps(results["pr_verification"], ensure_ascii=False, indent=2) + "\n"
    )
    print(
        "Layer records",
        len(results["layers"]),
        "Engram windows",
        len(results["engram_windows"]),
        "PR records",
        len(results["pr_verification"]),
        "timeline records",
        sum(len(t["events"]) for t in results["timelines"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ascend-repo",
        type=Path,
        default=discover_path((".", "vllm-ascend"), "vllm_ascend/platform.py"),
    )
    parser.add_argument(
        "--vllm-repo",
        type=Path,
        default=discover_path(("vllm/vllm", "vllm", "."), "vllm/config/cache.py"),
    )
    parser.add_argument(
        "--pr-reports",
        type=Path,
        default=discover_path(("reports", ".worktrees/report-origin/reports"), "deepseek_summary/summary.json"),
    )
    options = parser.parse_args()
    if any(value is None for value in (options.ascend_repo, options.vllm_repo, options.pr_reports)):
        parser.error("Cannot find source inputs; specify --ascend-repo, --vllm-repo and --pr-reports.")
    main(options.ascend_repo, options.vllm_repo, options.pr_reports)
