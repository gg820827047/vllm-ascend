#!/usr/bin/env python3
# HTML fragments retain their authored whitespace and long source URLs.
# ruff: noqa: E501
"""Build a self-contained Chinese report from pinned Git data and reviewed notes."""

import collections
import csv
import datetime as dt
import hashlib
import html
import io
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

from annotations import KIND, SEMANTIC, records

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GITHUB = "https://github.com/vllm-project/vllm-ascend"
EVIDENCE = {
    "e2e": "作者报告：模型 / 服务实测",
    "kernel": "作者报告：局部算子实测",
    "combined": "作者报告：联合 / 混合改动",
    "ported": "作者报告：兼容移植实测",
    "context": "配置对照，不能归因补丁",
    "mechanism": "机制明确，缺独立速度测量",
    "existing": "基线已有，不算新增",
    "local": "本地补丁审阅，无量化结论",
}


def e(value):
    return html.escape(str(value), quote=True)


def pr_link(number, label=None):
    return f'<a href="{GITHUB}/pull/{number}" target="_blank" rel="noopener">{e(label or f"#{number}")}</a>'


def csv_text(rows, fields):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def source(commit, path):
    return f"{GITHUB}/blob/{commit}/{quote(path, safe='/')}"


def commit_type(commit):
    title = commit["subject"].lower()
    if "[ci]" in title:
        return "CI / 构建验证"
    if "[doc]" in title or "[docs]" in title:
        return "文档 / 发布说明"
    if "[test]" in title or "[benchmark]" in title:
        return "测试 / Benchmark 工具"
    if "[revert]" in title or "revert " in title:
        return "回退 / 策略调整"
    if "[performance]" in title:
        return "显式 Performance 标签"
    if "[bugfix]" in title:
        return "修复"
    if "[feature]" in title:
        return "功能 / 使能"
    if "[refactor]" in title:
        return "重构"
    return "其他 / 上游对齐"


def file_area(path):
    for prefix, title in [
        ("vllm_ascend/worker/", "Worker / Runner"),
        ("vllm_ascend/attention/", "Attention"),
        ("vllm_ascend/models/", "模型实现"),
        ("vllm_ascend/distributed/", "分布式 / KV / EC"),
        ("vllm_ascend/ops/", "Python / Triton 算子"),
        ("vllm_ascend/quantization/", "量化"),
        ("vllm_ascend/patch/", "上游补丁"),
        ("csrc/", "AscendC / C++ 算子"),
        ("tests/", "测试"),
        ("docs/", "文档"),
        (".github/", "CI / GitHub"),
    ]:
        if path.startswith(prefix):
            return title
    return "配置 / 工具 / 其他"


def snippet(commit):
    """A bounded production-code diff excerpt; full original patch is linked."""
    chunks = re.split(r"(?=^diff --git )", (HERE / commit["patch"]).read_text(), flags=re.M)
    candidates = []
    for chunk in chunks[1:]:
        line = chunk.splitlines()[0]
        if not re.match(r"diff --git a/(vllm_ascend|csrc)/", line) or "/docs/" in line:
            continue
        hunks = re.split(r"(?=^@@ )", chunk, flags=re.M)[1:]
        for hunk in hunks:
            lines = hunk.splitlines()
            changed = [
                x
                for x in lines
                if x.startswith(("+", "-")) and x[1:].strip() and not x[1:].lstrip().startswith(("#", "//", "*"))
            ]
            if not changed:
                continue
            score = sum(
                bool(
                    re.search(
                        r"copy_|torch.mm|non_blocking|cache|buffer|fused|mask|graph|compile|event|contiguous|norm|prefetch|padding",
                        x,
                        re.I,
                    )
                )
                for x in changed
            )
            score -= sum(x.startswith(("+import ", "+from ", "-import ", "-from ")) for x in changed)
            candidates.append((score, line + "\n" + "\n".join(lines[:42])))
    return max(candidates, key=lambda x: x[0])[1] if candidates else "请查看完整提交补丁。"


def main():
    data = json.loads((HERE / "git_data.json").read_text())
    meta = data["meta"]
    notes = records()
    added = data["added"]
    old = data["old_only"]
    by_pr = {c["pr"]: c for c in added}
    assert len(by_pr) == len(added) == 364
    assert len(old) == 55
    assert len(notes) == len({x["pr"] for x in notes}) == 66
    assert all(x["pr"] in by_pr for x in notes)
    note_by_pr = {n["pr"]: n for n in notes}
    for c in added:
        c["type"] = commit_type(c)
        c["perf_kind"] = KIND[note_by_pr[c["pr"]]["kind"]] if c["pr"] in note_by_pr else "未列为性能重点"
        if c["equivalent_old_prs"]:
            c["dedup"] = "补丁等价"
            c["baseline_prs"] = c["equivalent_old_prs"]
        elif c["pr"] in SEMANTIC:
            c["dedup"] = "同源 / 分支适配"
            c["baseline_prs"] = [SEMANTIC[c["pr"]][0]]
        else:
            c["dedup"] = "未匹配等价项"
            c["baseline_prs"] = []
    for c in old:
        semantic_new = [n for n, (o, _) in SEMANTIC.items() if o == c["pr"]]
        c["mapped_new_prs"] = c["equivalent_new_prs"] or semantic_new
        c["dedup"] = "补丁等价" if c["equivalent_new_prs"] else "同源 / 分支适配" if semantic_new else "独立核对项"
        c["type"] = commit_type(c)
    unmatched = {
        18213: "旧分支镜像 memory dependency pin；新分支通过后续镜像/依赖升级演进，不能按标题判定缺失。",
        18152: "旧版 hybrid PD graph 修复；新侧存在 #18308→#18936 的实现替代链，非同一补丁，需按最终上游 graph API 理解。",
        18578: "旧分支镜像依赖配置；新分支对应各自 Docker/CANN/torch_npu pins。",
        18571: "v0.30.0rc2 专属 release note；文档历史内容差异。",
        19092: "旧分支 DeepSeek V4.1-1017 权重链接；新侧 #19242 同步部署指南，非逐补丁等价。",
    }
    for c in old:
        c["review"] = unmatched.get(c["pr"], "基线修改已有新侧对应；具体映射见 PR 链接。")
    patch_equal = sum(c["dedup"] == "补丁等价" for c in added)
    semantic_equal = sum(c["dedup"] == "同源 / 分支适配" for c in added)
    assert patch_equal == 44 and semantic_equal == 6
    assert sum(not c["mapped_new_prs"] for c in old) == 5

    # Recover exact target paths (numstat's human-readable rename syntax is not a path).
    status_raw = (
        subprocess.check_output(
            ["git", "diff", "--name-status", "-z", "--find-renames", meta["old_sha"], meta["new_sha"]], cwd=REPO
        )
        .decode()
        .rstrip("\0")
        .split("\0")
    )
    statuses = []
    i = 0
    while i < len(status_raw):
        status = status_raw[i]
        if status[0] in "RC":
            statuses.append((status, status_raw[i + 1], status_raw[i + 2]))
            i += 3
        else:
            statuses.append((status, status_raw[i + 1], status_raw[i + 1]))
            i += 2
    assert len(statuses) == len(data["net_files"]) == 1140
    for f, (status, old_path, new_path) in zip(data["net_files"], statuses):
        f.update(status=status, old_path=old_path, target_path=new_path, area=file_area(new_path))
        f["url"] = source(
            meta["old_sha"] if status == "D" else meta["new_sha"], old_path if status == "D" else new_path
        )
    net = data["net_files"]
    assert sum(f["added"] for f in net) == 113380
    assert sum(f["deleted"] for f in net) == 61855
    areas = collections.defaultdict(lambda: dict(files=0, added=0, deleted=0))
    for f in net:
        area = areas[f["area"]]
        area["files"] += 1
        area["added"] += f["added"]
        area["deleted"] += f["deleted"]

    for n in notes:
        c = by_pr[n["pr"]]
        n.update(
            sha=c["sha"],
            subject=c["subject"],
            date=c["date"],
            kind_label=KIND[n["kind"]],
            evidence_label=EVIDENCE[n["evidence"]],
            baseline_prs=c["baseline_prs"],
            patch=c["patch"],
        )
        cache = HERE / "evidence" / "prs" / f"{n['pr']}.json"
        n["api_verified"] = False
        if cache.exists():
            obj = json.loads(cache.read_text())
            assert obj["merged"] and obj["merge_commit_sha"] == c["sha"]
            n["api_verified"] = True
            n["source_fetched_at"] = obj["fetched_at"]

    types = collections.Counter(c["type"] for c in added)
    kinds = collections.Counter(n["kind"] for n in notes)
    metrics = collections.Counter(n["evidence"] for n in notes)
    explicit = [c for c in added if "[performance]" in c["subject"].lower()]
    assert len(explicit) == 23 and all(c["pr"] in note_by_pr for c in explicit)
    data.update(performance=notes, classification_counts=dict(kinds), type_counts=dict(types), areas=dict(areas))
    data["meta"].update(
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        api_sources=len(list((HERE / "evidence/prs").glob("*.json"))),
        performance_records=len(notes),
    )
    (HERE / "report_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    commit_export = [
        dict(
            c,
            baseline_prs=", ".join(map(str, c["baseline_prs"])),
            file_count=len(c["files"]),
            url=f"{GITHUB}/pull/{c['pr']}",
        )
        for c in added
    ]
    old_export = [
        dict(c, mapped_new_prs=", ".join(map(str, c["mapped_new_prs"])), url=f"{GITHUB}/pull/{c['pr']}") for c in old
    ]
    exports = {
        "commits.csv": csv_text(
            commit_export,
            [
                "sha",
                "pr",
                "date",
                "author",
                "subject",
                "type",
                "dedup",
                "baseline_prs",
                "perf_kind",
                "file_count",
                "url",
            ],
        ),
        "baseline_only.csv": csv_text(
            old_export, ["sha", "pr", "date", "subject", "dedup", "mapped_new_prs", "review", "url"]
        ),
        "files.csv": csv_text(net, ["status", "old_path", "target_path", "area", "added", "deleted", "binary", "url"]),
        "performance.csv": csv_text(
            notes,
            [
                "pr",
                "sha",
                "kind_label",
                "domain",
                "title",
                "mechanism",
                "applicable",
                "evidence_label",
                "result",
                "caveat",
                "api_verified",
            ],
        ),
    }
    for name, value in exports.items():
        (HERE / name).write_text("\ufeff" + value)

    lineage = f"""<div class="lineage" aria-label="分支关系"><div class="root">共同祖先<br><code>{meta["merge_base"][:12]}</code></div><div class="split">↗<br>↘</div><div><div class="branch new">v0.31.0rc · 364 独有提交<br><code>{meta["new_sha"][:12]}</code> · 2026-10-26</div><div class="branch old">v0.30.0rc2 · 55 独有提交<br><code>{meta["old_sha"][:12]}</code> · 2026-10-18</div></div></div>"""
    count_cards = "".join(
        f'<div class="stat"><strong>{num}</strong><span>{label}</span></div>'
        for num, label in [
            (364, "新侧独有提交 / PR"),
            (55, "旧侧独有提交 / PR"),
            (44, "补丁等价项"),
            (6, "同源适配项"),
            (1140, "净变更文件"),
            (len(notes), "性能及关联审阅记录"),
        ]
    )
    kinds_text = "；".join(f"{KIND[k]} {kinds[k]} 条" for k in KIND if kinds[k])
    area_rows = "".join(
        f'<tr><td>{e(area)}</td><td>{v["files"]}</td><td class="positive">+{v["added"]:,}</td><td class="negative">−{v["deleted"]:,}</td><td><div class="bar" style="width:{max(1, v["files"] / max(x["files"] for x in areas.values()) * 100):.2f}%"></div></td></tr>'
        for area, v in sorted(areas.items(), key=lambda x: x[1]["files"], reverse=True)
    )
    type_rows = "".join(
        f'<div class="type-item"><span>{e(k)}</span><strong>{v}</strong></div>' for k, v in types.most_common()
    )
    semantic_rows = "".join(
        f"<tr><td>{pr_link(n)}</td><td>{pr_link(o)}</td><td>{e(reason)}</td></tr>"
        for n, (o, reason) in SEMANTIC.items()
    )
    highlight_ids = [18506, 17263, 18074, 18717, 16996, 18390, 15571, 16542, 17068, 18293, 18907, 16730, 16731]
    measure_rows = "".join(
        f'<tr><td><a href="#perf-{n}">#{n}</a><br><small>{e(note_by_pr[n]["title"])}</small></td><td>{e(EVIDENCE[note_by_pr[n]["evidence"]])}</td><td>{e(note_by_pr[n]["result"])}</td><td>{e(note_by_pr[n]["caveat"])}</td></tr>'
        for n in highlight_ids
    )
    summaries = [
        (
            "执行引擎与上游对齐",
            "对齐 vLLM v0.31.0，扩大默认 MRV2 范围；V1、310P、xlite runner 目录隔离。这类迁移产生大量文件行差异，不能将行数解释为性能价值。",
            [18229, 17681, 18995, 19159, 18863, 19330],
        ),
        (
            "Attention 与算子",
            "新增/更换 FlashAttn、FlashMLA、mixed-quant SFA、CANN pooled indexing，融合 DSpark index 构建、MiniMax 准备/TopK 和 hc_pre 路径。硬件及 cache layout 条件较强。",
            [18717, 18742, 17354, 16730, 18295, 16542, 18294, 18293],
        ),
        (
            "并行、MoE 与内存",
            "PCP O projection 从权重交换转向 activation 交换并共享 scratch；恢复 SP MMRS 编译融合；优化 router dtype、事件和共享专家量化；修复共享 KV 池预算。",
            [18506, 19190, 18074, 17068, 17603, 18474, 19264, 19521],
        ),
        (
            "KV Pool 与传输",
            "减少 block/mask 扫描、lookup/save 开销，调整 KVPP layerwise 预取；扩展 decode offloading 和 encoder cache transfer，同时修复 transfer 失败、lease、cache consistency 等生命周期问题。",
            [17739, 18907, 18647, 14455, 16124, 19251, 19070, 19480],
        ),
        (
            "投机与 bounded replay",
            "引入 V4.1 SWA/decoder bounded replay 演进链、fixed-K PCP sharding、block verification 和 standalone draft model；收益需要接受率、图模式和 workload 共同验证。",
            [17742, 17496, 18936, 18112, 19138, 19364],
        ),
        (
            "多模态与工程支撑",
            "Qwen3.5-VL 图像预处理下沉 NPU，MRV2 encoder graph、CPU encoder cache offload；新增 runtime guard、profiling CI 与 release note fragment 工作流。",
            [16996, 15571, 13894, 16849, 18644, 18875, 18819],
        ),
    ]
    summary_html = "".join(
        f'<article class="summary-box"><h3>{e(title)}</h3><p>{e(body)}</p><div>{" · ".join(pr_link(n) for n in ids)}</div></article>'
        for title, body, ids in summaries
    )

    # Static cards remain useful with JavaScript disabled and print to PDF.
    cards = []
    for n in notes:
        c = by_pr[n["pr"]]
        prod = [f for f in c["files"] if f["path"].startswith(("vllm_ascend/", "csrc/")) and "/docs/" not in f["path"]]
        file_links = "".join(
            f'<li><a href="{source(c["sha"], f["path"])}" target="_blank" rel="noopener">{e(f["path"])}</a></li>'
            for f in prod[:8]
        )
        more = f"<li>另有 {len(prod) - 8} 个生产文件，完整文件清单见提交记录。</li>" if len(prod) > 8 else ""
        baseline = (
            ("<p><b>基线对应：</b>" + "、".join(pr_link(x) for x in c["baseline_prs"]) + "</p>")
            if c["baseline_prs"]
            else ""
        )
        query = " ".join(
            str(n[k]) for k in ["pr", "title", "domain", "mechanism", "applicable", "result", "caveat"]
        ).lower()
        cards.append(f'''<article class="perf-card" id="perf-{n["pr"]}" data-kind="{n["kind"]}" data-domain="{e(n["domain"])}" data-evidence="{n["evidence"]}" data-search="{e(query)}">
<div class="card-meta"><span class="pill {n["kind"]}">{e(KIND[n["kind"]])}</span><span>{e(n["domain"])}</span><span class="evidence-tag">{e(EVIDENCE[n["evidence"]])}</span></div>
<h3>{pr_link(n["pr"])} {e(n["title"])}</h3><p class="original">{e(c["subject"])}</p>
<dl><dt>优化机制 / 变化</dt><dd>{e(n["mechanism"])}</dd><dt>适用条件</dt><dd>{e(n["applicable"])}</dd><dt>证据与结果</dt><dd>{e(n["result"])}</dd><dt>解释边界</dt><dd>{e(n["caveat"])}</dd></dl>{baseline}
<details><summary>代码证据与来源 · {c["sha"][:10]} · {len(c["files"])} 个文件</summary><p>提交时间 {e(c["date"])}；<a href="{GITHUB}/commit/{c["sha"]}" target="_blank" rel="noopener">提交 diff</a> · <a href="{c["patch"]}">本地原始补丁</a>{(' · <a href="evidence/prs/' + str(n["pr"]) + '.json">PR 作者说明快照</a>') if n["api_verified"] else ""}</p><ul class="source-files">{file_links}{more}</ul><p class="muted">以下为提交中的局部 diff 节选，完整上下文见原始补丁。代码行与规模不代表性能收益。</p><pre>{e(snippet(c))}</pre></details></article>''')

    # A small paired source snapshot for toolchain context, not the current worktree.
    stack = """<table><thead><tr><th>配置项</th><th>v0.30.0rc2</th><th>v0.31.0rc</th></tr></thead><tbody>
<tr><td>.github/vllm-release-tag.commit</td><td>v0.30.0</td><td>v0.31.0</td></tr>
<tr><td>主 Dockerfile CANN base</td><td>9.2.0_20260930220649</td><td>9.2.0_20261022115200</td></tr>
<tr><td>主 Dockerfile torch_npu daily date</td><td>20260926.5</td><td>20261020.3</td></tr>
<tr><td>主 Dockerfile memcache date</td><td>20261009.15</td><td>20261020.1</td></tr>
<tr><td>requirements.txt</td><td colspan="2">torch 2.10.0、torch-npu 2.10.0.post4、triton-ascend 3.2.2 均相同</td></tr>
</tbody></table>"""
    stack += (
        '<p class="muted">requirements 与镜像安装路径有不同 pins；上表只描述两个快照中的配置，不能替代实际运行环境。'
    )
    for path in [".github/vllm-release-tag.commit", "Dockerfile", "requirements.txt"]:
        stack += f' {e(path)}：<a href="{source(meta["old_sha"], path)}">旧</a> / <a href="{source(meta["new_sha"], path)}">新</a>。'
    stack += "</p>"
    payload = dict(commits=added, baseline=old, files=net, notes=notes, exports=exports)
    payload_json = (
        json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    rendered = TEMPLATE
    replacements = {
        "TITLE": "vLLM Ascend 分支演进与性能分析",
        "CARDS": count_cards,
        "LINEAGE": lineage,
        "AREA_ROWS": area_rows,
        "TYPE_ROWS": type_rows,
        "SEMANTIC_ROWS": semantic_rows,
        "MEASURE_ROWS": measure_rows,
        "SUMMARY": summary_html,
        "PERF_CARDS": "".join(cards),
        "KINDS_TEXT": e(kinds_text),
        "STACK": stack,
        "GENERATED": e(data["meta"]["generated_at"]),
        "API_COUNT": str(meta["api_sources"]),
        "OLD_SHA": meta["old_sha"],
        "NEW_SHA": meta["new_sha"],
        "BASE_SHA": meta["merge_base"],
        "PAYLOAD": payload_json,
        "KIND_OPTIONS": "".join(f'<option value="{k}">{e(v)}</option>' for k, v in KIND.items()),
        "DOMAIN_OPTIONS": "".join(f"<option>{e(v)}</option>" for v in sorted({n["domain"] for n in notes})),
        "TYPE_OPTIONS": "".join(f"<option>{e(v)}</option>" for v in types),
    }
    for key, value in replacements.items():
        rendered = rendered.replace("@@" + key + "@@", value)
    assert not re.search(r"@@[A-Z_]+@@", rendered)
    rendered = re.sub(r"[ \t]+$", "", rendered, flags=re.M)
    (HERE / "report.html").write_text(rendered)
    manifest = {
        str(f.relative_to(HERE)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(HERE.rglob("*"))
        if f.is_file()
        and f.suffix in {".patch", ".json", ".csv", ".py", ".cjs", ".md", ".html"}
        and f.name != "manifest.json"
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                kinds=kinds,
                evidence=metrics,
                types=types,
                html_bytes=len(rendered.encode()),
                api_sources=meta["api_sources"],
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@@TITLE@@</title>
<style>
:root{--bg:#f4f6fa;--panel:#fff;--ink:#18243a;--muted:#5d6c83;--line:#dce3ee;--accent:#275cc6;--green:#15775e;--orange:#975316}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.75 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}button,input,select{font:inherit}button,.button{border:1px solid var(--line);border-radius:7px;background:white;color:var(--ink);padding:7px 13px;cursor:pointer}button:hover,.button:hover{background:#edf3ff;text-decoration:none}button:disabled{opacity:.4;cursor:default}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #82aaff;outline-offset:3px}.layout{display:grid;grid-template-columns:225px minmax(0,1fr);max-width:1700px;margin:auto}.sidebar{height:100vh;position:sticky;top:0;padding:30px 21px;background:#14213a;color:#dce6fa;overflow:auto}.brand{font-size:19px;font-weight:800;color:#fff;margin-bottom:8px}.side-label{color:#9aaccb;font-size:12px;letter-spacing:.1em;margin:24px 0 10px}.sidebar a{display:block;color:#dce6fa;padding:7px 9px;border-radius:5px;margin:2px 0}.sidebar a:hover{background:#223650;text-decoration:none}.sidebar .version{font-size:12px;color:#a8b9d7;line-height:1.6}.side-footer{font-size:12px;color:#a8b9d7;margin-top:32px}.main{min-width:0;padding:34px 44px 70px}header{margin-bottom:26px}.eyebrow{font-size:12px;letter-spacing:.15em;color:var(--accent);font-weight:700}h1{font-size:34px;letter-spacing:-.03em;line-height:1.35;margin:10px 0 14px}h2{font-size:25px;line-height:1.4;margin:0 0 16px;letter-spacing:-.02em}h3{font-size:18px;line-height:1.5;margin:8px 0 12px}p{margin:8px 0 15px}.subtitle{color:var(--muted);font-size:17px;max-width:960px}.muted,small{color:var(--muted)}small{font-size:12px}.tagline{display:flex;gap:9px;flex-wrap:wrap;margin-top:18px}.pill{display:inline-block;font-size:12px;font-weight:600;border-radius:5px;background:#edf2fb;padding:3px 8px;color:#405577}.pill.direct{background:#e8f5ee;color:#20694e}.pill.existing{background:#fff2db;color:#8d5b13}.pill.enable{background:#e9efff;color:#365eb2}.pill.related,.pill.superseded{background:#efedf3;color:#6e617c}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:24px 0}.stat{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px 20px}.stat strong{display:block;font-size:31px;font-weight:750;line-height:1.2;letter-spacing:-.04em}.stat span{font-size:13px;color:var(--muted)}section{margin-top:36px;padding-top:8px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:24px;margin:16px 0}.callout{border-left:4px solid var(--accent);background:#eaf0fb;padding:14px 19px;margin:18px 0;border-radius:0 8px 8px 0}.callout.amber{border-color:#c18b33;background:#fff6e7}.lineage{display:flex;align-items:center;justify-content:center;gap:20px;padding:24px 8px;font-size:14px}.root{text-align:center}.split{font-size:39px;color:#7890b5;line-height:1.5}.branch{border:1px solid #c9d8f4;border-radius:9px;padding:10px 20px;margin:10px 0}.branch.new{background:#edf4ff}.branch.old{background:#f4f6fa}code{font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;word-break:break-all}pre{background:#162238;color:#dce6f5;border-radius:8px;padding:17px;max-height:390px;overflow:auto;font:12px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.summary-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.summary-box{background:white;border:1px solid var(--line);border-radius:10px;padding:20px}.summary-box p{font-size:14px}.summary-box div{font-size:13px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px;text-align:left}th{font-size:12px;color:var(--muted);background:#f0f3f8;white-space:nowrap}th,td{padding:12px;border-bottom:1px solid var(--line);vertical-align:top}td{overflow-wrap:anywhere}tbody tr:last-child td{border-bottom:0}.positive{color:var(--green)}.negative{color:#aa4a51}.bar{height:9px;border-radius:3px;background:#6f99e5;min-width:2px;margin-top:8px}.types{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.type-item{display:flex;justify-content:space-between;gap:15px;padding:10px 13px;border:1px solid var(--line);border-radius:6px;font-size:13px}.controls{display:flex;align-items:center;flex-wrap:wrap;gap:9px;margin:16px 0}.controls input,.controls select{min-height:40px;border:1px solid #c7d1e1;border-radius:6px;padding:7px 10px;background:white;color:var(--ink)}.controls input{flex:1;min-width:200px}.control-label{font-size:12px;color:var(--muted)}.result-count{color:var(--muted);font-size:13px;margin:10px 0}.perf-card{background:white;border:1px solid var(--line);border-radius:10px;padding:23px 25px;margin:15px 0;scroll-margin-top:20px}.card-meta{display:flex;gap:9px;align-items:center;flex-wrap:wrap;font-size:12px;color:var(--muted)}.evidence-tag{margin-left:auto}.original{font-size:12px;color:var(--muted)}dl{display:grid;grid-template-columns:115px 1fr;margin:18px 0 8px;gap:8px 12px;font-size:14px}dt{font-weight:650;color:#3b5072}dd{margin:0}details{margin:12px 0 4px}summary{cursor:pointer;color:var(--accent);font-size:13px}details p,details li{font-size:12px}.source-files{padding-left:18px;overflow-wrap:anywhere}.pager{display:flex;align-items:center;justify-content:flex-end;gap:14px;margin-top:16px;font-size:13px}.inventory th:nth-child(1){width:105px}.inventory th:nth-child(3){width:150px}.inventory th:nth-child(4){width:150px}.inventory td{font-size:12px}.commit-title{font-size:13px}.download-row{display:flex;gap:10px;flex-wrap:wrap}.empty{text-align:center;padding:25px;color:var(--muted)}.footnote{font-size:12px;color:var(--muted);margin-top:28px}.file-list{max-height:260px;overflow:auto}.counts-badge{display:inline-block;background:#eaf0fb;border-radius:4px;padding:1px 5px;margin-left:5px}li{margin:6px 0}.print-button{float:right}.evidence-table td:nth-child(3){min-width:250px}.evidence-table td:nth-child(4){min-width:250px}[hidden]{display:none!important}@media(min-width:1450px){.stats{grid-template-columns:repeat(6,1fr)}.stat{padding:16px 13px}.stat span{font-size:12px}}@media(max-width:1000px){.layout{grid-template-columns:180px minmax(0,1fr)}.main{padding:28px 24px}.sidebar{padding:24px 12px}.summary-grid{grid-template-columns:1fr}.types{grid-template-columns:repeat(2,1fr)}}@media(max-width:720px){.layout{display:block}.sidebar{height:auto;position:static;padding:16px}.sidebar nav{display:flex;overflow:auto;gap:5px}.sidebar nav a{white-space:nowrap;font-size:12px}.side-label,.side-footer,.sidebar .version{display:none}.brand{font-size:16px;margin:0}.main{padding:22px 16px}h1{font-size:27px}.stats{grid-template-columns:repeat(2,1fr)}.lineage{gap:10px;justify-content:flex-start}.branch{padding:9px;font-size:12px}.root{font-size:12px}.split{font-size:26px}.perf-card{padding:18px}.evidence-tag{margin-left:0}dl{grid-template-columns:1fr;gap:3px}dd{margin-bottom:9px}.panel{padding:17px}.types{grid-template-columns:1fr}.print-button{float:none;margin-bottom:12px}}@media print{body{background:white;font-size:11px}.layout{display:block}.sidebar,.controls,.pager,.download-row,.print-button{display:none}.main{padding:0}.panel,.perf-card,.summary-box,.stat{box-shadow:none;break-inside:avoid}.summary-grid{display:block}.summary-box{margin:10px 0}.perf-card[hidden]{display:block!important}section{margin-top:25px}pre{max-height:none}a{color:inherit}h1{font-size:27px}.table-wrap{overflow:visible}.stats{grid-template-columns:repeat(3,1fr)}.inventory{font-size:9px}th,td{padding:7px}}
</style></head><body>
<div class="layout"><aside class="sidebar"><div class="brand">vLLM Ascend</div><div class="version">Branch Evolution Review<br>v0.30.0rc2 → v0.31.0rc</div><div class="side-label">报告导航</div><nav><a href="#overview">01 · 核心结论</a><a href="#scope">02 · 范围与去重</a><a href="#evolution">03 · 代码演进</a><a href="#measurements">04 · 性能数据解读</a><a href="#performance">05 · 性能 PR 审阅</a><a href="#commits">06 · 完整提交清单</a><a href="#baseline">07 · 旧分支独有项</a><a href="#files">08 · 文件净差异</a><a href="#validation">09 · 验证建议与复现</a></nav><div class="side-footer">中文分析报告<br>快照日期 2026-11-02<br>HTML 可直接离线打开<br>外部来源需联网访问</div></aside>
<main class="main"><header><button class="print-button" onclick="window.print()" type="button">打印 / 导出 PDF</button><div class="eyebrow">RELEASE BRANCH COMPARISON · 2026.11.02</div><h1>@@TITLE@@</h1><p class="subtitle">upstream/releases/v0.31.0rc 相比 upstream/releases/v0.30.0rc2：合入历史、实际代码净差异，以及可定位到提交的性能优化点。</p><div class="tagline"><span class="pill">仓库：vllm-project/vllm-ascend</span><span class="pill">全仓库范围</span><span class="pill">Git + PR 作者说明</span><span class="pill">离线报告 / 可筛选 / 可导出</span></div></header>
<section id="overview"><h2>01 · 核心结论</h2><div class="stats">@@CARDS@@</div><div class="callout"><b>364 是提交历史差集，不是 364 项全新能力。</b>其中 44 条与基线补丁等价，另 6 条是已回移的同源适配。剩下 314 条仅表示“未被本轮等价映射命中”，还包含文档、CI、重构、修复、回退与功能使能，不能直接叫作 314 项功能或优化。</div><p>新增性能工作主要落在 <b>DSpark/Attention 小算子融合、PCP/OTP 与 SP 通信、GLM/GDN 拷贝和清零、KV Pool 主机开销、MRV2 图执行及多模态预处理</b>。不同模型、硬件和开关命中不同路径，无法给出一个可信的“v0.31.0rc 整体提升百分比”。</p><p>提交标题中有 <b>23 条显式 [Performance]</b>，其中 <b>3 条已在基线有完全等价回移</b>，其余 20 条保留为新增优化提交候选。扩展审阅不局限于标签：本报告共记录 66 条性能及关联项，分类如下。</p><div class="panel"><p>@@KINDS_TEXT@@。</p><p class="muted">分类是本报告的代码机制判断；各行“作者实测”来自特定 PR 场景。本次没有运行 NPU benchmark，未测两分支的端到端性能。优化机制的确定性、路径使能和测得收益分别陈述。</p></div></section>
<section id="scope"><h2>02 · 比对范围、分叉与去重</h2><div class="panel">@@LINEAGE@@<div class="table-wrap"><table><tbody><tr><th>基线</th><td>upstream/releases/v0.30.0rc2<br><code>@@OLD_SHA@@</code></td></tr><tr><th>目标</th><td>upstream/releases/v0.31.0rc<br><code>@@NEW_SHA@@</code></td></tr><tr><th>共同祖先</th><td><code>@@BASE_SHA@@</code></td></tr><tr><th>采集校验</th><td>2026-11-02 通过 git ls-remote 核对两分支远端 tip 与本地引用一致。报告锁定 SHA；分支后续移动不改变本报告。</td></tr><tr><th>历史范围</th><td><code>OLD..NEW</code> 得到 364 条；<code>NEW..OLD</code> 得到 55 条。364 条各有一个不同的标题关联 PR。</td></tr><tr><th>净代码差异</th><td><code>git diff OLD NEW</code>，开启 rename detection；1140 文件，+113,380 / −61,855 行。GitHub 三点 compare 默认从 merge base 开始，不等同于此处 tip-to-tip 净差异。</td></tr><tr><th>补丁去重</th><td>两侧独有提交以 <code>git patch-id --stable</code> 匹配，44 对；再审阅标题同源但补丁不同的 6 对。patch-id 是文本补丁等价，不保证后续演进后的整个运行行为相同。</td></tr></tbody></table></div></div><h3>6 对同源适配，单独保留判断依据</h3><div class="panel table-wrap"><table><thead><tr><th>新侧 PR</th><th>基线回移 PR</th><th>审阅结论</th></tr></thead><tbody>@@SEMANTIC_ROWS@@</tbody></table></div><div class="callout amber"><b>已在基线的性能点：</b>#17555 ↔ #18094（KV save fence）、#18037 ↔ #18127（TurboQuant）、#18235 ↔ #18325（MiniMax QK/RoPE/cache 融合）为精确补丁等价；#17255 ↔ #18122（共享专家 overlap）主体已存在，属于同源适配。它们保留在清单中，但不计入新增直接优化。</div><div class="callout amber"><b>目标快照已撤销的路径：</b>#18902 撤回 A3 GMM 融合，#18128 删除 StoreKVBlock 与 reshape 开关，#19251 删除 RPC/hit 结果缓存，#19243 撤回 #18270 的 PCP-local profile 预算。它们在下方按“撤销 / 收缩”单列；#18851 还移除了 RoPE UB 覆盖变量。</div></section>
<section id="evolution"><h2>03 · 新侧代码主要演进</h2><div class="summary-grid">@@SUMMARY@@</div><h3 style="margin-top:24px">提交类型分布</h3><div class="types">@@TYPE_ROWS@@</div><p class="muted">类型按标题标签优先级分为互斥组，例如带 CI 的 BugFix 归 CI；这是导航分类，不代表性能判断。性能标签与类型分别显示，避免把测试 baseline 更新当成推理优化。</p><h3>运行栈配置也发生变化</h3><div class="panel table-wrap">@@STACK@@</div></section>
<section id="measurements"><h2>04 · 有数字的性能证据，应如何解读</h2><p>以下均为作者报告的历史测试。本报告核对了对应 PR 的 merge SHA 与新增提交 SHA，但未复测结果；不同测试的基线、模型、runner、并发和软件栈不一致，<b>表中数字不能相加或相乘</b>。</p><div class="panel table-wrap"><table class="evidence-table"><thead><tr><th>PR / 优化</th><th>证据层级</th><th>作者数据</th><th>限制与负面结果</th></tr></thead><tbody>@@MEASURE_ROWS@@</tbody></table></div><div class="callout amber">特别剔除误归因：#18596 的历史 −17.2% ITL 不是最终 mask-only 补丁的独立测量；#16730 与 #18295 共用联合结果；#18647 的三配置 TTFT 表不是补丁 A/B，KVPP+layerwise 组合仍有暴露开销。单算子 47× / 6.2× 也不是服务加速倍数。</div></section>
<section id="performance"><h2>05 · 性能 PR 与关联路径审阅</h2><p>每条记录列出机制、适用条件、证据和限制。4 条“基线已有”、关联适配、能力使能和被替代实现均明确标识，不作为独立新增优化累计。PR 数量也不等同于独立优化机制数量。</p><div class="controls"><input id="perf-search" aria-label="搜索性能记录" type="search" placeholder="搜索模型 / 硬件 / 优化机制 / PR 编号…"><select id="perf-kind" aria-label="性能分类"><option value="">全部分类</option>@@KIND_OPTIONS@@</select><select id="perf-domain" aria-label="性能领域"><option value="">全部领域</option>@@DOMAIN_OPTIONS@@</select><select id="perf-evidence" aria-label="性能证据"><option value="">全部证据</option><option value="measured">含作者速度实测</option><option value="unmeasured">无独立速度实测 / 其他</option></select><button id="perf-reset" type="button">清除筛选</button><button data-export="performance.csv" type="button">导出 CSV</button></div><div id="perf-count" class="result-count" aria-live="polite">共 66 条记录</div><div id="perf-records">@@PERF_CARDS@@</div><div id="perf-empty" class="empty" hidden>没有符合条件的记录。</div></section>
<section id="commits"><h2>06 · 全部 364 条新侧独有提交</h2><p>保留原始标题、提交哈希、类型、基线映射及文件清单。未列为性能重点的条目不等于“确定没有任何性能影响”，只是没有在本报告中认定为直接优化或明确相关路径。逐条文件变更可展开。</p><div class="controls"><input id="commit-search" aria-label="搜索提交" type="search" placeholder="搜索标题 / PR / SHA / 文件路径…"><select id="commit-type" aria-label="提交类型"><option value="">全部类型</option>@@TYPE_OPTIONS@@</select><select id="commit-dedup" aria-label="提交等价性"><option value="">全部等价性</option><option>未匹配等价项</option><option>补丁等价</option><option>同源 / 分支适配</option></select><select id="commit-perf" aria-label="提交性能范围"><option value="">全部提交</option><option value="yes">66 条性能及关联项</option><option value="no">其他提交</option></select><button data-export="commits.csv" type="button">导出完整 CSV</button></div><div id="commit-count" class="result-count" aria-live="polite"></div><div class="panel table-wrap"><table class="inventory"><thead><tr><th>PR / SHA</th><th>提交与文件</th><th>类型 / 性能</th><th>基线映射</th></tr></thead><tbody id="commit-body"></tbody></table><div class="pager"><button id="commit-prev" type="button">上一页</button><span id="commit-page"></span><button id="commit-next" type="button">下一页</button></div></div></section>
<section id="baseline"><h2>07 · 55 条旧分支独有提交的核对</h2><p>其中 44 条有精确等价新侧补丁，6 条有同源适配，余下 5 条主要是 release 文档、镜像依赖和 PD graph 的不同实现。<b>提交 SHA 不在新分支不代表功能被删除。</b>特别是 #18152 存在不同实现/上游语义演进，不能在未对齐运行栈时直接判定为回归。</p><div class="controls"><input id="baseline-search" aria-label="搜索旧分支提交" type="search" placeholder="搜索旧分支 PR / 标题 / 对应新 PR…"><select id="baseline-dedup" aria-label="旧分支映射类型"><option value="">全部映射类型</option><option>独立核对项</option><option>补丁等价</option><option>同源 / 分支适配</option></select><button data-export="baseline_only.csv" type="button">导出 CSV</button></div><div id="baseline-count" class="result-count" aria-live="polite"></div><div class="panel table-wrap"><table><thead><tr><th>旧分支 PR / SHA</th><th>原始提交</th><th>新侧对应 / 结论</th></tr></thead><tbody id="baseline-body"></tbody></table></div></section>
<section id="files"><h2>08 · 1140 个文件的净差异</h2><p>按两端快照统计，自动检测重命名。V1 runner 隔离、自带 kernel 删除、文档与单元测试都会影响行数；这些数字只描述变更规模。文件链接固定到对应旧/新 SHA。</p><div class="panel table-wrap"><table><thead><tr><th>区域</th><th>文件数</th><th>新增行</th><th>删除行</th><th style="width:20%">文件数分布</th></tr></thead><tbody>@@AREA_ROWS@@</tbody></table></div><div class="controls"><input id="file-search" aria-label="搜索文件" type="search" placeholder="搜索路径 / 模块，例如 worker/v1、hc_pre、kv_pool…"><select id="file-area" aria-label="文件区域"><option value="">全部区域</option></select><button data-export="files.csv" type="button">导出全部文件 CSV</button></div><div id="file-count" class="result-count" aria-live="polite"></div><div class="panel table-wrap"><table><thead><tr><th>状态</th><th>文件路径</th><th>区域</th><th>新增</th><th>删除</th></tr></thead><tbody id="file-body"></tbody></table><div class="pager"><button id="file-prev" type="button">上一页</button><span id="file-page"></span><button id="file-next" type="button">下一页</button></div></div></section>
<section id="validation"><h2>09 · 建议优先验证与复现方法</h2><div class="panel"><h3>按实际部署路径选择验证组合</h3><div class="table-wrap"><table><thead><tr><th>场景</th><th>优先审阅 PR</th><th>需要测量</th></tr></thead><tbody>
<tr><td>DeepSeek V4/V4.1 + PCP / DSpark</td><td>#18506、#19190、#17263、#16542、#18081、#18936</td><td>分开测 prefill/decode；PCP/DP/TP、spec K/接受率、Engram、graph 命中及 scratch/KV 容量。</td></tr>
<tr><td>GLM-5.3 / Flash / DFlash2</td><td>#18717、#16730+#18295、#18723、#18917、#19408、#19521</td><td>A3/A5 分层；C1/C8/C64；BF16/量化 cache；D2H 同步、launch 数、conv state/共享 buffer 正确性。</td></tr>
<tr><td>MiniMax-M3</td><td>#18294、#18666、#18949、#18074</td><td>512/513/2K/32K/256K token 边界；CANN 版本；TopK 周边 kernel 数；精度及长 prefill TTFT。</td></tr>
<tr><td>AscendStore / KVPP / P-D</td><td>#17739、#18907、#18647、#19251</td><td>hit/miss、高 churn、layerwise on/off；CPU 热点、RPC 数、TTFT P95/P99、真实 store eviction 与 transfer failure。</td></tr>
<tr><td>多模态与 310P</td><td>#16996、#15571、#18390</td><td>图像前处理 CPU、TTFT/encoder graph bucket；310P MTP FULL replay 命中与无 MTP 对照。</td></tr>
</tbody></table></div><p>建议做两组互补实验：① 固定权重、输入、采样和可兼容软件栈的单点 PR A/B；② 各版本受支持软件栈的升级整体 A/B。每组固定 runner、graph、量化、并行和缓存开关，完成相同 warmup，交替至少 3 组 paired runs，记录吞吐、TTFT/TPOT/ITL 分位、显存、接受率、精度和实际输出 token 数。当前报告不包含这些尚未执行的实验结果。</p></div><h3>固定 SHA 的本地复现</h3><pre>OLD=@@OLD_SHA@@
NEW=@@NEW_SHA@@
git log --reverse --oneline "$OLD..$NEW"
git log --reverse --oneline "$NEW..$OLD"
git rev-list --left-right --count "$OLD...$NEW"
git diff --find-renames --shortstat "$OLD" "$NEW"
git diff --find-renames --numstat "$OLD" "$NEW"
git log --no-merges -p "$OLD..$NEW" | git patch-id --stable
git log --no-merges -p "$NEW..$OLD" | git patch-id --stable</pre><p>重建本报告：在报告目录执行 <code>python build_report.py</code>，使用已有 <code>git_data.json</code>、原始补丁和人工分类。<code>collect.py</code> 用于重新采集当前远端分支，会得到重新采集时的快照；重新采集后需审阅并更新人工分类，不能直接沿用旧结论。</p><div class="download-row"><button data-export="commits.csv" type="button">提交 CSV</button><button data-export="performance.csv" type="button">性能 CSV</button><button data-export="baseline_only.csv" type="button">旧分支 CSV</button><button data-export="files.csv" type="button">文件 CSV</button><a class="button" href="report_data.json" download>完整分析 JSON</a><a class="button" href="manifest.json" download>证据 SHA-256 清单</a></div><div class="footnote"><p>证据：本地完整 Git 历史、419 份原始提交补丁、@@API_COUNT@@ 份公开 GitHub PR 作者说明快照。凡缓存 PR 均核对 merged=true 与 merge_commit_sha；不使用机器人评论作为性能结论。PR 说明可能在合入后编辑，记录抓取时间；代码结论以固定提交为准，作者数据作为场景证据。</p><p>生成时间（UTC）：@@GENERATED@@。本报告固定比较代码快照，文件存放于仓库 reports 目录。无联网脚本、无第三方字体或 CDN 依赖。筛选、分页和 CSV 导出可离线运行；完整原始补丁、JSON 与校验清单位于同目录，外部 GitHub 代码/PR 链接需要网络。</p></div></section>
<noscript><div class="callout amber">JavaScript 未启用：上方性能审阅与摘要仍可阅读；完整提交/文件清单请打开同目录 CSV 或 report_data.json。</div></noscript></main></div>
<script id="report-data" type="application/json">@@PAYLOAD@@</script>
<script>
'use strict';
const DATA=JSON.parse(document.getElementById('report-data').textContent);
const $=id=>document.getElementById(id);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const github='https://github.com/vllm-project/vllm-ascend';
const linkPR=n=>`<a target="_blank" rel="noopener" href="${github}/pull/${n}">#${n}</a>`;
const noteMap=new Map(DATA.notes.map(n=>[n.pr,n]));
const measured=new Set(['e2e','kernel','combined','ported']);
let commitPage=1,filePage=1;let printing=false;const PAGE=40;
function filterPerf(){let count=0;const q=$('perf-search').value.trim().toLowerCase(),kind=$('perf-kind').value,domain=$('perf-domain').value,ev=$('perf-evidence').value;document.querySelectorAll('.perf-card').forEach(card=>{const pass=(!q||card.dataset.search.includes(q))&&(!kind||card.dataset.kind===kind)&&(!domain||card.dataset.domain===domain)&&(!ev||(ev==='measured'?measured.has(card.dataset.evidence):!measured.has(card.dataset.evidence)));card.hidden=!pass;if(pass)count++;});$('perf-count').textContent=`显示 ${count} / ${DATA.notes.length} 条审阅记录`;$('perf-empty').hidden=count>0;}
['perf-search','perf-kind','perf-domain','perf-evidence'].forEach(id=>$(id).addEventListener('input',filterPerf));
$('perf-reset').addEventListener('click',()=>{['perf-search','perf-kind','perf-domain','perf-evidence'].forEach(id=>$(id).value='');filterPerf();});
function fileDetails(c){return `<details><summary>${c.files.length} 个文件 · ${esc(c.date.slice(0,10))} · ${esc(c.author)}</summary><p><a href="${github}/commit/${c.sha}" target="_blank" rel="noopener">GitHub 提交 diff</a> · <a href="${esc(c.patch)}">原始补丁</a></p><div class="file-list">${c.files.map(f=>`<div><code>${esc(f.path)}</code> <span class="positive">+${f.added}</span> <span class="negative">−${f.deleted}</span></div>`).join('')}</div></details>`;}
function renderCommits(){const q=$('commit-search').value.trim().toLowerCase(),type=$('commit-type').value,dedup=$('commit-dedup').value,perf=$('commit-perf').value;const list=DATA.commits.filter(c=>(!q||[c.subject,c.pr,c.sha,...c.files.map(f=>f.path)].join(' ').toLowerCase().includes(q))&&(!type||c.type===type)&&(!dedup||c.dedup===dedup)&&(!perf||(perf==='yes'?noteMap.has(c.pr):!noteMap.has(c.pr))));const pages=Math.max(1,Math.ceil(list.length/PAGE));commitPage=Math.min(commitPage,pages);$('commit-count').textContent=`匹配 ${list.length} / ${DATA.commits.length} 条提交`;$('commit-body').innerHTML=(printing?list:list.slice((commitPage-1)*PAGE,commitPage*PAGE)).map(c=>`<tr><td>${linkPR(c.pr)}<br><a href="${github}/commit/${c.sha}" target="_blank" rel="noopener"><code>${c.sha.slice(0,10)}</code></a></td><td><span class="commit-title">${esc(c.subject)}</span>${fileDetails(c)}</td><td>${esc(c.type)}<br>${noteMap.has(c.pr)?`<a href="#perf-${c.pr}">${esc(c.perf_kind)}</a>`:'<small>未列为性能重点</small>'}</td><td>${esc(c.dedup)}<br>${c.baseline_prs.map(linkPR).join(' / ')}</td></tr>`).join('')||'<tr><td colspan="4" class="empty">没有匹配提交。</td></tr>';$('commit-page').textContent=`第 ${commitPage} / ${pages} 页`;$('commit-prev').disabled=commitPage===1;$('commit-next').disabled=commitPage===pages;}
['commit-search','commit-type','commit-dedup','commit-perf'].forEach(id=>$(id).addEventListener('input',()=>{commitPage=1;renderCommits();}));
$('commit-prev').onclick=()=>{commitPage--;renderCommits();};$('commit-next').onclick=()=>{commitPage++;renderCommits();};
function renderBaseline(){const q=$('baseline-search').value.trim().toLowerCase(),dedup=$('baseline-dedup').value;const list=DATA.baseline.filter(c=>(!q||[c.subject,c.pr,c.sha,...c.mapped_new_prs].join(' ').toLowerCase().includes(q))&&(!dedup||c.dedup===dedup));$('baseline-count').textContent=`匹配 ${list.length} / ${DATA.baseline.length} 条旧侧提交`;$('baseline-body').innerHTML=list.map(c=>`<tr><td>${linkPR(c.pr)}<br><code>${c.sha.slice(0,10)}</code></td><td>${esc(c.subject)}${fileDetails(c)}</td><td><span class="pill">${esc(c.dedup)}</span><br>${c.mapped_new_prs.map(linkPR).join(' / ')}<p>${esc(c.review)}</p></td></tr>`).join('')||'<tr><td colspan="3" class="empty">没有匹配提交。</td></tr>';}
['baseline-search','baseline-dedup'].forEach(id=>$(id).addEventListener('input',renderBaseline));
[...new Set(DATA.files.map(f=>f.area))].sort().forEach(area=>{const option=document.createElement('option');option.textContent=area;$('file-area').append(option);});
function renderFiles(){const q=$('file-search').value.trim().toLowerCase(),area=$('file-area').value;const list=DATA.files.filter(f=>(!q||[f.path,f.old_path,f.target_path].join(' ').toLowerCase().includes(q))&&(!area||f.area===area));const pages=Math.max(1,Math.ceil(list.length/PAGE));filePage=Math.min(filePage,pages);$('file-count').textContent=`匹配 ${list.length} / ${DATA.files.length} 个文件`;$('file-body').innerHTML=(printing?list:list.slice((filePage-1)*PAGE,filePage*PAGE)).map(f=>`<tr><td>${esc(f.status)}</td><td><a href="${esc(f.url)}" target="_blank" rel="noopener"><code>${esc(f.target_path)}</code></a>${f.old_path!==f.target_path?`<br><small>原路径 ${esc(f.old_path)}</small>`:''}</td><td>${esc(f.area)}</td><td class="positive">${f.binary?'binary':'+'+f.added}</td><td class="negative">${f.binary?'binary':'−'+f.deleted}</td></tr>`).join('')||'<tr><td colspan="5" class="empty">没有匹配文件。</td></tr>';$('file-page').textContent=`第 ${filePage} / ${pages} 页`;$('file-prev').disabled=filePage===1;$('file-next').disabled=filePage===pages;}
['file-search','file-area'].forEach(id=>$(id).addEventListener('input',()=>{filePage=1;renderFiles();}));$('file-prev').onclick=()=>{filePage--;renderFiles();};$('file-next').onclick=()=>{filePage++;renderFiles();};
document.querySelectorAll('[data-export]').forEach(button=>button.addEventListener('click',()=>{const name=button.dataset.export;const blob=new Blob(['\ufeff',DATA.exports[name]],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1500);}));
function revealHash(){const el=document.querySelector(location.hash.startsWith('#perf-')?location.hash:'.__no_match');if(el&&el.hidden){$('perf-reset').click();el.scrollIntoView();}}
window.addEventListener('beforeprint',()=>{printing=true;renderCommits();renderFiles();});window.addEventListener('afterprint',()=>{printing=false;renderCommits();renderFiles();});window.addEventListener('hashchange',revealHash);filterPerf();renderCommits();renderBaseline();renderFiles();revealHash();
</script></body></html>"""

if __name__ == "__main__":
    main()
