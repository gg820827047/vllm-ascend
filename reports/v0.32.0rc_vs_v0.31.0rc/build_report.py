#!/usr/bin/env python3
# HTML fragments retain their authored whitespace and long source URLs.
# ruff: noqa: E501
"""Build the second pinned comparison using the first report's UI assets."""

import ast
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

from annotations32 import EVIDENCE, KIND, ORIGINS, records

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIRST = HERE.parent / "v0.31.0rc_vs_v0.30.0rc2"
GITHUB = "https://github.com/vllm-project/vllm-ascend"
OLD_SHA = "5e40bc723165bfc8d8b1ec245ce5908a45dfd9d2"
NEW_SHA = "d08fa9d4380684b64fd7a45a876fc080ecbd1d22"


def e(value):
    return html.escape(str(value), quote=True)


def pr_link(number):
    return f'<a href="{GITHUB}/pull/{number}" target="_blank" rel="noopener">#{number}</a>'


def source(sha, path):
    return f"{GITHUB}/blob/{sha}/{quote(path, safe='/')}"


def historical_source(sha, path):
    """Deleted files link to the parent tree; other files link to this commit."""
    result = subprocess.run(["git", "cat-file", "-e", f"{sha}:{path}"], cwd=REPO, stderr=subprocess.DEVNULL)
    if result.returncode:
        sha = subprocess.check_output(["git", "rev-parse", f"{sha}^"], cwd=REPO, text=True).strip()
    return source(sha, path)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO).decode("utf-8", "replace")


def csv_text(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def commit_type(commit):
    title = commit["subject"].lower()
    for marker, name in [
        ("[ci]", "CI / 构建验证"),
        ("[doc]", "文档 / 发布说明"),
        ("[test]", "测试 / Benchmark 工具"),
        ("[revert]", "回退 / 策略调整"),
        ("[performance]", "显式 Performance 标签"),
        ("[bugfix]", "修复"),
        ("[feature]", "功能 / 使能"),
        ("[refactor]", "重构"),
    ]:
        if marker in title:
            return name
    return "其他 / 上游对齐"


def area(path):
    for prefix, name in [
        ("vllm_ascend/worker/", "Worker / Runner"),
        ("vllm_ascend/attention/", "Attention"),
        ("vllm_ascend/models/", "模型实现"),
        ("vllm_ascend/distributed/", "分布式 / KV / EC"),
        ("vllm_ascend/ops/", "Python / Triton 算子"),
        ("vllm_ascend/quantization/", "量化"),
        ("vllm_ascend/patch/", "上游补丁"),
        ("vllm_ascend/_310p/", "310P"),
        ("csrc/", "AscendC / C++ 算子"),
        ("tests/", "测试"),
        ("docs/", "文档"),
        (".github/", "CI / GitHub"),
    ]:
        if path.startswith(prefix):
            return name
    return "配置 / 工具 / 其他"


def snippet(commit):
    chunks = re.split(r"(?=^diff --git )", (HERE / commit["patch"]).read_text(), flags=re.M)
    candidates = []
    for chunk in chunks[1:]:
        name = chunk.splitlines()[0]
        if not re.match(r"diff --git a/(vllm_ascend|csrc)/", name) or "/docs/" in name:
            continue
        for hunk in re.split(r"(?=^@@ )", chunk, flags=re.M)[1:]:
            changes = [line for line in hunk.splitlines() if line.startswith(("+", "-"))]
            score = sum(
                bool(
                    re.search(
                        r"copy_|cache|buffer|fused|mask|graph|compile|event|contiguous|norm|prefetch|padding",
                        line,
                        re.I,
                    )
                )
                for line in changes
            )
            candidates.append((score, name + "\n" + "\n".join(hunk.splitlines()[:38])))
    return max(candidates, key=lambda item: item[0])[1] if candidates else "完整变更见原始补丁。"


def exact_files(data):
    # -z parsing preserves real rename paths; no human-readable brace expansion.
    raw = git("diff", "--numstat", "-z", "--find-renames", OLD_SHA, NEW_SHA).rstrip("\0").split("\0")
    names = git("diff", "--name-status", "-z", "--find-renames", OLD_SHA, NEW_SHA).rstrip("\0").split("\0")
    statuses = []
    i = 0
    while i < len(names):
        status = names[i]
        if status[0] in "RC":
            statuses.append((status, names[i + 1], names[i + 2]))
            i += 3
        else:
            statuses.append((status, names[i + 1], names[i + 1]))
            i += 2
    result = []
    i = 0
    while i < len(raw):
        added, deleted, path = raw[i].split("\t", 2)
        i += 1
        if not path:
            old_path, path = raw[i : i + 2]
            i += 2
        else:
            old_path = path
        status, status_old, status_new = statuses[len(result)]
        assert (old_path, path) == (status_old, status_new)
        result.append(
            dict(
                path=path,
                target_path=path,
                old_path=old_path,
                status=status,
                added=int(added) if added.isdigit() else 0,
                deleted=int(deleted) if deleted.isdigit() else 0,
                binary=added == "-",
                area=area(path),
                url=source(OLD_SHA if status == "D" else NEW_SHA, old_path if status == "D" else path),
            )
        )
    assert len(result) == len(data["net_files"]) == 1094
    assert sum(f["added"] for f in result) == 73862
    assert sum(f["deleted"] for f in result) == 44125
    return result


def previous_template():
    # Reuse only a literal UI template, without importing/executing the old analysis.
    parsed = ast.parse((FIRST / "build_report.py").read_text())
    for node in parsed.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TEMPLATE" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("The previous report UI template is missing")


def main():
    data = json.loads((HERE / "git_data.json").read_text())
    meta = data["meta"]
    previous = json.loads((FIRST / "report_data.json").read_text())
    assert meta["old_sha"] == previous["meta"]["new_sha"] == OLD_SHA
    assert meta["new_sha"] == NEW_SHA
    added, old = data["added"], data["old_only"]
    assert len(added) == 260 and len(old) == 3
    by_pr = {c["pr"]: c for c in added}
    assert len(by_pr) == 260 and None not in by_pr
    notes = records()
    index = {n["pr"]: n for n in notes}
    assert len(notes) == len(index) == 71
    assert set(index) <= set(by_pr)
    explicit = [c for c in added if "[performance]" in c["subject"].lower()]
    assert len(explicit) == 22 and all(c["pr"] in index for c in explicit)
    assert sum(bool(c["equivalent_old_prs"]) for c in added) == 3
    for c in added:
        c.update(
            type=commit_type(c),
            baseline_prs=c["equivalent_old_prs"],
            dedup="补丁等价" if c["equivalent_old_prs"] else "未匹配等价项",
            perf_kind=KIND[index[c["pr"]]["kind"]] if c["pr"] in index else "未列为性能重点",
        )
    for c in old:
        c.update(
            mapped_new_prs=c["equivalent_new_prs"],
            dedup="补丁等价",
            type=commit_type(c),
            review="31 release回移与32中的main原始提交patch-id一致，属于共同已有修改。",
        )
        assert len(c["mapped_new_prs"]) == 1
    for n in notes:
        c = by_pr[n["pr"]]
        n.update(
            sha=c["sha"],
            subject=c["subject"],
            date=c["date"],
            patch=c["patch"],
            kind_label=KIND[n["kind"]],
            evidence_label=EVIDENCE[n["evidence"]],
            baseline_prs=c["baseline_prs"],
            api_verified=False,
            origin_prs=ORIGINS.get(n["pr"], []),
        )
        cache = HERE / "evidence" / "prs" / f"{n['pr']}.json"
        if cache.exists():
            obj = json.loads(cache.read_text())
            assert obj["merged"] and obj["merge_commit_sha"] == c["sha"], (n["pr"], obj["merge_commit_sha"])
            n.update(api_verified=True, source_fetched_at=obj["fetched_at"])
    cached_prs = list((HERE / "evidence" / "prs").glob("[0-9]*.json"))
    meta.update(
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        performance_records=len(notes),
        explicit_performance=len(explicit),
        exact_equivalents=3,
        semantic_equivalents=0,
        api_sources=sum(path.stem.isdigit() for path in cached_prs),
        comment_sources=1,
    )
    types = collections.Counter(c["type"] for c in added)
    kinds = collections.Counter(n["kind"] for n in notes)
    data.update(
        performance=notes, net_files=exact_files(data), type_counts=dict(types), classification_counts=dict(kinds)
    )
    areas = collections.defaultdict(lambda: dict(files=0, added=0, deleted=0))
    for f in data["net_files"]:
        group = areas[f["area"]]
        group["files"] += 1
        group["added"] += f["added"]
        group["deleted"] += f["deleted"]
    data["areas"] = dict(areas)
    exports = {
        "commits.csv": csv_text(
            [
                dict(
                    c,
                    baseline_prs=", ".join(map(str, c["baseline_prs"])),
                    file_count=len(c["files"]),
                    url=f"{GITHUB}/pull/{c['pr']}",
                )
                for c in added
            ],
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
            [dict(c, mapped_new_prs=", ".join(map(str, c["mapped_new_prs"]))) for c in old],
            ["sha", "pr", "date", "subject", "dedup", "mapped_new_prs", "review"],
        ),
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
        "files.csv": csv_text(
            data["net_files"], ["status", "old_path", "target_path", "area", "added", "deleted", "binary", "url"]
        ),
    }
    for name, body in exports.items():
        (HERE / name).write_text("\ufeff" + body)
    (HERE / "report_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    sections = sections_html(data, index, types, kinds, areas)
    template = previous_template()
    prefix = template.split('<section id="overview">')[0]
    prefix = prefix.replace("v0.30.0rc2 → v0.31.0rc", "v0.31.0rc → v0.32.0rc")
    prefix = prefix.replace(
        "upstream/releases/v0.31.0rc 相比 upstream/releases/v0.30.0rc2",
        "upstream/releases/v0.32.0rc 相比 upstream/releases/v0.31.0rc",
    )
    prefix = prefix.replace("@@TITLE@@", "vLLM Ascend 31 → 32 演进与性能分析")
    prefix = prefix.replace("<nav>", '<nav><a href="../index.html">← 30 → 31 → 32 总览</a>', 1)
    prefix = prefix.replace("代码演进</a>", "代码演进</a>", 1)
    tail = template[template.index("<noscript>") :]
    payload = dict(commits=added, baseline=old, files=data["net_files"], notes=notes, exports=exports)
    packed = (
        json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    tail = tail.replace("@@PAYLOAD@@", packed)
    tail = tail.replace("['e2e','kernel','combined','ported']", "['e2e','kernel','combined','historical','negative']")
    rendered = prefix + sections + tail
    assert not re.search(r"@@[A-Z_]+@@", rendered)
    rendered = re.sub(r"[ \t]+$", "", rendered, flags=re.M)
    (HERE / "report.html").write_text(rendered)
    manifest = {
        str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(HERE.rglob("*"))
        if path.is_file()
        and path.suffix in {".py", ".json", ".csv", ".patch", ".md", ".html", ".cjs"}
        and path.name != "manifest.json"
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                kinds=kinds,
                explicit=len(explicit),
                commits=len(added),
                old=len(old),
                files=len(data["net_files"]),
                api_sources=meta["api_sources"],
                html_bytes=len(rendered.encode()),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def sections_html(data, index, types, kinds, areas):
    """Assemble independent sections so old report conclusions remain unchanged."""
    meta = data["meta"]
    notes = data["performance"]
    by_pr = {c["pr"]: c for c in data["added"]}
    stats = "".join(
        f'<div class="stat"><strong>{num}</strong><span>{label}</span></div>'
        for num, label in [
            (260, "32侧独有提交 / PR"),
            (3, "31侧独有提交 / PR"),
            (3, "补丁等价项"),
            (257, "未匹配等价的历史提交"),
            (1094, "净变更文件"),
            (len(notes), "性能及关联审阅记录"),
        ]
    )
    kind_text = "；".join(f"{KIND[k]} {kinds[k]} 条" for k in KIND)
    type_html = "".join(
        f'<div class="type-item"><span>{e(k)}</span><strong>{v}</strong></div>' for k, v in types.most_common()
    )
    cards = []
    for n in notes:
        c = by_pr[n["pr"]]
        files = [f for f in c["files"] if f["path"].startswith(("vllm_ascend/", "csrc/")) and "/docs/" not in f["path"]]
        file_html = "".join(
            f'<li><a href="{historical_source(c["sha"], f["path"])}" target="_blank" rel="noopener">{e(f["path"])}</a></li>'
            for f in files[:7]
            if "=>" not in f["path"]
        )
        query = " ".join(
            str(n[k]) for k in ["pr", "title", "domain", "mechanism", "applicable", "result", "caveat"]
        ).lower()
        origin = (
            ("<p>原始main PR：" + "、".join(pr_link(x) for x in n["origin_prs"]) + "（不另计为release新增提交）。</p>")
            if n["origin_prs"]
            else ""
        )
        baseline = (
            ("<p>基线对应：" + "、".join(pr_link(x) for x in n["baseline_prs"]) + "</p>") if n["baseline_prs"] else ""
        )
        extra = (
            '<p>RecoverSSM原始验证评论：<a href="https://github.com/vllm-project/vllm-ascend/pull/20524#issuecomment-6554091747">局部A/B与整算法OFF/ON的不同基线</a> · <a href="evidence/prs/20524-comment-6554091747.json">评论快照</a></p>'
            if n["pr"] == 20524
            else ""
        )
        api = f' · <a href="evidence/prs/{n["pr"]}.json">PR作者说明快照</a>' if n["api_verified"] else ""
        cards.append(
            f'''<article class="perf-card" id="perf-{n["pr"]}" data-kind="{n["kind"]}" data-domain="{e(n["domain"])}" data-evidence="{n["evidence"]}" data-search="{e(query)}"><div class="card-meta"><span class="pill {n["kind"]}">{e(KIND[n["kind"]])}</span><span>{e(n["domain"])}</span><span class="evidence-tag">{e(EVIDENCE[n["evidence"]])}</span></div><h3>{pr_link(n["pr"])} {e(n["title"])}</h3><p class="original">{e(c["subject"])}</p><dl><dt>优化机制 / 变化</dt><dd>{e(n["mechanism"])}</dd><dt>适用条件</dt><dd>{e(n["applicable"])}</dd><dt>证据与结果</dt><dd>{e(n["result"])}</dd><dt>解释边界</dt><dd>{e(n["caveat"])}</dd></dl>{origin}{baseline}{extra}<details><summary>代码证据 · {c["sha"][:10]} · {len(c["files"])} 个文件</summary><p>{e(c["date"])}；<a href="{GITHUB}/commit/{c["sha"]}" target="_blank" rel="noopener">提交diff</a> · <a href="{c["patch"]}">本地原始补丁</a>{api}</p><ul class="source-files">{file_html}</ul><p class="muted">局部diff节选；完整文件清单见下方提交记录，完整上下文见原始补丁。</p><pre>{e(snippet(c))}</pre></details></article>'''
        )
    measurements = [
        19409,
        19007,
        19515,
        19550,
        19954,
        20116,
        20584,
        20578,
        20441,
        20524,
        20467,
        20542,
        19342,
        20258,
        20308,
    ]
    measure_rows = "".join(
        f'<tr><td><a href="#perf-{n}">#{n}</a><br><small>{e(index[n]["title"])}</small></td><td>{e(EVIDENCE[index[n]["evidence"]])}</td><td>{e(index[n]["result"])}</td><td>{e(index[n]["caveat"])}</td></tr>'
        for n in measurements
    )
    type_options = "".join(f"<option>{e(x)}</option>" for x in types)
    kind_options = "".join(f'<option value="{k}">{e(v)}</option>' for k, v in KIND.items())
    domain_options = "".join(f"<option>{e(x)}</option>" for x in sorted({n["domain"] for n in notes}))
    exact_rows = "".join(
        f"<tr><td>{pr_link(c['pr'])}</td><td>{'、'.join(pr_link(x) for x in c['equivalent_new_prs'])}</td><td>{e(c['subject'])}</td></tr>"
        for c in data["old_only"]
    )
    area_rows = "".join(
        f'<tr><td>{e(k)}</td><td>{v["files"]}</td><td class="positive">+{v["added"]:,}</td><td class="negative">−{v["deleted"]:,}</td></tr>'
        for k, v in sorted(areas.items(), key=lambda item: item[1]["files"], reverse=True)
    )
    summaries = [
        (
            "主机与元数据成本",
            "compiled embedding不再复制capacity buffer；RoPE改为flat base行gather；PCP跨KV group共享metadata，GDN混合批准备融合，UVA按区间追踪。",
            [19409, 19247, 19007, 19954, 20542, 20444],
        ),
        (
            "GLM编译与状态恢复",
            "GLM backbone/MTP接入torch.compile，补齐PCP与static-kernel路径；RecoverSSM从算子、框架到prefix/align再到commit优化逐步演进。",
            [19076, 20503, 20473, 20128, 20131, 20369, 20524],
        ),
        (
            "Attention/QLI与通信",
            "A5 QSFA与A2/A3 NoPE QSFA分别优化；QLI自适应TopK；DCP exchange/remap及uniform SP的EP gather/reduce减少布局重建。",
            [20116, 20584, 20578, 20441, 16350, 19515],
        ),
        (
            "模型范围扩展",
            "将V4的WKV堆叠扩展到V4.1；MiniMax fusion跨token复用权重并开放A5新量化融合；Qwen3.8/PLE引入模型、prefill dispatch、PP/SP、ETP及INT8表。",
            [19550, 20308, 20290, 16981, 20344, 20467, 20443, 20516, 20548],
        ),
        (
            "KV Pool与内存生命周期",
            "layerwise probe跳过HBM前缀，可选SSD预取；Step MRV2共享backing、310P预算和reload权重引用修复，完善PCP/DCP与Mamba传输依赖。",
            [19755, 19472, 19558, 19984, 17344, 19484, 20477, 19880],
        ),
        (
            "迁移与边界",
            "上游对齐到v0.32.0，requirements更新torch-npu/Triton/MemCache；退役legacy DSA-CP和部分V1/CPP分支，中途CANN9.3包升级已回退。",
            [19516, 20465, 19493, 18958, 18964, 20309],
        ),
    ]
    summary_html = "".join(
        f'<article class="summary-box"><h3>{e(title)}</h3><p>{e(body)}</p><div>{" · ".join(pr_link(n) for n in prs)}</div></article>'
        for title, body, prs in summaries
    )
    stack = f'''<table><thead><tr><th>配置</th><th>31 snapshot</th><th>32 snapshot</th></tr></thead><tbody><tr><td>vLLM release pin</td><td>v0.31.0</td><td>v0.32.0</td></tr><tr><td>requirements: torch</td><td>2.10.0</td><td>2.10.0</td></tr><tr><td>requirements: torch-npu</td><td>2.10.0.post4</td><td>2.10.0.post7</td></tr><tr><td>requirements: triton-ascend</td><td>3.2.2</td><td>3.6.0.post2</td></tr><tr><td>requirements: memcache / memfabric</td><td>1.2.0</td><td>1.3.0</td></tr><tr><td>主Docker base</td><td>CANN9.2.0_20261022115200</td><td>quay.io/ascend/cann:9.2.1</td></tr></tbody></table><p class="muted">这些是源码配置，不代表任一作者benchmark的实际依赖。Docker另有daily wheel安装路径，不能只读requirements推断运行栈。源码：<a href="{source(OLD_SHA, "requirements.txt")}">31 requirements</a> / <a href="{source(NEW_SHA, "requirements.txt")}">32 requirements</a>，<a href="{source(OLD_SHA, "Dockerfile")}">31 Dockerfile</a> / <a href="{source(NEW_SHA, "Dockerfile")}">32 Dockerfile</a>。</p>'''
    return f"""<section id="overview"><h2>01 · 核心结论</h2><p><a href="../index.html">30 → 31 → 32 三版本总览</a> · <a href="../v0.31.0rc_vs_v0.30.0rc2/report.html">第一段：30 → 31</a></p><div class="stats">{stats}</div><div class="callout"><b>32侧260条历史提交中，3条与31基线补丁等价，剩下257条未匹配等价。</b>257仍包括文档、CI、测试、修复、重构与能力使能，不是257项性能优化。显式[Performance]共22条，全部纳入人工审阅，另补[Perf]及没有性能标签的明确相关路径。</div><div class="panel"><p>{e(kind_text)}。</p><p>核心增量集中在主机/metadata工作量、GLM编译与RecoverSSM、QSFA/QLI、通信布局和Qwen新路径。WKV堆叠、MiniMax准备融合、KV mask缓存等是31已有工作的延伸，按模型/形状扩展说明。</p><p class="muted">本次未运行NPU benchmark。数字来自PR作者或原始验证评论；局部kernel、历史回移、联合栈和完整服务结果分开解释，不能得到可信的“32整体比31快多少”。</p></div></section>
<section id="scope"><h2>02 · 锁定快照与基线去重</h2><div class="panel"><div class="lineage"><div class="root">共同祖先<br><code>{meta["merge_base"][:12]}</code></div><div class="split">↗<br>↘</div><div><div class="branch new">32 · 260独有提交 · {NEW_SHA[:12]}</div><div class="branch old">31 · 3独有提交 · {OLD_SHA[:12]}</div></div></div><table><tbody><tr><th>基线</th><td>upstream/releases/v0.31.0rc<br><code>{OLD_SHA}</code> · {e(meta["old_tip"])}</td></tr><tr><th>目标</th><td>upstream/releases/v0.32.0rc<br><code>{NEW_SHA}</code> · {e(meta["new_tip"])}</td></tr><tr><th>连续性</th><td>本报告OLD SHA与上一份30→31报告的NEW SHA完全一致。报告锁定快照，后续branch移动不自动改变结论。</td></tr><tr><th>统计</th><td>提交范围OLD..NEW / NEW..OLD；代码净差异git diff OLD NEW，检测rename。1094文件，+73,862 / −44,125行。</td></tr><tr><th>采集</th><td>{e(meta["collected_at"])}（UTC），采集时通过ls-remote核对两端tip。</td></tr></tbody></table></div><h3>31 release的3条独有提交均有等价main原提交</h3><div class="panel table-wrap"><table><thead><tr><th>31回移PR</th><th>32中的原PR</th><th>提交</th></tr></thead><tbody>{exact_rows}</tbody></table></div><p class="muted">去重使用git patch-id --stable。这里无需另外猜测语义映射；补丁等价不保证后续提交演进后的全部运行行为相同。GitHub三点compare从merge base开始，不能代替两端tip净diff。</p></section>
<section id="evolution"><h2>03 · 代码演进与依赖变化</h2><div class="summary-grid">{summary_html}</div><h3 style="margin-top:24px">提交类型</h3><div class="types">{type_html}</div><p class="muted">标签优先分类互斥，CI+BugFix归CI；性能审阅是另一维。目录迁移与自带kernel删除会扩大行数，不等于算法收益。</p><div class="panel table-wrap">{stack}</div></section>
<section id="measurements"><h2>04 · 作者测量：收益、中性与退化</h2><p>各数据使用不同baseline、runner、batch、cache hit和依赖，不可相加或相乘。尤其注意：#20524的“commit局部A/B”与“ReplaySSM OFF/ON”是两种实验，#20584的release数字来自同源main实现，#20444最终default-on版本尚无独立速度验证。</p><div class="panel table-wrap"><table class="evidence-table"><thead><tr><th>PR</th><th>证据层级</th><th>数据</th><th>限制 / 退化</th></tr></thead><tbody>{measure_rows}</tbody></table></div><div class="callout amber">不隐藏负面结果：#20584列出的decode无收益；#20258短history入场可能更慢；#20308小batch有退化或波动；#19342探索模型对照约−0.90%且最终断言失败；#20524部分A3边界state case退化，整算法精度分数也有小幅变化。它们都不能被平均成一个版本加速百分比。</div></section>
<section id="performance"><h2>05 · {len(notes)} 条性能与关联路径审阅</h2><p>每条含机制、条件、量化边界、固定提交及原始patch。能力使能、基线已有、正确性约束和退役项分别标明；多PR组成一条机制时不重复累计收益。</p><div class="controls"><input id="perf-search" aria-label="搜索性能记录" type="search" placeholder="搜索模型 / 硬件 / PR / 机制…"><select id="perf-kind" aria-label="性能分类"><option value="">全部分类</option>{kind_options}</select><select id="perf-domain" aria-label="性能领域"><option value="">全部领域</option>{domain_options}</select><select id="perf-evidence" aria-label="性能证据"><option value="">全部证据</option><option value="measured">含作者测量（含中性/退化）</option><option value="unmeasured">机制 / 功能 / 其他</option></select><button id="perf-reset" type="button">清除筛选</button><button data-export="performance.csv" type="button">导出CSV</button></div><div id="perf-count" class="result-count" aria-live="polite">共{len(notes)}条</div><div id="perf-records">{"".join(cards)}</div><div id="perf-empty" class="empty" hidden>没有匹配记录。</div></section>
<section id="commits"><h2>06 · 全部260条32侧独有提交</h2><p>保留标题、SHA、日期、作者、文件及31映射。未列为性能重点不等于绝对没有性能影响。</p><div class="controls"><input id="commit-search" aria-label="搜索提交" type="search" placeholder="搜索标题 / PR / SHA / 文件…"><select id="commit-type" aria-label="提交类型"><option value="">全部类型</option>{type_options}</select><select id="commit-dedup" aria-label="提交等价性"><option value="">全部等价性</option><option>未匹配等价项</option><option>补丁等价</option></select><select id="commit-perf" aria-label="提交性能范围"><option value="">全部提交</option><option value="yes">{len(notes)}条性能及关联</option><option value="no">其他提交</option></select><button data-export="commits.csv" type="button">导出完整CSV</button></div><div id="commit-count" class="result-count" aria-live="polite"></div><div class="panel table-wrap"><table class="inventory"><thead><tr><th>PR / SHA</th><th>提交与文件</th><th>类型 / 性能</th><th>基线对应</th></tr></thead><tbody id="commit-body"></tbody></table><div class="pager"><button id="commit-prev" type="button">上一页</button><span id="commit-page"></span><button id="commit-next" type="button">下一页</button></div></div></section>
<section id="baseline"><h2>07 · 31侧独有的3条提交</h2><p>均已找到patch-id等价项；SHA不同由release回移产生，不代表修复在32缺失。</p><div class="controls"><input id="baseline-search" aria-label="搜索旧分支提交" type="search" placeholder="搜索31提交或32对应PR…"><select id="baseline-dedup" aria-label="旧分支映射类型"><option value="">全部映射</option><option>补丁等价</option></select><button data-export="baseline_only.csv" type="button">导出CSV</button></div><div id="baseline-count" class="result-count" aria-live="polite"></div><div class="panel table-wrap"><table><thead><tr><th>31 PR / SHA</th><th>原始标题</th><th>32对应 / 结论</th></tr></thead><tbody id="baseline-body"></tbody></table></div></section>
<section id="files"><h2>08 · 1094个文件的净差异</h2><p>精确解析NUL分隔rename记录；删除文件链接到31，其余链接到32。大范围V1/patch/AscendStore移动属于结构变化，不单独视为性能提升。</p><div class="panel table-wrap"><table><thead><tr><th>区域</th><th>文件</th><th>新增</th><th>删除</th></tr></thead><tbody>{area_rows}</tbody></table></div><div class="controls"><input id="file-search" aria-label="搜索文件" type="search" placeholder="搜索路径 / 模块…"><select id="file-area" aria-label="文件区域"><option value="">全部区域</option></select><button data-export="files.csv" type="button">导出CSV</button></div><div id="file-count" class="result-count" aria-live="polite"></div><div class="panel table-wrap"><table><thead><tr><th>状态</th><th>文件</th><th>区域</th><th>新增</th><th>删除</th></tr></thead><tbody id="file-body"></tbody></table><div class="pager"><button id="file-prev" type="button">上一页</button><span id="file-page"></span><button id="file-next" type="button">下一页</button></div></div></section>
<section id="validation"><h2>09 · 验证优先级与复现</h2><div class="panel"><p><b>同模型单点A/B：</b>DSV4.1优先#19409/#19247/#19550/#19007；GLM优先#19076/#19954/#20584/#20578，并将RecoverSSM开关与#20524算子修订分开；MiniMax优先#20308/#20290；Qwen优先区分#20444默认同步能力、#20467 dispatch及#20344局部kernel。</p><p><b>升级整体A/B：</b>分别使用各版本受支持运行栈，固定权重、prompt、采样、runner、并行、graph、量化、cache和warmup；交替至少3组成对测量。记录吞吐、TTFT/TPOT/真实SSE ITL、分位、输出长度、接受率、显存和精度。单独测短history/小batch及prefix hit差异，保留退化。当前报告未执行这些NPU测试。</p><p><b>迁移核对：</b>legacy enable_dsa_cp移除；memcache_dp_init_barrier移除且MemCache/MemFabric要求>=1.3.0；KVPP/CPP的V1支持需按新矩阵；torch-npu与Triton版本变化可能影响融合接口和编译缓存。</p></div><pre>OLD={OLD_SHA}
NEW={NEW_SHA}
git rev-list --left-right --count "$OLD...$NEW"
git log --reverse --oneline "$OLD..$NEW"
git log --reverse --oneline "$NEW..$OLD"
git diff --find-renames --shortstat "$OLD" "$NEW"
git log --no-merges -p "$OLD..$NEW" | git patch-id --stable
git log --no-merges -p "$NEW..$OLD" | git patch-id --stable</pre><p>在本目录运行<code>python build_report.py</code>从已有证据重建，需要当前Git仓库含两端SHA和相邻第一份报告（复用UI模板，不执行旧分析）。<code>collect.py --github</code>读取当时的远端分支，属于重新采样，需重新审阅人工注释及断言。</p><div class="download-row"><button data-export="commits.csv" type="button">提交CSV</button><button data-export="performance.csv" type="button">性能CSV</button><button data-export="baseline_only.csv" type="button">基线CSV</button><button data-export="files.csv" type="button">文件CSV</button><a class="button" href="report_data.json" download>完整JSON</a><a class="button" href="manifest.json" download>校验清单</a></div><div class="footnote"><p>证据：263份原始提交patch、{meta["api_sources"]}份公开PR作者说明和1份明确链接的验证评论快照。与本范围对应的缓存PR核对merged与merge SHA；两个main来源PR仅作上下文，不另计新增。PR正文可合入后编辑，保留抓取时间，最终代码结论以固定SHA为准。未采用机器人评论作为速度结论。</p><p>生成UTC：{e(meta["generated_at"])}。HTML核心内容、筛选与CSV离线可用；原始patch/JSON需同目录，GitHub链接需联网。<a href="../index.html">返回30→31→32总览</a>。</p></div></section>"""


if __name__ == "__main__":
    main()
