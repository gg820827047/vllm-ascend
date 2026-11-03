#!/usr/bin/env python3
# HTML fragments retain their authored whitespace and long source URLs.
# ruff: noqa: E501
"""Render the curated DeepSeek subset of the existing fixed-snapshot reports."""

import csv
import hashlib
import html
import json
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
REPO = REPORTS.parent
STAGES = ("v0.31.0rc_vs_v0.30.0rc2", "v0.32.0rc_vs_v0.31.0rc")
SUFFIXES = {".py", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".cu", ".cuh", ".cce"}
EXCLUDED_PARTS = {"test", "tests", "testing", "docs", "benchmarks", "benchmark", "golden"}
KINDS = {"direct": "直接优化 / 性能恢复", "enable": "性能路径使能", "candidate": "条件候选（不计合计）"}


def production(path):
    p = Path(path)
    return (
        path.startswith(("vllm_ascend/", "csrc/"))
        and p.suffix.lower() in SUFFIXES
        and not EXCLUDED_PARTS.intersection(part.lower() for part in p.parts)
        and not p.name.startswith("test_")
    )


def totals(rows):
    result = {"prs": len(rows)}
    for scope in ("production", "whole_pr"):
        for metric in ("added", "deleted"):
            result[f"{scope}_{metric}"] = sum(r[f"{scope}_{metric}"] for r in rows)
        result[f"{scope}_changed"] = result[f"{scope}_added"] + result[f"{scope}_deleted"]
    return result


def verify_git(commit):
    raw = subprocess.check_output(
        ["git", "show", "--format=", "--numstat", "--find-renames", commit["sha"]], cwd=REPO, text=True
    )
    actual = sorted(tuple(line.split("\t", 2)) for line in raw.splitlines())
    expected = sorted(
        ("-" if f["binary"] else str(f["added"]), "-" if f["binary"] else str(f["deleted"]), f["path"])
        for f in commit["files"]
    )
    if actual != expected:
        raise ValueError(f"Git numstat differs from cached evidence: {commit['sha']}")


def main():
    selection = json.loads((HERE / "selection.json").read_text())
    notes, commits, stage_for_pr, snapshots = {}, {}, {}, []
    for stage in STAGES:
        data = json.loads((REPORTS / stage / "report_data.json").read_text())
        snapshots.append(data["meta"])
        for note in data["performance"]:
            number = note["pr"]
            if number in notes:
                raise ValueError(f"Duplicate performance record: {number}")
            notes[number] = note
            stage_for_pr[number] = stage
        commits.update({c["sha"]: c for c in data["added"]})

    choices = []
    for kind, key in (("direct", "groups"), ("enable", "enable_groups")):
        choices.extend((number, kind, group) for group, numbers in selection[key].items() for number in numbers)
    choices.extend((number, "candidate", "需部署配置确认") for number in selection["candidates"])
    if len({number for number, _, _ in choices}) != len(choices):
        raise ValueError("Selections must be unique across both stages and categories")

    records = []
    for number, kind, group in choices:
        note = notes[number]
        commit = commits[note["sha"]]
        verify_git(commit)
        if commit.get("equivalent_old_prs"):
            raise ValueError(f"Selected PR already has patch-equivalent baseline: {number}")
        files = [f for f in commit["files"] if not f["binary"]]
        prod = [f for f in files if production(f["path"])]
        row = dict(note)
        row.update(
            kind=kind,
            group=group,
            stage="30→31" if stage_for_pr[number] == STAGES[0] else "31→32",
            relation="DeepSeek 专项" if number in selection["deepseek_specific"] else "条件适用的公共路径",
            deepseek_review=selection["overrides"].get(str(number), note["applicable"]),
            pr_url=f"https://github.com/vllm-project/vllm-ascend/pull/{number}",
            local_patch=f"../{stage_for_pr[number]}/{note['patch']}",
            production_added=sum(f["added"] for f in prod),
            production_deleted=sum(f["deleted"] for f in prod),
            whole_pr_added=sum(f["added"] for f in files),
            whole_pr_deleted=sum(f["deleted"] for f in files),
            production_files=prod,
            patch_review="关键生产 diff / 入口与限制条件审阅；非完整逐行审计",
            runtime_verified=False,
        )
        records.append(row)

    summary = {kind: totals([r for r in records if r["kind"] == kind]) for kind in KINDS}
    included = [r for r in records if r["kind"] != "candidate"]
    summary["included"] = totals(included)
    summary["by_stage"] = {
        stage: {
            kind: totals([r for r in included if r["stage"] == stage and r["kind"] == kind])
            for kind in ("direct", "enable")
        }
        for stage in ("30→31", "31→32")
    }
    summary["unique_production_files"] = len({f["path"] for r in included for f in r["production_files"]})
    data = {
        "scope": "既有 30→31→32 固定发布快照；非 2026-11-03 最新 main，主要覆盖 Ascend A2/A3/A5 的 DeepSeek 主干及共享路径",
        "review_date": "2026-11-03",
        "snapshots": snapshots,
        "method": "提交范围与 patch-id/同源回移排除 → 标题/正文/文件筛选 → 关键生产 diff、模型调用路径和开关核对；无全量逐行审计、NPU 精度或性能复测",
        "counting": "按唯一 PR 的 Git numstat 累加历史新增与删除；生产代码只取 vllm_ascend/、csrc/ 的源码，排除测试、文档、benchmark/golden；仍含注释空行。整 PR 取全部文本文件。重复编辑按次累计，非最终树净差或最小回移量。",
        "summary": summary,
        "records": records,
        "selection": selection,
    }
    (HERE / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    fields = [
        "pr",
        "stage",
        "kind",
        "group",
        "relation",
        "title",
        "deepseek_review",
        "production_added",
        "production_deleted",
        "whole_pr_added",
        "whole_pr_deleted",
        "evidence",
        "result",
        "caveat",
        "sha",
        "pr_url",
        "local_patch",
        "patch_review",
        "runtime_verified",
    ]
    with (HERE / "prs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    with (HERE / "production_files.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["pr", "stage", "kind", "sha", "path", "added", "deleted"])
        for row in records:
            for f in row["production_files"]:
                writer.writerow([row["pr"], row["stage"], row["kind"], row["sha"], f["path"], f["added"], f["deleted"]])
    render(data)
    names = (
        "selection.json",
        "build_report.py",
        "summary.json",
        "prs.csv",
        "production_files.csv",
        "report.html",
        "README.md",
    )
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}
    (HERE / "manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def render(data):
    e = html.escape
    records = data["records"]
    stat_rows = []
    for kind in ("direct", "enable", "included"):
        t = data["summary"][kind]
        label = KINDS.get(kind, "纳入合计")
        stat_rows.append(
            f"<tr><td>{label}</td><td>{t['prs']}</td><td>+{t['production_added']:,} / −{t['production_deleted']:,}</td>"
            f"<td>{t['production_changed']:,}</td><td>{t['whole_pr_changed']:,}</td></tr>"
        )
    group_rows = []
    for kind in ("direct", "enable"):
        for group in dict.fromkeys(r["group"] for r in records if r["kind"] == kind):
            rows = [r for r in records if r["kind"] == kind and r["group"] == group]
            t = totals(rows)
            links = "、".join(f'<a href="#pr-{r["pr"]}">#{r["pr"]}</a>' for r in rows)
            group_rows.append(
                f"<tr><td>{KINDS[kind]}</td><td>{e(group)}</td><td>{links}</td><td>{t['production_changed']:,}</td></tr>"
            )
    cards = []
    for r in records:
        source_links = "".join(
            f'<li><a href="https://github.com/vllm-project/vllm-ascend/blob/{r["sha"]}/{e(f["path"], quote=True)}">'
            f"{e(f['path'])}（+{f['added']}/−{f['deleted']}）</a></li>"
            for f in r["production_files"]
        )
        search = e(f"{r['pr']} {r['title']} {r['group']} {r['relation']} {r['deepseek_review']}".lower(), quote=True)
        cards.append(f'''<article id="pr-{r["pr"]}" data-kind="{r["kind"]}" data-search="{search}">
<small>{r["stage"]} · {e(KINDS[r["kind"]])} · {e(r["relation"])}</small>
<h3><a href="{r["pr_url"]}">#{r["pr"]}</a> {e(r["title"])}</h3>
<p><b>机制：</b>{e(r["mechanism"])}</p><p><b>DeepSeek 适用性：</b>{e(r["deepseek_review"])}</p>
<p><b>生产变更：</b>+{r["production_added"]:,} / −{r["production_deleted"]:,}；<b>整 PR：</b>+{r["whole_pr_added"]:,} / −{r["whole_pr_deleted"]:,}</p>
<p><b>原报告证据：</b>{e(r["result"])}</p><p><b>限制：</b>{e(r["caveat"])}</p>
<small>数值沿用原作者场景，不代表本次 DeepSeek 复测。{e(r["patch_review"])}。</small>
<details><summary>提交、原始补丁及生产文件</summary><p><code>{r["sha"]}</code> · <a href="{r["local_patch"]}">本地完整 patch</a></p><ul>{source_links}</ul></details></article>''')
    excluded = []
    for number, reason in data["selection"]["excluded"].items():
        excluded.append(
            f'<li><a href="https://github.com/vllm-project/vllm-ascend/pull/{number}">#{number}</a>：{e(reason)}</li>'
        )
    for key, title in (
        ("supporting_correctness", "配套正确性修复"),
        ("baseline_present", "30 基线已有主体"),
        ("withdrawn", "撤回或路径退役"),
    ):
        links = "、".join(
            f'<a href="https://github.com/vllm-project/vllm-ascend/pull/{n}">#{n}</a>' for n in data["selection"][key]
        )
        excluded.append(f"<li>{title}：{links}；不计新增优化代码合计。</li>")
    stage_counts = data["summary"]["by_stage"]
    stage_line = "；".join(
        f"{stage}：{values['direct']['prs']} 项直接优化 + {values['enable']['prs']} 项使能"
        for stage, values in stage_counts.items()
    )
    conditions = Counter(r["relation"] for r in records if r["kind"] == "direct")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepSeek 优化 PR 与代码量 · 30 → 31 → 32</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fb;color:#20304a;font:15px/1.8 system-ui,"Noto Sans CJK SC",sans-serif}}main{{max-width:1200px;margin:auto;padding:30px 24px 70px}}a{{color:#255fbe;text-decoration:none}}a:hover{{text-decoration:underline}}h1{{font-size:32px;line-height:1.4}}h2{{font-size:24px;margin-top:34px}}h3{{font-size:18px}}p{{margin:10px 0}}.muted,small{{color:#5e6c80}}.callout{{background:#e6efff;border-left:4px solid #2a66c3;padding:16px 20px;margin:20px 0}}article{{background:white;border:1px solid #d5dfee;border-radius:10px;padding:20px 24px;margin:15px 0;scroll-margin-top:20px;overflow-wrap:anywhere}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #dce3ef;vertical-align:top}}th{{background:#e8eef8;white-space:nowrap}}td{{overflow-wrap:anywhere}}.controls,nav{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}}input,select,button{{font:inherit;padding:8px 12px;border:1px solid #becde1;border-radius:5px;background:white;color:inherit}}input{{flex:1;min-width:180px}}button{{cursor:pointer}}code{{font-size:12px;overflow-wrap:anywhere}}li{{overflow-wrap:anywhere}}summary{{color:#255fbe;cursor:pointer}}[hidden]{{display:none!important}}@media(max-width:600px){{main{{padding:20px 14px}}h1{{font-size:26px}}article{{padding:16px}}td,th{{padding:9px;font-size:13px}}}}@media print{{body{{background:white}}main{{padding:0}}.controls,nav,button{{display:none}}article{{break-inside:avoid}}article[hidden]{{display:block!important}}}}
</style></head><body><main><nav><a href="../index.html">← 三版本总览</a><a href="prs.csv" download>PR 明细 CSV</a><a href="production_files.csv" download>逐文件行数 CSV</a><a href="summary.json">完整 JSON</a><button onclick="window.print()">打印 / PDF</button></nav>
<p class="muted">固定快照专题 · 2026-11-03 整理</p><h1>DeepSeek 相关优化与代码量<br>30 → 31 → 32</h1>
<p>{e(data["scope"])}。包括 DeepSeek 专项及在匹配配置下可适用的公共优化；不将 GLM、Qwen、MiniMax 专用路径自动归入。</p>
<p>30：<code>7d825229c7</code>（10-18）；31：<code>5e40bc7231</code>（10-26）；32：<code>d08fa9d438</code>（11-02）。{stage_line}。</p>
<div class="callout"><b>是“PR 信息 + 代码”组合筛选。</b>标题、描述与变更文件用于发现候选；关键生产 diff、模型入口和条件分支用于核对机制与适用性。不是每个 PR 的全链路逐行审计，也没有 NPU 精度或性能复测。代码显示少一次拷贝/启动，不等于实测吞吐提高。</div>
<h2>代码量</h2><div class="table-wrap"><table><thead><tr><th>统计范围</th><th>PR 数</th><th>生产代码 +新增 / −删除</th><th>生产变更行</th><th>整 PR 变更行</th></tr></thead><tbody>{"".join(stat_rows)}</tbody></table></div>
<p>37 项直接优化中：{conditions["DeepSeek 专项"]} 项 DeepSeek 专项、{conditions["条件适用的公共路径"]} 项条件适用的公共路径。另有 3 项候选不计入上述合计，见明细筛选。使能项之间可能属于同一机制的连续扩展，PR 数不代表独立优化数。</p>
<p><b>统计口径：</b>{e(data["counting"])}。例如 #16731 的生产变更 +6/−3798 主要是旧 QLI 删除；#18256 还包含非 DeepSeek 部分，不能把整 PR 当作最小回移代码。未在此合计撤销项、已有回移、纯重构和配套正确性修复。</p>
<h2>优化方向与完整 PR 清单</h2><div class="table-wrap"><table><thead><tr><th>类别</th><th>方向</th><th>PR</th><th>生产变更行</th></tr></thead><tbody>{"".join(group_rows)}</tbody></table></div>
<h2>筛选证据与适用条件</h2><p>本专题在原报告分类之上增加 DeepSeek 适用性复核。例如 hc_pre_v2/v3 都通过 run_hc_pre_fusion 调 aclnnHcPre，因此 #18293 纳入共享算子；#17068 的 BF16 router 快路径被 V4/V4.1 的 FP32 precast 绕过。#19247 主要涉及旧代 MLA/SFA RoPE；#20503 只有公共 MTP/static capture 范围部分可涉及 DeepSeek。</p>
<p>作者数据逐项保留原模型、硬件及测量层级。没有把 Qwen 的 #18074、GLM 的 #19247 测量改称 DeepSeek 数据；#19342 探索结果没有证实整模型提升，#18293、#16350 等也存在小 shape 无收益或退化。不同 PR 的百分比不能累加。</p>
<div class="controls"><input id="query" aria-label="搜索 PR、机制或条件" placeholder="搜索 PR、机制或适用条件"><select id="kind" aria-label="筛选分类"><option value="">全部（含候选）</option><option value="direct">直接优化 / 性能恢复</option><option value="enable">性能路径使能</option><option value="candidate">条件候选，不计合计</option></select><button id="reset">重置</button></div><p id="count" class="muted"></p>
<div id="cards">{"".join(cards)}</div><h2>未计入合计的相关项</h2><ul>{"".join(excluded)}</ul><p class="muted">这是对既有人工性能记录的 DeepSeek 子集复核，并非对仓库所有提交完成了穷尽性语义证明。原始 patch、作者来源和上下文见两阶段详报；本专题可通过 python reports/deepseek_summary/build_report.py 重建并校验 Git numstat。</p>
</main><script>
'use strict';const cards=[...document.querySelectorAll('#cards article')];const q=document.getElementById('query'),kind=document.getElementById('kind');function filter(){{let n=0;for(const card of cards){{const hit=(!kind.value||kind.value===card.dataset.kind)&&(!q.value.trim()||card.dataset.search.includes(q.value.trim().toLowerCase()));card.hidden=!hit;if(hit)n++;}}document.getElementById('count').textContent=`显示 ${{n}} / ${{cards.length}} 项（其中 3 项为未纳入合计的候选）；上方统计为固定选集。`;}}q.addEventListener('input',filter);kind.addEventListener('change',filter);document.getElementById('reset').onclick=()=>{{q.value='';kind.value='';filter();}};filter();
</script></body></html>"""
    (HERE / "report.html").write_text(page)


if __name__ == "__main__":
    main()
