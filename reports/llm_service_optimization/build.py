#!/usr/bin/env python3
"""Build the offline service optimization map and its editable task exports."""

import csv
import html
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_FIELDS = [
    ("id", "ID"),
    ("category_name", "主题"),
    ("stage_names", "请求位置"),
    ("title", "子事项"),
    ("kind", "任务类型"),
    ("priority", "优先级"),
    ("status", "当前状态"),
    ("owner", "责任人"),
    ("role_name", "建议责任角色"),
    ("collaborator_names", "协同角色"),
    ("dependency_names", "前置任务"),
    ("current", "已有证据与范围"),
    ("action_text", "分析或实施步骤"),
    ("deliverable", "交付物"),
    ("expected_benefit", "预期收益"),
    ("metric", "验证指标"),
    ("acceptance", "验收条件"),
    ("evidence_names", "证据索引"),
]


def escape(value):
    return html.escape(str(value), quote=True)


def enrich(data):
    stage_names = {s["id"]: s["title"] for s in data["stages"]}
    stage_names["X"] = "跨阶段"
    ids = {t["id"] for t in data["tasks"]}
    assert len(ids) == len(data["tasks"]), "Duplicate task ID"
    evidence_ids = {e["id"] for e in data["evidence"]}
    for task in data["tasks"]:
        assert set(task["depends"]) <= ids, task["id"]
        assert set(task["evidence"]) <= evidence_ids, task["id"]
        assert set(task["stages"]) <= set(stage_names), task["id"]
        task["category_name"] = data["categories"][task["category"]]
        task["stage_names"] = " / ".join(f"{s} {stage_names[s]}" for s in task["stages"])
        task["role_name"] = data["roles"][task["owner_role"]]
        task["collaborator_names"] = " / ".join(data["roles"][r] for r in task["collaborators"])
        task["dependency_names"] = " / ".join(task["depends"]) or "—"
        task["action_text"] = "；".join(task["actions"])
        task["evidence_names"] = " / ".join(task["evidence"]) or "—"
    # A task may require an earlier analysis; reject circular experiment plans.
    by_id = {t["id"]: t for t in data["tasks"]}

    def visit(task_id, path):
        assert task_id not in path, f"Cyclic dependency: {path} -> {task_id}"
        for dep in by_id[task_id]["depends"]:
            visit(dep, path | {task_id})

    for task_id in by_id:
        visit(task_id, set())
    for row in data["observability"]["rows"]:
        assert set(row["tasks"]) <= ids, row["title"]
    for finding in data["recent"]["findings"]:
        assert set(finding["tasks"]) <= ids, finding["id"]
        assert set(finding["evidence"]) <= evidence_ids, finding["id"]
    for project in data["peer_review"]["projects"]:
        assert set(project["tasks"]) <= ids, project["id"]
        assert set(project["evidence"]) <= evidence_ids, project["id"]
    for item in data["multistream_review"]["conclusions"]:
        assert set(item["tasks"]) <= ids, item["topic"]
    note_ids = [note["id"] for note in data["multistream_review"]["notes"]]
    assert len(note_ids) == len(set(note_ids)), "Duplicate mechanism note ID"


def review_table_html(table):
    header = "".join(f"<th>{escape(value)}</th>" for value in table["headers"])
    rows = "".join("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>" for row in table["rows"])
    return (
        f'<h3>{escape(table["title"])}</h3><div class="table-wrap">'
        f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def review_note_html(note):
    paragraphs = "".join(f"<p>{escape(value)}</p>" for value in note["paragraphs"])
    tables = "".join(review_table_html(table) for table in note.get("tables", []))
    code = f"<pre><code>{escape(note['code'])}</code></pre>" if note.get("code") else ""
    links = " · ".join(f'<a href="{escape(link["source"])}">{escape(link["title"])}</a>' for link in note["links"])
    return (
        f'<details id="mechanism-{escape(note["id"])}" class="mechanism-note">'
        f'<summary>{escape(note["title"])}</summary><p class="muted">{escape(note["scope"])}</p>'
        f"{paragraphs}{code}{tables}<p>{links}</p></details>"
    )


def multistream_review_html(data):
    section = data["multistream_review"]
    conclusions = {
        "title": "阶段结论与后续归属",
        "headers": ["事项", "当前状态", "已知证据与边界", "下一步 / 台账"],
        "rows": [
            [item["topic"], item["status"], item["finding"], item["next"] + " / " + "、".join(item["tasks"])]
            for item in section["conclusions"]
        ],
    }
    handoff = " / ".join(f'<a href="#task-{task}">{task}</a>' for task in ["A05", "A06", "C03", "A08", "B08"])
    steps = "".join(f"<li>{escape(step)}</li>" for step in section["next_steps"])
    notes = "".join(review_note_html(note) for note in section["notes"])
    return (
        f"<h2>{escape(section['title'])}</h2><p>{escape(section['scope'])}</p>"
        f'<p class="note"><b>{escape(section["verdict"])}</b></p>'
        f'<p class="note warn">{escape(section["hypothesis_limit"])}</p>'
        f"{review_table_html(conclusions)}<p>跟进任务：{handoff}</p>"
        f"<h3>下一轮只验证这些问题</h3><ol>{steps}</ol>"
        f"<h3>原理与profiling对照笔记（持续更新）</h3><p>展开查看，表中时间口径分别标注。</p>{notes}"
    )


def review_table_markdown(table):
    headers = table["headers"]
    lines = [
        f"### {table['title']}",
        "",
        "| " + " | ".join(map(markdown_cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(markdown_cell, row)) + " |" for row in table["rows"])
    return lines + [""]


def multistream_review_markdown(data):
    section = data["multistream_review"]
    lines = [
        f"## {section['title']}",
        "",
        section["scope"],
        "",
        section["verdict"],
        "",
        section["hypothesis_limit"],
        "",
    ]
    for item in section["conclusions"]:
        lines.extend(
            [
                f"### {item['topic']}：{item['status']}",
                "",
                item["finding"],
                "",
                f"后续：{item['next']}。台账：{' / '.join(item['tasks'])}。",
                "",
            ]
        )
    lines.extend(["### 下一轮验证", ""])
    lines.extend(f"{index}. {step}" for index, step in enumerate(section["next_steps"], 1))
    lines.append("")
    for note in section["notes"]:
        lines.extend([f"### {note['title']}", "", note["scope"], ""])
        for paragraph in note["paragraphs"]:
            lines.extend([paragraph, ""])
        if note.get("code"):
            lines.extend(["```text", note["code"], "```", ""])
        for table in note.get("tables", []):
            lines.extend(review_table_markdown(table))
        lines.extend([" · ".join(f"[{link['title']}]({link['source']})" for link in note["links"]), ""])
    return lines


def recent_html(data):
    section = data["recent"]
    timing = section["timing"]
    report_links = "".join(
        f'<p><a href="{escape(report["source"])}">{escape(report["title"])}</a> · {escape(report["detail"])}</p>'
        for report in section.get("reports", [])
    )
    rows = "".join(
        "<tr>" + "".join(f"<td>{escape(row[key])}</td>" for key in ["boundary", "time", "meaning"]) + "</tr>"
        for row in timing["rows"]
    )
    cards = []
    for finding in section["findings"]:
        facts = "".join(f"<li>{escape(fact)}</li>" for fact in finding["facts"])
        tasks = " / ".join(f'<a href="#task-{task}">{task}</a>' for task in finding["tasks"])
        evidence = " / ".join(f'<a href="#evidence-{item}">{item}</a>' for item in finding["evidence"])
        cards.append(
            f'<article id="finding-{finding["id"]}"><h3>{escape(finding["title"])}</h3>'
            f"<p><b>{escape(finding['level'])}</b> · {escape(finding['stage'])}</p>"
            f"<ul>{facts}</ul><p class='note warn'>{escape(finding['limit'])}</p>"
            f"<p>跟进：{tasks}<br>证据：{evidence}</p></article>"
        )
    return (
        f"<h2>{escape(section['title'])}</h2><p>{escape(section['scope'])}</p>"
        f"{report_links}"
        f"<h3>{escape(timing['title'])}</h3><p>{escape(timing['reference'])}</p>"
        '<div class="table-wrap"><table><thead><tr><th>边界</th><th>时间</th><th>解读</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div><div class='evidence-grid recent-cards'>{''.join(cards)}</div>"
    )


def recent_markdown(data):
    section = data["recent"]
    timing = section["timing"]
    lines = [
        f"## {section['title']}",
        "",
        section["scope"],
        "",
        *[f"[{report['title']}]({report['source']})：{report['detail']}" for report in section.get("reports", [])],
        "",
        f"### {timing['title']}",
        "",
        timing["reference"],
        "",
        "| 边界 | 时间 | 解读 |",
        "| --- | --- | --- |",
    ]
    for row in timing["rows"]:
        lines.append("| " + " | ".join(markdown_cell(row[key]) for key in ["boundary", "time", "meaning"]) + " |")
    for finding in section["findings"]:
        lines.extend(["", f"### {finding['title']}", "", f"{finding['level']}。位置：{finding['stage']}。", ""])
        lines.extend(f"- {fact}" for fact in finding["facts"])
        lines.extend(
            [
                "",
                f"边界：{finding['limit']}",
                "",
                f"跟进：{' / '.join(finding['tasks'])}。证据：{' / '.join(finding['evidence'])}。",
            ]
        )
    return lines + [""]


def peer_html(data):
    section = data["peer_review"]
    takeaways = "".join(f"<li>{escape(item)}</li>" for item in section["takeaways"])
    rules = "".join(f"<li>{escape(item)}</li>" for item in section["measurement_rules"])
    experiments = "".join(f"<li>{escape(item)}</li>" for item in section["first_experiments"])
    summary = "".join(f"<p>{escape(item)}</p>" for item in section["daily_summary"])
    rows = []
    for project in section["projects"]:
        tasks = " / ".join(f'<a href="#task-{task}">{task}</a>' for task in project["tasks"])
        evidence = " / ".join(f'<a href="#evidence-{item}">{item}</a>' for item in project["evidence"])
        rows.append(
            f'<tr id="peer-{project["id"]}"><th scope="row">{project["id"]} {escape(project["title"])}'
            f'<p class="muted">对方报告：{escape(project["reported"])}</p></th>'
            f"<td><b>{escape(project['priority'])}</b><p>{escape(project['placement'])}</p>"
            f"<p>{escape(project['verdict'])}</p></td><td>{escape(project['match'])}<p>{evidence}</p></td>"
            f"<td>{escape(project['next'])}<p>跟进：{tasks}</p></td></tr>"
        )
    return (
        f"<h2>{escape(section['title'])}</h2><p>{escape(section['scope'])}</p>"
        f"<p class='note warn'>{escape(section['provenance'])}</p>"
        f"<h3>对我们最有价值的启发</h3><ul>{takeaways}</ul>"
        '<div class="table-wrap"><table id="peer-table"><thead><tr>'
        "<th>对方项目与原文结果</th><th>位置与优先级</th><th>我们的profiling / 代码证据</th><th>下一步</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        f"<h3>收益如何比较</h3><ul>{rules}</ul><h3>首批实验顺序</h3><ol>{experiments}</ol>"
        f'<div class="note"><h3>今天的进展 · 可直接用于简报</h3>{summary}</div>'
    )


def peer_markdown(data):
    section = data["peer_review"]
    lines = [f"## {section['title']}", "", section["scope"], "", section["provenance"], ""]
    lines.extend(f"- {item}" for item in section["takeaways"])
    lines.extend(
        [
            "",
            "| 对方项目与原文结果 | 位置与优先级 | 我们的证据 | 下一步与任务 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for project in section["projects"]:
        cells = [
            f"{project['id']} {project['title']}；对方报告：{project['reported']}",
            f"{project['placement']}；{project['priority']}；{project['verdict']}",
            f"{project['match']}；证据 {' / '.join(project['evidence'])}",
            f"{project['next']}；跟进 {' / '.join(project['tasks'])}",
        ]
        lines.append("| " + " | ".join(map(markdown_cell, cells)) + " |")
    lines.extend(["", "### 收益比较口径", ""])
    lines.extend(f"- {item}" for item in section["measurement_rules"])
    lines.extend(["", "### 首批实验顺序", ""])
    lines.extend(f"{i}. {item}" for i, item in enumerate(section["first_experiments"], 1))
    lines.extend(["", "### 今天的进展", ""])
    for item in section["daily_summary"]:
        lines.extend([item, ""])
    return lines


def observability_html(data):
    section = data["observability"]
    rows = []
    for row in section["rows"]:
        links = " / ".join(f'<a href="#task-{task_id}">{task_id}</a>' for task_id in row["tasks"])
        cells = "".join(f"<td>{escape(row[field])}</td>" for field in ["profiling", "collect", "measure"])
        rows.append(f"<tr><th scope='row'>{escape(row['title'])}</th>{cells}<td>{links}</td></tr>")
    notes = "".join(f"<li>{escape(note)}</li>" for note in section["notes"])
    return (
        f"<h2>{escape(section['title'])}</h2><p>{escape(section['intro'])}</p>"
        f"<p class='note warn'>{escape(section['priority'])}</p>"
        '<div class="table-wrap"><table id="observability-table"><thead><tr>'
        "<th>位置</th><th>单独看 worker profiling</th><th>需要补采</th><th>计时口径</th><th>任务</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div><ul>{notes}</ul>"
    )


def observability_markdown(data):
    section = data["observability"]
    lines = [
        f"## {section['title']}",
        "",
        section["intro"],
        "",
        section["priority"],
        "",
        "| 位置 | 单独看worker profiling | 需要补采 | 计时口径 | 任务 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in section["rows"]:
        cells = [row[k] for k in ["title", "profiling", "collect", "measure"]] + [" / ".join(row["tasks"])]
        lines.append("| " + " | ".join(map(markdown_cell, cells)) + " |")
    lines.extend([""] + [f"- {note}" for note in section["notes"]] + [""])
    return lines


def task_row(task):
    actions = "".join(f"<li>{escape(a)}</li>" for a in task["actions"])
    evidence = " ".join(f'<a href="#evidence-{e}">{e}</a>' for e in task["evidence"])
    stages = " ".join(f'<span class="tag">{s}</span>' for s in task["stages"])
    dependencies = " ".join(f'<a href="#task-{d}">{d}</a>' for d in task["depends"]) or "—"
    return f"""<tr id="task-{task["id"]}" data-task="{task["id"]}">
<th scope="row"><b>{task["id"]}</b><span class="priority">{task["priority"]}</span></th>
<td>{stages}<div class="muted">{escape(task["category_name"])}</div><span class="tag">{task["kind"]}</span></td>
<td class="task-detail"><details><summary>{escape(task["title"])}</summary><ol>{actions}</ol>
<p><b>交付：</b>{escape(task["deliverable"])}</p><p><b>验收：</b>{escape(task["acceptance"])}</p></details></td>
<td><b>{escape(task["owner"])}</b><div>{escape(task["role_name"])}</div>
<small>协同：{escape(task["collaborator_names"]) or "—"}</small></td>
<td><span class="status">{task["status"]}</span><p>{escape(task["current"])}</p>{evidence}</td>
<td><p>{escape(task["expected_benefit"])}</p><small><b>测量：</b>{escape(task["metric"])}</small></td>
<td>{dependencies}</td></tr>"""


def stage_button(stage):
    return f"""<button class="stage" type="button" data-stage="{stage["id"]}" aria-pressed="false">
<span class="stage-id">{stage["id"]}</span><strong>{escape(stage["title"])}</strong>
<span>{escape(stage["subtitle"])}</span></button>"""


def options(values):
    return "".join(f'<option value="{escape(k)}">{escape(v)}</option>' for k, v in values)


def render_html(data):
    stages = {s["id"]: stage_button(s) for s in data["stages"]}
    history = "".join(
        f"<tr><th scope='row'>{h['id']} {escape(h['title'])}</th>"
        f"<td>{escape(h['stage'])}</td><td><b>{escape(h['status'])}</b><p>{escape(h['conclusion'])}</p></td>"
        f"<td><a href='#evidence-{h['evidence']}'>{h['evidence']}</a><br>{escape(h['followup'])}</td></tr>"
        for h in data["history"]
    )
    evidence = "".join(
        f"<article id='evidence-{e['id']}'><h3>{e['id']} · {escape(e['title'])}</h3>"
        f"<p>{escape(e['detail'])}</p><a href='{escape(e['source'])}'>对应资料 ↗</a></article>"
        for e in data["evidence"]
    )
    roles = "".join(f"<li><b>{escape(name)}</b>：具体责任人待认领</li>" for name in data["roles"].values())
    category_cards = "".join(
        f'<button type="button" class="theme" data-category="{key}"><b>{escape(data["categories"][key])}</b>'
        f"<span>{sum(t['category'] == key for t in data['tasks'])} 项 · 点击筛选</span></button>"
        for key in ["A", "B", "C", "D", "F"]
    )
    substitutions = {
        "TITLE": escape(data["title"]),
        "DATE": escape(data["date"]),
        "SCOPE": escape(data["scope"]),
        "COUNT": str(len(data["tasks"])),
        "FOUNDATION": str(sum(t["category"] in ["G", "R"] for t in data["tasks"])),
        "CANDIDATES": str(sum(t["kind"] == "优化候选" for t in data["tasks"])),
        "THEMES": category_cards,
        "OBSERVABILITY": observability_html(data),
        "RECENT": recent_html(data),
        "MULTISTREAM_REVIEW": multistream_review_html(data),
        "PEER_REVIEW": peer_html(data),
        "HISTORY": history,
        "EVIDENCE": evidence,
        "ROLES": roles,
        "TASK_ROWS": "\n".join(task_row(t) for t in data["tasks"]),
        "CATEGORY_OPTIONS": options(data["categories"].items()),
        "STAGE_OPTIONS": options([(s["id"], f"{s['id']} {s['title']}") for s in data["stages"]] + [("X", "X 跨阶段")]),
        "STATUS_OPTIONS": options((x, x) for x in dict.fromkeys(t["status"] for t in data["tasks"])),
        "ROLE_OPTIONS": options(data["roles"].items()),
        "SCRIPT": (ROOT / "interactions.js").read_text(),
        "DATA": json.dumps(data, ensure_ascii=False).replace("</", "<\\/"),
        **stages,
    }
    page = (ROOT / "page_template.html").read_text()
    for key, value in substitutions.items():
        page = page.replace(f"@@{key}@@", value)
    assert "@@" not in page, "Unresolved HTML placeholder"
    return page


def markdown_cell(value):
    return str(value).replace("|", " / ").replace("\n", " ")


def render_markdown(data):
    """Keep the shareable plan and full task detail together in one document."""
    lines = [
        f"# {data['title']}",
        "",
        f"更新：{data['date']}。{data['scope']}",
        "",
        "## 当前主线：P侧Device对照与验证",
        "",
        "全局简版见 [执行位置、成本与任务](../dsv41_a3_a5_prefill/stage_brief.html)，"
        "实现证据见 [Device模块与源码对照](../dsv41_a3_a5_prefill/device_module_comparison/report.html)。",
        "",
        "- P0 / G04：待采集A3 2P2D、P侧32DP×TP1、128k输入profiling；部署/压测负责采集，性能分析负责复表。",
        "- P1 / B03：Device-Metadata窗口/索引复用与融合，先核对等价组；Host耗时对照暂缓。",
        "- P1 / B11：共享专家融合进MegaMoE，先核对A3 INT8/CANN支持；与B08共享专家多流独立。",
        "",
        "采集与接口核对可并行；128k和TP1不保证逐step query/历史KV、实际EP和专家负载相同。"
        "A5旧样本未强制均衡，不可比行只保留实现差异；当前无新增性能收益实测。个人责任待认领。",
        "",
        "## 全局位置、四类主线与边界观测",
        "",
        "```text",
        "S0 网关/API接入、排队、分词 → S1 引擎入队/等待、调度、缓存判定",
        "                         ↓",
        "P: S2 Host准备 → S3 Prefill模型层 → S4 采样/输出回传",
        "                         ↓",
        "S5 P/D KV交接（P2P） ↔ Store共享缓存读写旁路（可选）",
        "                         ↓",
        "D: S6 KV就绪/调度准备 → S7 Decode / draft / verify ↺",
        "                         ↓ 每轮采样复用S4",
        "S8 反分词/流式发送 → 客户端",
        "X: DP/TP/EP/PCP/PP、资源与多流依赖横跨多个阶段",
        "```",
        "",
        "这是逻辑主路径，不表示所有事件严格串行，也不表示所有请求都经过Store。首token由P或D返回、D是否补算尾token/恢复窗口，以实际协议与代码为准。一个请求可跨多个step，一个step也可能合批多个请求；dummy不等于warmup。",
        "",
        "| 用户事项 | 主要位置 | 核心边界 |",
        "| --- | --- | --- |",
        "| ① A3同步点 | S1/S2/S4/S6，及S3/S7跨流依赖 | 区分Host阻塞、设备流依赖、通信等待、输出线程wait；"
        "只优化暴露在关键路径上的可消除部分 |",
        "| ② 算子/性能/引擎 | S2准备、S3目标层、S7生成层 | 模型侧优先复用/融合接口/布局/多流/合法切分；"
        "框架侧关注输入准备、Runner、下发与图；重型核内问题交接算子方 |",
        "| ③ 快慢卡 | X，落到S2/S3/S5/S7 | 分开同工作量计算速度、晚到rank、通信链路与真实硬件异常 |",
        "| ④ 切分配置 | S1/S2/S3/S5/S6/S7 | 固定资源预算与负载，"
        "比较DP/TP/EP、batch/chunk、P/D配比、缓存、MTP和执行路径 |",
        "| ⑤ 网关排队/KV传输（P2） | S0/S1/S5/S6/S8 | "
        "F01分开网关真队列、API前处理和引擎等待；F02分开P/D交接与Store读写，联合请求级观测 |",
        "",
        "## 各阶段看什么",
        "",
    ]
    for stage in data["stages"]:
        lines.extend(
            [
                f"### {stage['id']} {stage['title']}",
                "",
                stage["meaning"],
                "",
                f"代码边界：{stage['code']}。观测：{stage['observe']}。",
                "",
            ]
        )
    lines.extend(recent_markdown(data))
    lines.extend(multistream_review_markdown(data))
    lines.extend(peer_markdown(data))
    lines.extend(observability_markdown(data))
    lines.extend(
        [
            "## 已有工作和证据等级",
            "",
            "代码集成、trace观察到、功能通过、性能收益通过分别记录；本台账没有新增任何NPU性能实验，也没有将既有观察改写成已兑现的加速百分比。",
            "",
            "| 已有工作 | 请求位置 | 当前证据 | 范围与后续 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for h in data["history"]:
        lines.append(
            f"| {h['id']} {h['title']} | {h['stage']} | {h['status']} | "
            f"{h['conclusion']}；{h['evidence']}；后续 {h['followup']} |"
        )
    lines.extend(
        [
            "",
            "## 责任与预期收益口径",
            "",
            "责任人按建议角色分配，具体人名均待认领，不代表人员承诺。分析/验证/归档任务本身的直接性能收益为0，产物是可复查证据和决策；优化候选的收益为假设，须实测。",
            "",
            "仅可用关键路径中未被其他工作覆盖、且能够消除的时间估计时延空间；不能把所有同步时长、算子累加、Free占比或上游PR收益直接相加。吞吐、排队与缓存改变后，还需要重新测客户端指标。",
            "",
            "教学示例（非实测）：100ms关键路径中某段10ms，若该段加速2倍、其余条件不变，则总延迟约95ms，减少5%；如果这段原本完全被其他工作覆盖，端到端收益可能为0。",
            "",
            "## 子事项总表",
            "",
            "P0先形成证据与回归基础；P1做受控实验和低风险候选；P2为后置专题、跨组件依赖或外部协同。"
            "F01/F02初始列P2，若成为TTFT主因则提升；点击HTML中的任务可展开实施步骤。",
            "",
            "| ID/优先级 | 位置/类型 | 子事项 | 责任人（建议角色） | 状态与已有证据 | 预期收益 / 指标 | 前置 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for t in data["tasks"]:
        cells = [
            f"{t['id']} / {t['priority']}",
            f"{'/'.join(t['stages'])} / {t['kind']}",
            t["title"],
            f"{t['owner']}（{t['role_name']}）",
            f"{t['status']}；{t['current']}",
            f"{t['expected_benefit']} 指标：{t['metric']}",
            t["dependency_names"],
        ]
        lines.append("| " + " | ".join(map(markdown_cell, cells)) + " |")
    lines.extend(["", "## 子事项实施与验收", ""])
    for t in data["tasks"]:
        lines.extend(
            [
                f"### {t['id']} {t['title']}",
                "",
                f"位置：{t['stage_names']}。责任人：{t['owner']}；"
                f"建议角色：{t['role_name']}；协同：{t['collaborator_names'] or '—'}。",
                "",
            ]
        )
        lines.extend(f"{i}. {a}" for i, a in enumerate(t["actions"], 1))
        lines.extend(
            [
                "",
                f"交付物：{t['deliverable']}",
                "",
                f"验收：{t['acceptance']}",
                "",
                f"前置：{t['dependency_names']}。证据：{t['evidence_names']}。",
                "",
            ]
        )
    lines.extend(
        [
            "## 建议启动顺序",
            "",
            "1. G01/G02/G03 + A01/A02/B01/C01：补齐真实基线、请求级结果和关联，产出同步/算子/rank三张证据清单。",
            "2. A05/A06 + C02/C03/B05：优先验证元数据复用与Host热点，同时查Mega等待和异常MatMul；"
            "A07异步开关只在支持组合中受控对照。",
            "3. B02/B03/A08 + D01/D02/D03：再推进可用融合、多流/切分与拓扑/预算；每次只改变计划变量。",
            "4. D04/D05/D06/D07 + R01/R02：缓存、SWA、MTP和执行路径分项验收；逐项回写实测与负结果，再评估组合。",
            "5. 11月27日当前投入为G04采集与B03/B11核对并行，Host耗时对照暂缓；"
            "上面保留全项目通用顺序。多流阶段分析已归档，B08共享专家辅助流兼容扩展实验后置；"
            "D08独立记录冷启动、缓存重启与实例就绪时间，和热服务时延分开。",
            "6. F01/F02作为P2补充专题，与主线复用request_id与时间戳采集；"
            "进入P/D容量或Store收益结论前，先补齐所依赖的排队与KV交接数据。",
            "",
            "## 证据索引与适用范围",
            "",
            "E03/E05是2026-11-13对同批原始数据库的补充查询摘要，原始数据路径/提取方法沿用A3/A5报告；本页没有附加新的全量同步统计或硬件遥测，不应理解为已排除所有同步/慢卡可能。",
            "",
        ]
    )
    for e in data["evidence"]:
        lines.extend([f"### {e['id']} {e['title']}", "", e["detail"], "", f"[对应资料]({e['source']})", ""])
    return "\n".join(lines)


def main():
    data = json.loads((ROOT / "tasks.json").read_text())
    enrich(data)
    (ROOT / "index.html").write_text(render_html(data))
    (ROOT / "plan.md").write_text(render_markdown(data))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([title for _, title in CSV_FIELDS])
    for task in data["tasks"]:
        writer.writerow([task.get(key, "") for key, _ in CSV_FIELDS])
    (ROOT / "tasks.csv").write_text(buffer.getvalue(), encoding="utf-8-sig", newline="")
    print(f"Built {len(data['tasks'])} tasks, {len(data['stages'])} stages; HTML / Markdown / CSV")


if __name__ == "__main__":
    main()
