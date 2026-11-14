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
            "5. F01/F02作为P2补充专题，与主线复用request_id与时间戳采集；"
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
