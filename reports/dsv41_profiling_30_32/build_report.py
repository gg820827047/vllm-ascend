#!/usr/bin/env python3
# Long HTML fragments preserve their authored whitespace in the generated report.
# ruff: noqa: E501
"""Build a fully offline Chinese profiler analysis and learning report."""

import html
import json

import markdown
from analyze import HERE

LABELS = {
    "v030_unlabelled": "11/03 · 0.30 · P 特征侧",
    "v032_p_tp4": "11/07 · 0.32 · P / TP4",
    "v032_d_tp2": "11/07 · 0.32 · D / TP2",
    "v032_other_host11": "11/09 · 0.32 · host11 / TP8",
    "v032_other_host35": "11/09 · 0.32 · host35 / TP2",
}


def esc(value):
    return html.escape(str(value))


def main():
    analysis = json.loads((HERE / "data/analysis.json").read_text())
    evidence = json.loads((HERE / "data/evidence.json").read_text())
    reps = [r for r in analysis["ranks"] if r["rank"]["tp"] == 0]
    rank_rows = []
    for r in analysis["ranks"]:
        s = r["step_summary"][0]
        stage = float(s["Stage"])
        rank_rows.append(
            f"<tr><td>{esc(LABELS[r['id']])}</td><td>{r['rank']['dp']}</td><td>{r['rank']['tp']}</td>"
            f"<td>{r['kernel_count']:,}</td><td>{stage / 1e6:.3f}</td>"
            f"<td>{float(s['Free']) / stage * 100:.1f}%</td><td>{float(s['Computing']) / stage * 100:.1f}%</td></tr>"
        )
    catalog = []
    for top in evidence["topology"]:
        r = next(r for r in reps if r["id"] == top["dataset"] and r["rank"] == top["rank"])
        number = sum(x["id"] == r["id"] and x["rank"]["dp"] == r["rank"]["dp"] for x in analysis["ranks"])
        inference = (
            "DP2、PCP1（采到DP0/1，结合EP/TP确定）"
            if r["id"] == "v032_other_host11"
            else f"DP×PCP={top['dp_if_no_cp']}；PCP1时DP={top['dp_if_no_cp']}"
        )
        catalog.append(
            f"<tr><td>{esc(LABELS[r['id']])}<br>采到 DP{r['rank']['dp']}</td><td>{number}</td>"
            f"<td>{top['tp_size']}</td><td>{top['ep_size']}</td><td>{esc(inference)}</td></tr>"
        )
    cards = []
    for p in evidence["pr_verification"]:
        state = p["trace_verdict"]
        tone = "verified" if state == "机制已验证" else "limited"
        cards.append(f'''<details class="pr-card {tone}" data-status="{esc(state)}" data-search="{esc(str(p["pr"]) + " " + p["title"] + " " + p["group"] + " " + state)}">
<summary><span class="badge {tone}">{esc(state)}</span><b>#{p["pr"]} · {esc(p["title"])}</b><span>{esc(p["stage"])}</span></summary>
<p><strong>本次证据：</strong>{esc(p["trace_evidence"])}</p><p><strong>下一步：</strong>{esc(p["next_validation"])}</p>
<p><strong>原优化机制：</strong>{esc(p["mechanism"])}</p><p><strong>适用条件：</strong>{esc(p["applicable"])}</p>
<p class="muted">原报告说明：{esc(p.get("deepseek_review") or p.get("caveat", ""))}</p>
<a href="{esc(p["pr_url"])}" target="_blank" rel="noreferrer">查看 GitHub PR</a></details>''')
    sources = "".join(
        f'<details class="source"><summary>{esc(s["path"])} · {esc(s["revision"][:12])}</summary>'
        f'<pre>{esc(s["text"])}</pre><a href="{esc(s["url"])}">固定版本源码</a></details>'
        for s in evidence["sources"]
    )
    coverage = []
    colors = ["#4263eb", "#e99c37", "#dce4ed"]
    for r in reps:
        s = r["step_summary"][0]
        stage = float(s["Stage"])
        segments = "".join(
            f'<span style="width:{float(s[key]) / stage * 100:.4f}%;background:{color}" '
            f'title="{esc(key)}: {float(s[key]) / 1000:.3f} ms"></span>'
            for key, color in zip(["Computing", "Communication(Not Overlapped)", "Free"], colors)
        )
        coverage.append(
            f'<div class="coverage-row"><div>{esc(LABELS[r["id"]])} · DP{r["rank"]["dp"]}</div>'
            f'<div class="coverage-bar">{segments}</div><span>{stage / 1e6:.3f}s</span></div>'
        )
    # Keep full shapes and representative trace intervals for offline inspection.
    payload = {
        "labels": LABELS,
        "ranks": reps,
        "layers": evidence["layers"],
        "timelines": evidence["timelines"],
        "engram": evidence["engram_windows"],
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    tutorial = markdown.markdown((HERE / "tutorial.md").read_text(), extensions=["tables", "fenced_code"])
    tutorial = tutorial.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    replacements = {
        "__DATA__": serialized,
        "__PR_CARDS__": "\n".join(cards),
        "__SOURCES__": sources,
        "__RANK_ROWS__": "".join(rank_rows),
        "__CATALOG__": "".join(catalog),
        "__COVERAGE__": "".join(coverage),
        "__TUTORIAL__": tutorial,
        "__RANK_COUNT__": str(len(analysis["ranks"])),
    }
    template = (HERE / "report_template.html").read_text()
    for key, value in replacements.items():
        template = template.replace(key, value)
    (HERE / "report.html").write_text(template)
    print(HERE / "report.html", "bytes", len(template.encode()))


if __name__ == "__main__":
    main()
