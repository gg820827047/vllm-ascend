#!/usr/bin/env python3
# Long HTML fragments and source URLs preserve their authored whitespace.
# ruff: noqa: E501
"""Build a linked 0.30 → 0.31 → 0.32 overview from two immutable reports."""

import ast
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STAGE1 = "v0.31.0rc_vs_v0.30.0rc2"
STAGE2 = "v0.32.0rc_vs_v0.31.0rc"
GITHUB = "https://github.com/vllm-project/vllm-ascend"

# Domain, baseline state, 31 delta, 31 reviewed PRs, 32 delta, 32 reviewed PRs.
MATRIX = [
    (
        "DSpark context projection",
        "DSpark已有context路径，各draft层独立执行WKV projection。",
        "DeepSeek V4多draft层WKV堆叠，减少同输入的小GEMM。",
        [17263],
        "同类堆叠扩展至V4.1，并跨层复用RoPE lookup；这是模型覆盖扩展。",
        [19550],
    ),
    (
        "Embedding / RoPE / UVA",
        "保留原embedding exchange、RoPE registration与UVA fallback。",
        "PCP O projection改为activation exchange并跨层共享scratch。",
        [18506, 19190],
        "消除compiled embedding整容量复制、flat RoPE按行gather、UVA按区间追踪。",
        [19409, 19247, 20542],
    ),
    (
        "CPU metadata / 同步",
        "各路径的原metadata与CPU精确长度同步构成对照。",
        "GLM A5 DFlash2设备长度与metadata复用，RuntimeGuard静态关闭快路径。",
        [18717, 18644],
        "PCP跨group共享metadata、融合GDN mixed metadata，Qwen合格consumer默认省D2H同步。",
        [19007, 19954, 20444],
    ),
    (
        "GLM KDA / GDN 布局",
        "原GDN/GLM convolution/state实现。",
        "减少QKV layout copy、卷积state原位更新、跨层共享conv输出并减少预清零。",
        [18723, 18917, 19408],
        "再去除KDA padding mask；310P用专用masked_fill/index_copy代替IndexPut。",
        [19620, 19295],
    ),
    (
        "RecoverSSM",
        "未在该快照检出本报告跟踪的GLM KDA RecoverSSM算子。",
        "31没有新报告所跟踪的GLM RecoverSSM框架链；不得与SWA bounded replay混同。",
        [],
        "算子→MRV2/PD集成→prefix align→aligned commit优化；多PR是一条机制，OFF/ON与局部A/B分开。",
        [20128, 20131, 20369, 20524],
    ),
    (
        "torch.compile / static kernel",
        "V4.1主干编译入口尚未接入本次新增decorator；并非所有模型都不编译。",
        "DeepSeek V4.1主干接入torch.compile并补动态embedding维。",
        [18081, 18191],
        "扩展GLM backbone/MTP，补齐static shape范围和安装包编译兼容、PCP custom-op boundary与FX多matcher注册。",
        [19076, 20503, 20473, 19756],
    ),
    (
        "SWA bounded replay",
        "PCP基础能力已有回移；本报告未检出swa_bounded_replay开关。",
        "V4.1 encoder/decoder replay与V1/V2协议接入，配置组合有严格gate。",
        [17742, 17496, 18936],
        "先扩大到PCP（DP1），再允许PCP+DP协调；eager等限制仍需核对。",
        [19321, 20210],
    ),
    (
        "QSFA / QLI",
        "量化SFA/QLI已有基础实现；不能把原生算子名当新增收益。",
        "CANN pooled indexer+Triton adapters、paged mask与FlashMLA/mixed quant路径。",
        [16730, 18295, 18596, 18742, 17354],
        "A5与A2/A3 NoPE分别优化、kernel内tail-zero、QLI按实际TopK缩宽，并可选FP8 LI。",
        [20116, 20584, 20578, 20441, 20447],
    ),
    (
        "通信 / MoE",
        "共享专家overlap主体已通过#18122存在；TurboQuant与KV fence也已有回移。",
        "BF16 router+FP32 accumulation、按需event、W8A8 MXFP共享activation融合、MMRS融合与EPLB异步map。",
        [17068, 17603, 18474, 18074, 18020],
        "uniform SP的EP gather/reduce去恒等重建、DCP布局/发布融合、HIXL按消费event等待。",
        [19515, 16350, 19342],
    ),
    (
        "MiniMax-M3",
        "QK norm/RoPE/cache融合已通过#18325在基线。",
        "提高融合范围到256K、融合TopK周边、迁移CANN IndexScore。",
        [18666, 18294, 18949],
        "原融合kernel跨token复用norm权重；A5改AscendC IndexScore并融合MXFP8 SwiGLU。",
        [20308, 20290],
    ),
    (
        "KV Pool / SSD",
        "AscendStore、KVPP与layerwise基础已有。",
        "按步跨层缓存masked block runs、reachable mask/lookup/save优化、全局预取窗口。",
        [17739, 18907, 18647],
        "HBM prefix跳过probe、可选SSD prefetch、去掉MemCache启动barrier；模块拆分保留31缓存实现。",
        [19755, 19472, 19370, 20140],
    ),
    (
        "KV 容量 / 权重生命周期",
        "Step V1 overlay等基线能力已存在。",
        "释放死scale、OTP scratch复用、Gemma4及GLM共享pool预算恢复。",
        [17105, 19190, 19264, 19521],
        "Step MRV2共享backing、310P预算修复、ModelSlim reload释放旧权重，builder适配multi-allocation。",
        [19558, 19984, 17344, 20203],
    ),
    (
        "Qwen3.8 / PLE",
        "不以不同模型之间的数据做版本A/B。",
        "本报告未将Qwen3.8/PLE链作为31新增项；Qwen3.5 VL预处理另见多模态。",
        [],
        "新模型、PLE prefill、24K逻辑KV dispatch、PP/SP、ETP表分片、INT8表及W4A8 Linear。",
        [16981, 20344, 20467, 20443, 20516, 20548, 20507],
    ),
    (
        "投机与多模态",
        "已有MTP/投机基础，具体模型与runner覆盖不同。",
        "block verification、standalone draft、encoder ACL Graph及Qwen3.5 VL预处理下沉NPU。",
        [19138, 19364, 15571, 16996],
        "增加Gemma4/MultiModule MTP与NGram NPU，VL encoder disaggregation默认V2。",
        [17492, 19296, 18061, 19183],
    ),
    (
        "退役 / 回退 / 依赖",
        "各发布分支存在独立回移，箭头不等于祖先关系。",
        "撤销A3 GMM融合、StoreKVBlock、RPC结果缓存与PCP-local profile预算。",
        [18902, 18128, 19251, 19243],
        "退役legacy DSA-CP、KVPP/CPP部分V1分支；CANN9.3临时包升级回退，最终base9.2.1。",
        [19493, 18958, 18964, 20309],
    ),
]

MARKERS = [
    ("_masked_runs_cache", "masked block runs缓存"),
    ("_REACHABLE_MASK_CACHE_SIZE", "reachable mask缓存"),
    ("use_shared_conv_output", "GLM共享卷积输出"),
    ("_cos_sin_flat", "RoPE flat base"),
    ("can_skip_qwen_cpu_length_sync", "Qwen CPU length能力gate"),
    ("kda_recoverssm_verify", "GLM RecoverSSM算子"),
    ("npu_swiglu_mx_quant", "MXFP8 SwiGLU算子接入"),
    ("QSA_FIA_MAX_LOGICAL_KV_LENGTH", "QSA逻辑KV dispatch阈值"),
    ("memcache_dp_init_barrier", "旧MemCache DP barrier配置"),
    ("enable_dsa_cp", "legacy DSA-CP配置"),
]


def e(value):
    return html.escape(str(value), quote=True)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO).decode("utf-8", "replace")


def pr_link(number):
    return f'<a href="{GITHUB}/pull/{number}" target="_blank" rel="noopener">#{number}</a>'


def stage_links(stage, numbers):
    return " · ".join(f'<a href="{stage}/report.html#perf-{n}">#{n}</a>' for n in numbers) or "—"


def ui_style():
    parsed = ast.parse((HERE / STAGE1 / "build_report.py").read_text())
    for node in parsed.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TEMPLATE" for t in node.targets):
            template = ast.literal_eval(node.value)
            return re.search(r"<style>([\s\S]*?)</style>", template).group(1)
    raise ValueError("Missing UI source")


def main():
    first = json.loads((HERE / STAGE1 / "report_data.json").read_text())
    second = json.loads((HERE / STAGE2 / "report_data.json").read_text())
    assert first["meta"]["new_sha"] == second["meta"]["old_sha"]
    versions = [
        dict(label="v0.30.0rc2", sha=first["meta"]["old_sha"], date=first["meta"]["old_tip"]),
        dict(label="v0.31.0rc", sha=first["meta"]["new_sha"], date=first["meta"]["new_tip"]),
        dict(label="v0.32.0rc", sha=second["meta"]["new_sha"], date=second["meta"]["new_tip"]),
    ]
    old, middle, new = [v["sha"] for v in versions]
    left, right = map(int, git("rev-list", "--left-right", "--count", f"{old}...{new}").split())
    cumulative_shas = set(git("rev-list", f"{old}..{new}").split())
    stage1_only = [c for c in first["added"] if c["sha"] not in cumulative_shas]
    assert left == 55 and right == 621
    assert {c["pr"] for c in stage1_only} == {19521, 19522, 19696}
    assert len(first["added"]) + len(second["added"]) - len(stage1_only) == right
    cumulative = dict(
        old_only=left,
        new_only=right,
        shortstat=git("diff", "--shortstat", old, new).strip(),
        first_stage_shas_not_reachable_from_32=[dict(pr=c["pr"], sha=c["sha"]) for c in stage1_only],
    )
    assert cumulative["shortstat"] == "1779 files changed, 180728 insertions(+), 99466 deletions(-)"
    matrix = [
        dict(domain=row[0], baseline=row[1], stage31=row[2], prs31=row[3], stage32=row[4], prs32=row[5])
        for row in MATRIX
    ]
    notes1 = {n["pr"] for n in first["performance"]}
    notes2 = {n["pr"] for n in second["performance"]}
    assert all(set(row["prs31"]) <= notes1 and set(row["prs32"]) <= notes2 for row in matrix)
    markers = []
    for token, label in MARKERS:
        snapshots = []
        for version in versions:
            run = subprocess.run(
                ["git", "grep", "-n", "-F", token, version["sha"], "--", "vllm_ascend"],
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            assert run.returncode in {0, 1}
            hits = []
            for line in run.stdout.splitlines():
                _, path, number, _ = line.split(":", 3)
                hits.append(
                    dict(
                        path=path,
                        line=int(number),
                        url=f"{GITHUB}/blob/{version['sha']}/{quote(path, safe='/')}#L{number}",
                    )
                )
            snapshots.append(
                dict(
                    label=version["label"],
                    match_count=len(hits),
                    files=sorted({h["path"] for h in hits}),
                    examples=hits[:3],
                )
            )
        markers.append(dict(token=token, label=label, snapshots=snapshots))
    out = dict(
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        versions=versions,
        cumulative=cumulative,
        stages=[
            dict(name="30→31", directory=STAGE1, meta=first["meta"], kinds=first["classification_counts"]),
            dict(name="31→32", directory=STAGE2, meta=second["meta"], kinds=second["classification_counts"]),
        ],
        matrix=matrix,
        markers=markers,
    )
    (HERE / "overview_data.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["domain", "baseline_30", "delta_31", "prs_31", "delta_32", "prs_32"])
    for row in matrix:
        writer.writerow(
            [
                row["domain"],
                row["baseline"],
                row["stage31"],
                ",".join(map(str, row["prs31"])),
                row["stage32"],
                ",".join(map(str, row["prs32"])),
            ]
        )
    matrix_csv = buffer.getvalue()
    (HERE / "evolution_matrix.csv").write_text("\ufeff" + matrix_csv)
    version_cards = "".join(
        f'<div class="stat"><strong style="font-size:25px">{v["label"]}</strong><span>{e(v["date"][:10])}</span><p><a href="{GITHUB}/commit/{v["sha"]}"><code>{v["sha"][:12]}</code></a></p></div>'
        for v in versions
    )
    comparison_rows = [
        ("新侧独有提交 / PR", 364, 260),
        ("旧侧独有提交 / PR", 55, 3),
        ("补丁等价匹配", 44, 3),
        ("另外审阅的同源适配", 6, 0),
        ("未命中这两类映射的历史提交", 314, 257),
        ("文件净差异", 1140, 1094),
        ("新增行", 113380, 73862),
        ("删除行", 61855, 44125),
        ("显式[Performance]提交", 23, 22),
        ("其中基线已有等价性能提交", 3, 0),
        (
            "直接优化 / 性能恢复审阅记录",
            first["classification_counts"]["direct"],
            second["classification_counts"]["direct"],
        ),
        (
            "性能相关能力 / 路径使能",
            first["classification_counts"]["enable"],
            second["classification_counts"]["enable"],
        ),
        ("全部性能及关联审阅记录", len(first["performance"]), len(second["performance"])),
    ]
    comparison_html = "".join(
        f"<tr><td>{e(label)}</td><td>{a:,}</td><td>{b:,}</td></tr>" for label, a, b in comparison_rows
    )
    matrix_html = "".join(
        f'<tr class="matrix-row" data-search="{e(json.dumps(row, ensure_ascii=False).lower())}"><td><b>{e(row["domain"])}</b></td><td>{e(row["baseline"])}</td><td>{e(row["stage31"])}<p>{stage_links(STAGE1, row["prs31"])}</p></td><td>{e(row["stage32"])}<p>{stage_links(STAGE2, row["prs32"])}</p></td></tr>'
        for row in matrix
    )
    marker_html = "".join(
        "<tr><td>"
        + e(row["label"])
        + "<br><code>"
        + e(row["token"])
        + "</code></td>"
        + "".join(
            "<td>"
            + (
                "未检出"
                if not snapshot["match_count"]
                else str(snapshot["match_count"])
                + "处匹配<br>"
                + "".join(
                    f'<a href="{hit["url"]}"><code>{e(hit["path"])}</code></a><br>' for hit in snapshot["examples"]
                )
            )
            + "</td>"
            for snapshot in row["snapshots"]
        )
        + "</tr>"
        for row in markers
    )
    rejected = "、".join(pr_link(c["pr"]) for c in stage1_only)
    style = (
        ui_style()
        + ".matrix th:nth-child(1){width:15%}.matrix th{min-width:180px}.matrix th:first-child{min-width:130px}.matrix td{font-size:13px}.version-stats{grid-template-columns:repeat(3,1fr)}@media(max-width:720px){.version-stats{grid-template-columns:1fr}.matrix th{min-width:230px}}"
    )
    csv_payload = json.dumps(matrix_csv, ensure_ascii=False).replace("<", "\\u003c")
    rendered = f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>vLLM Ascend 30 → 31 → 32 代码与性能演进</title><style>{style}</style></head><body><div class="layout"><aside class="sidebar"><div class="brand">vLLM Ascend</div><div class="version">30 → 31 → 32<br>Code & Performance Evolution</div><div class="side-label">三版本总览</div><nav><a href="#overview">01 · 快照与结论</a><a href="#stages">02 · 两阶段统计</a><a href="#matrix">03 · 性能演进矩阵</a><a href="#markers">04 · 实现符号追踪</a><a href="#evidence">05 · 证据解读</a><a href="{STAGE1}/report.html">进入 30 → 31 详报</a><a href="{STAGE2}/report.html">进入 31 → 32 详报</a><a href="deepseek_summary/report.html">DeepSeek 优化与代码量</a><a href="pd_configuration_comparison/report.html">A3 P/D 配置与迁移</a></nav><div class="side-footer">日期 2026-11-02<br>固定SHA / 离线浏览<br>外部来源需联网</div></aside><main class="main"><header><button class="print-button" onclick="window.print()">打印 / 导出PDF</button><div class="eyebrow">THREE RELEASE SNAPSHOTS · 2026.11.02</div><h1>30 → 31 → 32<br>代码与性能演进</h1><p class="subtitle">以 v0.30.0rc2、v0.31.0rc、v0.32.0rc 三个固定发布分支快照为依据，跟踪合入历史、去重后的增量线索与性能机制演进。</p><div class="tagline"><span class="pill">vllm-project/vllm-ascend</span><span class="pill">两阶段详报</span><span class="pill">跨版本机制矩阵</span><span class="pill">可追溯代码与作者证据</span></div></header>
<section id="overview"><h2>01 · 固定快照与主要结论</h2><div class="stats version-stats">{version_cards}</div><div class="callout"><b>两个阶段共用完全相同的31 SHA。</b>30→31详报保持原快照，31→32新增详报沿用相同基线，因此机制可以逐段追踪。箭头表示比较顺序，并不表示这些release分支彼此是祖先关系。</div><p><b>31主要建立或扩展：</b>DSpark/Attention融合、PCP O-projection通信、GLM/GDN拷贝与清零、KV Pool主机开销、编译和多模态路径。</p><p><b>32主要继续推进：</b>metadata/D2H/UVA及embedding/RoPE搬运成本、GLM compile与RecoverSSM、A2/A3/A5 QSFA/QLI、SP/DCP布局，以及Qwen3.8/PLE新模型与并行/量化能力。</p><div class="download-row"><a class="button" href="{STAGE1}/report.html">30 → 31 详报</a><a class="button" href="{STAGE2}/report.html">31 → 32 详报</a><a class="button" href="deepseek_summary/report.html">DeepSeek 优化专题</a><a class="button" href="pd_configuration_comparison/report.html">P/D 配置对照与迁移</a><button id="download-matrix">演进矩阵CSV</button><a class="button" href="overview_data.json" download>总览JSON</a></div><div class="callout amber">没有执行新的NPU benchmark。PR作者的局部、模型、历史回移、联合栈和开关对照不能混成一个整体版本加速百分比。直接优化记录数也不是独立优化机制数，两个阶段的百分比与PR数量均不可简单相加。</div></section>
<section id="stages"><h2>02 · 两个阶段的统计对照</h2><div class="panel table-wrap"><table><thead><tr><th>指标</th><th>30 → 31</th><th>31 → 32</th></tr></thead><tbody>{comparison_html}</tbody></table></div><p>30→31的44条patch-id等价与6条同源适配，均标出基线回移；31→32的3条旧侧独有提交全部patch-id等价。其余314/257条仍含文档、CI、测试、重构和修复，不能称作同数目的新功能或性能优化。</p><div class="panel"><h3>30 → 32 总体净差异：单独计算，而非把两段相加</h3><p>直接对比两端快照：32侧独有<b>621</b>个提交，30侧独有<b>55</b>个；<b>1779</b>个文件、<b>+180,728 / −99,466</b>行。文件重命名、模块拆分与kernel移除均计入代码规模。</p><p>364+260=624，但31中的三个release回移SHA（{rejected}）不在32的祖先历史里，32保留对应main原提交，所以直接历史范围得到621。这个差值说明<b>SHA不再出现不代表改动缺失</b>。</p><pre>git rev-list --left-right --count {old}...{new}
git diff --find-renames --shortstat {old} {new}</pre></div></section>
<section id="matrix"><h2>03 · 按机制追踪 30 → 31 → 32</h2><p>30列描述本轮基线状态；31/32列只列该段审阅结论及对应详报入口。同一行可包含一串配套PR，避免将“框架接入、能力扩展、kernel微调”视作多套可叠加收益。</p><div class="controls"><input id="matrix-search" type="search" aria-label="搜索演进矩阵" placeholder="搜索模型 / 机制 / PR，例如 GLM、PCP、19550…"><button id="matrix-reset">清除搜索</button></div><p id="matrix-count" class="result-count" aria-live="polite"></p><div class="panel table-wrap"><table class="matrix"><thead><tr><th>机制 / 模块</th><th>30 基线</th><th>31 增量</th><th>32 增量</th></tr></thead><tbody>{matrix_html}</tbody></table></div></section>
<section id="markers"><h2>04 · 关键实现符号的三端定位</h2><p>这是固定源码上的文本匹配，不是完整支持矩阵；“未检出”只表示该字面符号没有出现。用途是追踪实现保留、迁移和新增，例如31的masked-runs cache在32移到layerwise模块，不能把目录删除误读成优化被撤销。</p><div class="panel table-wrap"><table><thead><tr><th>符号 / 机制</th><th>30</th><th>31</th><th>32</th></tr></thead><tbody>{marker_html}</tbody></table></div></section>
<section id="evidence"><h2>05 · 怎样比较性能证据</h2><div class="summary-grid"><article class="summary-box"><h3>继承与扩展</h3><p>31的#17263对V4 WKV堆叠，32的#19550扩展V4.1；31的#18235基础MiniMax融合已在30回移，31#18666提高token范围，32#20308再优化kernel。它们不是三次独立端到端提速。</p></article><article class="summary-box"><h3>局部改善不等于模型同幅改善</h3><p>32的GDN metadata阶段可下降约57%～58%，所列自然decode吞吐改善约0.19%～1.92%；V4.1 WKV/RoPE局部改善约41%/63%，模型场景吞吐仅+0.86%。具体条件见详报。</p></article><article class="summary-box"><h3>保留退化与未验证项</h3><p>32 A2/A3 QSFA的所测decode无收益；bincount短history有退化；MiniMax小batch会波动；HIXL探索结果未证实改善；Qwen默认CPU length优化的最终tree尚无独立速度验证。</p></article><article class="summary-box"><h3>严格区分对照基线</h3><p>RecoverSSM commit优化的局部base/head，与同source的ReplaySSM OFF/ON不是同一实验。后者还改变cache几何与命中量，TTFT不能宣称纯decode增益。回移前的测量也不能冒充release checkout复测。</p></article></div><div class="panel"><p><b>建议验证：</b>同模型/同依赖的单点PR A/B与各发布版本受支持栈的整体升级A/B分别执行；固定runner、并行、graph、量化、cache、采样和输入，交替多轮，记录TTFT/TPOT/SSE ITL、吞吐、输出长度、接受率、显存、精度及退化。Qwen新模型的可用性无法直接变成该模型在31上的同配置对照。</p><p>各阶段报告中都有全部提交、反向提交、文件净差异、人工PR分析、原始patch、作者来源与CSV。总览数据见<a href="overview_data.json">overview_data.json</a>，重建命令<code>python reports/build_overview.py</code>。三版本仅跟踪上述固定SHA，后续分支移动需重新采样。</p></div><p class="footnote">生成UTC：{e(out["generated_at"])}。本页复用首份报告的视觉样式；原30→31报告未改动。核心阅读、搜索与矩阵CSV导出离线可用；外部源码/PR需要网络。</p></section></main></div>
<script>'use strict';const rows=[...document.querySelectorAll('.matrix-row')];function filterMatrix(){{const q=document.getElementById('matrix-search').value.trim().toLowerCase();let n=0;rows.forEach(row=>{{const hit=!q||row.dataset.search.includes(q);row.hidden=!hit;if(hit)n++;}});document.getElementById('matrix-count').textContent=`显示 ${{n}} / ${{rows.length}} 条机制路径`;}}document.getElementById('matrix-search').addEventListener('input',filterMatrix);document.getElementById('matrix-reset').onclick=()=>{{document.getElementById('matrix-search').value='';filterMatrix();}};document.getElementById('download-matrix').onclick=()=>{{const text={csv_payload};const url=URL.createObjectURL(new Blob(['\ufeff',text],{{type:'text/csv;charset=utf-8'}}));const a=document.createElement('a');a.href=url;a.download='evolution_matrix.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1500);}};filterMatrix();</script></body></html>'''
    rendered = re.sub(r"[ \t]+$", "", rendered, flags=re.M)
    (HERE / "index.html").write_text(rendered)
    names = ["index.html", "overview_data.json", "evolution_matrix.csv", "build_overview.py"]
    manifest = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}
    (HERE / "overview_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(versions=versions, cumulative=cumulative, matrix_rows=len(matrix), marker_rows=len(markers)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
