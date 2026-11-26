#!/usr/bin/env python3
"""Render the execution boundary audit as a standalone, offline HTML report."""
# Narrative paragraphs and URLs remain intact.
# ruff: noqa: E501

import csv
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "evidence.json").read_text())
ASCEND = "https://github.com/gg820827047/vllm-ascend/blob/36b73a9c431d12e7b462f34c89de1da9b18b278d/"
CORE = "https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/"


def esc(value):
    return html.escape(str(value))


def link(path, line, label, core=False):
    return f'<a href="{CORE if core else ASCEND}{path}#L{line}" target="_blank" rel="noopener">{esc(label)}</a>'


def table(head, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{x}</th>" for x in head)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def phase(sample, name):
    return next(p for p in sample["phases"] if p["id"] == name)


def ms(value):
    return f"{value:.3f}"


def device_span(p):
    d = p["device"]
    return (
        f"+{ms(d['start_ms'])} → +{ms(d['end_ms'])}<br><small>包络 {ms(d['span_ms'])} ms；{d['count']} 个关联任务</small>"
        if d["count"]
        else "未关联到硬件任务"
    )


def task_detail(task):
    return f"{esc(task['type'])}<br>+{ms(task['start_ms'])} → +{ms(task['end_ms'])} ms<br><small>Stream {task['stream']} / Task {task['task']} / Connection {task['connection']}<br>kernel_details.csv 第 {task['csv_line']} 行</small>"


def phase_timeline(sample):
    colors = ["#64748b", "#8b5cf6", "#176db5", "#14a594", "#a85576", "#de9b28", "#b85b22", "#a1376c", "#5352a6"]
    left, scale = 190, 0.8
    out = ['<div class="plot"><svg viewBox="0 0 1010 365" role="img" aria-label="Host 与 Device 阶段范围，不能相加">']
    for t in range(0, 1001, 100):
        x = left + t * scale
        out.append(
            f'<path d="M{x} 22V345" stroke="#e1e8ef"/><text x="{x}" y="16" font-size="11" fill="#526477">{t}</text>'
        )
    for i, p in enumerate(sample["phases"]):
        y = 32 + i * 34
        out.append(f'<text x="4" y="{y + 14}" font-size="11">{esc(PHASES[p["id"]][0])}</text>')
        for row, start, end, opacity in [(0, p["host_start_ms"], p["host_end_ms"], 0.45)] + (
            [(1, p["device"]["start_ms"], p["device"]["end_ms"], 1)] if p["device"]["count"] else []
        ):
            out.append(
                f'<rect x="{left + start * scale}" y="{y + row * 12}" width="{max(1, (end - start) * scale)}" height="9" fill="{colors[i]}" opacity="{opacity}"><title>{esc(p["id"])} {"Host" if row == 0 else "Device envelope"}: {start:.6f}–{end:.6f} ms</title></rect>'
            )
    out.append(
        '<text x="190" y="360" font-size="11">上：Host wall；下：关联 Device 任务包络。长包络可以包含其他阶段的任务与等待。</text></svg></div>'
    )
    return "".join(out)


PHASES = {
    "worker_prepare": (
        "Worker 状态/输入/元数据",
        "execute_model → prepare_inputs / prepare_attn；含 DP 同步及可能的 KV connector 钩子",
        "不能全部解释为 CPU 计算；可能包含等待。",
    ),
    "target_input": (
        "模型入口至第 0 层",
        "ForCausalLM.forward → Model.forward；Engram/embedding/HC 初始状态等",
        "部分 embedding 已在 Worker 准备期执行，不能仅按这一段统计全部 embedding。",
    ),
    "target_layers": (
        "目标模型 40 层",
        "DeepseekV41DecoderLayer_0…39 → forward；每层两对 HC Pre/Post",
        "设备包络含 AICPU 通信辅助任务；需与首个 HC Pre 的计算边界分开。",
    ),
    "target_tail": (
        "目标层后 collapse / norm",
        "DeepseekV41Model.forward → hc_collapse → norm",
        "不属于第 39 层 HC Post；旧的 461 ms 未包含本段。",
    ),
    "runner_after_forward": (
        "forward 后至 sample_tokens",
        "Runner 收尾；A5 在本段 compute_logits；A3 对应工作移入 sample",
        "按语义把 LM head 合并比较，不能仅按不同 Runner 的函数名字比较。",
    ),
    "target_sampling": (
        "目标采样与状态更新",
        "sample_tokens → sample / _sample，直到 proposer 入口",
        "A5 的随机数任务提前在独立流运行；426 ms 包络不等于采样开销。",
    ),
    "draft_prepare": (
        "草稿准备与 context KV",
        "propose → metadata / precompute_and_store_context_kv → draft 输入",
        "含草稿第 0 层前的 embedding；Device 包络可与主模型执行重叠。",
    ),
    "draft_layers": (
        "草稿 3 层",
        "DeepseekV41DSparkModel.forward → DSparkDecoderLayer_0…2",
        "A3 的 DispatchFFNCombine 与 A5 的分立 MoE 路径不同。",
    ),
    "draft_head_and_output": (
        "草稿 head / 采样 / 收尾",
        "hc_collapse / compute_logits / _sample_sequential / 输出状态，直到 sample_tokens 返回",
        "混有输出状态及 post_forward 钩子；只审计可关联的本 Worker 硬件任务，不声称覆盖网络传输完成。",
    ),
}


def main():
    """Render audited boundaries, workload differences, and the A3 experiment plan."""
    a, b = D["samples"]["A3"], D["samples"]["A5"]
    intro = """
<p class="eyebrow">DSV4.1 · 执行边界审计 · 2026-11-26</p>
<h1>模型执行内的 A3 / A5 差距与优化路线</h1>
<p class="lead">目标：排除 tokenizer、网关及 Scheduler 初次排队，定位 Worker 内哪些工作耗时、为什么不同，以及 A3 该优先验证什么。所有数字关联到原始 task、采集调用栈和代码语义，避免把 Host 返回、设备执行与跨 rank 等待混为一项。</p>
<div class="notice"><b>先明确结论：</b>现有包足以确认实现路径与成本构成的差异，尚不能给出相同工作负载下的硬件加速比。两边的 token 行数、请求数、TP/EP、量化、Runner、CANN 和参与计算的 DP 状态均不同。不能通过除以 token 数或对齐 MegaMoE 输出 shape 消除这些差异。</div>
<p><a href="../../llm_execution_code_walkthrough/report.html">Scheduler → Ascend → DeepSeekV4.1 代码地图</a> · <a href="../report.html">A3/A5 原报告</a> · <a href="../host_device_timing/report.html">Host/Device 原专题</a> · <a href="../../llm_service_optimization/index.html">优化总览</a> · <a href="evidence.json">完整边界、调用栈 JSON</a> · <a href="a3_tasks.csv">A3 任务明细 CSV</a> · <a href="a5_tasks.csv">A5 任务明细 CSV</a></p>
"""
    scope = """
<h2 id="scope">1. 比较范围与样本身份</h2>
<p><b>T0 取本地 Model Runner.execute_model 入口。</b>因此不计 Scheduler 初次排队、tokenizer、API 和 Executor 向 Worker 传递调度结果的 IPC。Worker 内发生的 DP 同步、输入准备及模型所需等待仍保留；草稿、采样单列。最后一个关联任务的结束是设备证据，不用 Python 函数返回替代。</p>
<p>这是一次 rank-local chunk/step 的审计，不是整条 128k 请求的 Prefill 总耗时。若要研究请求从首次准入到最终 Prefill 完成，需要 request_id 与每个 chunk 的对应关系；批执行成本不能简单除以请求数。最近口述的“128k、64 并发”尚未有新归档路径，本页不把旧采集宣称为该受控实验。</p>
"""
    scope += table(
        ["条件", "A3", "A5"],
        [
            ["原始包", "P.rar", "vllm_profile_prefill-c192-bs12.zip"],
            ["样本", "DP0/TP0；本地 step index=1（第 2 个窗口）", "DP0/TP0；本地 step index=2（第 3 个窗口）"],
            ["拓扑", "P DP4×TP4，EP16；16 rank 均有数据", "P DP8×TP1，EP8；8 rank 均有数据"],
            ["实际 Runner", "worker/v2/model_runner.py；MRV2", "worker/model_runner_v1.py；MRV1（dsnext）"],
            [
                "采集可见输入维度",
                "Attention/HC：1848→1024；MegaMoE：462→256，local experts=24",
                "Attention/HC 与 MegaMoE：11791→768，local experts=48",
            ],
            [
                "可见 batch 信息",
                "metadata 的请求数组为 8；草稿 query 行数 40",
                "metadata 的请求数组为 6；草稿 query 行数 30",
            ],
            [
                "MoE 量化与共享专家",
                "w8a8_dynamic；INT8 routed weights；选中 MegaMoE 无 shared 权重输入",
                "w4a8_mxfp4；FLOAT4_E2M1 routed weights；MegaMoE 含 FP8 shared 权重输入",
            ],
            ["CANN", "9.2.1", "9.3.0"],
            [
                "时间原点",
                f"SESSION 起点后 {a['worker_entry_session_ms']:.6f} ms",
                f"SESSION 起点后 {b['worker_entry_session_ms']:.6f} ms",
            ],
            [
                "仍缺的锁定信息",
                "镜像 digest、采集时 vLLM/Ascend SHA、完整启动参数、该 step 的 SchedulerOutput 与上下文长度分布",
                "同左；A5 无法重跑时保留为观察基线，不杜撰受控硬件结论",
            ],
        ],
    )
    scope += '<p class="muted">1848/11791 等为实际张量中的行数，不能仅凭 HC/Mega shape 推断整个请求长度。MoE SP 还会改变局部行数；HcPre 与 MegaMoE 不总在同一切分位置。</p>'

    boundaries = '<h2 id="boundaries">2. 先把开始与结束找准确</h2>'
    boundaries += table(
        ["锚点（相对本样本 T0，ms）", "A3", "A5", "取值依据"],
        [
            ["Model Runner 入口", "0", "0", "采集 Python execute_model scope；两个嵌套 execute_model 不能相加"],
            [
                "前一轮关联任务最晚结束",
                ms(a["previous_work_before_first_target"]["end_ms"]),
                ms(b["previous_work_before_first_target"]["end_ms"]),
                "Host 归属早于 T0 的任务；含 AICPU 辅助任务，不当作当前 Host 开销",
            ],
            [
                "首个 HC Pre 的 Host wrapper 入口",
                "223.594840",
                "44.683850",
                "PyTorch HC wrapper；与内部 aclnn launch 有嵌套关系，不是同一事件",
            ],
            ["首个 HC Pre 的 Device 开始", "235.027351", "309.687555", "关联到 Stream/Task 的实际计算任务"],
            [
                "第 39 层最后 HC Post 的 Device 结束",
                "690.784237",
                "888.060299",
                "40 层 decoder block 结束；不含后续 collapse/norm/head",
            ],
            [
                "forward tail 的最后任务结束",
                ms(phase(a, "target_tail")["device"]["end_ms"]),
                ms(phase(b, "target_tail")["device"]["end_ms"]),
                "关联 norm / collapse / 状态写回，不使用 Host module 返回时间",
            ],
            [
                "草稿第 2 层最后任务结束",
                ms(phase(a, "draft_layers")["device"]["end_ms"]),
                ms(phase(b, "draft_layers")["device"]["end_ms"]),
                "不含后续草稿 logits/采样",
            ],
            [
                "已关联 Worker 路径的最后任务结束",
                ms(a["model_path_device"]["end_ms"]),
                ms(b["model_path_device"]["end_ms"]),
                "含采样/输出状态的可见任务；不声称网络或客户端响应已完成",
            ],
            [
                "execute_model 的 Host 返回",
                ms(a["runner_return_ms"]),
                ms(b["runner_return_ms"]),
                "只是下发/Host 控制结束；草稿位于后续 sample_tokens",
            ],
            [
                "sample_tokens 的 Host 返回",
                ms(a["sample_return_ms"]),
                ms(b["sample_return_ms"]),
                "依然不能替代 Device 完成",
            ],
        ],
    )
    boundaries += '<div class="notice"><b>此前口径修正：</b>461.233 ms 是 A3 40 层的关联任务包络（从 +229.551 ms 的 allreduce AICPU 辅助任务算起），不是完整主模型执行。若统一使用首个 HC Pre → 第 39 层最后 HC Post，A3 为 <b>455.757 ms</b>，A5 为 <b>578.373 ms</b>。同样，旧的 +733.695 / +890.322 ms 只到草稿层末尾；本次已追到后续可见采样/状态任务 +734.612 / +891.443 ms。这些仍不能直接算硬件快慢。</div>'
    boundaries += "<p>下表为可在 Insight 中核对的设备锚点。软件显示零点可能采用 cluster/SESSION 的不同基准；若时间偏移，优先按 <b>rank 文件名 + Stream ID + Task ID</b> 找同一任务，再以 execute_model 入口归零。Connection ID 是数据库 raw runtime connection，可能与某个 PyTorch wrapper 的 group ID 不同。</p>"
    anchor_rows = []
    for label, s in D["samples"].items():
        first = next(
            x["task"] for x in s["operator_anchors"] if x["phase"] == "target_layers" and x["task"]["type"] == "HcPre"
        )
        for name, t in [
            ("目标第 0 层 HC Pre", first),
            ("目标第 39 层 HC Post", phase(s, "target_layers")["device"]["last"]),
            ("目标 tail 最后任务", phase(s, "target_tail")["device"]["last"]),
            ("草稿第 2 层最后任务", phase(s, "draft_layers")["device"]["last"]),
            ("本路径最后可见任务", s["model_path_device"]["last"]),
        ]:
            anchor_rows.append(
                [
                    label,
                    name,
                    task_detail(t),
                    f"SESSION +{s['worker_entry_session_ms'] + t['start_ms']:.6f} → +{s['worker_entry_session_ms'] + t['end_ms']:.6f} ms",
                ]
            )
    boundaries += table(["设备", "位置", "任务身份 / 相对 T0", "SESSION 坐标（仅本 rank）"], anchor_rows)

    phases = '<h2 id="phases">3. Worker 内细分：Host 和 Device 分开看</h2><p>上层 Python scope 含子调用；下面使用互斥的 Host 归属区间。Device 包络由同一阶段关联任务的最早开始/最晚结束确定，跨阶段可重叠；因此 <b>不同阶段 Device 包络不能相加</b>。没有记录到 kernel 也不等于硬件完全空闲。</p>'
    phases += table(
        ["阶段", "A3 Host wall / ms", "A5 Host wall / ms", "A3 Device", "A5 Device", "如何解释"],
        [
            [
                PHASES[p["id"]][0],
                f"+{ms(p['host_start_ms'])} → +{ms(p['host_end_ms'])}<br><b>{ms(p['host_wall_ms'])} ms</b>",
                f"+{ms(q['host_start_ms'])} → +{ms(q['host_end_ms'])}<br><b>{ms(q['host_wall_ms'])} ms</b>",
                device_span(p),
                device_span(q),
                PHASES[p["id"]][2],
            ]
            for p, q in zip(a["phases"], b["phases"])
        ],
    )
    phases += "<h3>A3 时间线</h3>" + phase_timeline(a) + "<h3>A5 时间线</h3>" + phase_timeline(b)
    phases += "<p>A5 的目标采样包络是 +462.118→888.656 ms，但其 kernel 累计只有约 0.409 ms：独立流先准备随机数，主流待 logits 就绪后采样。将 426.538 ms 记为“采样太慢”会得出错误结论。A3 草稿准备/收尾的长包络同样含多流跨度，必须按任务与依赖核对。</p>"

    evidence = '<h2 id="code">4. 拆解依据：采集调用栈与代码语义</h2><p><b>采集栈是执行事实，源码链接是固定版本的语义对照。</b>采集包没有锁定两份镜像的源码 SHA；尤其 A5 是 dsnext 私有路径，不能假装下列 CANN 源码逐行等同于采集实现。相同函数名、行号也不能证明版本完全相同。原始 Python 嵌套区间与同线程包含关系用于恢复调用层级，不是 with_stack 的原生 callchain。</p>'
    evidence += table(
        ["边界或阶段", "采集事实", "代码对应与含义"],
        [
            [
                "Worker 主入口",
                "A3 model_runner.py(704): execute_model；core gpu/model_runner.py(1718)；A5 model_runner_v1.py(3293)",
                link("vllm_ascend/worker/v2/model_runner.py", 706, "Ascend V2 execute_model")
                + " → "
                + link("vllm/v1/worker/gpu/model_runner.py", 1719, "core execute_model", True)
                + "；core vllm/v1 目录表示 Engine V1，不能据此当成 MRV1",
            ],
            [
                "元数据",
                "A3 prepare_attn +100.702→160.589；replay _build_metadata +162.500→216.777 ms",
                link("vllm_ascend/worker/v2/model_states/default.py", 416, "prepare_attn")
                + " / "
                + link("vllm_ascend/models/deepseek_v41/decoder_replay_state.py", 346, "replay _build_metadata")
                + " / "
                + link("vllm_ascend/attention/dsa_v41.py", 95, "build_replay_swa_visible_lens"),
            ],
            [
                "目标 40 层",
                "nn.Module: DeepseekV41DecoderLayer_0…39，独立于 3 个 DSparkDecoderLayer；各有 2 对 HC Pre/Post",
                link("vllm_ascend/models/deepseek_v41/model.py", 1560, "Model.forward 循环")
                + " → "
                + link("vllm_ascend/models/deepseek_v41/model.py", 922, "DecoderLayer.forward：Attention + FFN"),
            ],
            [
                "主干收尾",
                "第 39 层 HC Post 之后仍有 collapse / norm / ScatterUpdate 等",
                link("vllm_ascend/models/deepseek_v41/model.py", 1664, "hc_collapse → norm")
                + "；LM head 的 compute_logits 是另一层语义边界",
            ],
            [
                "Q/KV 与 Attention",
                "vllm::dsa_v41_forward → _prepare_inputs_and_caches / multistream_preprocess / _forward_attention",
                link("vllm_ascend/attention/dsa_v41.py", 755, "Attention forward")
                + " / "
                + link("vllm_ascend/attention/dsa_v41.py", 491, "多流 Q/KV preprocess")
                + "；两条分支仍在 Attention 前 join，Cube 资源争用不能靠多流消除",
            ],
            [
                "MoE",
                "A3 W8A8 _apply_cann_mega_moe；草稿 DispatchFFNCombine；A5 W4A8_MXFP4 fused_shared / GMM",
                link("vllm_ascend/models/deepseek_v41/model.py", 277, "DeepseekV41MoE.forward")
                + " / "
                + link("vllm_ascend/ops/fused_moe/moe_comm_method.py", 593, "fused_experts 分支")
                + "；A5 栈中的 shared fusion 不应算成 A3 相同算子的硬件提速",
            ],
            [
                "采样与草稿",
                "execute_model 返回后还有 sample_tokens；草稿的 Model/DecoderLayer scope 单独可见",
                link("vllm/v1/worker/gpu/model_runner.py", 2088, "sample_tokens", True)
                + " → "
                + link("vllm/v1/worker/gpu/model_runner.py", 2239, "speculator.propose", True)
                + " → "
                + link("vllm_ascend/models/deepseek_v41/dspark.py", 279, "DSparkModel.forward 三层"),
            ],
        ],
    )
    evidence += "<p>关联方法：<code>PYTORCH_API.connectionId（group） → CONNECTION_IDS.connectionId（raw） → TASK.connectionId → kernel_details.csv（精确 startNs 匹配）</code>。A5 少量自定义 kernel 走同一主线程的 CANN launch 回退关联。一个 Enqueue/Dequeue/Runtime launch/Device kernel 是不同阶段，不能当四次执行。</p>"
    evidence += "<p>下方可展开实际记录的调用栈，包含量化分支、HC 边界和任务 ID；完整边界及所有父级见 evidence.json。不是凭算子名称或 PR 描述归因。</p>"
    for label, s in D["samples"].items():
        for an in s["operator_anchors"]:
            t = an["task"]
            kept = [p for p in an["python_parents"] if "vllm_ascend/" in p or "nn.Module: Deepseek" in p]
            evidence += f"<details><summary>{label} / {esc(an['phase'])} / {esc(t['type'])} / +{ms(t['start_ms'])} ms</summary><p>{task_detail(t)}</p><pre>{esc(chr(10).join(kept))}</pre><p>输出 shape：<code>{esc(t['output'])}</code></p><p>输入 dtype：<code>{esc(t['dtype'])}</code></p></details>"

    findings = """<h2 id="differences">5. 差距项：已确认什么，不能推出什么</h2>"""
    findings += "<h3>目标 40 层的算子成本账本</h3><p>累计值来自该层范围内实际任务驻留时间；一个融合 kernel 的驻留也可能包含 EP 等待。下表各类可能在不同 stream 重叠，因此不能相加为模型时延。算法、dtype 和 shape 不同，同行用于对照成本位置，不表示同一 kernel 的硬件速度差。</p>"
    kernel_groups = [
        ("MegaMoE（含其内部通信/等待）", lambda t: t == "MegaMoe"),
        ("Attention core：SparseFlashMla / MQ-SMLA", lambda t: t == "SparseFlashMla" or "MqsmlaKernel" in t),
        ("Indexer 相关（不含投影 MatMul）", lambda t: "indexer" in t.lower() or "Indexer" in t),
        ("HC Pre", lambda t: t == "HcPre"),
        ("HC Post", lambda t: t in ["HcPost", "MhcPost"]),
        ("QuantBatchMatmulV3", lambda t: t == "QuantBatchMatmulV3"),
        ("普通 MatMulV2 / V3", lambda t: t in ["MatMulV2", "MatMulV3"]),
        (
            "TransposeBatchMatMul / TransposeQuantBatchMatMul",
            lambda t: t in ["TransposeBatchMatMul", "TransposeQuantBatchMatMul"],
        ),
        ("RmsNorm / RmsNormCast", lambda t: t in ["RmsNorm", "RmsNormCast"]),
        ("RmsNormDynamicMxQuant 融合", lambda t: t == "RmsNormDynamicMxQuant"),
        ("DynamicQuant / DynamicMxQuant", lambda t: t in ["DynamicQuant", "DynamicMxQuant"]),
        ("InplacePartialRotaryMul", lambda t: t == "InplacePartialRotaryMul"),
        (
            "KV 写回/压缩融合（部分）",
            lambda t: t in ["ScatterNdUpdateSk", "KvCompressEpilogV2"] or "FusedPostScatter" in t,
        ),
        ("显式 TP hcom AllGather / AllReduce", lambda t: t in ["hcom_allGather_", "hcom_allReduce_"]),
        ("AICPU collective 辅助任务（单独列）", lambda t: t in ["allgatherAicpuKernel", "allreduceAicpuKernel"]),
        ("TensorMove", lambda t: t == "TensorMove"),
    ]
    costs = {
        label: [
            r for r in csv.DictReader((HERE / f"{label.lower()}_tasks.csv").open()) if r["phase"] == "target_layers"
        ]
        for label in ["A3", "A5"]
    }
    cost_rows = []
    for title, predicate in kernel_groups:
        cells = []
        for label in ["A3", "A5"]:
            rows = [r for r in costs[label] if predicate(r["type"])]
            cells.append(f"{len(rows)} 次 / {sum(float(r['end_ms']) - float(r['start_ms']) for r in rows):.3f} ms")
        cost_rows.append([title] + cells)
    findings += table(["成本类别", "A3 次数 / 驻留累计", "A5 次数 / 驻留累计"], cost_rows)
    findings += "<p>全量类型及每条任务的 input/output shape、dtype、Host 归属和任务身份见 CSV。部分功能融合前后边界不同，例如 Norm+Quant 不能只用独立 RmsNorm 的调用次数判定是否省了全部归一化。</p>"
    findings += table(
        ["差距项", "本次证据", "对 A3 的实际启发", "不能据此断言"],
        [
            [
                "元数据与重复小算子",
                "旧的 Worker→首 HC wrapper 口径：223.595 vs 44.684 ms。A3 的 prepare_attn 59.887 ms、replay metadata 54.277 ms，二者不重叠；前置 RepeatInterleave 42 次、Index/IndexCheck 各 176 次；A5 前置 metadata fused kernel 6 次。",
                "优先去掉同一步中相同请求/同缓存组的重复展开；复用 visible lens、positions、RoPE 与组共享计划，再评估可用 metadata fusion。",
                "不能把 178.911 ms 差额或 114.164 ms 两段 wall 全当成可兑现收益：含 DP 同步/等待，且两份软件不同。",
            ],
            [
                "Host/Device 衔接与前轮积压",
                "A5 前轮关联任务直到 +308.726 ms；当前主模型首 HC 在 +309.688 ms。A3 的 metadata wrapper +99.861→216.806 ms 内，原审计仅有 19.392 ms 可见任务覆盖，约 97.553 ms 未记录到 kernel。",
                "A3 先证明是否存在等待数据、重复构造、CPU 调度脱核运行或下发受阻，再优化暴露空泡。用 CPU 调度采样与 Runtime/SQ/CQ 时序区分原因。",
                "Host 很短而 Device 晚启动，不一定是 Host bound；开 async_scheduling 也不保证重复工作或依赖自动消失。",
            ],
            [
                "MegaMoE 大项",
                "目标 40 次：A3 累计 299.701 ms；A5 305.172 ms。第一层各 rank 的到达时间相差 35.763 / 102.178 ms，而结束分散仅 0.031 / 0.692 ms。",
                "先看 EP 到齐与路由负载，减少首层及每层等待，再判断真正 FFN 算力；框架可做 DP 负载匹配、批次组织、路由/布局/路径选择。",
                "MegaMoE 时间不是纯 GMM 时间；A5 首次 100.456 ms 不能当硬件算力慢，更不能当作 A3 的 100 ms 可优化空间。",
            ],
            [
                "草稿 MoE 路径",
                "A3 3 层设备包络 36.818 ms，其中 DispatchFFNCombine 3 次累计 34.832 ms；A5 3 层包络 1.002 ms，走 MoeDistribute / GMM 等路径；query 行数 40 vs 30。",
                "单独追踪 draft EP 到达、实际专家 token 数和分支选择；在 A3 上受控比较合法的 MoE 路径与 DSpark 块大小。这是本次重新拆出的明确候选。",
                "不能把约 35.8 ms 差值全说成算子性能差或保证提速；量化、EP、路由和通信实现均不同。",
            ],
            [
                "TP/SP 通信与工作切分",
                "A3 目标层有 80 次 AllGather、40 次 AllReduce，hcom 包络累计 18.607 ms；A5 TP1 无这类显式 TP collective。",
                "比较 Attention 输出投影到 MoE 的布局、收集/切片顺序和是否可消减重复通信；拓扑试验独立于算子实现试验。",
                "不是 A5 把所有通信融合掉；MegaMoE 内 EP 通信仍存在。不能把所有 hcom/AICPU 时间重复加到总时延。",
            ],
            [
                "Attention / 量化 / 融合",
                "A3 SparseFlashMla 累计 40.173 ms；A5 MQ-SMLA、DSL Indexer、MXFP 量化路径不同，且头数/上下文/行数不同。A5 可见 Norm+MXQuant、KvCompressEpilog、shared expert fusion。",
                "按 Q_a、Q_b、KV、压缩、Indexer、SWA/global attention、o_proj 逐段映射；只替换 A3 硬件和 dtype 真正支持的融合实现。",
                "不能只拿 SparseFlashMla 与 MQ-SMLA 时长相减；也不能直接搬 A5 FP4/FP8 kernel 期待 A3 生效。",
            ],
            [
                "HC 与尾部写回",
                "HC Pre+Post：A3 累计 37.149 ms；A5 41.414 ms。目标 tail：A3 3.369 ms vs A5 0.121 ms；A3 tail 的 5 次 ScatterUpdate 累计 2.608 ms。",
                "检查 aux hidden / MTP buffer 的有效行复制、布局转换与写回次数；HC 融合及少分配/少拷贝做第二梯队。",
                "尾部差距也受请求数、布局和 buffer 策略影响；不能直接把 3.25 ms 都认成无用操作。",
            ],
            [
                "多流与快慢卡",
                "A3 Q/KV 仍有局部重叠，现有审计该段累计约 6.538 ms；TP2 偶发晚到与 EP 等待关联。首个 HC 提交→执行 11.267 ms 中约 11.258 ms 被其他可见任务覆盖。",
                "先检查关键路径和依赖，再判断多流能否掩盖；持续相同 kernel/相同 shape 的跨卡退化，结合频率/温度/交换 rank 才能定硬件慢卡。",
                "少重叠不自动等于坏卡/CPU 抖动；一个 EventWait 或 0.1 ms 重叠不是最终优化收益。",
            ],
        ],
    )
    findings += "<h3>EP 第一层 MegaMoE 到达图</h3><p>每组以该组最早 MegaMoE 开始为 0，使用 SESSION epoch 纳秒对齐所有 rank。这里比较“同组是否到齐”，不是跨机器比较绝对时钟。</p>"
    for dataset, v in D["ep_first_mega_alignment"].items():
        findings += f"<details><summary>{dataset}：到达跨度 {v['arrival_spread_ms']:.3f} ms / 结束跨度 {v['completion_spread_ms']:.3f} ms</summary>"
        findings += (
            table(
                ["rank", "模式", "开始 / ms", "结束 / ms", "驻留 / ms"],
                [
                    [esc(r["rank"]), r["kind"], ms(r["start_ms"]), ms(r["end_ms"]), ms(r["end_ms"] - r["start_ms"])]
                    for r in v["ranks"]
                ],
            )
            + "</details>"
        )

    findings += "<h3>如何判 bound：从现象走到可行动结论</h3>"
    findings += table(
        ["假设", "必须补齐的证据", "对应动作"],
        [
            [
                "Host 准备/下发受限",
                "后继任务的数据依赖已满足、设备有可执行空隙，但 Host 尚未提交；结合 Python/CANN 与 CPU on-core/off-CPU、runqueue、线程亲和性证明原因。",
                "减少重复元数据/分配/Python 调用，独立准备提前或异步；CPU 抢占与 NUMA 问题分别处理。",
            ],
            [
                "设备排队或流依赖",
                "Host 已提交，Device 较晚执行；其间有前一步任务、同流排队或 EventWait。A5 当前首 HC 的延后有前轮 Device 积压这一证据。",
                "找真正占队列或生产数据的任务；优化错误依赖、生产者时延或积压，不能把 launch→start 都记为 Host 工作。",
            ],
            [
                "EP/TP 通信或到齐等待",
                "同层各 rank 的到达/结束、消息量、链路带宽、专家 token 分布与 kernel 内 PMU。当前 MegaMoE 的结束集中、到达分散符合等待参与，不能量化纯 GMM 占比。",
                "先减少 rank skew/负载差，再评估通信算法、拓扑、量化传输及分发/计算重叠。",
            ],
            [
                "Cube/Vector 计算受限",
                "相同 shape/dtype/layout 的算量、相应引擎实际利用率及 PMU；不同架构的百分比定义必须先对齐。",
                "框架侧选更合适融合/布局与合法路径，尝试能互补资源的并行；真正 kernel 算力优化再交算子团队。",
            ],
            [
                "HBM/访存受限",
                "实际字节流量、DRAM 带宽、缓存/搬运等待与算术强度；不能仅凭某个 MTE ratio 下结论。",
                "消减冗余读写、临时 buffer 和布局转换，融合访问；检查 offload/Engram 数据本地性。",
            ],
            [
                "小 kernel / launch 密度受限",
                "大量短任务的 Host 下发时间与依赖间隙；证明融合减少实际关键路径，而不只降低调用次数。",
                "先复用、批量处理与可用融合，再考虑新增算子；核对动态 shape、图模式及编译成本。",
            ],
        ],
    )

    plan = """<h2 id="plan">6. A3 发力顺序与验收计划</h2><p>责任人以下按职责建议，尚未指定具体姓名。预期收益采用可验收的成本项，不把非受控差额当收益承诺。P0 为证据补齐，P1 为优先闭环，P2 为后续候选。</p>"""
    plan += table(
        ["优先级 / 工作包", "具体事项与交付", "建议牵头", "收益判断 / 验收"],
        [
            [
                "P0 · 冻结比较合同",
                "锁定已有 A5 的语义工作包：每请求 query/context/computed length、KV 命中、SWA 截断点、draft 数、dtype、布局、TP/EP、batch。A3 重采在可行范围复现同逻辑工作；导出 SchedulerOutput 和版本指纹。",
                "MDE + 测试",
                "交付每一步指纹与排除 dummy 的清单。只对匹配项作结论；A5 不可复跑时保留观测基线。",
            ],
            [
                "P0 · 完整执行边界",
                "目标 forward、target head/sample、draft prepare/forward/head、输出状态分别关联到 Device 完成；单独标记 Worker 内 connector 等待。核对本报告最后任务之后有无跨线程未关联任务。",
                "框架 + profiling",
                "不能用 host 返回替代完成；建立每批 Worker completion P50/P90、core busy、exposed gap 与跨 rank skew。",
            ],
            [
                "P1 · 元数据去重/复用",
                "优先分析 build_replay_swa_visible_lens、SWA/global/cache-group 共享、RoPE/positions；统计同输入重复构造次数。先 step 内复用，再考虑融合 kernel，避免跨 step 状态失效。进一步评估与前一步 Device 重叠准备独立的 batch plan；依赖 DP 全局 shape 的字段仍在同步后确定。",
                "vLLM-Ascend 框架",
                "目标是减少 42 次 RepeatInterleave 与重复索引，以及 metadata 的暴露 Host wall。验收必须同时减少 kernel 数/Host 工作和 Worker completion，验证缓存隔离与精度；不盲目移除 DP 同步。",
            ],
            [
                "P1 · EP 到达与 MegaMoE 等待",
                "对 40 层逐层采集每 rank arrival/end、expert_tokens；将 DP token/上下文差异与 Host 下发晚到分开；再选择负载均衡、batch 组织或布局方案。",
                "分布式 / MoE",
                "先降低 arrival spread 和最慢 rank completion。同步等待不是固定 35.8 ms 可收益；不能累加每卡等待作为集群收益。",
            ],
            [
                "P1 · 草稿路径专项",
                "定位 A3 3 次 DispatchFFNCombine 中的分发、计算、回收与到齐。检查草稿 EP 拓扑和合法路由分支，比较块大小/算法时固定上下文及采样设置。",
                "投机解码 + MoE",
                "聚焦当前 34.832 ms 成本；收益待 A3 A/B，兼顾接受率、正确性和总吞吐。仅 P 侧变快不代表 D 侧 TPOT 一定改善。",
            ],
            [
                "P2 · Attention/模型融合",
                "逐项对齐 Q/KV/Indexer/压缩/Attention/o_proj 的 shape、dtype、head、窗口与历史长度；按硬件支持评估 norm+quant、RoPE/量化、输出投影与写回融合。",
                "模型适配 + 算子",
                "一次只验证一个路径；区分消减 Host 下发与缩短 Device 执行。A5 专用格式不作为 A3 可用性证据。",
            ],
            [
                "P2 · TP/SP 与拓扑",
                "保持软件及量化一致，比较 TP 输出通信、MoE SP 的局部行数、重复 gather/scatter；若内存允许才尝试与 A5 的 DP8×TP1/EP8 对齐。",
                "分布式 + 部署",
                "两种目标分别报告：相同算子/工作包延迟；相同卡数的 P 服务效率。改卡数、TP/EP 或 PCP 后不能称为单算子硬件对照。",
            ],
            [
                "P2 · Host/多流/卡一致性",
                "对晚到点补 perf sched / CPU on-core、off-CPU、线程 affinity、NUMA、Runtime 下发；与 SQ/CQ/EventWait 对齐。固定 case 多轮，跨卡交换角色，结合频率/温度。",
                "Host 性能 + 运维",
                "只优化暴露在关键路径上的准备/等待。绑核以无 profiler 的服务指标和多轮 Worker completion 验收，不能只看重叠百分比。",
            ],
            [
                "P2 · HC/head/状态搬运",
                "审查 HC pre/post、aux hidden、MTP buffer、lm_head 和 sampling 的复制/布局/分配；单独分析 A3 tail 5 次 ScatterUpdate。Engram lookup/传输看与主干的依赖与重叠。",
                "模型 + 框架",
                "目标减少无效写回与临时 tensor；保留草稿/缓存一致性。累计 kernel 时间与最终收益分别报告。",
            ],
        ],
    )
    plan += """<h3>最小实验闭环</h3><ol><li><b>固定 A3 基线：</b>代码/镜像、量化、TP/EP、batch、每请求上下文、SWA/缓存状态、DSpark、采样、CPU 绑定不变。先在无 profiler 条件记录服务指标，再用短窗口 profiling 定位。</li><li><b>一次只改一个因素：</b>先元数据复用或草稿路径其一。固定 warmup、同 seed/相同请求，不用 prefix/Store 命中变化充当收益。</li><li><b>至少多次交错 A/B：</b>报告每步分布、跨 rank 最大完成时间、长尾和精度；预先固定统计区间，避免只选最漂亮的一步。高开销 Python/module tracing 可能放大 Host 成本，前后采集配置必须一致。</li><li><b>以完成时间验收：</b>kernel 数减少、重叠增多只是中间证据；最终要缩短同工作包 Worker 完成时间，并在真实 128k/64 并发验证 TTFT/吞吐和稳定性。若瓶颈转移，把新关键路径作为下一轮输入。</li></ol>
<div class="notice"><b>当前优先级建议：</b>先闭环“元数据重复工作”和“MegaMoE / 草稿 MoE 的到齐与分支成本”；随后再做局部多流、融合和拓扑。现有证据不支持先把主要精力押在固定坏卡或盲目搬运 A5 算子上。</div>"""

    layers = '<h2 id="layers">7. 每层设备边界（便于 Insight 逐层复核）</h2><p>这些是按采集 module 区间归属的硬件包络；早启动的 AICPU 辅助任务会导致相邻层包络重叠。不要把 43 行的包络相加。</p>'
    layers += table(
        ["层 / 角色", "A3 Host / ms", "A3 Device / ms", "A5 Host / ms", "A5 Device / ms"],
        [
            [
                f"{x['layer']} / {x['role']}",
                f"+{ms(x['host_start_ms'])}→{ms(x['host_end_ms'])}",
                device_span(x),
                f"+{ms(y['host_start_ms'])}→{ms(y['host_end_ms'])}",
                device_span(y),
            ]
            for x, y in zip(a["layers"], b["layers"])
        ],
    )
    footer = '<h2 id="reproduce">8. 复现与证据边界</h2><p>原始包不变。audit_execution.py 从选择性解压的两个 SQLite 数据库与 CSV 重建边界，build_report.py 只渲染已有 evidence.json。主报告仍保留完整 24 rank 加历史数据共 28 rank 的结果。本页所有设备关联任务都能在 CSV 下载中按行号/Stream/Task 查验。</p>'
    footer += table(
        ["样本", "数据库及硬件任务覆盖"],
        [
            [
                label,
                f"<code>{esc(s['database_archive_member'])}</code><br>整个选中 rank：{s['correlation']['matched_rows']}/{s['correlation']['total_rows']} 行可关联；本 Worker 窗口分配 {s['correlation']['selected_rows']} 行。该设备时间窗未关联 {s['correlation']['unmapped_in_device_window']} 行，类型 {esc(s['correlation']['unmapped_types'])}。A3 这 1 行是 +176.700 ms 的 Maximum，约 3.12 μs，尚无充分 Host 关联，不强行补归属。外部 HIXL/网络完成不由本报告承诺。",
            ]
            for label, s in D["samples"].items()
        ],
    )
    footer += "<p>kernel_union / aicore_union 是区间并集；kernel_sum 是驻留时间累计。AICPU、hcom 与 AI Core 可重叠，不能称全部为数学计算时间。库内多核驻留可含等待，判断 compute/memory/communication/Host bound 还需 PMU、字节量/算量、依赖和 CPU 调度证据。</p><p>CSV 行号从表头为第 1 行计算；相对毫秒由 epoch 整数纳秒相减后换算，避免绝对浮点时间丢精度。资料截至本次归档，没有新增 NPU 实测或收益承诺。</p>"

    body = intro + scope + boundaries + phases + evidence + findings + plan + layers + footer
    nav = (
        "<nav>"
        + "".join(
            f'<a href="#{x}">{t}</a>'
            for x, t in [
                ("scope", "范围"),
                ("boundaries", "起止证据"),
                ("phases", "阶段拆分"),
                ("code", "调用栈/代码"),
                ("differences", "差距项"),
                ("plan", "优化计划"),
                ("layers", "逐层边界"),
                ("reproduce", "复现"),
            ]
        )
        + "</nav>"
    )
    style = """*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#173047;font:16px/1.8 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1440px;margin:auto;background:white;padding:35px 45px 70px}h1{font-size:34px;line-height:1.35}h2{font-size:25px;margin-top:50px;border-top:2px solid #dde6ef;padding-top:18px;scroll-margin-top:70px}h3{font-size:19px}a{color:#1462a5}p{max-width:1160px}.lead{font-size:18px;color:#456079}.eyebrow{color:#56718c;letter-spacing:1px;font-size:12px}.notice{background:#fff5df;border-left:4px solid #cc992e;padding:16px 22px;margin:22px 0}.scroll,.plot{overflow:auto;margin:18px 0}table{width:100%;min-width:780px;border-collapse:collapse;font-size:13px;line-height:1.7}th,td{padding:12px;border-bottom:1px solid #dce5ed;text-align:left;vertical-align:top}th{background:#edf3f9}small,.muted{color:#5e7286;font-size:12px}code,pre{font-size:12px;overflow-wrap:anywhere;white-space:pre-wrap;background:#f0f4f8}pre{padding:14px;border-radius:6px;max-height:500px;overflow:auto}details{border:1px solid #dbe5ee;padding:12px 16px;margin:10px 0;border-radius:6px}summary{cursor:pointer;font-size:14px;color:#155e9a}nav{position:sticky;top:0;background:#142f49;z-index:1;display:flex;justify-content:center;flex-wrap:wrap;padding:8px;gap:18px}nav a{color:white;text-decoration:none;font-size:13px}svg{min-width:800px;width:100%;display:block}li{margin:12px 0}@media(max-width:700px){main{padding:20px 16px}h1{font-size:27px}h2{font-size:22px}nav{gap:10px}}@media print{nav{display:none}main{padding:0}table{min-width:0;font-size:10px}.scroll{overflow:visible}details{break-inside:avoid}}"""
    output = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>模型执行范围 A3/A5：边界、差距与 A3 优化路线</title><style>{style}</style></head><body>{nav}<main>{body}</main></body></html>'
    (HERE / "report.html").write_text(output)
    print("Built", HERE / "report.html")


if __name__ == "__main__":
    main()
