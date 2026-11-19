#!/usr/bin/env python3
"""Build a source-timestamp audit of a Host-limited multistream example."""
# Preserve the HTML narrative and embedded SVG/JavaScript as literal markup.
# ruff: noqa: E501

import csv
import gzip
import html
import json
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RIDS = ["a3_recent-dp2-tp0", "a3_recent-dp2-tp2"]
M = {rid: json.loads((ROOT / (rid + ".json")).read_text()) for rid in RIDS}
C = {rid: json.loads(gzip.decompress((ROOT / (rid + "-records.json.gz")).read_bytes())) for rid in RIDS}
K = json.loads((ROOT / "kernel_connections.json").read_text())
BAD = RIDS[1]
GOOD = RIDS[0]
ORIGIN = M[BAD]["origin_ns"]
ANCHOR = next(k for k in K if k["rank"] == BAD and k["layer"] == 0 and k["kernel"] == "MatMulV2")["kernel_start_ns"]
COMMON_SHIFT = (ORIGIN - M[BAD]["common_origin_ns"]) / 1e6
CLOCK_AUDIT = json.loads((ROOT / "clock_audit.json").read_text())


def ms(ns, origin=ORIGIN):
    return (int(ns) - int(origin)) / 1e6


def absolute_us(ns):
    return format(Decimal(int(ns)) / 1000, ".3f")


def union(intervals):
    out = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(e, out[-1][1])
        else:
            out.append([s, e])
    return sum(e - s for s, e in out)


def classify(name):
    if name.endswith(": current_stream"):
        return "current_stream"
    if name.endswith(": record_event"):
        return "Stream.record_event"
    if "profiler_interface.py" in name and name.endswith(": wrapper"):
        return "profiler Event包装"
    if name.endswith(": wait_event"):
        return "Stream.wait_event（异步依赖）"
    if name.endswith(": wait_stream"):
        return "Stream.wait_stream（异步依赖）"
    return None


phase_rows = []
category_rows = []
for rid in RIDS:
    m = M[rid]
    r = C[rid]
    names = {int(k): v for k, v in r["strings"].items()}
    start = r["origin_ns"] + round(m["target_host_enter_session_ms"] * 1e6)
    end = r["origin_ns"] + round(m["target_host_end_session_ms"] * 1e6)
    py = [(s, e, t, n, conn) for s, e, t, n, conn in r["python"] if start <= s < end]
    categories = {}
    for s, e, t, n, _ in py:
        key = classify(names[n]) if t == 50003 else None
        if key:
            categories.setdefault(key, []).append((s, e))
    cann = [
        (max(s, start), min(e, end))
        for s, e, t, n, tid, c, depth in r["cann"]
        if tid == r["main_tid"] and s < end and e > start
    ]
    sync = [
        names[n]
        for s, e, t, n, tid, c, depth in r["cann"]
        if tid == r["main_tid"] and s < end and e > start and "synchroniz" in names[n].lower()
    ]
    category_rows.append(
        {
            "rank": rid,
            "target_host_ms": (end - start) / 1e6,
            "cann_main_union_ms": union(cann) / 1e6,
            "categories": {
                key: {"count": len(v), "inclusive_union_ms": union(v) / 1e6} for key, v in categories.items()
            },
            "main_thread_cann_synchronizations": sync,
        }
    )
    specs = [
        ("DP调度协调", ": dispatch_cg_and_sync_dp", "/patch/worker/"),
        ("Attention metadata", ": prepare_attn", "model_states/default.py"),
        ("Replay metadata", ": _build_metadata", "decoder_replay_state.py"),
        ("metadata外层含上述子阶段", ": prepare_attn", "engram/model_state.py"),
    ]
    for title, suffix, required in specs:
        x = next(
            x
            for x in m["selected_scopes"]
            if x["name"].endswith(suffix)
            and required in x["name"]
            and x["start_session_ms"] < m["target_host_enter_session_ms"]
        )
        phase_rows.append(
            {
                "rank": rid,
                "phase": title,
                "start_common_ms": x["start_common_ms"],
                "end_common_ms": x["end_common_ms"],
                "start_session_ms": x["start_session_ms"],
                "end_session_ms": x["end_session_ms"],
                "duration_ms": x["duration_ms"],
                "name": x["name"],
            }
        )

layer_rows = []
for rid in RIDS:
    for layer in [0, 10, 21]:
        subset = [k for k in K if k["rank"] == rid and k["layer"] == layer]
        kv = next(k for k in subset if k["kernel"] == "MatMulV2" and k["stream"] == 37)
        qnorm = next(k for k in subset if k["kernel"] == "RmsNorm" and k["stream"] == 47)
        layer_rows.append(
            {
                "rank": rid,
                "layer": layer,
                "kv_end_session_ms": ms(kv["kernel_end_ns"], M[rid]["origin_ns"]),
                "qnorm_enqueue_session_ms": ms(qnorm["enqueue_start_ns"], M[rid]["origin_ns"]),
                "qnorm_launch_end_session_ms": ms(qnorm["launches"][-1]["end_ns"], M[rid]["origin_ns"]),
                "qnorm_device_start_session_ms": ms(qnorm["kernel_start_ns"], M[rid]["origin_ns"]),
                "enqueue_after_kv_end_us": (qnorm["enqueue_start_ns"] - kv["kernel_end_ns"]) / 1000,
                "qnorm_csv_line": qnorm["csv_line"],
                "qnorm_task": qnorm["task_id"],
            }
        )

# Short-window points use the bad rank's original integer timestamps.
r = C[BAD]
names = {int(k): v for k, v in r["strings"].items()}
lo = ORIGIN + round(1920400 * 1000)
hi = ORIGIN + round(1921000 * 1000)
points = []


def add(label, s, e, thread, details):
    points.append(
        {
            "label": label,
            "start_ns": int(s),
            "end_ns": int(e),
            "start_session_ms": ms(s),
            "end_session_ms": ms(e),
            "duration_us": (int(e) - int(s)) / 1000,
            "start_absolute_us": absolute_us(s),
            "start_common_ms": ms(s, M[BAD]["common_origin_ns"]),
            "thread": thread,
            "details": details,
        }
    )


for s, e, t, n, c in r["python"]:
    name = names[n]
    if not lo <= s < hi:
        continue
    wanted = t == 50003 and (
        name == "torch_npu/npu/streams.py(56): record_event"
        or name == "torch_npu/npu/streams.py(209): record"
        or name == "nn.Module: AscendRMSNorm_1"
        or name == "torch_npu/profiler/profiler_interface.py(49): wrapper"
        or name == "torch/autograd/profiler.py(806): __exit__"
    )
    if wanted:
        add(name, s, e, "Host TID1601", "Python记录的墙钟，含包装/可能调度停顿；不能当CPU运行时间")
for s, e, t, n, tid, c, depth in r["cann"]:
    name = names[n]
    if tid == r["main_tid"] and lo <= s < hi and name == "aclnnRmsNormGetWorkspaceSize":
        add(name, s, e, "Host TID1601", "其中Tiling是嵌套子区间，不能再次相加")
for k in K:
    if (
        k["rank"] != BAD
        or k["layer"] != 0
        or k["kernel"] not in ["MatMulV2", "RmsNorm", "QuantBatchMatmulV3", "ScatterNdUpdateSk"]
    ):
        continue
    if k["kernel"] == "RmsNorm" and k["stream"] != 47:
        continue
    desc = f"CSV行{k['csv_line']} / Stream{k['stream']} / Task{k['task_id']}"
    add(k["enqueue_name"] + " CPU提交", k["enqueue_start_ns"], k["enqueue_end_ns"], "Host TID1601", desc)
    for launch in k["launches"]:
        add("launch → " + k["kernel"], launch["start_ns"], launch["end_ns"], "Runtime TID" + str(launch["tid"]), desc)
    add("Device " + k["kernel"], k["kernel_start_ns"], k["kernel_end_ns"], "Device Stream" + str(k["stream"]), desc)
points.sort(key=lambda x: x["start_ns"])

# Explicitly preserve physical timestamps and the known CPU wrapper/profiler caveat.
summary = {
    "date": "2026-11-18",
    "source_archive": "P.rar",
    "reference_rank": GOOD,
    "investigated_rank": BAD,
    "bad_main_native_tid": 1601,
    "bad_launch_native_tid": 3591,
    "bad_session_origin_ns": ORIGIN,
    "comparison_common_origin_ns": M[BAD]["common_origin_ns"],
    "anchor": {
        "csv_line": 4168,
        "stream": 37,
        "task_id": 15140,
        "kernel": "MatMulV2",
        "start_ns": ANCHOR,
        "start_absolute_us": absolute_us(ANCHOR),
        "start_session_ms": ms(ANCHOR),
    },
    "phase_comparison": phase_rows,
    "host_categories": category_rows,
    "short_window_points": points,
    "layer_readiness": layer_rows,
    "clock_audit": "clock_audit.json",
    "limits": [
        "Session-relative ms is not asserted to be the default zero in MindStudio Insight. Match the same Task/Stream/Connection IDs before calibration. A single anchor only applies if Host and Device share the same display offset.",
        "Common-origin comparison uses recorded absolute session timestamps within the same collection host; no cross-machine clock claim.",
        "Python/self wall includes possible CPU scheduling/GIL/profiler overhead; _host_sys is empty and no CPU scheduler trace/affinity/frequency was collected.",
        "No main-thread CANN synchronization API was observed in this target interval; this does not prove absence of every implicit driver wait.",
        "Profile config includes CPU/NPU, with_modules=True, PIPE_UTILIZATION; recorded Event wrappers add measurement work. No unprofiled NPU A/B is available.",
    ],
    "new_npu_experiments": False,
}
(ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
with (ROOT / "inspection_points.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(points[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(points)

esc = lambda x: html.escape(str(x), quote=True)


def timing_cell(value):
    return f'<td class="time" data-ms="{value:.9f}">{value:.6f}</td>'


MODEL_PHASE_HTML = """<section id="model-phases"><h2>为什么Prefill调用DecoderLayer？</h2>
<p>trace中的nn.Module是PyTorch模块。DeepseekV41DecoderLayer描述模型层的结构，Prefill和增量生成都会调用它；模块名不能用来判断当前请求阶段。</p>
<div class="table"><table><tr><th>服务阶段</th><th>这次送入模型什么</th><th>经过哪些模型层</th></tr>
<tr><td>Prefill</td><td>批量处理prompt或其中一个chunk，构建所需KV并计算后续token预测</td><td>使用模型层的因果Attention；bounded replay生效时，后段可只处理尾窗，不能把它理解成所有token必经所有层</td></tr>
<tr><td>Decode</td><td>增量处理新token并读取历史KV；MTP等路径可一次处理多个候选token</td><td>继续调用同一组DecoderLayer</td></tr></table></div>
<p>V4.1官方架构为CED：20层因果Encoder接20层Decoder，Decoder的全局KV来源于Encoder的最终隐藏状态。两段均可使用DecoderLayer这一代码类，不能仅由类名推断执行阶段。模型的Encoder/Decoder分段与部署P/D实例不是一一对应，bounded replay是否生效仍须核对实际开关和布局。</p>
<p>Tokenizer.encode/decode又是另一层含义：文本与token ID之间的转换。识别本trace的真实Prefill、dummy和生成前向，需结合execute_model/_dummy_run上下文、Attention元数据、输入shape与任务关联。</p>
<p><a href="https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/models/deepseek_v41/model.py#L780">固定源码：DeepseekV41DecoderLayer</a> · <a href="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash">官方CED / CSA2说明</a> · <a href="../../llm_service_optimization/index.html#mechanism-qkv">Q/KV原理与13个任务对照</a> · <a href="../../llm_service_optimization/index.html#multistream-review">多流阶段结论</a></p></section>"""

# Static scientific timeline; x axis uses the bad rank SESSION clock.
svg = [
    '<svg viewBox="0 0 1140 320" role="img" aria-label="第0层主线程、runtime launch和设备计算的时间关系"><rect width="1140" height="320" fill="#f7fafc"/>'
]
for tick in [1920.45, 1920.55, 1920.65, 1920.75, 1920.85, 1920.95]:
    x = 225 + (tick - 1920.45) * 1600
    svg.append(
        f'<line x1="{x}" x2="{x}" y1="25" y2="280" stroke="#dce5ed"/><text x="{x}" y="305" font-size="12" text-anchor="middle">{tick:.2f}ms</text>'
    )
lanes = [
    ("Host record_event", "torch_npu/npu/streams.py(56): record_event", "#c58c25"),
    ("Host Q Norm module", "nn.Module: AscendRMSNorm_1", "#3a65ba"),
    ("Host workspace/tiling", "aclnnRmsNormGetWorkspaceSize", "#617fbe"),
    ("Host Q Norm enqueue", "aclnnRmsNorm CPU提交", "#3a65ba"),
    ("Runtime Q Norm launch", "launch → RmsNorm", "#8262b5"),
    ("Device KV / Stream37", "Device MatMulV2", "#279b93"),
    ("Device Q Norm / Stream47", "Device RmsNorm", "#e47752"),
]
for i, (label, key, color) in enumerate(lanes):
    point = next(p for p in points if p["label"] == key)
    y = 35 + i * 34
    x = 225 + (point["start_session_ms"] - 1920.45) * 1600
    width = (point["end_session_ms"] - point["start_session_ms"]) * 1600
    svg.append(
        f'<text x="12" y="{y + 17}" font-size="13">{esc(label)}</text><rect x="{x:.3f}" y="{y}" width="{width:.3f}" height="24" rx="3" fill="{color}"><title>{point["start_session_ms"]:.6f}–{point["end_session_ms"]:.6f}ms / {point["duration_us"]:.3f}µs</title></rect>'
    )
svg.append("</svg>")
parts = [
    '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A3 DP2/TP2 Host下发追踪</title>',
    "<style>body{overflow-wrap:anywhere;font:15px/1.8 system-ui,sans-serif;color:#193149;background:#f2f5f9;margin:0}main{max-width:1320px;margin:auto;padding:24px}section{background:white;border:1px solid #d7e1eb;border-radius:10px;padding:22px;margin:18px 0}h1{font-size:29px}h2{font-size:22px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #dce4eb;text-align:left;vertical-align:top;overflow-wrap:anywhere}.table{overflow:auto}.note{border-left:4px solid #c39029;background:#fff7e6;padding:14px}code{overflow-wrap:anywhere;background:#edf2f7;padding:2px 5px}a{color:#174fa2}input,button{font:inherit;padding:7px;border:1px solid #aabccc;border-radius:5px}input{width:180px}svg{min-width:800px;width:100%}.time{white-space:nowrap}@media(max-width:600px){main{padding:12px}section{padding:14px}}</style><main><h1>A3 DP2/TP2：元数据、主线程与实际launch的时序追踪</h1>",
    "<p>2026-11-18 · 数据为既有P.rar。追踪PID/TID1601主线程，实际launch位于TID3591；同组TP0的主线程为1096、launch线程3604。未执行新的NPU实验。</p>",
    '<p><a href="../../llm_service_optimization/index.html#recent">← 返回LLM服务优化总览</a> · <a href="#model-phases">DecoderLayer与Prefill/Decode的区别</a></p>',
    MODEL_PHASE_HTML,
    "<section><h2>可确认的原因与当前边界</h2><ul><li>直接机制已确认：Host元数据阶段晚结束，进入模型后仍逐层下发偏慢，独立Q/KV分支不能同时就绪。不是少创建了辅助流。</li>",
    "<li>相同请求/算子调用规模下，TP2的初始metadata外层比TP0多25.566ms；40层Host推进多63.670ms。DP协调反而更早结束，不能归咎于DP同步等待更长。</li>",
    "<li>整体多流计算重叠：TP2为0.088462ms，TP0为7.420534ms。但目标设备区间分别为447.408ms和463.346ms；重叠少不等于这张NPU的算子更慢，其他rank的排队/等待也会改变重叠形态。</li>",
    "<li>具体热点包括42次build_replay_swa_visible_lens、Event record/profiler包装、current_stream的对象/设备查询和CANN workspace/tiling；这些记录不能简单相加。</li>",
    "<li>现有数据没有CPU线程调度、亲和和频率记录，不能进一步断言物理CPU慢、绑核冲突或GIL竞争。也不能确认无profiler时是否存在同等退化。</li></ul></section>",
    "<section><h2>1924ms的同名MatMul复核：任务与坐标都需要对应</h2>",
    "<p>用户提供Stream37 / Task15092 / connection_id55127。原始trace与数据库一致：它属于前一轮_dummy_run的第37层，输入6 token；本报告分析的Task15140属于后续有效prefill的第0层，输入1848 token。</p>",
    '<div class="table"><table><tr><th>Task / connection_id</th><th>上下文</th><th>设备SESSION ms</th><th>设备共同坐标ms</th><th>launch共同坐标ms</th></tr>'
    + "".join(
        f"<tr><td>{x['task_id']} / {x['connection_id']}</td><td>{esc(x['context'])}</td><td>{x['device_session_ms']:.6f}</td><td>{x['device_common_ms']:.6f}</td><td>{x['launch_common_ms']:.6f}</td></tr>"
        for x in CLOCK_AUDIT["matched_task_comparison"]
    )
    + "</table></div>",
    "<p>共同坐标以DP0/TP0 SESSION为零点，比本rank SESSION坐标增加233.589ms。它与用户描述的大致位置吻合；尚未读取用户UI的精确零点和偏移配置，不能宣称这就是Insight的默认坐标。</p>",
    "<p>共同坐标1913.928329–1924.258976ms确实有EVENT WAIT（Task15091），之后才是Task15092。该MatMul的launch提前97.230776ms，属于队列/依赖等待场景；不能与Task15140的Host供给不足现场混用。</p>",
    "<p>无需猜零点：在同一Stream37中，本报告Task15140比用户选中的Task15092晚<b>229.884513ms</b>。按用户已有锚点向后找这一段即可；还需核对Task ID15140和connection_id69013。</p></section>",
    "<section><h2>Insight定位方法：匹配同一任务后再校准</h2>",
    "<p>打开P.rar中<code>dp2_pp0_tp2_dcp0_ep10_rank2_1601_20261111115927221_ascend_pt</code>。找Stream37的MatMulV2，Task ID <b>15140</b>，<code>kernel_details.csv</code>第4168行。</p>",
    f"<p>原始Start Time(us)：<b>{absolute_us(ANCHOR)}</b>；本rank SESSION零点后的时间为<b>{ms(ANCHOR):.6f}ms</b>。本页不假设Insight默认采用这个零点。</p>",
    f'<p><label>Task15140在你Insight里显示的开始时间（ms）：<input id="anchor" type="number" step="0.000001" value="{ms(ANCHOR):.6f}"></label> <button id="common">采用共同坐标（+233.589ms）</button> <button id="reset">恢复SESSION时间</button></p><p id="offset"></p>',
    "<p>输入同一个任务的锚点后，下表起止时刻平移，持续时间不变。单锚点换算仅适用于Host、CANN和Device使用相同显示偏移的情况；若分别移动过泳道，需先恢复共同偏移，或各自校准，不能用移位后的横向距离推断跨泳道延迟。</p>",
    '<p>本页后面的跨rank对照表始终用DP0TP0 SESSION作为共同零点。<a href="https://www.hiascend.com/document/detail/zh/mindstudio/830/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0053.html">官方文档：卡与泳道的时间偏移设置</a>。</p></section>',
    "<section><h2>关键现场：第0层KV完成后，Q Norm为什么没及时跟上？</h2>",
    "<p>KV矩阵乘结束后，Stream.record_event外层包装仍在执行约57.86µs；Python Q Norm模块到74.90µs后才进入，随后还需算子包装、workspace与tiling、CPU提交、runtime launch。直到315.486µs后，Q Norm才真正开始在设备执行。</p>",
    '<p class="note">此前“Host晚约233µs进入Q Norm”的说法应精确为：低层aclnnRmsNorm提交比KV结束晚232.880µs。Python层Q Norm模块入口更早，晚74.900µs。两者是不同边界。</p>',
    '<div class="table">'
    + "".join(svg)
    + "</div><p>图中固定使用TP2 SESSION坐标，表格可按Insight锚点校准。橙色Host包装尚未退出时，KV设备任务已经完成；主线程之后继续准备Q Norm。</p>",
    '<div class="table"><table><thead><tr><th>记录位置</th><th>校准后开始ms</th><th>校准后结束ms</th><th>墙钟µs</th><th>线程/节点</th></tr></thead><tbody>',
]
for point in points:
    if not (1920.4 <= point["start_session_ms"] <= 1921.0):
        continue
    parts.append(
        "<tr><td>"
        + esc(point["label"])
        + "</td>"
        + timing_cell(point["start_session_ms"])
        + timing_cell(point["end_session_ms"])
        + f"<td>{point['duration_us']:.3f}</td><td>{esc(point['thread'])}<br>{esc(point['details'])}</td></tr>"
    )
parts += [
    "</tbody></table></div>",
    "<p>Event record外层169.57µs，真实Event.record Python方法26.99µs，其内底层record_event CPU节点11.92µs。trace明确出现profiler_interface.wrapper、record_function.__enter__/__exit__和pytree分派检查；不能把外层时长当作设备同步等待。</p>",
    "<p>Q Norm侧：Python模块入口→底层npu_rms_norm→aclnnRmsNormGetWorkspaceSize(60.38µs，含26.16µs tiling)→CPU提交→runtime launch。实际launch结束到设备开始约58.956µs；这也不能全归为Host计算。</p>",
    "<p>第二个失配：KV Scatter在1922.249403ms结束，Q_b的CPU提交到1922.642460ms才开始（晚393.057µs），runtime launch为1922.675430–1922.695880ms，设备到1922.743692ms才执行。对应CSV4173/4174、Task15145/5911。</p></section>",
    '<section><h2>初始元数据：同DP的TP0/TP2对照</h2><p>下表用共同绝对时钟平移到DP0TP0 SESSION零点。外层metadata包含两个子阶段，不能重复相加。两个worker进入execute_model仅差约12.49µs。</p><div class="table"><table><tr><th>Rank</th><th>阶段</th><th>共同坐标起止ms</th><th>持续ms</th></tr>',
]
for x in phase_rows:
    parts.append(
        f"<tr><td>{esc(x['rank'])}</td><td>{esc(x['phase'])}</td><td>{x['start_common_ms']:.3f}–{x['end_common_ms']:.3f}</td><td>{x['duration_ms']:.3f}</td></tr>"
    )
parts += [
    "</table></div><p>DP协调：TP2约85.112ms，TP0约90.698ms；结束反而提前约4.34ms。之后Attention metadata：69.025 vs56.220ms；Replay metadata：67.727 vs51.508ms；合计导致首次HcPre的Host入口TP2晚约22.063ms。</p>",
    "<p>build_replay_swa_visible_lens均为42次：记录的Python self wall在TP2约48.642ms、TP0约37.444ms。self wall不是CPU实际运行时间，可能包含调度与采集开销；该函数还调用多个小tensor/索引操作，可评估按相同输入/组复用，需先核对各组replay floor与有效长度，不能假设42次完全等价。</p>",
    '<p>仅查看TP2时，下面的metadata时间同样支持用前面的锚点校准：</p><div class="table"><table><tr><th>TP2阶段</th><th>校准后开始ms</th><th>校准后结束ms</th><th>持续ms</th></tr>',
]
for x in phase_rows:
    if x["rank"] == BAD:
        parts.append(
            "<tr><td>"
            + esc(x["phase"])
            + "</td>"
            + timing_cell(x["start_session_ms"])
            + timing_cell(x["end_session_ms"])
            + f"<td>{x['duration_ms']:.3f}</td></tr>"
        )
parts += [
    "</table></div></section>",
    "<section><h2>为何正常rank有重叠，而TP2没有：队列提前量不同</h2><p>TP0第0层Q Norm runtime launch早在本rank1900.516320–1900.525670ms完成，设备到1920.306425ms才执行；KV runtime launch也早完成，两条流上的任务都已就绪。</p>",
    "<p>TP2对应Q Norm runtime launch为1920.856210–1920.864140ms，此时KV早已在1920.607610ms完成。该点没有可以同时运行的两份独立工作。正常rank的排队提前量也可能包含跨rank/前序设备等待，更多重叠不自动等于更快。</p>",
    "<p>40层Host推进：TP0为383.783ms，TP2为447.453ms，多63.670ms；主要算子次数一致。相邻dummy窗口里TP2也更慢：step2为359.576 vs304.186ms，step3为363.293 vs309.874ms，说明不只这一次元数据构造受影响。</p>",
    '<p>TP2在第10层和第21层也出现同样的就绪失配，排除仅第0层的一次偶发现象。下表均为TP2坐标，CSV行为kernel_details.csv中Q Norm的位置：</p><div class="table"><table><tr><th>层</th><th>KV设备结束ms</th><th>Q Norm CPU提交ms</th><th>Q Norm launch结束ms</th><th>Q Norm设备开始ms</th><th>提交晚于KV结束µs</th><th>CSV行 / Task</th></tr>',
]
for row in layer_rows:
    if row["rank"] == BAD:
        parts.append(
            f"<tr><td>{row['layer']}</td>"
            + "".join(
                timing_cell(row[k])
                for k in [
                    "kv_end_session_ms",
                    "qnorm_enqueue_session_ms",
                    "qnorm_launch_end_session_ms",
                    "qnorm_device_start_session_ms",
                ]
            )
            + f"<td>{row['enqueue_after_kv_end_us']:.3f}</td><td>{row['qnorm_csv_line']} / {row['qnorm_task']}</td></tr>"
        )
parts += [
    "</table></div>",
    '<div class="table"><table><tr><th>范围</th><th>TP0</th><th>TP2</th><th>说明</th></tr><tr><td>CANN主线程API区间并集</td><td>73.875ms</td><td>89.220ms</td><td>多15.345ms；嵌套workspace与tiling只计一次</td></tr>',
]
for key in ["current_stream", "Stream.record_event", "profiler Event包装"]:
    g = category_rows[0]["categories"][key]
    b = category_rows[1]["categories"][key]
    parts.append(
        f"<tr><td>{esc(key)}</td><td>{g['count']}次 / {g['inclusive_union_ms']:.3f}ms</td><td>{b['count']}次 / {b['inclusive_union_ms']:.3f}ms</td><td>墙钟/包含下级；与其他行可能嵌套，禁止相加</td></tr>"
    )
parts += [
    "</table></div><p>目标区间未见主线程CANN Synchronize API，不支持“主线程被一个长NPU同步卡住”的简单解释；也不能排除所有隐式等待。</p></section>",
    "<section><h2>下一步定位与优化顺序</h2><ol><li>同负载做无profiler/轻量NPU采集对照：当前profiler会包装Event.record/wait，短kernel容易被Host记录开销扰动。先验证异常rank、EP最后到达与低重叠是否稳定复现。</li>",
    "<li>若复现，补主线程CPU调度/迁核/频率、亲和与其他线程争用；本次_host_sys为空，没有这些证据。不能仅凭Python self wall断言GIL或绑核冲突。</li>",
    "<li>代码侧优先减少同step/SWA组重复visible_lens与元数据构建；核对现有缓存current_stream接口能否替代重复对象/设备查询。区分正确性依赖与多余包装，不能直接删event或wait。</li>",
    "<li>为预处理下发做最小实验：复用安全的stream/event对象或减少包装/批量下发，验证Q/KV在设备就绪前均已提交。图捕获与异步buffer/event寿命必须独立检查；Prefill不能直接照搬Decode图模式。</li>",
    "<li>验收不仅看重叠：同时看最后到达rank、层/step墙钟、TTFT/TPOT和吞吐，带profiler用于定位，不带profiler确认收益。</li></ol>",
    '<p><a href="https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/profiler/profiler_interface.py#L43-L60">官方Event profiler包装实现</a>；采集安装版本为2.10.0.post7，实际包装和调用栈以本trace为准。</p>',
    '<p><a href="summary.json">汇总与时间锚点JSON</a> · <a href="clock_audit.json">原始JSON连线及用户任务复核</a> · <a href="inspection_points.csv">定位时间表CSV</a> · <a href="kernel_connections.json">CPU→launch→device关联</a> · <a href="inspect_host.py">主线程复核脚本</a> · <a href="correlate_launches.py">连接ID关联脚本</a></p></section></main>',
    f'<script>const base={ms(ANCHOR):.9f};const input=document.getElementById("anchor");function update(){{const value=Number(input.value);if(!Number.isFinite(value))return;const delta=value-base;document.querySelectorAll(".time").forEach(el=>el.textContent=(Number(el.dataset.ms)+delta).toFixed(6));document.getElementById("offset").textContent="当前平移量："+delta.toFixed(6)+"ms；原始SESSION锚点："+base.toFixed(6)+"ms。";}}input.addEventListener("input",update);document.getElementById("reset").onclick=()=>{{input.value=base;update();}};update();</script></html>',
]
parts[-1] = parts[-1].replace(
    "update();</script>",
    f'document.getElementById("common").onclick=()=>{{input.value=(base+{COMMON_SHIFT:.9f}).toFixed(6);update();}};update();</script>',
)
(ROOT / "report.html").write_text("".join(parts))
print("Built report.html, summary.json, inspection_points.csv; anchor", absolute_us(ANCHOR), "us")
