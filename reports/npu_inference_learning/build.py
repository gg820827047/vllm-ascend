#!/usr/bin/env python3
"""Build a standalone learning guide; no model loading or benchmark execution."""

import html
import json
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent

MODULES = [
    {
        "name": "业务与指标",
        "label": "定义要交付的体验",
        "question": "用户在等什么？2000 日活对应多大的实际负载？",
        "topics": [
            "Agent 多次调用、上下文增长与突发",
            "TTFT / TPOT / ITL 与 p95 / p99",
            "排队、到达率、SLO 与 goodput",
        ],
        "code": "客户端结果、/metrics、vllm/v1/metrics/loggers.py",
        "experiment": "固定一次流式请求，记录首输出、后续到达、实际 token 数和请求 ID。",
        "anchor": "service-target",
    },
    {
        "name": "模型与张量",
        "label": "解释一轮计算",
        "question": "模型一层怎样处理 token？数据的 shape 在哪里变化？",
        "topics": [
            "Q/K/V、Attention、Softmax、Norm、位置编码",
            "KV Cache、MLA、SWA、稀疏选择",
            "MoE / Router / 专家、Engram、draft / verify",
        ],
        "code": "models/deepseek_v41/model.py → attention/dsa_v41.py",
        "experiment": "用已有 trace 标注一层模型，解释每个主要算子的 shape 和作用。",
        "anchor": "knowledge-map",
    },
    {
        "name": "vLLM 引擎",
        "label": "跟踪请求状态",
        "question": "一个请求为什么跨多个 step？哪些请求会被一起执行？",
        "topics": [
            "continuous batching、token budget、chunked prefill",
            "prefix caching、KV block、抢占与恢复",
            "Scheduler → Executor → MRV2 → 采样",
        ],
        "code": "v1/engine/core.py → core/sched/scheduler.py → worker/gpu/model_runner.py",
        "experiment": "比较单并发和多并发，跟踪本轮新 token、历史长度和物理缓存块。",
        "anchor": "code-tour",
    },
    {
        "name": "Ascend 执行",
        "label": "把源码与任务对应",
        "question": "Python 调用如何变成 NPU 上的一个任务？",
        "topics": [
            "平台插件、模型注册与 Attention backend",
            "torch_npu / CANN / custom op",
            "eager、图捕获与重放、元数据、CPU 下发",
        ],
        "code": "platform.py → worker/v2/ → attention/、ops/ → csrc/",
        "experiment": "选一条 SparseFlashMla，找到 Python 入口、host tiling 与核内 Softmax。",
        "anchor": "code-tour",
    },
    {
        "name": "并行与硬件",
        "label": "解释数据与等待",
        "question": "哪部分在算，哪部分在搬运，哪部分在等其他 rank？",
        "topics": [
            "TP / DP / EP / PP、SP、PCP / DCP",
            "Cube / Vector、HBM、片上存储与数据搬运",
            "HCCL、NUMA、通信组、Roofline 与关键路径",
        ],
        "code": "parallel config、fused_moe/、operator tiling / kernel",
        "experiment": "对一处 collective 比较所有相关 rank 的到达时间，再分析核内指标。",
        "anchor": "optimization-map",
    },
    {
        "name": "部署与交付",
        "label": "保证可复现与稳定",
        "question": "如何让别人部署相同环境，并发现和恢复性能退化？",
        "topics": [
            "两个包、依赖、镜像 digest 与模型版本",
            "容量曲线、超时取消、健康检查与回滚",
            "精度、压力、故障恢复与版本升级验证",
        ],
        "code": "Dockerfile.a3、配置、启动脚本、监控与实验清单",
        "experiment": "让同事按文档复现同一请求，核对加载路径、实际开关与指标。",
        "anchor": "assets",
    },
    {
        "name": "网关与 Mooncake",
        "label": "理解上下游边界",
        "question": "实例间谁持有 KV，加载何时完成，等待发生在哪里？",
        "topics": [
            "路由、流式合包、排队、限流与重试",
            "P/D 交接、Transfer Engine 与 Store",
            "缓存来源、缓存组、异步完成与失败回退",
        ],
        "code": "Hybrid / MultiConnector / AscendStoreConnector 接口",
        "experiment": "画出一次 P/D 请求，标注 KV 来源、ready 时刻、必要恢复与首输出。",
        "anchor": "code-tour",
    },
]

PHASES = [
    ("部署与指标", "复现环境和一条流式请求，解释实际 TTFT / TPOT。"),
    ("模型与张量", "完成一张模型层、算子、shape 对照图。"),
    ("引擎与 KV", "跟踪一个请求跨 step 的调度、缓存和状态。"),
    ("测量与定位", "用关键路径和 peer 证据解释一次变慢。"),
    ("修改与验证", "完成正确性、局部基准及端到端验证。"),
    ("服务与赋能", "交付容量报告、运维手册和可复现教学案例。"),
]

SYMPTOMS = {
    "ttft": {
        "title": "TTFT 大：先分清排队、prefill 与交接",
        "checks": [
            "客户端和 engine 的测量边界相同吗？网关有没有排队/缓冲？",
            "是否只有高负载才慢？num_requests_waiting 与抢占是否增长？",
            "按实际未缓存输入、chunk、SWA 生效形状比较；P/D 是否等待 KV 就绪？",
        ],
        "next": "先看客户端结果和队列，再选对应的稳态 prefill / 恢复区间。不要把最长 kernel 自动当成 TTFT 根因。",
    },
    "itl": {
        "title": "TPOT / ITL 大：分布、batch 与每步路径一起看",
        "checks": [
            "一个流式事件有几个 token？投机或合包是否改变了 ITL 口径？",
            "按活跃请求数、历史长度、图命中和长短混合情况分组。",
            "检查关键路径上的 Attention / MoE、CPU 空洞、未覆盖通信与最晚到达 peer。",
        ],
        "next": "先找一次异常输出间隔对应的 engine step。更大 batch 可能增加吞吐，也可能拉长单次间隔。",
    },
    "idle": {
        "title": "NPU 空洞：先确认是否有可执行工作",
        "checks": [
            "这段是否真的有就绪请求？是否处在 warmup、导出或负载结束？",
            "CPU 调度、整理输入、同步或 H2D 是否推迟了下发？",
            "是不是在等另一条 stream、远端 KV 或 collective peer？",
        ],
        "next": "低利用率本身不能区分 CPU、通信、访存或没有负载；先找前驱依赖。",
    },
    "comm": {
        "title": "通信任务很长：区分等人和搬数据",
        "checks": [
            "通信组、消息大小、collective 顺序和各 rank 的时间是否匹配？",
            "最晚到达 rank 之前在做什么，专家 token 是否失衡？",
            "看 wait / transit 与未重叠部分，字段为空或带宽为 0 时不补猜。",
        ],
        "next": "只有 peer 及时到达且传输仍慢，才进一步把调查重点放到链路、协议和消息组织。",
    },
    "memory": {
        "title": "显存高 / OOM / 抢占：先区分容量与带宽",
        "checks": [
            "权重、KV、activation、workspace、图和预留各占多少？",
            "实际驻留 token、缓存组、分配失败、抢占和重算如何变化？",
            "prefix / Store 命中是否真实恢复所需状态？失败是否转为重算？",
        ],
        "next": "显存占用高不等于 HBM 带宽吃满。容量问题与 kernel 访存带宽问题需要不同证据。",
    },
    "kernel": {
        "title": "某个 kernel 慢：先固定工作量，再判断核内 bound",
        "checks": [
            "shape、dtype、stride、阶段和设备范围是否相同？首轮与稳态分开了吗？",
            "它占关键路径多少？是否被其他 stream 的工作覆盖？",
            "再取计算/搬运计数器和实际流量，结合 Roofline 与 tiling 查效率。",
        ],
        "next": "时间线给出位置和依赖；完整 Roofline 或源代码热点通常需要额外的单算子采集。",
    },
}


def render() -> str:
    body = markdown.markdown(
        (ROOT / "guide.md").read_text(),
        extensions=["tables", "fenced_code", "attr_list"],
    )
    body = body.partition("</h1>")[2]
    body = body.replace("<table>", '<div class="table-scroll"><table>')
    body = body.replace("</table>", "</table></div>")
    for scheme in ("https://", "http://"):
        body = body.replace(f'<a href="{scheme}', f'<a target="_blank" rel="noopener noreferrer" href="{scheme}')
    modules = "".join(
        f'<button class="map-button" role="tab" id="map-{i}" aria-controls="module-detail" aria-selected="false">'
        f"<span>{i + 1:02d}</span><strong>{html.escape(module['name'])}</strong>"
        f"<em>{html.escape(module['label'])}</em></button>"
        for i, module in enumerate(MODULES)
    )
    phases = "".join(
        f'<label class="phase"><input type="checkbox" aria-label="完成阶段 {i + 1}：{html.escape(title)}">'
        f"<span><strong>{i + 1:02d} · {html.escape(title)}</strong>"
        f"<small>{html.escape(description)}</small></span></label>"
        for i, (title, description) in enumerate(PHASES)
    )
    navigation = [
        ("interactive-map", "交互知识地图"),
        ("learning-path", "学习阶段与实验"),
        ("code-tour", "0.32 源码阅读路线"),
        ("service-target", "2000 日活与容量"),
        ("optimization-map", "优化拆分与 bound"),
        ("profiling-workflow", "采集与分析动作"),
        ("change-workflow", "修改与验证闭环"),
        ("assets", "学习资产与赋能"),
        ("references", "材料和版本依据"),
    ]
    links = "".join(f'<a href="#{anchor}">{label}</a>' for anchor, label in navigation)
    data = json.dumps({"modules": MODULES, "symptoms": SYMPTOMS}, ensure_ascii=False).replace("</", "<\\/")
    values = {
        "CSS": (ROOT / "style.css").read_text(),
        "JS": (ROOT / "interactions.js").read_text(),
        "MODULES": modules,
        "PHASES": phases,
        "LINKS": links,
        "DATA": data,
        "BODY": body,
    }
    template = (ROOT / "page_template.html").read_text()
    parts = template.split("@@")
    return "".join(values[part] if index % 2 else part for index, part in enumerate(parts))


if __name__ == "__main__":
    target = ROOT / "index.html"
    target.write_text(render())
    print(f"Generated {target.name}: {target.stat().st_size:,} bytes")
