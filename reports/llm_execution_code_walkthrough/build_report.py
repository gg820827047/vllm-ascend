#!/usr/bin/env python3
"""Build an offline, source-pinned walkthrough of Scheduler -> Ascend -> DSv4.1."""
# Keep explanatory prose and diagram labels intact.
# ruff: noqa: E501

import html
import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
WORKTREES = HERE.parents[2]
REPOS = {
    "core": {
        "name": "vLLM",
        "path": WORKTREES / "vllm-core-v032",
        "sha": "5df63a7203cdb9328d824a27b682e77c7b9928be",
        "url": "https://github.com/vllm-project/vllm/blob/",
        "color": "#e7f0fc",
    },
    "ascend": {
        "name": "vllm-ascend",
        "path": WORKTREES / "cann-scheduler-admission",
        "sha": "36b73a9c431d12e7b462f34c89de1da9b18b278d",
        "url": "https://github.com/gg820827047/vllm-ascend/blob/",
        "color": "#e6f3e9",
    },
}

# Explicit line ranges are deliberate: show the actual call site as well as
# its implementation, without importing torch or loading the NPU plugin.
SOURCES = [
    (
        "entry",
        "startup",
        "ascend",
        "setup.py",
        505,
        513,
        "插件入口声明",
        "安装 vllm-ascend 后，Python 包元数据声明 platform_plugins 和 general_plugins。vLLM 据此发现插件，不需要 Scheduler 手工 import DeepSeek 模型。",
    ),
    (
        "platform",
        "startup",
        "ascend",
        "vllm_ascend/__init__.py",
        83,
        87,
        "平台注册：register",
        "返回 NPUPlatform 的完整类名。vLLM 将其解析为当前平台实例；这是启动选型，不是每个请求都执行的跳转。",
    ),
    (
        "platform-resolve",
        "startup",
        "core",
        "vllm/platforms/__init__.py",
        270,
        285,
        "vLLM 选择外部平台",
        "平台插件被激活后，vLLM 调用其注册函数，获得 NPUPlatform 类名。",
    ),
    (
        "worker-select",
        "startup",
        "ascend",
        "vllm_ascend/platform.py",
        1557,
        1575,
        "平台配置选择 Worker",
        "标准 A3/A5 路径由 hardware_profile.default_worker_cls 选择 NPUWorker；Xlite/310P 等有另外的类，本文不展开。",
    ),
    (
        "worker-profile",
        "startup",
        "ascend",
        "vllm_ascend/device/hardware_profile.py",
        294,
        306,
        "标准 Worker 类名",
        "标准 worker 类名为 vllm_ascend.worker.worker.NPUWorker。平台选择的是一个 Python 类，不是一个独立服务。",
    ),
    (
        "wrapper-init",
        "startup",
        "core",
        "vllm/v1/worker/worker_base.py",
        275,
        299,
        "WorkerWrapperBase.init_worker",
        "加载 general plugins，然后按 parallel_config.worker_cls 动态解析 Worker 类。稍后用 worker_class(**kwargs) 创建实际 NPUWorker。",
    ),
    (
        "runner-select",
        "startup",
        "ascend",
        "vllm_ascend/worker/worker.py",
        593,
        611,
        "NPUWorker.init_device 选择 Runner",
        "use_v2_model_runner 为真时创建 worker/v2/model_runner.py 的 NPUModelRunner；否则创建 worker/v1/model_runner.py 的 NPUModelRunner。类名相同，模块不同。",
    ),
    (
        "model-register",
        "startup",
        "ascend",
        "vllm_ascend/models/__init__.py",
        46,
        55,
        "模型 architecture 注册",
        "DeepseekV41ForCausalLM 映射到 deepseek_v41/vl_model.py 的 AscendDeepseekV41ForCausalLM。即使只服务文本，注册入口仍是这一外层 wrapper。",
    ),
    (
        "ascend-load",
        "startup",
        "ascend",
        "vllm_ascend/worker/v2/model_runner.py",
        339,
        354,
        "Ascend MRV2.load_model",
        "Ascend 在加载前后增加适配，主体继续 super().load_model，复用 vLLM 的模型加载流程。",
    ),
    (
        "core-load",
        "startup",
        "core",
        "vllm/v1/worker/gpu/model_runner.py",
        386,
        399,
        "vLLM Runner 加载模型实例",
        "model_loader.load_model 返回模型对象，保存到 Runner.self.model。这个对象已经是注册后的 Ascend DeepSeekV4.1 类。",
    ),
    (
        "model-initialize",
        "startup",
        "core",
        "vllm/model_executor/model_loader/utils.py",
        74,
        98,
        "initialize_model 创建模型",
        "没有外部指定 model_class 时，经 get_model_architecture 取得模型类，最终执行 model_class(vllm_config=..., prefix=...)。",
    ),
    (
        "architecture",
        "startup",
        "core",
        "vllm/model_executor/model_loader/utils.py",
        299,
        308,
        "architecture → ModelRegistry → 类",
        "读取 hf_config.architectures，然后用 model_config.registry.resolve_model_cls 查注册表。它决定的是模型 Python 类；量化配置和权重加载另有流程。",
    ),
    (
        "engine-loop",
        "runtime",
        "core",
        "vllm/v1/engine/core.py",
        1547,
        1557,
        "EngineCoreProc.run_busy_loop",
        "EngineCore 循环处理输入队列并调用 _process_engine_step。每个 DP 的 EngineCore 维护自己的请求与调度状态。",
    ),
    (
        "engine-step",
        "runtime",
        "core",
        "vllm/v1/engine/core.py",
        657,
        683,
        "EngineCore.step：总组织者",
        "调用 Scheduler.schedule，再发 execute_model，必要时发 sample_tokens，最后 update_from_output。batch queue 路径使用同样的主要对象，但跨 step 的等待顺序不同。",
    ),
    (
        "batch-step",
        "runtime",
        "core",
        "vllm/v1/engine/core.py",
        698,
        745,
        "step_with_batch_queue：流水化组织",
        "队列有空位时继续调度并异步提交，队列满或暂时无新批次时回收较早的结果。不能把这段代码简化成每一步都先完全执行完再 schedule 下一步。",
    ),
    (
        "scheduler",
        "runtime",
        "core",
        "vllm/v1/core/sched/scheduler.py",
        591,
        617,
        "Scheduler.schedule：决定本 step 算什么",
        "选择请求和 token 份额、申请 KV block、处理缓存相关约束。本函数主要产生计划，不执行 DeepSeek 的矩阵乘。AsyncScheduler 继承此 schedule，在后续更新钩子中增加异步状态管理。",
    ),
    (
        "plan",
        "runtime",
        "core",
        "vllm/v1/core/sched/output.py",
        232,
        253,
        "SchedulerOutput：本 step 执行计划",
        "num_scheduled_tokens 是 request_id → 本步 token 数，total_num_scheduled_tokens 是总和；另有新请求和 Worker 已知请求的增量资料、投机 token 与 connector 元数据。scheduled_cached_reqs 不等于 prefix cache 命中的请求。它不是模型计算结果。",
    ),
    (
        "executor",
        "runtime",
        "core",
        "vllm/v1/executor/multiproc_executor.py",
        341,
        365,
        "MultiprocExecutor.execute_model / sample_tokens",
        "把方法名字符串与参数交给 collective_rpc；所有需要参与该执行组的 worker 接收计划，不是在 Executor 里直接运行 DeepSeek.forward。",
    ),
    (
        "rpc-send",
        "runtime",
        "core",
        "vllm/v1/executor/multiproc_executor.py",
        425,
        435,
        "collective_rpc 发往进程间队列",
        "rpc_broadcast_mq.enqueue 发送方法名、参数及输出 rank。此处跨进程；单机 multiproc 情况不是远程 HTTP，也不是 Mooncake 的 KV 数据传输。",
    ),
    (
        "rpc-recv",
        "runtime",
        "core",
        "vllm/v1/executor/multiproc_executor.py",
        1043,
        1059,
        "WorkerProc._execute_worker_rpc",
        "从队列读到 execute_model 等方法名，用 getattr(self.worker, method) 找到方法再调用。这里的 self.worker 是 WorkerWrapperBase。",
    ),
    (
        "wrapper-execute",
        "runtime",
        "core",
        "vllm/v1/worker/worker_base.py",
        365,
        372,
        "WorkerWrapperBase.execute_model",
        "wrapper 处理通用事项后调用 self.worker.execute_model。这里 wrapper.self.worker 的真实类型是启动时创建的 NPUWorker，这是运行时从 vLLM 接口进入 Ascend 实现的位置。",
    ),
    (
        "npu-worker",
        "runtime",
        "ascend",
        "vllm_ascend/worker/worker.py",
        874,
        909,
        "NPUWorker.execute_model",
        "负责 worker 生命周期、并行/PP 数据与 profiling 边界等，然后调用 self.model_runner.execute_model。通常一个此类进程对应一个逻辑 NPU。",
    ),
    (
        "ascend-execute",
        "runtime",
        "ascend",
        "vllm_ascend/worker/v2/model_runner.py",
        742,
        761,
        "NPUModelRunner.execute_model：进入 vLLM 父类",
        "super().execute_model 调用 vLLM 的 GPUModelRunner 实现，但 self 仍然是 Ascend NPUModelRunner 实例，设备仍为 NPU，覆写方法仍通过动态绑定生效。",
    ),
    (
        "core-execute",
        "runtime",
        "core",
        "vllm/v1/worker/gpu/model_runner.py",
        1804,
        1823,
        "GPUModelRunner.execute_model：通用执行骨架",
        "父类执行逻辑负责 batch、输入、注意力元数据、图模式、forward 与状态记录。其中 self.prepare_inputs 可回到 Ascend 的覆写实现。GPU 这个类名不表示这里改用 GPU 设备。",
    ),
    (
        "state-patch",
        "runtime",
        "ascend",
        "vllm_ascend/patch/worker/patch_model_state.py",
        20,
        31,
        "另一个接入点：ModelState 工厂替换",
        "替换 vLLM model_runner 模块内 init_model_state 的绑定。DeepSeekV4.1 的模型提供 get_model_state_cls，选择 EngramModelState；其 prepare_attn 再复用 AscendModelState 的注意力元数据实现。",
    ),
    (
        "metadata",
        "runtime",
        "ascend",
        "vllm_ascend/worker/v2/model_states/default.py",
        416,
        429,
        "AscendModelState.prepare_attn",
        "这是此前 profiling 里元数据构造的一条入口；它与目标模型的 decoder layer 不属于同一阶段。DeepSeekV4.1 的 EngramModelState 会包裹这条调用。",
    ),
    (
        "dispatch",
        "runtime",
        "core",
        "vllm/v1/worker/gpu/model_runner.py",
        2020,
        2045,
        "图模式分派 / self.model(**model_inputs)",
        "Eager 模式直接调用现有模型实例；PIECEWISE/FULL/ubatch 走图或批次管理分支。Prefill 使用 FULL_DECODE_ONLY 配置时仍可能走 Eager，不能凭参数名认定每次都图回放。",
    ),
    (
        "outer-model",
        "model",
        "ascend",
        "vllm_ascend/models/deepseek_v41/vl_model.py",
        270,
        284,
        "外层 wrapper.forward → language_model",
        "调用 self.language_model。文本输入也会看到此 wrapper；走到 vl_model.py 不等于这次请求执行了视觉模型。",
    ),
    (
        "llm-forward",
        "model",
        "ascend",
        "vllm_ascend/models/deepseek_v41/model.py",
        1770,
        1805,
        "LLM wrapper.forward → self.model",
        "调用 language_model 内部的 DeepseekV41Model。这里的 self.model 与 Runner.self.model 不同：前者是 backbone，后者是最外层模型实例。",
    ),
    (
        "backbone",
        "model",
        "ascend",
        "vllm_ascend/models/deepseek_v41/model.py",
        1607,
        1625,
        "DeepseekV41Model.forward：层循环",
        "按模型配置遍历 decoder layers；本次采集目标模型为 40 层。启用 bounded replay 时，后段可经过 decoder_replay_layers / _run_replay_layers，最终仍执行相应 DecoderLayer.forward。",
    ),
    (
        "layer",
        "model",
        "ascend",
        "vllm_ascend/models/deepseek_v41/model.py",
        922,
        964,
        "DeepseekV41DecoderLayer.forward",
        "一层包含 HC Pre → Norm → Attention → HC Post，然后 HC Pre → Norm → MoE/MLP → HC Post。Prefill 与 Decode 复用这段模型结构，差别在 token/上下文/缓存/元数据及运行模式。",
    ),
    (
        "attention",
        "model",
        "ascend",
        "vllm_ascend/models/deepseek_v41/model.py",
        774,
        778,
        "DeepseekV41Attention.forward",
        "torch.ops.vllm.dsa_v41_forward 是一个显式自定义算子边界；vllm 命名空间不表示实现一定在 vLLM 仓库，也不表示它是一个单独的 NPU 融合 kernel。",
    ),
    (
        "dsa-op",
        "model",
        "ascend",
        "vllm_ascend/attention/dsa_v41.py",
        59,
        84,
        "自定义 op 注册及 dsa_v41_forward",
        "Ascend 自己将 Python 函数注册到 torch.ops；运行时从 forward_context 中找到层，再调用其 v41_impl.forward。",
    ),
    (
        "dsa-impl",
        "model",
        "ascend",
        "vllm_ascend/attention/dsa_v41.py",
        775,
        799,
        "AscendDSAV41Impl.forward",
        "准备 Q/KV/缓存，选择稀疏索引，执行注意力，逆 RoPE，再投影输出。每段可能对应多个 Torch/CANN/Device 任务，不能把整个 dsa_v41_forward 当一个硬件 kernel。",
    ),
    (
        "moe",
        "model",
        "ascend",
        "vllm_ascend/models/deepseek_v41/model.py",
        277,
        314,
        "DeepseekV41MoE.forward",
        "经 self.experts / FusedMoEFactory 进入路由、共享专家和专家计算路径；通信实现及量化方法继续决定后端算子。",
    ),
    (
        "moe-backend",
        "model",
        "ascend",
        "vllm_ascend/ops/fused_moe/moe_comm_method.py",
        593,
        633,
        "MoE 后端：MegaMoE / DispatchFFNCombine 等",
        "这是当前 CANN 路径的具体分支之一，并非所有 MoE/所有硬件都必经。实际选型受量化、MC2、draft 拓扑及上下文控制。",
    ),
    (
        "tail",
        "model",
        "ascend",
        "vllm_ascend/models/deepseek_v41/model.py",
        1656,
        1670,
        "目标层结束之后：collapse / norm",
        "最后一层 HC Post 后仍有辅助 hidden buffer 写回、HC collapse 与最终 norm。因此 40 层设备包络不是完整模型 forward 的边界。",
    ),
    (
        "worker-sample",
        "return",
        "ascend",
        "vllm_ascend/worker/worker.py",
        943,
        952,
        "NPUWorker.sample_tokens",
        "EngineCore 通过第二个 RPC 指令调用此方法，再转交 Runner.sample_tokens。Wrapper 没有专门覆写的方法可经 __getattr__ 转发给 NPUWorker。",
    ),
    (
        "ascend-sample",
        "return",
        "ascend",
        "vllm_ascend/worker/v2/model_runner.py",
        434,
        446,
        "Ascend MRV2.sample_tokens → super",
        "Ascend 处理状态恢复等适配后，再调用 vLLM 父类 sample_tokens。采样和草稿与前面的 execute_model 是两条相接的调用链。",
    ),
    (
        "core-sample",
        "return",
        "core",
        "vllm/v1/worker/gpu/model_runner.py",
        2088,
        2105,
        "vLLM sample_tokens 读取 forward 状态",
        "从 execute_model_state 取 hidden_states 与 batch 元数据。主模型 forward 一般先提供 hidden states，之后才计算 logits 并采样。",
    ),
    (
        "logits",
        "return",
        "core",
        "vllm/v1/worker/gpu/model_runner.py",
        1607,
        1620,
        "sample → model.compute_logits",
        "没有 sampler sharding 的分支中，先取需要 logits 的 hidden 行，再调用模型 compute_logits。随后 sampler 进行采样或投机验证。",
    ),
    (
        "propose",
        "return",
        "core",
        "vllm/v1/worker/gpu/model_runner.py",
        2225,
        2257,
        "可选 speculator.propose",
        "配置了 speculator 时，采样路径还会调草稿模型；DSpark 适配进入 AscendDSparkSpeculator。后续草稿层属于另一模型实例，不是主模型多了三层。",
    ),
    (
        "draft",
        "return",
        "ascend",
        "vllm_ascend/models/deepseek_v41/dspark.py",
        279,
        307,
        "DeepseekV41DSparkModel.forward",
        "草稿模型自己的 embedding、layer 循环及 HC collapse。此次采集看到三个 draft layer；候选生成/草稿 KV 更新由具体阶段与配置决定。",
    ),
    (
        "update",
        "return",
        "core",
        "vllm/v1/core/sched/scheduler.py",
        2026,
        2044,
        "Scheduler.update_from_output",
        "模型结果回来后更新请求进度、输出 token、完成状态等，并组成要返回前端的 EngineCoreOutputs；未完成请求还会参加下一 step。",
    ),
    (
        "v1",
        "alternate",
        "ascend",
        "vllm_ascend/worker/v1/model_runner.py",
        2390,
        2410,
        "MRV1 的另一路 execute_model",
        "选择 V1 时，NPUWorker 仍相同，但进入本文件。它的 vLLM 父类来自旧的 gpu_model_runner.py，不能把 MRV2 的方法顺序逐行套过去。旧采集里的 model_runner_v1.py 是此前目录命名。",
    ),
]


def escape(value):
    return html.escape(str(value))


def source_data():
    cache = {}
    result = []
    for key, group, repo, path, start, end, title, description in SOURCES:
        r = REPOS[repo]
        if (repo, path) not in cache:
            cache[repo, path] = subprocess.check_output(
                ["git", "-C", str(r["path"]), "show", f"{r['sha']}:{path}"], text=True
            ).splitlines()
        lines = cache[repo, path]
        assert 1 <= start <= end <= len(lines), (key, path, start, end)
        snippet = "\n".join(f"{i:4d}  {lines[i - 1]}".rstrip() for i in range(start, end + 1))
        result.append(
            dict(
                id=key,
                group=group,
                repo=repo,
                path=path,
                start=start,
                end=end,
                title=title,
                description=description,
                snippet=snippet,
                url=f"{r['url']}{r['sha']}/{path}#L{start}",
            )
        )
    return result


def graph(name, nodes, edges):
    header = """digraph G {
graph [rankdir=TB, bgcolor="transparent", pad="0.2", nodesep="0.4", ranksep="0.45", fontname="Noto Sans CJK SC"];
node [shape=box, style="rounded,filled", color="#bccbda", fontname="Noto Sans CJK SC", fontsize=13, margin="0.16,0.12"];
edge [fontname="Noto Sans CJK SC", fontsize=11, color="#5b6f84", arrowsize=0.7];
"""
    q = lambda x: json.dumps(x, ensure_ascii=False)
    parts = [header]
    for key, label, repo in nodes:
        color = REPOS[repo]["color"] if repo in REPOS else "#fff2dc"
        parts.append(f"{q(key)} [label={q(label)},fillcolor={q(color)},URL={q('#src-' + key)},tooltip={q(label)}];")
    for src, dst, label, style in edges:
        parts.append(f"{q(src)} -> {q(dst)} [label={q(label)},style={q(style)}];")
    parts.append("}")
    dot = "\n".join(parts)
    (HERE / f"{name}.dot").write_text(dot)
    svg = subprocess.check_output(["dot", "-Tsvg"], input=dot, text=True)
    # Parse Graphviz XML so standalone headers disappear and embedded IDs remain unique.
    ElementTree.register_namespace("", "http://www.w3.org/2000/svg")
    ElementTree.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ElementTree.fromstring(svg)
    for element in root.iter():
        if "id" in element.attrib:
            element.set("id", f"{name}-{element.attrib['id']}")
    root.set("role", "img")
    root.set("aria-label", name)
    svg = ElementTree.tostring(root, encoding="unicode")
    (HERE / f"{name}.svg").write_text(svg)
    return f'<div class="diagram">{svg}</div>'


def table(head, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{x}</th>" for x in head)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def main():
    """Render the three diagrams, reader guide, and pinned source excerpts."""
    data = source_data()
    by_id = {s["id"]: s for s in data}
    ref = lambda key, text=None: f'<a href="#src-{key}">{escape(text or by_id[key]["title"])}</a>'
    startup = graph(
        "startup",
        [
            ("entry", "vllm-ascend 安装包\nentry_points", "ascend"),
            ("platform", "platform_plugins\nregister → NPUPlatform", "ascend"),
            ("worker-select", "worker_cls 选型\nNPUWorker", "ascend"),
            ("wrapper-init", "vLLM WorkerWrapperBase\n按 worker_cls 创建 Worker", "core"),
            ("runner-select", "NPUWorker.init_device\n选择 MRV2 / MRV1", "ascend"),
            ("model-register", "general_plugins\nModelRegistry 注册 architecture", "ascend"),
            ("architecture", "hf_config.architectures\nresolve_model_cls", "core"),
            ("core-load", "model_loader.load_model\n实例保存为 Runner.model", "core"),
            ("outer-model", "AscendDeepseekV41ForCausalLM\n已创建的模型实例", "ascend"),
        ],
        [
            ("entry", "platform", "平台插件", "solid"),
            ("platform", "worker-select", "平台更新配置", "solid"),
            ("worker-select", "wrapper-init", "类名绑定", "dashed"),
            ("wrapper-init", "runner-select", "创建/初始化", "solid"),
            ("entry", "model-register", "模型插件", "solid"),
            ("model-register", "architecture", "注册表映射", "dashed"),
            ("runner-select", "core-load", "Runner.load_model", "solid"),
            ("architecture", "core-load", "提供模型类", "dashed"),
            ("core-load", "outer-model", "实例化并加载权重", "solid"),
        ],
    )
    runtime = graph(
        "runtime",
        [
            ("engine-step", "① EngineCore\nstep / step_with_batch_queue", "core"),
            ("scheduler", "② Scheduler.schedule\n生成 SchedulerOutput", "core"),
            ("executor", "③ MultiprocExecutor.execute_model\ncollective_rpc", "core"),
            ("rpc-recv", "④ WorkerProc / WorkerWrapperBase\n接收命令并转发", "core"),
            ("npu-worker", "⑤ NPUWorker.execute_model\nself.model_runner", "ascend"),
            ("ascend-execute", "⑥ Ascend NPUModelRunner V2\nexecute_model", "ascend"),
            ("core-execute", "⑦ vLLM GPUModelRunner\nexecute_model 通用骨架", "core"),
            ("metadata", "Ascend / EngramModelState\n输入与 Attention 元数据", "ascend"),
            ("outer-model", "⑧ Runner.self.model(...)\nDeepSeekV4.1 Ascend 实例", "ascend"),
            ("worker-sample", "⑨ 第二条 RPC：sample_tokens\nlogits / sampling / 可选 draft", "ascend"),
            ("update", "⑩ Scheduler.update_from_output\n更新进度并输出 / 下一步", "core"),
        ],
        [
            ("engine-step", "scheduler", "调用", "solid"),
            ("scheduler", "executor", "计划返回 EngineCore 后提交", "solid"),
            ("executor", "rpc-recv", "跨进程消息队列", "dashed"),
            ("rpc-recv", "npu-worker", "调用已绑定 Worker", "solid"),
            ("npu-worker", "ascend-execute", "普通 Python 方法调用", "solid"),
            ("ascend-execute", "core-execute", "super().execute_model", "solid"),
            ("core-execute", "metadata", "动态绑定的准备方法", "solid"),
            ("metadata", "outer-model", "准备好后继续父类 forward", "solid"),
            ("outer-model", "worker-sample", "execute 返回后由 EngineCore 再提交", "dashed"),
            ("worker-sample", "update", "ModelRunnerOutput 经 Executor 返回", "dashed"),
            ("update", "engine-step", "仍有未完成请求", "dashed"),
        ],
    )
    model = graph(
        "model",
        [
            ("outer-model", "AscendDeepseekV41ForCausalLM\nvl_model.py 的外层 wrapper", "ascend"),
            ("llm-forward", "self.language_model\nAscendDeepseekV41LLMForCausalLM", "ascend"),
            ("backbone", "language_model.self.model\nDeepseekV41Model：遍历主干层", "ascend"),
            ("layer", "DeepseekV41DecoderLayer.forward\nAttention 子层 + FFN 子层", "ascend"),
            ("attention", "本层 Attention\nself.self_attn(...) / Attention.forward", "ascend"),
            ("dsa-op", "torch.ops.vllm.dsa_v41_forward\nAscend 注册的自定义 op 边界", "ascend"),
            ("dsa-impl", "AscendDSAV41Impl.forward\nQ/KV → Indexer → Attention → o_proj", "ascend"),
            ("moe", "本层 FFN\nself.mlp(...) / MoE.forward", "ascend"),
            ("moe-backend", "FusedMoE / quant / comm 分派\nMegaMoE、GMM 等后端", "ascend"),
            ("tail", "所有目标层之后\n辅助 hidden 写回 / collapse / norm", "ascend"),
            ("core-sample", "返回 hidden_states\n随后经 sample_tokens 计算 logits / 采样", "core"),
        ],
        [
            ("outer-model", "llm-forward", "forward", "solid"),
            ("llm-forward", "backbone", "forward", "solid"),
            ("backbone", "layer", "层循环 / bounded replay 后段包装", "solid"),
            ("layer", "attention", "先执行", "solid"),
            ("attention", "dsa-op", "调用", "solid"),
            ("dsa-op", "dsa-impl", "注册的 Python 实现", "solid"),
            ("dsa-impl", "moe", "Attention 返回后继续本层 FFN", "solid"),
            ("moe", "moe-backend", "路由 / 量化 / 通信选型", "solid"),
            ("moe-backend", "tail", "完成各层循环后", "dashed"),
            ("tail", "core-sample", "回到 Runner", "dashed"),
        ],
    )

    intro = """<p class="eyebrow">源码地图 · CANN 0.32 核对快照 · 2026-11-26</p>
<h1>从 Scheduler 到 DeepSeekV4.1，代码怎样走过去</h1>
<p class="lead">先记住：vLLM 负责组织请求与执行，vllm-ascend 提供 NPU 平台、Worker、Runner 和模型实现。启动时选好对象，每个 step 再沿这些对象调用；中间会通过继承多次复用 vLLM 的代码。</p>
<div class="note">阅读顺序：先看图 1 的启动绑定，再看图 2 的每步执行，最后进入图 3 的模型层。蓝色为 vLLM，绿色为 vllm-ascend。点图中节点可跳到代码索引，再展开固定版本源码片段。</div>
<p><a href="../dsv41_a3_a5_prefill/execution_scope/report.html">回到模型执行耗时拆解</a> · <a href="../llm_service_optimization/index.html">性能优化总览</a> · <a href="sources.json">源码索引 JSON</a></p>
"""
    part1 = (
        '<h2 id="startup">图 1：启动时如何选中 Ascend 和模型类</h2><p>这张图表达配置与对象绑定关系，不要求两个插件分支严格按图中某一先后顺序完成。平台注册、模型注册是两个不同机制；它们都在实际 forward 之前准备好。</p>'
        + startup
    )
    part1 += "<p>模型目录的 <code>config.json → architectures</code> 例如 <code>DeepseekV41ForCausalLM</code>，通过 ModelRegistry 映射到 Ascend 实现。模型路径名称、<code>--served-model-name</code> 不是选择 Python 模型类的直接开关。<code>--quantization ascend</code> 主要选择量化处理，也不是激活整个 NPU 插件的唯一开关。</p>"
    part2 = (
        '<h2 id="runtime">图 2：每个 step 从 schedule 到执行，再回到 Scheduler</h2><p>主线以 <b>MRV2 + 标准 multiproc + 文本生成</b> 为例。图中“计划返回后提交”“返回后再提交”表示 EngineCore 组织的数据流，不是 Scheduler 或模型直接调用下一个框。实际函数调用看下面代码索引。</p>'
        + runtime
    )
    part2 += "<p><b>跨进程与跨仓库不同：</b>Executor→WorkerProc 经过进程间消息队列；WorkerWrapper→NPUWorker 是同一个 Worker 进程里的 Python 方法调用；Ascend Runner→vLLM Runner 是 <code>super()</code> 父类调用。它们不需要 HTTP 跳转，也不是 KV 通过 Mooncake 传输。</p>"
    part2 += "<h3>本 step 传下去的是什么</h3><pre>SchedulerOutput（执行计划）\n  num_scheduled_tokens: {request_id: 本步要计算多少 token}\n  total_num_scheduled_tokens: 上述数量的总和\n  scheduled_new_reqs / scheduled_cached_reqs: 请求、token、KV block 等更新\n  scheduled_spec_decode_tokens: 待验证的候选 token\n  kv_connector_metadata: 本步 KV 加载 / 保存计划\n\nModelRunnerOutput（执行结果）\n  sampled_token_ids、request 对应关系、投机统计、connector 完成信息等\n\nEngineCoreOutputs（更新请求后向前端输出）\n  token / 完成状态 / 统计等</pre>"
    part2 += "<p><b>AsyncScheduler 不等于另一套模型：</b>未配置自定义 scheduler_cls 时，async_scheduling 决定 Scheduler / AsyncScheduler 类。AsyncScheduler 继承 Scheduler.schedule，增加异步占位与结果更新；EngineCore 的 batch queue 决定怎样流水化提交/回收，二者有关但不是同一个开关。Ascend 自定义 scheduler 启用时可以替换默认类。</p>"
    part2 += "<h3>简化伪代码：用来理解职责，不是可直接运行的实现</h3><pre># EngineCore 内；非 batch queue 路径的简化示意\nplan = scheduler.schedule()\nfuture = executor.execute_model(plan, non_block=True)  # IPC 发给 Worker\nresult = future.result()\nif result is None:\n    result = executor.sample_tokens(grammar_output)   # 再发一条 Worker RPC\noutputs = scheduler.update_from_output(plan, result)\n\n# 已绑定的 Ascend Worker 进程内\nNPUWorker.execute_model(plan):\n    return self.model_runner.execute_model(plan, intermediate_tensors)\n\nNPUModelRunnerV2.execute_model(...):\n    ...\n    return_or_postprocess(super().execute_model(...))\n\n# vLLM 父类的通用骨架；self 仍是 Ascend Runner\nGPUModelRunner.execute_model(...):\n    self.prepare_inputs(...)            # 可回到 Ascend 覆写\n    self.model_state.prepare_attn(...)  # 已选好的 Ascend / Engram ModelState\n    hidden_states = self.model(...)    # 已加载的 Ascend DSv4.1 实例，Eager 分支\n    self.execute_model_state = ...\n    return None                         # 普通生成、最后 PP rank 的典型路径</pre>"
    part3 = (
        '<h2 id="model">图 3：为什么 self.model 会进入 deepseek_v41</h2><p>Runner 不会在每一步猜“现在是什么模型”。加载阶段已经创建了具体对象；<code>nn.Module.__call__</code> 负责 hooks 等通用事项，再调用该对象的 <code>forward</code>。下面展示文本请求的主要计算关系，省略视觉塔、LoRA 与部分并行分支。</p>'
        + model
    )
    part3 += "<p><b>Prefill / Decode 都可以经过 DecoderLayer.forward。</b>DecoderLayer 是模型结构名称，不表示“这里只能在 D 节点执行”。Prefill 的长 query/chunk、Decode 的短 query、KV 历史长度与元数据会让后端分支和 shape 不同。</p>"
    objects = table(
        ["在哪个对象里", "这个属性的实际内容", "不要混淆"],
        [
            ["EngineCore.model_executor", "MultiprocExecutor", "负责进程间提交和结果收集，不是模型实例"],
            ["WorkerProc.worker", "WorkerWrapperBase", "通用包装器，不是最里面的 NPUWorker"],
            ["WorkerWrapperBase.worker", "NPUWorker", "这一步完成 vLLM Worker 接口到 Ascend Worker 实现的委派"],
            [
                "NPUWorker.model_runner",
                "Ascend NPUModelRunner V1 或 V2",
                "Runner 负责 batch、缓存/元数据、图与采样等执行组织",
            ],
            ["Runner.model", "AscendDeepseekV41ForCausalLM 外层 wrapper", "不是某一个 decoder layer"],
            [
                "Runner.model.language_model",
                "AscendDeepseekV41LLMForCausalLM",
                "语言模型 wrapper，内部还有 backbone 与 lm_head",
            ],
            [
                "Runner.model.language_model.model",
                "DeepseekV41Model",
                "这个 self.model 是 backbone，和 Runner.self.model 不是同一层对象",
            ],
            ["backbone.layers[i]", "DeepseekV41DecoderLayer", "本配置目标层数为 40；模型层数由配置决定"],
        ],
    )
    part3 += "<h3>最容易绕晕的 self.model 对照</h3>" + objects
    part4 = '<h2 id="return">采样与 DSpark 为什么又走一条链</h2><pre>EngineCore\n  → Executor.sample_tokens  ──IPC──→ WorkerProc / Wrapper\n  → NPUWorker.sample_tokens\n  → Ascend NPUModelRunnerV2.sample_tokens\n  → vLLM 父类 GPUModelRunner.sample_tokens\n      → sample → 已加载模型.compute_logits → sampler / 投机验证\n      → [配置了 speculator 时] speculator.propose\n          → Ascend DSpark 适配 → 另一份 DSpark 草稿模型实例.forward\n          → 草稿 logits / 采样\n      → ModelRunnerOutput 经 RPC 返回 EngineCore\n  → Scheduler.update_from_output → 完成 / 下一 step</pre>'
    part4 += "<p>因此 profiling 看到两层 execute_model、多层 wrapper.forward，不是做了多遍主模型。目标模型 40 层与草稿 3 层是不同模型实例的调用。本次采样路径也不能自动解释为每个中间 Prefill chunk 都向用户输出了一个 token；哪些输出有效由请求进度与调度/采样逻辑决定。</p>"
    part5 = '<h2 id="variants">MRV1、MRV2、图模式与 P/D 放在哪里理解</h2>'
    part5 += table(
        ["情况", "路径变化", "保持的结构"],
        [
            [
                "MRV2（本文主线）",
                "Ascend worker/v2/model_runner.py → vLLM worker/gpu/model_runner.py；准备过程使用 Ascend ModelState",
                "Scheduler → Executor → Worker → Runner → 具体模型",
            ],
            [
                "MRV1",
                "Ascend worker/v1/model_runner.py → vLLM 旧 worker/gpu_model_runner.py；旧 trace 可能叫 model_runner_v1.py",
                "同一个 NPUWorker 选择另一个 Runner；不能逐行套用 MRV2 的内部方法顺序",
            ],
            [
                "Eager / PIECEWISE / FULL",
                "Runner 的 forward 分派不同；FULL 主要回放捕获的图，不一定每次重新出现完整 Python 层栈",
                "模型仍是加载时确定的模型；图源于捕获的计算",
            ],
            [
                "P / D 分离",
                "P 和 D 各自运行一套上述 EngineCore/Worker/模型；connector 在 scheduler / runner 钩子处处理 KV 计划、加载、保存",
                "不是从 P 的某个 Python forward 直接跳进 D 的 forward",
            ],
            [
                "P 4DP×4TP / D 8DP×2TP",
                "常规部署分别有 4 / 8 个 DP EngineCore，及各 16 个 NPUWorker；EP 还可跨 DP 组通信",
                "每个 DP EngineCore 调度本组请求；TP Worker 参与同一执行计划，独立拥有本 rank 的权重/专家分片与缓存",
            ],
        ],
    )
    part5 += "<p><code>vllm/v1/...</code> 是 V1 Engine 命名空间，不等于 Model Runner V1；<code>GPUModelRunner</code> 的类名也不意味着 Ascend 正在使用 GPU。当前快照的 Runner 选择还支持环境变量显式覆盖与特性回退，判断实际路径以有效配置和日志/trace 为准。</p>"
    profile = '<h2 id="profiling">把这张代码地图套回 profiling</h2>'
    profile += table(
        ["你在 trace / 日志看到什么", "在此图中属于哪里", "怎样读"],
        [
            [
                "SchedulerAdmission / Scheduler.schedule",
                "EngineCore 的调度阶段",
                "是决策与计划构造；不能把请求排队时间当 schedule 函数 CPU 耗时",
            ],
            [
                "Ascend worker/v2/model_runner.py execute_model + vLLM gpu/model_runner.py execute_model",
                "子类调用父类，处于同一个 Worker 的一条嵌套执行链",
                "外层含内层，不能相加",
            ],
            [
                "prepare_inputs / prepare_attn / _build_metadata",
                "Runner 与 ModelState 的输入/元数据阶段",
                "在目标 decoder 层之前；部分工作和前轮 Device 执行可重叠",
            ],
            [
                "nn.Module: AscendDeepseekV41ForCausalLM / LLMForCausalLM",
                "外层 wrapper 与语言模型 wrapper",
                "包含主干层；不表示额外执行了一遍 40 层",
            ],
            [
                "nn.Module: DeepseekV41DecoderLayer_0…39",
                "目标主干层循环",
                "Prefill 也走这里；完整 forward 在层末尾之后还可能有 collapse / norm",
            ],
            [
                "vllm::dsa_v41_forward",
                "Ascend 注册的 Attention 自定义 op 边界",
                "内部是多个 Q/KV/Indexer/Attention/投影任务，不是一个硬件 kernel",
            ],
            [
                "MegaMoe / DispatchFFNCombine / GroupedMatmul",
                "MoE 后端分派后的具体 Device 任务",
                "通过连接 ID 对回具体层；驻留可能包含通信/跨 rank 等待",
            ],
            [
                "sample_tokens / DSparkDecoderLayer_0…2",
                "后续采样 / 草稿实例执行链",
                "不应并入主模型 40 层的区间，也不以 Host 返回代替 Device 完成",
            ],
        ],
    )
    profile += "<p>例如前次 A3 DP0/TP0 的主干计算锚点是首 HC Pre +235.027351 ms（Stream 47 / Task 45774）到目标第 39 层末 HC Post +690.784237 ms（Stream 47 / Task 47475），相对该次 Worker 入口。它对应图 3 的目标层区域，不包括完整输入准备、目标 tail、采样和草稿。详见上方链接的执行边界报告。</p>"
    source_sections = '<h2 id="sources">代码索引：点开看真正发生调用的位置</h2>'
    source_sections += "<p>链接固定到已核对提交，片段通过 git show 从同一提交生成。源码读图不依赖线上 main 是否又变化；A5 的 dsnext 镜像不保证与本页逐行相同。索引展示的行号属于本页固定快照。</p>"
    for s in data:
        r = REPOS[s["repo"]]
        source_sections += f'<article id="src-{s["id"]}"><h3><span class="tag {s["repo"]}">{r["name"]}</span> {escape(s["title"])}</h3><p>{escape(s["description"])}</p><p class="path"><a href="{s["url"]}" target="_blank" rel="noopener">{escape(s["path"])}:{s["start"]}</a></p><details><summary>展开固定版本源码片段</summary><pre><code>{escape(s["snippet"])}</code></pre></details></article>'
    reading = (
        '<h2 id="reading">建议先读这 8 处</h2><ol>'
        + "".join(
            f"<li>{ref(key)}</li>"
            for key in [
                "engine-step",
                "scheduler",
                "executor",
                "rpc-recv",
                "wrapper-execute",
                "ascend-execute",
                "dispatch",
                "layer",
            ]
        )
        + "</ol><p>读懂这条主线后，再回看插件注册、ModelState patch、MoE 后端和草稿链。这样每读一个文件，都知道它在请求执行的哪个位置。</p>"
    )
    footer = (
        "<footer>vLLM："
        + REPOS["core"]["sha"]
        + "<br>vllm-ascend："
        + REPOS["ascend"]["sha"]
        + "<br>本页为代码关系梳理，不修改运行时功能，不新增性能测量。</footer>"
    )
    body = intro + part1 + part2 + part3 + part4 + part5 + profile + reading + source_sections + footer
    nav = (
        "<nav>"
        + "".join(
            f'<a href="#{key}">{label}</a>'
            for key, label in [
                ("startup", "启动绑定"),
                ("runtime", "每步执行"),
                ("model", "模型内部"),
                ("return", "采样/草稿"),
                ("variants", "路径区别"),
                ("profiling", "对应 profiling"),
                ("reading", "阅读顺序"),
                ("sources", "源码索引"),
            ]
        )
        + "</nav>"
    )
    css = """*{box-sizing:border-box}body{margin:0;background:#eff3f7;color:#19334b;font:16px/1.85 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1380px;margin:auto;padding:38px 44px 80px;background:white}h1{font-size:34px;line-height:1.35}h2{margin-top:48px;border-top:2px solid #e0e8f0;padding-top:18px;font-size:25px;scroll-margin-top:70px}h3{font-size:19px}p{max-width:1160px}a{color:#1663a5}.lead{font-size:18px;color:#46617b}.eyebrow,.path,footer{font-size:12px;color:#56718a}.path{overflow-wrap:anywhere}.note{background:#fff5df;border-left:4px solid #cd9b36;padding:16px 20px;margin:24px 0}.diagram,.scroll{overflow:auto;margin:20px 0}.diagram{background:#fbfcfe;padding:18px;border:1px solid #e1e8f0;border-radius:8px}.diagram svg{width:auto;max-width:100%;height:auto;display:block;margin:auto}pre{font:12px/1.8 ui-monospace,monospace;padding:15px;background:#f0f4f8;white-space:pre-wrap;overflow-wrap:anywhere;max-height:550px;overflow:auto}code{font-size:13px;background:#f0f4f8}table{width:100%;min-width:760px;border-collapse:collapse;font-size:13px}th,td{padding:11px;text-align:left;border-bottom:1px solid #dce5ee;vertical-align:top}th{background:#edf3f9}article{padding:10px 0 20px;border-bottom:1px solid #e1e8ef;scroll-margin-top:70px}article p{margin:10px 0}summary{cursor:pointer;color:#1663a5;font-size:13px}.tag{padding:3px 8px;border-radius:4px;font-size:12px;margin-right:6px}.core{background:#e7f0fc}.ascend{background:#e6f3e9}nav{display:flex;justify-content:center;gap:18px;flex-wrap:wrap;padding:9px;position:sticky;top:0;background:#17334b;z-index:1}nav a{color:white;text-decoration:none;font-size:13px}footer{margin-top:45px}li{margin:7px 0}@media(max-width:720px){main{padding:22px 16px}h1{font-size:27px}h2{font-size:22px}nav{gap:10px}.diagram{padding:8px}.diagram svg{min-width:620px;max-width:none}}@media print{nav{display:none}main{padding:0}.diagram svg{max-width:100%}details pre{max-height:none}.scroll{overflow:visible}table{min-width:0;font-size:10px}}"""
    (HERE / "report.html").write_text(
        f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scheduler → Ascend → DeepSeekV4.1 代码地图</title><style>{css}</style></head><body>{nav}<main>{body}</main></body></html>'
    )
    (HERE / "sources.json").write_text(
        json.dumps(
            dict(revisions={k: {x: v[x] for x in ["name", "sha", "url"]} for k, v in REPOS.items()}, sources=data),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("Built", HERE / "report.html", "with", len(data), "source anchors")


if __name__ == "__main__":
    main()
