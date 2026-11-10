# NPU 大模型推理：知识地图与性能优化实践

版本：2026-11-10。面向刚开始系统学习大模型、准备使用 Ascend NPU 部署 DeepSeek 等模型，并负责推理服务性能与代码优化的工程师。

本文的目标是形成一条可反复实践、也能教给别人的路径：**能部署 → 能解释执行过程 → 能测量 → 能定位 → 能修改 → 能证明收益 → 能维护服务**。

代码阅读基线固定为：vLLM `v0.32.0 / 5df63a7203cdb9328d824a27b682e77c7b9928be`；升级后的 vllm-ascend CANN 分支 `d0128bcd099125d6a6bdbf9cd75799c2c5ed4a80`，其合入的上游发布提交为 `21609b11ee574c477e5f36c925921e2355df160c`。这是一份学习与实验手册，不是对其他版本、其他 NPU 或未测试配置的兼容承诺。

## 1. 学什么：以一条请求贯穿知识地图 {#knowledge-map}

你的主线应是：**用户请求如何变成一批 token，如何被调度，如何变成 NPU 上的计算与通信，最后如何影响用户等待时间。** 每个名词都放回这条链路理解。

| 层次 | 核心问题 | 必学内容 | 先掌握到什么程度 |
| --- | --- | --- | --- |
| 业务与指标 | 用户究竟在等什么？ | Agent 多轮调用、输入输出长度、并发与到达率、TTFT/TPOT/ITL、p95/p99、SLO、goodput | 能写一张业务负载与验收表 |
| 模型计算 | 每轮前向做了什么？ | token、张量 shape、Transformer、Q/K/V、Attention、Softmax、FFN/MoE、Norm、位置编码、KV Cache | 能解释一层模型的输入输出及主要开销 |
| 推理引擎 | 为什么这几个请求被放在一起？ | continuous batching、token budget、chunked prefill、prefix caching、抢占、block table、slot mapping、模型执行器与采样 | 能跟踪一个请求跨多个调度 step 的状态 |
| Ascend 执行 | Python 怎么变成 NPU 任务？ | 插件注册、模型适配、Attention backend、torch_npu、CANN、图捕获与重放、自定义算子 | 能从 trace 中的任务定位到实际调用代码 |
| 并行与硬件 | 多张卡如何协作？ | TP/DP/EP/PP、SP、PCP/DCP、Cube/Vector、HBM/片上存储、数据搬运、HCCL、CPU/NUMA | 能区分计算、内存、通信、等待和主机瓶颈 |
| 部署与运维 | 如何稳定交付性能？ | 镜像/依赖/模型版本、监控、容量、超时取消、滚动变更、回滚、故障演练 | 能复现环境，发现退化，恢复服务 |
| 网关与 Mooncake（了解边界） | 时间和 KV 在实例间怎么流动？ | 路由、限流、流式返回、P/D 交接、Transfer Engine、Store、命中与失败回退 | 能判断引擎外的等待，向对应模块提出有证据的问题 |

建议精力分配的起点：模型与张量 20%，引擎/Ascend 代码 30%，测量与性能分析 35%，部署和外围组件 15%。这是学习取舍，不是固定工时要求。性能分析也依赖前面几项，不能直接从算子耗时榜开始。

### 1.1 一次请求的执行路径

```text
Agent / 客户端
  → 网关、鉴权与路由
  → vLLM API、模板处理与分词
  → EngineCore：入队、调度、KV 块分配
  → ModelRunner：整理本轮 token、位置、block table、slot mapping
  → 模型层：Attention / MoE / Engram 等
  → torch_npu / CANN / 自定义算子
  → NPU：计算、访存、通信
  → 采样与状态更新 → 流式返回
```

单个请求会经过很多轮调度；每轮可以同时服务多个请求。一个 batch、一个 step、一个模型前向、一个用户请求、一个输出 token，必须分别理解。投机解码还可能一轮验证并输出多个 token。

P/D 分离是在这条链路中加入两个引擎实例及 KV 交接：P 负责主要的 prompt 计算，D 利用恢复的状态继续生成。首 token 究竟由 P 首步产生还是 D 恢复后产生，要看代理与协议实现。D 仍可能执行少量恢复、重算或窗口重建计算。**P/D 是部署分工，prefill/decode 是计算阶段，两者不能简单按进程名称一一对应。**

vLLM 的调度器使用统一 token budget 管理请求进度；理解 prefill/decode 两种计算特征，并不意味着调度器只有两个严格分开的循环。参考：[引擎架构](https://docs.vllm.ai/en/latest/design/arch_overview/)。

### 1.2 最容易混淆的概念

| 概念 | 准确理解 | 你应该能回答的问题 |
| --- | --- | --- |
| token | 分词器的离散单位，与字符数、字节数、代码行数不相等 | 相同字节长度的两段代码，token 数为什么不同？ |
| KV Cache | 保存历史 token 对后续 Attention 有用的状态；decode 仍要计算新 token 的各层 | 复用什么，省掉什么，又必须继续算什么？ |
| prefix cache | 跨请求复用已有前缀计算状态；不能消除当前请求正常 decode 所需的历史 KV | 重复请求是否真的命中同一个缓存域？ |
| MLA | 经典机制通过低维联合表示压缩 K/V；当前 V4.1 的 SparseFlashMla 还要按其共享 KV 算子合同理解 | 压缩的是表示维度，还是参与注意力的 token 范围？ |
| SWA | 滑动窗口注意力：某些注意力分支只访问最近一段历史；模型可以同时保留长程分支 | 窗口由模型结构决定，为什么不能随意改小？ |
| DSA / 稀疏 Attention | 通过索引/选择等机制，只对部分候选历史位置计算注意力 | Indexer 的成本能否超过被省掉的 Attention 成本？ |
| MoE / EP | Router 选择专家；EP 把专家分布到不同 rank，伴随分发、专家计算与合并 | 激活专家少，为什么模型仍需要存放大量权重？ |
| 量化 | 降低权重、激活或 KV 的表示精度；三者是不同开关 | 当前 GEMM 真正使用什么 dtype 和 kernel？ |
| 图重放 | 复用捕获的执行流程，减少主机逐个下发算子的开销 | 是哪些 batch 命中了图，哪些回退到 eager？ |
| SWA replay | 与图重放不同，涉及窗口状态恢复，以及符合条件时后段层的尾窗计算 | 启动日志、恢复状态和实际 shape 是否同时证明生效？ |
| TP / DP / EP | TP 分割部分层计算；DP 管理不同请求批次；EP 分布专家 | EP 开启后，DP 之间是否仍参与同一个专家通信域？ |
| SP / PCP / DCP | 分别涉及中间 token 布局、prefill 上下文并行、decode 上下文并行 | 切分的是激活、上下文计算，还是 KV 状态？ |
| 引擎 V1 / MRV1 / MRV2 | `vllm/v1/` 是引擎目录；Model Runner V1/V2 是执行器实现，两者命名层次不同 | MRV2 为什么仍然可以位于 `vllm/v1/worker/`？ |

先学 Transformer 推理所需的张量与计算，再补充相关数学。暂时不需要从预训练、反向传播、RLHF 或从零训练大模型开始。

### 1.3 Engram 和 draft：它们分别在做什么

**Engram 是模型的条件记忆查表模块。** 它根据相邻 token 组成的短片段（n-gram）计算索引，从学习得到的参数表里取出向量，再通过投影和门控与当前隐藏状态结合。可以把它理解为模型在计算过程中访问的一张“短片段记忆表”，用来补充局部模式和知识表示。[Engram 官方说明](https://github.com/deepseek-ai/Engram)。

```text
最近的一段 token → n-gram / hash → 记忆表 lookup
                               ↓
当前隐藏状态 ─────────────→ 投影与门控融合 → 后续模型层
```

记忆表属于模型参数，与一次请求产生的 KV 状态分别管理。你之前的 `engram_config.cpu_offload=true` 控制这类表的主机内存存放；`dp_shared_memory=true` 在满足条件的同机进程间共享相应表分片。实际执行仍有查表、数据访问、可能的反量化、投影和门控成本。性能分析应分别看 hash/lookup、数据准备与模型层的重叠，以及访存、同步和融合开销，不能把 Engram lookup 当成一整轮 Attention。

**draft 是投机解码中的“候选生成”。** 先由较便宜的模型或专用预测模块提出后续几个 token，再由目标主模型集中验证；验证接受的前缀继续作为输出，被拒的位置由验证算法修正。希望通过一轮输出更多有效 token，减少昂贵主模型逐步执行的次数。[vLLM 投机解码说明](https://docs.vllm.ai/en/latest/features/speculative_decoding/)。

```text
草稿模块提议：A B C
          ↓
主模型批量验证 → 接受若干 token，必要时修正 → 继续生成
```

draft 的实现不一定是另一个独立加载的完整小模型，也可以是配套预测层。当前 V4.1 路径有 [DSpark 草稿模块](https://github.com/gg820827047/vllm-ascend/blob/d0128bcd099125d6a6bdbf9cd75799c2c5ed4a80/vllm_ascend/models/deepseek_v41/dspark.py)，实际块数由对应模型配置确定；草稿层数、提议 token 数和最终接受 token 数分别是不同概念。prefill 时也可能为草稿模块准备上下文状态，因此 P 侧出现 draft 调用需要结合阶段判断。

draft 性能重点是：提议花了多少时间，主模型验证花了多少时间，一轮实际产出了多少有效 token，以及接受率和 batch/图模式怎样变化。草稿或验证成本太高、接受率太低时，投机路径可能使 TPOT 变差。标准的精确验证以保持目标分布为设计目标；若启用近似验证策略，还需单独评估质量。

| 项目 | Engram | draft |
| --- | --- | --- |
| 主要作用 | 查找学习得到的记忆向量，并融入模型计算 | 提出候选 token，供主模型验证 |
| 关注状态 | token 短片段、索引、表向量和隐藏状态 | 草稿上下文、候选、验证与接受结果 |
| profiling 重点 | hash、lookup、数据访问、投影、门控与重叠 | draft / verify 时长、接受长度及有效输出速度 |
| 与用户指标的联系 | 是模型前向的一部分，开销可能影响 prefill 和 decode | 主要尝试改善持续生成效率，实际收益依负载而定 |

## 2. 怎么学：六个阶段，每阶段有验收物 {#learning-path}

阶段长短由基础和可用 NPU 决定。建议每阶段以 3–5 个半天为一轮学习单元，再根据实测问题迭代；复杂并行和自定义算子可以占用多轮。**以验收结果推进，不用“读完多少篇文章”代表学会。**

若张量基础还不牢，先用小数组补齐矩阵乘、reshape/transpose、广播、dtype、stride，以及异步设备调用与同步的含义。能够自己解释一个小 Attention 的输入输出，再把相同概念对应到融合算子；这比一开始阅读整个高性能 kernel 更容易建立正确模型。

| 阶段 | 学习与实验 | 验收标准 | 沉淀的资产 |
| --- | --- | --- | --- |
| ① 跑通并建立指标 | 先用适配良好的较小模型理解请求流程，再在目标 DS 上复现；固定镜像、两个仓库提交和模型配置；发一个流式请求 | 能解释实际输入/输出 token 数、TTFT、TPOT，能复现启动 | 一页部署说明、环境清单、一条可复现请求 |
| ② 读懂一轮计算 | 学张量 shape、Attention/MoE；看小模型/目标模型的单次稳态前向 | 能指出 Q/K/V、Attention、MoE、输出投影和采样所在范围；知道 Softmax 可能已融合 | 一张“模型层 → 算子 → shape”对照图 |
| ③ 跟踪引擎状态 | 从 API 跟到调度器、KV 管理和 MRV2；比较单请求、并发请求、长短输入混合 | 能解释一个请求为什么分成多轮 prefill、为什么被等待或抢占 | 一张请求生命周期图和一次调度记录 |
| ④ 建立分析闭环 | 无 profiler 的容量基线；一次短 profiling；关联客户端、引擎、CPU/NPU 与 peer ranks | 对一个变慢现象提出可证伪假设，用证据缩小到一个模块 | 一份瓶颈定位案例与实验卡 |
| ⑤ 改代码与算子 | 先选重复元数据处理、CPU 同步、小算子融合等证据充分的问题；再考虑 AscendC/Triton | 通过功能/精度验证；microbenchmark 有收益；端到端确认收益或解释无收益 | 一个可复现 patch、基准、精度报告、回滚方法 |
| ⑥ 服务化与赋能 | 真实 Agent 流量回放、容量曲线、P/D 配比、尾延迟、故障恢复和版本升级 | 以业务 SLO 和 goodput 交付；别人可以用文档复现实验 | 运维手册、容量报告、教学案例与版本化知识库 |

### 2.1 前十个实验，按这个顺序做

1. **单请求、无 profiler。** 固定输入和输出，保存逐次流式到达时间及实际 token 数。
2. **输入长度扫描。** 例如 1K/8K/32K，输出固定 128；这些是教学采样点，不是生产分布。
3. **输出长度扫描。** 输入固定，输出例如 128/512；观察 TTFT 与后续生成时间分别如何变化。
4. **并发扫描。** 例如 1/2/4/8，找到 batching 变好以及排队开始恶化的区间。
5. **到达率扫描。** 以固定/随机到达间隔施压，观察等待队列是否持续增长，找到 SLO 下的容量。
6. **缓存实验。** 无复用、同前缀复用、真实命中分布分组测试；分别记录命中 token 和实际计算量。
7. **单个稳态前向的 trace。** 从 CPU 排队与下发一直看到 NPU stream，再定位模型层。
8. **SWA 机制验证。** 在可支持的固定配置下对比 OFF/ON，检查后段层 shape、准确性和请求级指标。
9. **多 rank 等待实验。** 找一处 collective，比较最早/最晚到达的 rank，识别等待来自哪里。
10. **一个小改动闭环。** 把已证明多余的工作移除或融合，验证局部与端到端收益，并写成案例。

前六项的性能数据在关闭 profiler 时采集。第七项以后用短 trace 解释机制。最终性能结论仍回到关闭 profiler 的基线。

## 3. 从哪里读代码：固定到 0.32 的阅读路线 {#code-tour}

vLLM 提供通用服务、调度和执行框架；vllm-ascend 是平台插件，注册 Ascend 平台、设备执行、模型和算子适配；CANN/torch_npu 承接设备运行时与计算。两个 Python 包需要配套，单独重装插件不会自动把旧 vLLM 主包升级到目标版本。

推荐阅读顺序是一次调用的先后顺序，而不是把目录逐个读完。每处先回答：输入是什么，输出是什么，持有哪些跨 step 状态，在哪儿异步，失败如何处理。

| 顺序 | 仓库与入口 | 阅读目标 |
| --- | --- | --- |
| 1 | [vLLM API](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/entrypoints/openai/api_server.py)；[Chat serving](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/entrypoints/openai/chat_completion/serving.py) | 请求校验、模板、流式输出与引擎接口 |
| 2 | [AsyncLLM](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/async_llm.py)、[EngineCore](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/core.py) | 请求如何进入 engine；`step()` 如何连起调度、执行与状态更新 |
| 3 | [Scheduler](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py) 的 `schedule()`、`update_from_output()` | token budget、等待与运行队列、chunk、抢占、完成条件 |
| 4 | [KVCacheManager](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_manager.py)、[BlockPool](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/block_pool.py)、[KV spec](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_cache_interface.py) | 逻辑 token 与物理块、引用计数、缓存组、前缀命中、分配/回收 |
| 5 | [MultiprocExecutor](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/executor/multiproc_executor.py)、[vLLM MRV2](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/model_runner.py) | RPC、每轮输入、持久请求状态、采样与设备异步执行 |
| 6 | [Ascend 平台](https://github.com/gg820827047/vllm-ascend/blob/d0128bcd099125d6a6bdbf9cd75799c2c5ed4a80/vllm_ascend/platform.py)、[MRV2](https://github.com/gg820827047/vllm-ascend/blob/d0128bcd099125d6a6bdbf9cd75799c2c5ed4a80/vllm_ascend/worker/v2/model_runner.py)、[model state](https://github.com/gg820827047/vllm-ascend/blob/d0128bcd099125d6a6bdbf9cd75799c2c5ed4a80/vllm_ascend/worker/v2/model_states/default.py) | 平台选项、`prepare_inputs()`、block table/slot mapping、图模式和 replay 状态 |
| 7 | [V4.1 模型层](https://github.com/gg820827047/vllm-ascend/blob/d0128bcd099125d6a6bdbf9cd75799c2c5ed4a80/vllm_ascend/models/deepseek_v41/model.py)、[DSA backend](https://github.com/gg820827047/vllm-ascend/blob/d0128bcd099125d6a6bdbf9cd75799c2c5ed4a80/vllm_ascend/attention/dsa_v41.py)、[fused MoE](https://github.com/gg820827047/vllm-ascend/tree/d0128bcd099125d6a6bdbf9cd75799c2c5ed4a80/vllm_ascend/ops/fused_moe) | 对照实际模型 config，识别 Attention/Indexer/MoE/Engram 的状态和调用 |
| 8 | [SparseFlashMla 调用与算子源码](https://github.com/gg820827047/vllm-ascend/tree/d0128bcd099125d6a6bdbf9cd75799c2c5ed4a80/csrc/attention/sparse_flash_mla) | Python/torch op → host tiling → kernel；最后深入热点的搬运与流水 |
| 9（按需） | [Hybrid connector](https://github.com/gg820827047/vllm-ascend/blob/d0128bcd099125d6a6bdbf9cd75799c2c5ed4a80/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py)、[MultiConnector](https://github.com/gg820827047/vllm-ascend/blob/d0128bcd099125d6a6bdbf9cd75799c2c5ed4a80/vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py)、[Store](https://github.com/gg820827047/vllm-ascend/blob/d0128bcd099125d6a6bdbf9cd75799c2c5ed4a80/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py) | KV 的所有权、缓存组、异步完成、失败回退；先理解接口合同 |

MRV2 的关键阅读材料：[Model Runner V2 设计](https://docs.vllm.ai/en/stable/design/model_runner_v2/)。其中持久请求状态与每轮输入分离的思想，能帮助理解异步调度和元数据维护成本；实现细节仍以固定提交为准。

多卡阅读先画实际通信组。例如 4DP×4TP 并开启 EP 时，DP 管理不同请求批次，而专家通信通常跨这些 rank；不能直接当成四份完全隔离的服务副本。某个 DP 没有业务 token，也可能需要参与专家通信。用启动配置和 profiler 的 group metadata 核实实际组成员，再判断谁在等谁。

### 3.1 每次读代码都留下这六项

1. 请求 ID / sequence 状态是在哪个对象里保存的？
2. 本轮新处理 token 数、历史长度、padding 后长度分别是多少？
3. block table 把哪段逻辑位置映射到哪个缓存组和物理块？
4. 一个函数返回后，NPU 是否已经完成，还是只完成下发？
5. 当前优化的进入条件、回退条件、实际选中路径是什么？
6. 中途取消、抢占、加载失败时，状态由谁清理或重建？

### 3.2 网关和 Mooncake，先理解到边界即可

网关至少要知道：路由是否保持会话/缓存亲和，是否排队和限流，流式输出有没有被缓冲，取消是否传到引擎，重试是否放大流量。你不必先实现完整网关，但需要拿到请求 ID 和各阶段时间。

Mooncake 先区分两层：**Transfer Engine** 搬运数据；**Store** 管理可复用对象、放置与回收。Master 管理元数据/对象位置，KV 数据分布在提供存储资源的节点上，不能把 Master 理解成集中保存所有 KV 的内存池。具体组件部署和元数据服务关系随版本变化，以安装版本为准。[项目说明](https://github.com/kvcache-ai/Mooncake)、[Store 设计](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/store/mooncake-store.md)。

MultiConnector 是多个 connector 的组合；仅有 Hybrid 可以用于 P/D 数据交接，加入 Store 才增加相应缓存池能力。还需要核对实际请求选中了哪个来源、恢复了哪些缓存组、什么时候可以安全继续计算。

## 4. 服务目标：把 2000 日活转换成可测负载 {#service-target}

**2000 日活无法直接换算卡数。** 一个用户一次 Agent 任务可能产生多轮 LLM 调用、工具调用和重试；上下文可能逐轮增长；同一请求还可能有并行分支。需要测量的是请求到达过程、token 分布、缓存状态和目标服务质量。

先填写 `service_profile_template.json`。至少区分下面三类业务，而不是把它们合并成一个平均数：

| 业务类型 | 典型特征（须用实际数据确认） | 重点指标 |
| --- | --- | --- |
| 交互式代码问答/补全 | 用户等待反馈，可能较短输入与输出 | TTFT、ITL 尾部、成功率 |
| 长上下文 Agent | 多轮仓库/工具结果进入 prompt，上下文增长且可能复用前缀 | 分长度的 TTFT、命中率、实际 prefill 工作量、整体任务完成时间 |
| 后台批量任务 | 吞吐和成本重要，可以采用不同优先级 | SLO 下的吞吐、完成时间与资源成本 |

### 4.1 一个容量换算示例

以下全部是教学假设，不是你的实际负载或推荐配置：

```text
2000 用户 × 每人每天 10 个任务 × 每任务 6 次 LLM 调用
= 每天 120000 次 LLM 请求

若集中在 10 小时：平均到达率约 3.33 req/s
若实测高峰是均值的 5 倍：高峰约 16.7 req/s
若该稳定负载下，单次请求平均驻留 15 秒：
平均系统内请求数 L ≈ λ × W ≈ 250
```

`L = λW` 是稳定系统中均值关系。这里算的是排队加运行中的平均请求数，不是 250 个用户，也不是 batch size。不能把 p99 时延塞进公式推导 p99 并发；已经过载、队列持续增长时也不能据此宣称系统稳定。

进一步用实际分布估算：输出 token 需求约为 `到达率 × 平均实际输出 token 数`；prefill 需求需要考虑命中后仍需计算的输入、窗口重建等实际工作量。跨实例传输和模型结构会影响时间，不能仅按“token 总数”线性预估所有成本。

最终以负载扫描得到的容量曲线确定部署：横轴为实际到达率，纵轴分别画 p95/p99 TTFT、TPOT、ITL、队列、错误/超时和 goodput。达到业务 SLO 且队列不持续增长的区间，才是可交付容量。为突发、维护或故障保留多少余量，需要结合部署目标测量。

同时建立内存预算：`权重 + KV + activation + workspace + 图相关内存 + 运行时预留`。MoE 的激活参数量影响计算量，驻留权重仍由实际模型放置和精度决定；不能用“每 token 只激活多少参数”估算全部权重显存。V4.1 有多种缓存组、共享和压缩布局，KV 预算应依据实际 spec 与分配结果计算。`max_model_len` 是请求长度上限，不等于每个请求已经占用了同样长的 KV。

### 4.2 指标字典：必须先统一测量边界

| 指标 | 定义与解释 | 常见误读 |
| --- | --- | --- |
| 客户端 TTFT | 发起请求到收到首个有效内容输出的时间；包括网络、排队及链路开销 | 把一个 NPU prefill kernel 时长当成 TTFT |
| 请求级 TPOT | 通常为 `(最后内容到达时间 - 首内容到达时间) / (实际输出 token 数 - 1)`；输出 ≤1 个 token 时不宜作通常 TPOT 对比 | 使用配置中的 max_tokens 作为实际输出数；把结束标记当最后 token |
| ITL / inter-output latency | 相邻流式输出事件之间的时间间隔 | 投机解码或流式合包时，一个事件可能包含多个 token；平均 ITL 与请求级 TPOT 不一定相等 |
| E2E latency | 请求发起到响应完成；注意是否包含最终 usage/结束事件 | 与 Agent 整个任务时长混为一谈 |
| queue time | 引擎或网关中对应边界的等待时间 | 把多个层次的排队时间相加时重复统计 |
| throughput | req/s、输入/输出 tok/s；分别列出真实完成量 | 把总 tokens/s 作为交互体验的唯一目标 |
| goodput | 每秒满足约定时延和成功条件的请求数 | 总吞吐上升而 p99 和成功率变差，仍声称扩容成功 |
| Agent 任务时延 | 多次模型与工具调用构成的整个任务关键路径 | 单次请求快 10%，不意味着 Agent 任务也快 10% |

指标实现和统计权重可随版本变化；Prometheus 与压测工具也可能对单 token 响应采用不同处理。先核对安装版本的实际输出。[生产指标](https://docs.vllm.ai/en/latest/usage/metrics/)、[延迟指标语义](https://docs.vllm.ai/en/latest/design/metrics/)。

至少看这些引擎指标：`num_requests_running`、`num_requests_waiting`、KV 使用量、prefix 命中、抢占次数、queue/prefill/decode 时间及首 token、输出间隔直方图。具体指标名以运行服务的 `/metrics` 和固定源码为准，不能从“没有这个字段”直接推断“没有这个行为”。

## 5. 性能提升从哪里拆：先确定慢在什么层 {#optimization-map}

先把 TTFT 分解为前端/网络、排队、prefill、必要的 KV 交接与恢复、首输出路径；把后续输出间隔分解为调度等待、模型执行、通信/同步、状态与采样、返回链路。它们可能重叠，不能用所有函数的 inclusive duration 直接相加。

| 拆分点 | 要采什么证据 | 可能的改进方向 | 必须防止的副作用 |
| --- | --- | --- | --- |
| 排队/容量与业务分级 | 到达率、运行/等待队列、各长度段 p95/p99、取消/重试 | 准入、请求分类、并发/预算、实例数与 P/D 配比 | 用吞吐换掉交互 SLO；限流/拒绝被排除出统计 |
| Prefill 工作量 | 实际未缓存 token、chunk 数、层级 shape、Indexer/Attention/MoE 时长 | 前缀复用、chunk budget、符合条件的 SWA replay、合适的并行 | 命中不真实；恢复窗口错误；长 prompt 干扰 decode |
| Decode 每步成本 | 活跃请求数、上下文长度、权重/KV 读取、图命中、接受 token 数 | batching、量化、图重放、Attention 内存访问、投机解码 | batch 变大但 ITL 恶化；投机接受率低或 draft 成本过高 |
| CPU 与元数据 | CPU self time、下发间隙、block table/slot mapping、H2D、同步 API | 预分配、增量更新、减少对象与复制、异步调度、避免不必要的设备同步 | 生命周期和竞态；把本来能重叠的工作串行化 |
| MoE 和通信 | Router 分布、各专家 token 数、各 rank 到达时间、collective wait/transit | EP/TP 组合、负载均衡、融合分发合并、通信拓扑与重叠 | 等待来自 peer 计算，却只调网络；稀疏负载下融合退化 |
| KV 容量与恢复 | 缓存组、使用量、块分配、抢占/重算、传输完成与失败 | 并发/长度预算、KV dtype、复用、分层缓存、分配与恢复策略 | 只看占用率；混淆不同 cache layout；覆盖共享块 |
| 算子内部效率 | 同 shape/dtype 的核耗时、Cube/Vector/MTE、实际访存、tiling 与尾块 | 数据复用、融合、分块、流水、拷贝与计算重叠、必要时新算子 | 局部更快但增加全局同步/显存；只优化一个 shape |
| P/D 与 Store 交接 | 请求 ID、P 完成、send/load 完成、D 可运行、缓存来源、链路时间 | 传输粒度与并行、恢复衔接、减少复制、匹配 P/D 容量 | 仅有 P trace；异步未完成就使用 KV；缓存池与 P/D 传输混淆 |

不是所有问题都值得写 kernel。优先处理可证明的多余工作、排队或同步，再定位高价值的关键路径算子。

### 5.1 识别不同的 bound：证据、证伪与行动

**bound 是对“某个负载、某个配置、某段关键路径”的判断。** 一个服务可以同时有通信等待、CPU 下发和内存瓶颈；随着 batch 和长度变化，主导因素也会改变。

先建立直觉：长 prefill 一轮处理很多 token，矩阵的 token 维通常更大，更容易复用权重和利用矩阵计算单元；小 batch decode 每次只推进少量 token，却仍需读取权重和历史状态，常见访存、下发或通信开销占比更高。这些只是候选方向。batch、上下文、MoE、稀疏选择、量化和投机会改变结论，不能直接给所有 prefill 标成 compute-bound、所有 decode 标成 memory-bound。

| 候选瓶颈 | 支持证据 | 需要排除什么 / 做什么对照 | 对应动作 |
| --- | --- | --- | --- |
| 计算吞吐限制 | 同精度、同形状下计算单元持续繁忙，实测运算吞吐接近匹配的可持续上限 | 核内时间是否其实在搬运/同步？小 shape 是否利用不足？ | 合适的计算 kernel/tiling、量化、减少计算量 |
| 内存带宽限制 | 实际 HBM 读写带宽接近可持续上限，且时间随需搬运数据量变化 | 数据是否来自缓存？带宽计数口径和单/多 die 是否一致？ | 减少字节、提高复用、融合、优化 KV 读取 |
| 访存延迟/访问低效 | 带宽没有吃满，但不规则访问、依赖链、小任务或搬运 stall 明显 | 不是 CPU 未及时供给任务，也不是跨 rank 等待 | 合并访问、预取、调整数据布局、提高并行度 |
| 通信限制 | 关键路径存在未被覆盖的通信；结合参与 rank、消息大小、wait/transit 分析 | 大部分时间是否在等晚到 rank，或者 EP 负载失衡？ | 调整拓扑/并行、消息组织、重叠、负载均衡 |
| CPU / launch 限制 | 有就绪工作但 NPU 出现空洞，CPU 调度/构造输入/下发在关键路径 | 没有请求、流控、采集导出、其他 stream 正在工作？ | 优化输入准备、减少同步、图模式与异步执行 |
| KV/内存容量限制 | 高负载下分配受阻、抢占、重算或 OOM；与驻留 token 分布相关 | 高显存占用不一定意味着慢；先看真实抢占和重新计算 | 缩小驻留预算、布局/精度、复用/分层、扩容 |
| 排队/负载限制 | 到达率超过持续完成能力，等待队列增长，p95/p99 恶化 | 单请求慢还是仅负载升高后慢？压测端是否也堵塞？ | 容量/准入与调度，必要时优化当前饱和资源 |
| I/O / 缓存恢复限制 | H2D/远端加载/存储访问处于依赖路径，D 等待 KV 就绪 | 异步是否真正重叠？加载失败是否触发重算？ | 传输与恢复优化、减少复制和等待 |

**NPU 利用率低不能直接推出 memory-bound；显存使用率高也不是 HBM 带宽利用率高。** `Computing` 分类可能包含融合通信等待；一个 `MegaMoE` 长任务不能直接证明矩阵乘法忙了同样长时间。

### 5.2 Roofline：用一个模型建立计算与访存直觉

对一个确定的 kernel、shape、dtype 和存储层级，估算或测量：

```text
F = 运算量（例如常规矩阵乘约为 2 × M × N × K）
B = 实际通过目标存储层级搬运的字节数
I = F / B                         # 算术强度，FLOPs/byte
P_observed = F / kernel_time
BW_observed = B / kernel_time

P_roofline = min(P_compute, BW_sustained × I)
T_lower_bound = max(F / P_compute, B / BW_sustained)
```

`B` 需要计入缓存效果及真实读写；仅用张量大小做的估算不等于实测 HBM 流量。算力上限要与数据类型、实际执行单元、设备范围和运算口径匹配。不能拿 INT8 峰值衡量 BF16 kernel，也不能把单 rank 的计数与整机峰值混用。多种流水和通信融合的 kernel，单一 Roofline 只是起点。

教学例子：假设可持续算力 100 TFLOP/s、HBM 带宽 2 TB/s；某核有 20 GFLOPs、搬运 2 GB，则 `I=10`，带宽侧上界为 20 TFLOP/s。如果它约用 1 ms，实测接近 2 TB/s，支持带宽限制判断；若它用了 4 ms，算力和带宽都远未到顶，需要继续查访问效率、任务粒度、流水依赖与等待。以上数字不代表 A3 规格。

若要判断核内 bound，需要针对性的硬件计数器或单算子分析；普通 timeline 不能直接提供完整 Roofline 输入。参考：[Roofline 原理](https://crd.lbl.gov/assets/Uploads/ECP21-Roofline-1-intro.pdf)、[Ascend 单算子分析](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/optool/atlasopdev_16_0082.html)。工具界面和支持指标必须匹配现场 CANN/设备版本。

## 6. 怎么采集和分析 profiling：一套固定动作 {#profiling-workflow}

### 6.1 第零步：先保存“实验身份证”

每次保存：镜像名与 digest、vLLM/Ascend Git SHA、实际导入路径、模型/tokenizer/config 版本、权重/KV/激活精度、CANN/torch/torch_npu/Mooncake 版本、NPU 型号、rank 与设备映射、TP/DP/EP/PP/PCP/DCP、环境变量、完整启动参数，以及负载、缓存状态、时钟、采集窗口。

尤其记录开关的**实际生效状态**：MRV1/MRV2、SWA replay、FlashComm1/SP、图模式、static kernel、MC2、投机与 drafter。命令行写了一个值，不代表所有 batch 都执行同一条路径。

### 6.2 第一步：关闭 profiler，建立服务基线

先测低并发的服务下限，再测并发和到达率曲线。按输入长度、输出长度、命中状态分别统计 p50/p95/p99，不只保存平均值。对实际 Agent 至少保存分阶段请求与工具耗时，找到任务关键路径。

下面是用于理解机制的单机合成示例；模型名、地址、tokenizer 路径需要对应现场服务。它只压测已运行的服务，不修改服务的 SWA 或 prefix cache 配置。

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8989 \
  --endpoint /v1/completions \
  --model dsv41 \
  --tokenizer /data/models/DeepSeek-V4.1-Flash-1017-w8a8 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 128 \
  --random-range-ratio 0 \
  --ignore-eos \
  --num-prompts 32 \
  --max-concurrency 1 \
  --request-rate inf \
  --seed 42 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --save-result --save-detailed \
  --result-dir /workspace/bench/c1
```

这里请求率为无限、并发限制为 1，形成持续单并发测量；32 条请求只适合起步定位，不能支撑可靠的生产 p99 结论。`--ignore-eos` 与长度设置用于控制合成实验，仍应核对实际计数。改变 `--max-concurrency` 做并发扫描；做容量实验时再改变 `--request-rate`，并检查客户端真实发送率和引擎真实到达率。

有限并发的闭环压测在服务变慢时会自动降低发送速度，可能掩盖过载；容量分析还需要带到达率控制的实验。压测端并发上限过低也可能让实际到达率达不到目标值。参数以当前安装版本的 `vllm bench serve --help` 为准；合成 token 用于形状和机制实验，真实代码与 Agent 回放用于生产验收。

缓存实验要明确 OFF/ON：测未缓存 prefill 时固定关闭 prefix cache，或使用无复用输入并核对实际命中；测生产命中时按真实复用分布建组。相同 seed 可使实验可复现，但跨轮同输入可能命中先前缓存；warmup 不计入正式统计，仍可能改变图、编译与缓存状态。

### 6.3 第二步：只采一段有明确问题的短窗口

启动服务时，在现有 `vllm serve` 命令中加入：

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/workspace/profiles/run-001","torch_profiler_with_stack":false}'
```

先完成模型加载、图捕获与必要 warmup，再开始稳态负载。然后在独立终端控制短采集：

```bash
curl --fail -X POST http://127.0.0.1:8989/start_profile
# 让已经准备好的、可复现的负载继续运行，覆盖目标 batch / step。
curl --fail -X POST http://127.0.0.1:8989/stop_profile
```

首次学习可以只覆盖少量稳态迭代；问题低频发生时，用更轻的监控先找触发窗口，再做针对性采集。Python 调用栈、内存等详细信息按需要逐项增加，不要一次打开所有高开销功能。

P/D 场景需要分别配置 P 和 D，并控制各实例的采集入口；确认输出包含相关 DP/TP/EP peer rank。普通业务代理不一定转发 profiler 控制接口。需要关联 request ID、角色、rank、时间窗口；跨机器时间线要校准时钟，不能直接相减未经验证的时间戳。[Ascend 采集指南](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/performance_and_debug/service_profiling_guide.html)。

### 6.4 第三步：按固定顺序看文件

| 文件/视图 | 看什么 | 能回答什么 |
| --- | --- | --- |
| 客户端结果 + `/metrics` | 目标请求、队列、各时延分布、成功/取消/重试 | 用户慢是否与引擎排队和执行对应？ |
| `step_trace_time.csv` | 计算、未覆盖通信与 Free 等时间覆盖 | 从整体上先找空洞、长等待、异常窗口 |
| `trace_view.json` | CPU、CANN、NPU stream、通信和 peer 的依赖关系 | 请求经过的路径在等谁？哪些工作实际重叠？ |
| `kernel_details.csv` | 类型、起止、shape、dtype、stream、core | 是哪个形状、哪个阶段的任务变慢？ |
| `operator_details.csv` / `op_statistic.csv` | self/inclusive、计数、最大值与分位数 | 热点候选来自哪段调用，是否重复统计？ |
| `communication.json` / matrix | collective、消息大小、wait/transit 与参与 rank | 是传输慢、启动延迟，还是 peer 晚到？ |
| profiler DB / 元数据 | 通信组、rank、拓扑与采集信息 | 目录名字是否真实反映部署规模？ |
| 针对性单算子数据 | 各计算/搬运流水、实际流量、核心负载、源代码热点 | 为什么这个 kernel 还没接近适合它的上界？ |

不是每个版本都导出相同文件或列；缺少某项数据就保留“待验证”，不要补猜。当前迁移分支的 NPU profiler wrapper 使用 Level1 和 PipeUtilization；这不意味着默认 trace 已包含每个 Roofline 指标，细节需检查实际导出和额外采集能力。

### 6.5 第四步：打开 MindStudio Insight 后的十个动作

1. 先定位数据集、P/D、rank、模型版本和采集起止，确认采到了稳态业务。
2. 缩放看整体：初始化、有效计算、长空洞、长通信、采集停止/导出分别在哪里？
3. 选一个可以解释的 batch / 前向区间，记录实际请求数、新 token 数、历史长度和 padding。
4. 从 CPU 调用及关联关系向下看 CANN 和 NPU，区别“下发结束”与“设备完成”。
5. 沿模型层找到 Attention、MoE/Engram 等区域，核对 shape，而非仅搜名字。
6. 点最长的关键路径事件，找它的前驱、后继；若它被别的工作覆盖，先算未覆盖部分。
7. 若涉及 collective，查看同一通信组其他 rank 的到达与完成，排查 straggler。
8. 对同一 shape/dtype/角色/阶段做聚合；首轮、warmup、padding、draft 单列。
9. 定位对应源码和优化门槛，写一个可证伪的原因假设。
10. 做受控对照；最终关闭 profiler 重测，检查精度、SLO、goodput 和失败路径。

### 6.6 目前这些现象分别能证明什么

| 现象 | 已经支持的判断 | 还不能直接证明 |
| --- | --- | --- |
| 没有独立 Softmax，出现 SparseFlashMla | 可能由融合 Attention 内部完成；需用实现交叉确认 | Softmax 被省略；单独 Softmax 耗时为零 |
| 后段层的行数从 8188 变为 128 | 相应层执行了尾窗裁剪，实际计算形状发生变化 | 同等百分比的 TTFT/TPOT 提速 |
| MegaMoE 单次约 17.87 秒，Gloo 长等待重叠 | 存在同步/等待候选，需要查看 peer、采集边界和调用顺序 | 专家 GEMM 真的计算了 17 秒；网络就是根因 |
| 看到三个重复主干层序列 | 采集到三组可识别的模型前向 | 一定是三个用户请求或三个输出 token |
| 出现很小的 shape | 可能是尾窗、decode/draft、padding、warmup 或某个特定分支 | 它必然是 warmup |
| `SWA bounded replay` 开关默认 true | 配置默认请求启用该能力 | 当前并行和图模式下后段裁剪必然生效 |

这些案例来自你之前的 0.30/0.32 数据分析；它们是学习材料，不是对当前服务的重新测量。原报告位于同一仓库的 `reports/dsv41_profiling_30_32/`。

当前固定基线的 [decoder replay 进入条件](https://github.com/gg820827047/vllm-ascend/blob/d0128bcd099125d6a6bdbf9cd75799c2c5ed4a80/vllm_ascend/models/deepseek_v41/decoder_replay_layers.py) 会在 SP、PP>1、DCP>1 或 microbatching 下禁用后段层尾窗裁剪；图模式只接受 NONE 或 FULL_DECODE_ONLY。你此前配置的 FlashComm1 在 TP/EP 条件下会开启 SP，因此默认 replay 开关为 true 也不保证裁剪生效。做 replay OFF/ON 对照时，两组需要采用相同的、支持裁剪的并行与图配置；不能用“FlashComm1 ON、replay OFF”与“FlashComm1 OFF、replay ON”两组结果直接计算 replay 收益。这条限制针对当前后段裁剪实现，不表示模型滑动窗口 Attention 被关闭。

## 7. 怎么动手优化：先形成证据，再改实现 {#change-workflow}

### 7.1 一张实验卡应该有的内容

直接复制 `experiment_template.md`。一张卡只研究一个主问题：

```text
现象：在哪种输入/并发/命中状态下，哪个指标超过什么目标？
边界：客户端、engine、rank、step 和 kernel 的时间如何关联？
假设：哪项工作导致关键路径变长？什么结果能够证伪？
证据：trace 区间 + shape/dtype + peer 状态 + 源码条件。
改动：只改变哪个因素，其他因素如何固定？
正确性：输出/数值、缓存恢复、失败/取消/抢占是否仍然正确？
性能：局部、阶段、请求和 SLO goodput 分别变化多少？
结论：确认、部分支持、证伪或数据不足；下一步是什么？
```

至少重复多个独立运行，比较分布和波动；变化小于自然波动时不能确认收益。交错执行 A/B、控制缓存与热身状态，有助于减少时间漂移。高负载下先确认真实完成量、丢弃/超时和发送率一致。

### 7.2 优化优先级与收益上限

假设一个模块占请求关键路径的比例为 `f`，被加速 `s` 倍，并且其他工作和重叠关系不变，则：

```text
整体加速比 = 1 / ((1 - f) + f / s)
```

例如模块占 40%，自身加速 2 倍，整体最多约快到 1.25 倍，时延降低约 20%。这个 `f` 必须来自未重复统计的关键路径，不能拿多 stream 算子累加占比替代。排队系统、流水重叠和共享资源变化会让实际结果偏离这个简化模型。

### 7.3 什么时候值得写自定义算子

当你已经证明：热点位于有足够权重的关键路径；现有实现有可解释的多余搬运/调度/计算；数据布局和数学合同清楚；框架层调整无法解决或收益不足。优先从可独立验证的小核入手，逐步进入复杂 Attention/MoE。

| 层次 | 可以尝试什么 | 先验证什么 |
| --- | --- | --- |
| Python / 引擎 | 增量元数据、预分配、减少重复构造、去掉不必要同步 | 请求生命周期、异步依赖、抢占和取消 |
| torch / 图层 | 复用 buffer、减少临时张量、合并小操作、改善图命中 | shape/dtype/alias、图捕获与 eager 回退 |
| Triton-Ascend / AscendC | 数据布局、归约/索引/融合、tiling、搬运流水 | 数学正确性、边界/尾块、动态形状、目标硬件支持 |
| 高复杂度融合 | Attention 或通信计算融合 | 多 rank 合同、不同长度/batch、量化、并行布局与失败恢复 |

算子学习至少掌握：HBM 与片上存储、Cube/Vector、数据搬运与同步、tiling、对齐和尾块、流水与复用。先能解释一段已存在的 kernel 如何搬运和计算，再修改它。AscendC 与 Triton 的取舍按设备支持、热点性质、现有实现和团队维护能力决定。

microbenchmark 应复刻真实 shape、dtype、stride、布局、padding 与调用方式；warmup 后重复测量，使用正确的设备计时/同步边界，避免只测到异步下发。最后必须回到真实服务验证，单核更快不保证整体更快。

### 7.4 两道验收：准确性与服务收益

准确性：小张量与参考实现、目标精度误差、真实模型的任务级评价；涉及缓存时还要覆盖 prefix 命中/不命中、异步恢复完成、加载失败、取消、抢占与重调度；涉及并行时覆盖实际 TP/EP 和不同 rank 的 token 分布。模型生成可能非确定，不能只用长输出逐字一致来判断数值正确性。

性能：核级延迟 → step/阶段时间 → 客户端 TTFT/TPOT/ITL → SLO goodput 与资源成本。记录没有改善或退化的配置。适配完成、路径启用、局部提速、端到端提速，是四个不同结论。

## 8. 如何变成可复用的学习资产 {#assets}

建议每次只增加一个可以复现的知识单元，而不是写一篇堆概念的长总结。统一采用“问题 → 最小背景 → 实验 → 证据 → 结论 → 适用范围 → 复现与下一步”的结构。

| 资产 | 必须包含什么 | 新人如何使用 |
| --- | --- | --- |
| 概念卡 | 一句话定义、所在执行层、一个例子、一个常见误读、来源与版本 | 遇到名词时能放回执行链路 |
| 源码导读 | 固定 SHA、调用入口、状态、关键函数、进入/回退条件 | 跟一个请求走完整条路径 |
| 实验卡 | 环境、输入、控制变量、步骤、原始数据和结论 | 同事能在自己的机器复现 |
| 瓶颈案例 | 症状、排查顺序、peer/shape/关键路径、证伪过程 | 学会分析方法，而不是背一个参数 |
| 优化报告 | 修改、精度、局部与端到端收益、适用形状、限制、回滚 | 判断能否进入自己的服务 |
| 运维与容量手册 | 部署清单、SLO、容量曲线、告警、故障演练与升级验证 | 维持服务并识别退化 |

证据标记建议统一为：**已观察**（日志/trace 出现）、**机制已验证**（源码与输入输出/shape 对得上）、**收益已验证**（受控端到端测量）、**假设待验证**、**不适用/被证伪**。一条 PR 被归类为性能优化，并不意味着它在当前模型与拓扑上产生了收益。

面向“2000+ 日活”的交付，至少还需要能复现的镜像和模型版本、基础监控、超时/取消、健康检查与回滚、合理的队列和准入策略，以及峰值和部分故障下的容量验证。你可以先掌握与引擎的接口和证据边界，再按实际问题深入外围系统。

### 8.1 你当前最值得先交付的三项

1. **一份固定环境的单机性能基线。** 分输入长度与并发，记录客户端指标和实际运行配置；先去掉 profiler 的干扰。
2. **一张模型层、trace 名称与源码的对应图。** 用已有数据把 HcPre、SparseFlashMla、Indexer、MoE、Engram、draft/verify 放回模型结构。
3. **一份可复现的 SWA 或长等待案例。** 保持 SP/图模式等条件一致，先证明路径与工作量，再验证客户端时延；把缺少的证据明确列出。

这三项完成后，你就拥有部署基准、执行地图和定位方法，后续每次优化都能复用它们。

## 9. 材料、模板与版本说明 {#references}

阅读顺序建议：先本手册的执行链路和指标，再看官方架构与 MRV2，然后带着一次具体实验读采集和算子分析文档。

| 材料 | 用途 | 版本提醒 |
| --- | --- | --- |
| [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/) | API、EngineCore、Scheduler 与执行器 | main/latest 会变化；实现以固定 SHA 为准 |
| [Model Runner V2](https://docs.vllm.ai/en/stable/design/model_runner_v2/) | 请求状态与每轮输入、CPU 开销的来源 | 文档与本地实现交叉阅读 |
| [Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/) / [Metrics Design](https://docs.vllm.ai/en/latest/design/metrics/) | 监控、TTFT/ITL/TPOT 口径 | 对照现场 `/metrics` 和压测工具 |
| [Ascend Service Profiling Guide](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/performance_and_debug/service_profiling_guide.html) | 在线采集、输出文件与 P/D 采集 | 参数、控制端点和工具需匹配安装版本 |
| [vLLM Profiling](https://docs.vllm.ai/en/latest/contributing/profiling/) | 分析器与开销概念 | NVIDIA 工具不能直接当作 NPU 工具使用 |
| [Ascend 单算子分析](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/optool/atlasopdev_16_0082.html) | 核内计数器、流水与 Roofline | 该链接为工具概念入口；现场按 CANN 版本选文档 |
| [MindStudio 算子调优入门](https://www.hiascend.com/document/detail/en/mindstudio/2610/visualization_tool/MindStudioInsight/docs/en/quick_start/operator_tuning_quick_start.md) | 实机和仿真、Timeline/Source 的联系 | 不假定现场已经安装此版本 |
| [Roofline 教学材料](https://crd.lbl.gov/assets/Uploads/ECP21-Roofline-1-intro.pdf) | 算术强度、不同存储层级与性能上界 | 原理可迁移，设备峰值必须重新标定 |
| [Mooncake](https://github.com/kvcache-ai/Mooncake) | Transfer Engine 与 Store 的边界 | 组件布局与 API 依现场版本 |

配套资产：`index.html` 是无需联网即可阅读和操作的页面；`experiment_template.md` 可复制填写实验；`service_profile_template.json` 用于补齐负载与 SLO；`manifest.json` 记录本手册的固定代码依据。外部链接需要联网，交互控件的示例参数不是硬件规格或实测结果。
