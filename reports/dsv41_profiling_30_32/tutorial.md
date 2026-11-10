# 从一次请求走到一条 kernel

你的目标不是把算子耗时榜背下来，而是能回答三件事：请求慢在哪一段，这一段在等谁，改动是否缩短了请求真正经过的路径。

## 1. 先建立三层视角

| 视角 | 主要对象 | 回答的问题 |
| --- | --- | --- |
| 请求 | 排队、P prefill、KV 传输、D decode、网络返回 | 用户的 TTFT、TPOT、ITL 为什么变化？ |
| 模型与调度 | 一个 batch、主干层、Attention、MoE、Engram、draft/verify | 本轮究竟处理多少请求、多少 token？ |
| 执行 | Python/CPU、CANN API、NPU stream、计算/通信任务 | 谁在执行，谁在等待，哪些任务真正重叠？ |

从上往下定位，再从下往上验证。你这批是 worker 级 profiling，能观察后两层，缺少代理/客户端的请求级链路，因此不能从一个 trace 的总长度直接算出 TTFT 或 TPOT。

一个便于定位的 P/D 请求路径是：代理路由 → P 排队/调度 → prompt prefill → KV 交接及 D 等待就绪 → D 的首步计算 → 首个 token 返回 → 后续 decode。实际部署可能有重算、窗口恢复、投机验证和缓存命中，需要结合请求 ID 判断哪些阶段发生过。P、D 不同机器的绝对时间戳还需要时钟对齐，不能直接拼接成本报告的端到端关键路径。

## 2. Prefill、decode 和一个 step 都是什么

**Prefill** 的新计算 token 数通常多；矩阵的 token 维较大。**Decode** 每个活跃请求通常只推进少量 token，但投机解码会一次验证多个位置，因此不能用“输入第一维大于 1”来判断它是 prefill。还要看本轮新 token 数、历史上下文长度、请求数和 draft/verify 元数据。

**调度 step** 是一次调度/模型执行的工作单元；一个请求可以跨很多个 step，一个 step 也可以包含多个请求。D 端可能有恢复或重算计算；它的具体作用要看数据来源和调度状态，不能仅凭出现了 prefill 型矩阵就判断 P/D 配置错误。

本次 `step_trace_time.csv` 的 Step 列为空，trace 也缺少清晰的 ProfilerStep 打点。报告中“3 组主干前向”来自重复的层序列，不是 3 个请求，也不是已知输出 token 数。

DeepSeek-V4.1 的一层并不只有 Attention。你会反复看到 mHC/HcPre → 归一化/投影 → DSA Attention/Indexer → O projection → mHC/HcPost → MoE，某些层还有 Engram。DSpark/draft 会额外前向；融合和图执行会改变可见的包装层，因此不同版本的 CPU 事件数量不能机械相减。

## 3. 先打开哪些文件

| 文件 | 第一次看什么 | 常见误读 |
| --- | --- | --- |
| `step_trace_time.csv` | Computing、未重叠通信、Free 的覆盖分布 | 把采集窗口当一次请求延迟；把 Computing 当纯矩阵计算 |
| `op_statistic.csv` | 类型计数、总时间、最大值，找主要记录 | 把多个 stream 的累加时间当墙钟时间 |
| `kernel_details.csv` | Type、开始/时长、shape、dtype、core、stream | 只按 Type 比平均值，忽略 shape 和通信父子记录 |
| `operator_details.csv` | Host Self/Total、Device Self/Total | 把父算子和子算子总时长再加一遍 |
| `trace_view.json` | CPU → CANN → NPU 的先后关系和空洞 | 把 CPU 返回当 NPU 已完成；把短 API 调用当低延迟 |
| `communication.json` | 同类 collective 的 elapsed/wait/transit 和长尾 | 看到 bandwidth=0 就说网络带宽为零 |
| `ascend_pytorch_profiler_*.db` | `META_DATA` 中的通信组成员 | 从只导出 DP0 的目录推断部署 DP=1 |

本次 DB 中保存了 TP、EP、MC2 组，这是拓扑的重要证据。vLLM 0.32 的 EP 组跨 DP×PCP×TP，因此 EP16/TP4 只能直接推出 DP×PCP=4；只有再确认 PCP=1，才能说 DP=4。第二批已经采到了 DP0、DP1 且 TP8/EP16，结合完整成员可确认这一侧是 DP2、PCP1。目录里的 dcp0 是 rank 坐标，不是 DCP size。

## 4. 用 MindStudio Insight 走一遍

本报告自带可缩放时间线，不用安装额外工具也能先练习。要查看完整的 CPU/CANN/NPU 对应关系，再用 MindStudio Insight 导入原始 trace。

先只取一个 rank，避免一次展开全部 8 GiB 数据：

```bash
cd /home/gengc/code/open_source/vllm/reports/dsv41_profiling_30_32
python extract_one.py --dataset v032_p_tp4 --dp 0 --tp 0 \
  --output /tmp/dsv41-p32-rank0-trace.json
```

在 MindStudio Insight 的导入入口选择此 JSON，进入 Timeline。菜单名称可能随版本变化，但要找的对象一致：

1. 先缩放到整段，看是否包含加载、首轮、长空闲和尾部采集停止。
2. 找到 Python、CANN 和 Ascend Hardware 的泳道。展开 NPU stream，点击事件，查看名称、起止时间、shape/core 和关联流。
3. 搜索 `vllm::dsa_v41_forward`，定位 Attention 包装；再看对应设备任务。CPU 包装的时长不等于设备任务的完成时间。
4. 搜索 `HcPre`，找主干重复序列；用 shape 识别 8188 → 128 的层边界。
5. 搜索 `MegaMoe` 和 `gloo:all_reduce`，对齐同一台机器不同 DP 的长等待区间。先问依赖和同步发生了什么，再考虑优化 kernel。
6. 对长空洞，向上看 CPU 是否正处于同步、元数据构建、分词/调度、导出分析或没有请求；向下看是否有通信/拷贝被另一条 stream 覆盖。

也可用 Perfetto 在本地浏览器打开此 JSON；它能展示通用时间线，Ascend 专属分析通常由 MindStudio Insight 提供。

## 5. 本次的四个练习

### 练习 A：证明 SWA 路径执行

在 11 月 7 日 0.32 的 P rank0，按开始时间排列 HcPre：每组主干前向前 21 层是 8188 行，后 19 层是 128 行。HcPre 每层调用两次，3 组前向得到 126 条大 shape 和 114 条小 shape。Attention 的 query shape 也能交叉检查。

这证明后半主干按尾窗口运行。固定这组 batch、假设其余逻辑不变，主干 layer-token 行数减少约 46.8%；这个数不是 FLOPs 减少比例，更不是 TTFT 提速比例。早期层、通信、索引、Engram、图/元数据及调度时间仍然存在。

这是模型特定的路径：全局 KV-source 层已经生成后续层复用的全局 KV，之后的层在尾窗子批次上执行。不能把这种裁剪直接推广到任意模型，也需要对输出质量做回归。

特别注意：**decoder 尾窗裁剪**与**prefix hit 后补算 SWA 窗口**是两条相关但不同的现象。这里确认了前者；没有请求级 cache-hit/replay-start，不能顺便宣称后者也已验证。

### 练习 B：证明 Engram 冗余清零消失

不要直接比较两版总 `ZerosLike` 数量。SWA 本身会产生其他输出清零，整个模型仍需要 zeros。

把每次 `_engram_host_uva_gather_dequant_kernel` 的准备区间单独取出。0.30 六个窗口里都能看到 `aten::contiguous → aten::zeros → lookup`；0.32 的六个对应窗口变成 `aten::contiguous → lookup`。再对照 #19657 中 `torch.zeros` 改为 `torch.empty` 的生产 diff，才有“调用点—代码—执行”三者对应的证据。

这确认一项机制生效；绝对收益还要在相同 token、local heads 和 TP/DP 交换策略下测量。

### 练习 C：17 秒 MegaMoE 是不是 GEMM 很慢

第二批 host11 的 DP1 有一条 17.8706 秒的 MegaMoe，CPU 的一次 Gloo all_reduce 为 17.9575 秒，时间上高度重叠。对端 DP0 还有长空闲。MegaMoE 融合了专家分发/计算/合并，记录时长可能包含等待其他 rank，并不等于 Tensor Core 持续计算了 17 秒。

排查优先级是：核对这些 rank 的时间范围、EP collective 顺序、每个 DP 的实际执行轮数、空闲 rank 的参与机制，以及 profiler 启停/导出是否阻塞了 peer。当前不能仅凭相邻事件判断谁先阻塞谁，也不能确认死锁。最有价值的下一次采集是所有 EP rank 同步进入稳定窗口，并保留采集开始/停止标记。

### 练习 D：平均值与首轮长尾

同一批 P rank0 的 SparseFlashMlaMetadata 有一条约 757 ms，而中位数约 234 μs。先定位这条异常出现在哪一轮，再排查首次初始化、排队/同步和采集影响。不能直接把平均值 58 ms 当稳态 metadata 成本，也不能把异常删掉后不在端到端结果中解释。

## 6. 指标怎样算才不会误导

```text
设备记录累加时间 = 各条记录 duration 之和
覆盖时间 = 所有 [start, end] 区间的并集长度
未重叠通信 = 通信覆盖中没有被计算覆盖的部分
请求 TPOT = (最后一个 token 时间 - 第一个 token 时间) / (输出 token 数 - 1)
```

父子 scope 的 inclusive 时间会重叠；多个 stream 的时间也会重叠。本次 CSV 里还同时包含 HCCL 逻辑 op 和底层 AIV/AICPU task。例如同一个 AllReduce 的 139.442 ms 会出现两条不同粒度的记录，不能相加成 278.884 ms 的请求开销。

TTFT 必须从请求入口到首 token 测量。ITL 是相邻输出事件的间隔，需要说明是服务端 token 时间、客户端 chunk 时间还是流式聚合后的间隔；一个网络 chunk 可能有多个 token。投机 step 的时长必须结合实际接受/输出的 token 数归一化。请求 TPOT 的均值与把所有请求合并后的加权均值也不是一个指标。

`Wait Time`、CPU 同步和 Free 都是线索，不是可以直接删掉的时间。正确的依赖保护数据完整性。优化目标是减少被请求关键路径暴露的等待，或者创造有用的计算重叠。

## 7. 一套可以复用的分析流程

| 顺序 | 动作 | 应留下的产物 |
| --- | --- | --- |
| 1 | 保存镜像 digest、两个仓 SHA、CANN/torch_npu/算子版本、启动参数 | 可复现的运行清单 |
| 2 | 固定请求、输入/输出长度、并发/请求率、缓存冷热与预热 | 客户端原始请求和结果 |
| 3 | 先测无 profiler 的 TTFT/TPOT/ITL、吞吐、成功率、接受率 | 真实端到端基线 |
| 4 | 所有相关 rank 在稳定阶段采短窗口，标记 prefill/decode/draft/verify | 可对齐的时间线 |
| 5 | 看覆盖与长尾，选最影响请求的一段 | 一段明确的关键路径候选 |
| 6 | 下钻到 CPU scope、NPU shape、通信 peer 与模型层 | 可以复现的瓶颈解释 |
| 7 | 核对 PR 的条件和实际命中路径，再做单变量 A/B | 机制证据和收益证据分开记录 |
| 8 | 关闭 profiler 复测，并做质量/稳定性回归 | 可以发布的优化结论 |

第一次不要同时开齐 record_shapes、with_stack、memory 和全部硬件计数器。先轻量采集稳定窗口；对定位到的短区间再开启 shapes/stack；要分析带宽或 Cube 利用率，再补所需的硬件指标。采集本身也会扰动 Python、同步与多 DP 步进。

## 8. 怎样验证一个 PR 的收益

先画出条件链：硬件 → 模型 → dtype/量化 → Runner → 图模式 → 并行策略 → flag → 目标调用点。条件链有一项不满足，就不能拿 PR 的数字解释当前数据。

本报告的 #16731 是通用 device adaptor QLI 替换，但 V4.1 trace 仍走自定义 QLI v3；#18786 改的是通用 SFA 的 V up-projection，而 V4.1 的同名 TransposeBatchMatMul 主要属于 DSA O-projection。**同名算子不等于同一个优化入口。**

同一个 Type 也可能有不同 shape、dtype、layout、量化方式和运行 phase。先做严格匹配，再看中位数、长尾、调用次数与总量。严格匹配不了，就把它当路径/工作量差异，不能计算“这个 kernel 快了百分之多少”。

理想证据依次是：配置条件满足 → 调用点/图确实命中 → 工作量或任务结构按预期变化 → 同条件耗时改善 → 无 profiler 的端到端指标改善 → 质量和失败率合格。本次能确认部分机制，但没有同条件单 PR A/B，不能把版本差异拆成各 PR 的独立百分比。

## 9. 下一轮建议先做什么

1. **先补齐可比数据**：0.30 的 D 侧、原始请求结果、实际启动配置和镜像 digest。相同用例不保证每个调度 batch 相同。
2. **先排除大停顿**：对第二批 P DP2/TP8，所有 EP rank 同步采集，标明采集启停。先解释 17 秒等待，再考虑十几微秒的算子优化。
3. **隔离 SWA**：在同一 0.32 镜像和受支持拓扑上只切 replay，记录是否连带改变 SP/图路径，比较相同请求的 TTFT、TPOT 与质量。不能把关联路由变化隐去。
4. **隔离 Engram**：相同 shape/本地 heads 的 zeros/empty 对照，外加完整请求测试。
5. **再验证其他候选**：QLI、mHC、通信融合、Store、显存释放分别设计所需采集；不要同时切换十个优化开关。

## 参考资料

- [Ascend PyTorch Profiler 文件与字段说明](https://www.hiascend.com/doc_center/source/zh/canncommercial/70RC1/devtools/auxiliarydevtool/atlasprofiling_16_0266.html)：文件、CPU/设备时间与内存记录的含义。旧版页面的界面与新版本可能不同。
- [Ascend msprof-analyze 集群分析说明](https://github.com/Ascend/msprof-analyze/blob/master/docs/en/user_guide/cluster_analyse_instruct.md)：多 rank、通信覆盖、慢 rank/慢链路分析及指标完整性。
- [MindStudio Insight 服务化调优说明](https://www.hiascend.com/document/detail/en/mindstudio/2610/visualization_tool/MindStudioInsight/docs/en/user_guide/service_optimization.md)：worker profiling 与请求级 serving trace 是不同层次的数据。
- 代码依据来自本地固定发布快照，报告的源码卡片保留版本、文件路径和 GitHub 链接；当前镜像的精确 digest/SHA 尚未提供。
