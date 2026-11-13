#!/usr/bin/env python3
"""Build a standalone A3/A5 report from confirmed archive identities."""
# Keep the HTML/Markdown narrative paragraphs intact for offline rendering.
# ruff: noqa: E501

import gzip
import json

import markdown
from analyze import HERE


def main():
    cache = HERE / "data/report_data.json"
    if cache.exists():
        data = json.loads(cache.read_text())
    else:
        with gzip.open(HERE / "data/report_data.json.gz", "rt", encoding="utf-8") as stream:
            data = json.load(stream)
    byid = {r["id"]: r for r in data["ranks"]}
    pref = data["preferred"]
    a, p = byid[pref["a_id"]], byid[pref["p_id"]]
    sa, sp = a["steps"][pref["a_step"]], p["steps"][pref["p_step"]]

    def get(s, typ):
        return next((op for op in s["operators"] if op["type"] == typ), dict(count=0, sum_us=0))

    source = "https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/"
    ratio = 100 * (1 - sp["host"]["count"] / sa["host"]["count"])
    body = f"""
<div class="eyebrow">DEEPSEEK V4.1 · A3 / A5 PREFILL · 2026-11-13</div>
# A3 与 A5：Prefill 算子、调度与优化证据

<p class="lead">以 <strong>P.rar（A3）</strong> 与 <strong>vllm_profile_prefill-c192-bs12.zip（A5）</strong> 为主对照，旧的 0.30/0.32 ZIP 仅作 A3 历史参照。分析覆盖有效 execute_model 的输入准备、40 层主干、draft 调用及跨 rank 等待，回答哪些融合已发生、哪些差异来自部署，以及 A3 下一步应优化什么。</p>

<div class="cards"><div><b>A3：DP4×TP4</b><span>EP16 · INT8 routed weights · MRV2 · CANN 9.2.1</span></div><div><b>A5：DP8×TP1</b><span>EP8 · MXFP4 routed weights · MRV1 · CANN 9.3.0</span></div><div><b>40 次融合 / 前向</b><span>A5 RMSNorm + MX 动态量化，调用栈与硬件事件相互印证</span></div><div><b>1641 → 221</b><span>代表前向的输入准备硬件任务数量；不是同配置硬件 A/B</span></div></div>

## 核心结论

1. **A5 确实使用了不同的融合路径。** 每个有效主干有 40 个 `RmsNormDynamicMxQuant`，调用栈定位到 `_normalize_q → maybe_fuse_rms_norm_mx_quant`；另外可见带 shared-expert 权重的 MegaMoE、`KvCompressEpilogV2`、MXFP 量化 Attention。证据等级和可移植边界见第 3 节。
2. **不能把所有差距都算作融合收益。** A3 为 TP4，A5 为 TP1；A3 的每轮 80 次 AllGather、40 次 AllReduce 在 A5 中消失，主要是部署拓扑改变。量化格式、CANN 版本、Runner 也都不同。
3. **A5 的全段 Free 更低，这次数据支持这一观察。** A3 P.rar 的 16 rank Free 中位数约 36.06%，A5 的 8 rank 约 3.55%。但 dummy rank 可以在 MegaMoE 内部等待，因而显示“忙”；Free 低不等于同等请求完成更快。
4. **A3 的输入准备是明确的实现优化点。** 代表前向有 42 次 RepeatInterleave、各 176 次 Index/IndexCheck；A5 同类前置窗口没有这条重复展开路径。A3 的调用栈已定位到 `build_replay_swa_visible_lens`，可先做 step 内复用，再融合索引与窗口边界计算。
5. **未实锤 A3 存在固定物理慢卡。** 这次 A3 P.rar 覆盖完整 EP16，可比较各 DP 内四个 TP；HcPre/HcPost 中位数跨度低于约 3.5%。个别 rank 的 Free/MegaMoE 差异显著，但更符合等待与下发节奏差异，不能直接判坏卡。

## 1. 数据身份、实际拓扑与比较边界

<div id="inventory"></div>

主对照共 **24 个 rank**：A3 16 个，A5 8 个。加上历史 A3 ZIP 中的 DP0/TP0–3，共处理 **28 个 rank、122 个 rank-local 前向窗口**。A3 P.rar 重建出 20 个有效 prefill、40 个 dummy 窗口；A5 为 24 个有效 prefill、26 个 dummy。它们是各 rank 的执行窗口数量，不是去重后的客户端请求数。

A5 `NPU_INFO` 明确为 Ascend950，设备 info 记录 32 个 AI Core / 64 个 AIV、配置频率 1650 MHz；A3 记录 Ascend910B，25 / 50 核、1850 MHz。频率来自采集元数据，不能替代运行时降频监控。两边 torch_npu 都为 2.10.0.post7。

**实际 Runner 与代码路径不同：** A3 栈来自 `/vllm-workspace/vllm-ascend/.../worker/v2/model_runner.py`；A5 来自 `/opt/vllm-ascend-dsnext/.../worker/model_runner_v1.py`。不能因为都称 0.32，就当成同一份实现。精确插件/vLLM SHA、模型 revision、启动脚本尚未在这些包中找到；文件名的 c192 / bs12 也不能替代实际调度记录。

量化差异同样能从算子输入确认：A3 MegaMoE 的专家矩阵为 INT8，当前 EP16 每 rank 的可见专家权重数为 24；A5 为 FP4_E2M1 配合 E8M0 scale，EP8 每 rank 为 48，且有 1 个 shared-expert 的 FP8 权重输入。减少 TP/EP 的 rank 数会改变单 rank 的工作量，不能把每张卡的时长直接换算为集群吞吐或硬件倍数。

### 本报告默认选哪一次有效执行

<div id="selected-steps"></div>

A3 选 DP0/TP0 本地第 2 轮；A5 选 DP0 本地第 3 轮。后者处于采集中段，并且能对齐全部 EP8，避开第一轮与停止边界。两边分别观察到 1848→1024 与 11791→768 的 HcPre token 维度；第 21 层起，多个请求的尾窗合并后仍可大于 128。这里的维度不直接等于客户端完整输入长度，也不自行乘除 TP。**下文的时长用于找机制与瓶颈候选，不给出受控硬件加速比。**

## 2. 一次 execute_model 的分解与 Free 的正确读法

先找 Host `execute_model/_dummy_run`，再用 43 次 `vllm::dsa_v41_forward` 与 86 次 HC Pre/Post 交叉校验：前 40 次是主干，后 3 次是 draft 调用，不是模型有 43 个不同的主干层。A3 使用 `npu_hc_pre_v3/npu_hc_post`；A5 使用 `npu_hc_pre_v2/mhc_post`。Enqueue/Dequeue 是一次下发的不同阶段，不计为多次模型执行。

硬件关联优先使用 `PYTORCH_API → CONNECTION_IDS → TASK`。A5 的 C++ MQ-SMLA/Indexer kernel 有些不经过 PyTorch op 记录，但保留同一主线程的 CANN launch；这部分再通过 `TASK.connectionId → CANN_API` 关联，**没有用“最近的时间”猜归属**。余下主要是异步 HIXL Get/Put，保留在整段视图，不强行分配到某个主干。每轮都检查到主干 80 个 HcPre 和 40 个 MegaMoE。

Host scope 与 NPU 时间分别展示：A3 默认 execute_model 的 Host 包含时间约 632.9 ms，A5 约 453.4 ms；对应 40 层硬件包络约 {sa["metrics"]["span_us"] / 1000:.2f} / {sp["metrics"]["span_us"] / 1000:.2f} ms。NPU 异步执行可能晚于函数返回，draft 还可能由后续调用发起，**这些数字不能相加当作 TTFT**。

<div id="free-summary"></div>

全段导出数据里，A3 Free 范围约 12.94%–57.48%，A5 约 2.07%–19.35%。默认有效主干窗口重算为 A3 约 {sa["metrics"]["free_pct"]:.2f}%、A5 约 {sp["metrics"]["free_pct"]:.2f}%。因此差异有明显的“选哪个 rank、选整段还是有效执行”依赖。

MindStudio 的 Free 是既未计算也未通信的时间，不是 CPU 自身耗时，更不是矩阵乘利用率。[官方时间线说明](https://www.hiascend.com/document/detail/en/mindstudio/830/GUI_baseddevelopmenttool/MindStudioInsight/Insight_userguide_0034.html) 本报告的任务包络按硬件区间取并集，包含 AICPU 协调任务；HCCL 子任务不再与大算子重复累计。整段导出 Free 与本报告裁剪后的重算结果分列，避免混用口径。

**A5 导出 Communication=0 不代表没有通信。** 它的 EP8 数据交换包含在 MegaMoE 的通算融合事件内，仍会等待对端。长融合算子可以把等待“包”进 Computing：下面同时给出 A3、A5 的同层到达/结束证据。

<div id="mega-evidence"></div>

A3 首个共同轮次中，dummy DP0 的第 0 层 MegaMoE 为 44.24 ms，active DP1 约 0.49 ms，完成时刻却只差约 5 μs。A5 中段共同轮次也能看到：早到的 dummy DP1 约 104.82 ms，而后到的 active DP2 约 5.58 ms，完成时刻相差不足 0.5 ms。**这是融合事件包含跨 rank 进度等待的证据，不是物理卡相差几十倍的证据。** dummy 参与 collective 可能是必要的，不能直接跳过。

### 不要把各 rank 的“本地第 N 轮”硬拼在一起

<div id="round-coverage"></div>

A5 DP5 的第一份完整前向，已经是其他 rank 的第 2 个共同轮次。报告按同一 EP 组第 0 层完成时刻聚类对齐，2 ms 为聚类阈值，并显示实际组内跨度与覆盖人数；不是依靠本地索引。A5 首轮只覆盖 7/8，末轮只覆盖 3/8，末轮还出现约 1.49 s 的长等待，不能用它判断稳态慢卡。中间共同轮次才用于跨 rank 诊断。

## 3. 主干算子：哪些融合已生效，哪些不是融合

<div id="operator-comparison"></div>

默认两轮前 40 层的硬件计算类任务数为 A3 **{sa["metrics"]["kernel_count"]}**、A5 **{sp["metrics"]["kernel_count"]}**；A5 并不是所有 kernel 数都更少。对应 CPU op 事件数约 {sa["host"]["count"]} / {sp["host"]["count"]}，A5 少约 {ratio:.1f}%。CPU op 计数含嵌套 wrapper，不是物理下发次数，也不能单独证明 Host 墙钟时间按此比例下降。

| 变化 | 实际证据 | 可确认的含义 | A3 可否直接照搬 |
|---|---|---|---|
| RMSNorm + MX 动态量化 | A5 每有效主干 40 个 RmsNormDynamicMxQuant；栈定位到 normalize_q | 已确认的融合，省去该路径的中间写回与分开量化调用 | 不能直接调用 A5 MXFP 算子；应按 A3 INT8 精度/scale 合同验证对应融合 |
| shared expert 并入 MegaMoE | A5 MegaMoE 输入含 routed FP4 与 shared FP8 权重/scale；A3 此路径为 INT8，shared MLP 另有算子 | A5 走带 shared-expert 参数的融合 ABI；需结合当前源码检查其执行分支 | 当前源码有硬件、量化类型与 ABI 门槛，不能只改开关 |
| KV 写入 epilogue | A5 有 KvCompressEpilogV2，按 cache kind 打包量化后写 cache；A3 是另一套 scatter/量化路径 | 写入接口和缓存量化布局发生了变化，不能当作等价 kernel 替换 | 可以借鉴一次写入完成量化/布局转换的思路；必须保留 A3 KV 格式 |
| Attention / Indexer | A3 SparseFlashMla + VllmAscendQuantLightningIndexerV2；A5 MQ-SMLA/QLI custom kernel | 不同量化 ABI 与实现；仅凭名称不能计算融合收益 | 先对齐 q/KV dtype、布局和长度，再决定能否复用 |
| HC Pre/Post | 两边各 80 次；API v3/v2、HcPost/MhcPost 不同 | 是不同实现/ABI，不是次数减少 | 同 shape 做专项对照；两次 HC 分别用于 Attention/MoE，不能删一半 |
| AllGather/AllReduce 消失 | A3 每主干 80 / 40 次，A5 为 0 | 主要是 TP4→TP1 的拓扑效应，不应算成融合算子收益 | 是否降低 TP 要看权重容量、精度、单卡工作量和集群吞吐 |
| FusedMatMul / TopKHash | A5 每主干各 40 次 | 有实现变更；FusedMatMul 名字本身不能说明合并了哪些后处理 | 需要实际算子属性或对应源码，不由名字猜融合内容 |

### RMSNorm 融合的调用栈与移植边界

<div id="fusion-stack"></div>

这条栈将硬件事件与 `ops/rms_norm_mx_quant.py:maybe_fuse_rms_norm_mx_quant → dsa_v41.py:_normalize_q` 关联起来。[官方 RmsNormDynamicMxQuant 接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnRmsNormDynamicMxQuant.md)明确描述了两步融合及减少搬入搬出；该版本产品表列出 950 支持、A3 不支持。因此 A3 应评估 INT8 对应的 RMSNorm+动态量化实现，而不是移植 MXFP 接口本身。

当前 CANN 迁移分支已有[INT8 / MXFP 分支的融合调用参考]({source}vllm_ascend/attention/dsa_v1.py#L2875)及[shared-expert ABI 检查]({source}vllm_ascend/ops/fused_moe/mega_moe.py#L36)。链接是当前可查源码，不是已经证明与 A5 `/opt/vllm-ascend-dsnext` 完全一致的 commit。

### 同 shape 的局部对照

<div id="exact-shapes"></div>

这里选择两边后 19 层均为 128-token 的前向，严格要求 Type、完整 shape、dtype 一致。共找到 {len(data["exact_shapes"])} 组。HcPre 的中位数约 66.77 / 17.49 μs，说明这个局部硬件实现组合有明显差异；但两边调用 API、架构和 CANN 不同，不能把差异全部归为某一个融合。HcPost/MhcPost 因 ABI 不同没有被冒充为“完全匹配”的同算子。

## 4. 输入准备：A3 最直接的实现发力点

<div id="prep-table"></div>

A3 默认前向的输入准备粗窗口约 **223.59 ms**，A5 约 **44.68 ms**。A3 其中有 88.76 ms 的 Gloo，约 69.16 ms 与前轮 NPU 任务重叠，约 19.60 ms 未被这些任务覆盖；未覆盖也不自动等于可消除。A3 扣除 Gloo 作粗参照约 134.84 ms，A5 不含对应 Gloo。两个 Runner 的入口工作不同，不能直接将 178.91 ms 全归为融合收益。

更明确的是实际下发工作：A3 前置阶段有 **1641 个计算类硬件任务**，A5 为 **221 个**；RepeatInterleave 为 42→0，Index / IndexCheck 为 176/176→7/0，Range 为 89→0。A5 仍有 Tile、FloorMod、FloorDiv 和专门的 ring table / metadata kernel，并非“不做输入准备”。

<div id="parent-stack"></div>

A3 的栈能直接定位 `prepare_attn → build_attn_metadata → MetadataBuilder.build → build_replay_swa_visible_lens → repeat_interleave`。旧 A3 ZIP 的大 shape 8188 场景中，42 次 RepeatInterleave 累计约 21.43 ms；本次 A3 P.rar 的较小 shape 1848 中累计约 7.47 ms。两者都是 A3 数据，不能把两次 A3 之间的差异说成 A5 优势。

### 建议实现方案：先复用，再融合

1. **按 step 复用可见长度结果。** 相同 window_size、query_start_loc、seq_lens、replay_start、num_tokens 与坐标语义的组，只生成一次。缓存键必须包含 step 的内容版本，不能只靠 tensor 地址；压缩率、裁剪后的 query 坐标不同的组不能误共用。
2. **融合展开与边界计算。** 使用按 request / token 分块的 kernel，直接从 query_start_loc 求 request 内偏移，再算 position 和 `position - max(position-window+1, replay_start, 0) + 1`，一次输出 int32 visible_lens，省去显式 repeat_interleave、Range、Index 和临时 tensor。
3. **把通用部分与每缓存组部分拆开。** request-id、token-position 等 batch 通用量共享；真正依赖窗口与缓存组的边界保留独立。先确认 42 次里哪些输入完全相同，不直接要求“42 全部变 1”。

累计 7.47 或 21.43 ms 是局部候选成本，不是承诺相同的墙钟收益。正确性要覆盖 chunked prefill、prefix 命中、replay_start、多请求不同长度、0 query、dummy/padding、128-token 裁剪和不同窗口组。先逐元素对齐，再测输入准备关键路径，最后测 TTFT 与精度。

### A5 的哪些思路更适合转为 A3 优化

| 优先级 | A3 候选 | A5 启发 / 本次证据 | 实现与验证重点 |
|---|---|---|---|
| P1 | visible_lens 去重与融合 | A3 42 次展开、A5 无此路径；准备任务数差异大 | step 内去重后做单 kernel；保持 replay 下界和跨组坐标正确 |
| P1 | INT8 Q-norm + 动态量化融合 | A5 已有 40 次 norm+MX quant 融合 | 为 A3 INT8 保留输出/scale 数值语义；确认归一化结果是否还被 Indexer 复用，必要时双输出 |
| P2 | RoPE/量化/cache scatter 边界融合 | A5 有 KvCompressEpilogV2，TensorMove 数量较少 | 根据 producer/consumer 依赖选择一段融合，保持 A3 block/slot、KV dtype 和压缩状态布局 |
| P2 | metadata 按等价缓存组复用 | 两个 Runner 和 builder 路径差异大 | 比较 builder 输入，避免每层重复生成；不能因返回缓存而读到前一 step 的可变数据 |
| P2 | HcPre/HcPost tiling 与邻接布局转换 | 80 次均保留；同 shape HcPre 差异明显 | 微基准扫 token、核数、K/M 分块；按输出消费者决定是否值得融合 |

## 5. A3 快慢卡：完整 EP16 下的检查

<div id="stability"></div>

同一个 DP 内四个 TP 处理相同的调度 token，适合先作对照。A3 中段共同轮次里，HcPre 中位数跨度约 0.56%–2.95%，HcPost 约 1.11%–3.48%；没有显示一张卡在所有基础算子上都固定慢。MatMulV2 有个别组约 22% 的跨度，RmsNorm 可达约 14%，需要同 shape、逐层重复测量，不能由混合 shape 的中位数直接判定硬件异常。

DP2/TP2 全段 Free 约 57.48%，明显高于同组其他 TP；但其 MegaMoE 反而更短，基础 HcPre 没有同级退化。这种组合更像进入融合算子的时间不同、Host 间隙或等待分布变化，不支持“更高 Free 就是卡更慢”的推论。跨 DP 的实际 token 不同，也不能直接比较它们的绝对前向时间。

建议下一轮同时记录调度 token / 请求数量、各层进入与完成时刻、expert token 直方图，再对疑似卡做同 shape 的 HcPre / GEMM 微基准，配合频率、温度、功耗、ECC 与网络重传。当前只有配置频率，没有完整的运行时硬件健康数据，因此结论是“未实锤固定慢卡”，不是“证明所有硬件完全正常”。

## 6. MegaMoE：大头在哪里，还能怎么优化

<div id="mega-layer-chart"></div>

默认主干中，A3 的 40 个 MegaMoE 累计约 {get(sa, "MegaMoe")["sum_us"] / 1000:.2f} ms，占主干包络约 {100 * get(sa, "MegaMoe")["sum_us"] / sa["metrics"]["span_us"]:.1f}%；A5 约 {get(sp, "MegaMoe")["sum_us"] / 1000:.2f} ms、约 {100 * get(sp, "MegaMoe")["sum_us"] / sp["metrics"]["span_us"]:.1f}%。这些比例包含融合内部通信/等待，并且多 stream 可能重叠，不是可全部消除的串行成本。

A3 第 0–20 层 HcPre 为 1848 token，MegaMoE 经 TP/SP 后可见输入为 462；后 19 层 HcPre 为 1024，MegaMoE 输入为 256。A5 默认轮为 11791 / 768，TP1 下对应 MegaMoE 输入同为这些维度。单 rank 工作量、每 rank 专家数、权重精度均不同，不能比较平均 MegaMoE 时长就推断谁的 FFN 更快。

先按以下顺序拆解：

| 方向 | 证据与行动 | 怎样确认有效 |
|---|---|---|
| 先到者等待 | 两种硬件都出现长 dummy MegaMoE；对齐同一共同轮次和层，往最晚到达的 rank 前面追 | 比较最晚到达时刻、完成时刻及未被其他工作覆盖的等待，而非只追短单卡长条 |
| DP 负载与专家负载 | A5 同轮各 DP query token 从几百到 24512，差异很大；A3 也有 active/dummy 混合 | 分别记录调度 token、真实/填充 token 和每专家 token 数；调度/路由均衡后测端到端吞吐与尾延迟 |
| 大 prefill 与尾窗分流 | 第 21 层前后 workload 改变，不能全用一个平均 kernel 时间 | 在相同 shape、精度与专家分布下，比较已支持的路径和 tiling；不能简单关闭 MegaMoE |
| 算子内部流水 | 需要区分 Dispatch / FFN / Combine 与等待；现有一个事件不足以定位 | 追加 CANN 支持的阶段打点、带宽/计数器与专家分布，确认通信、HBM 或算力谁在限制 |
| shared expert 边界 | A5 有融合 ABI，A3 仍有独立 shared MLP；硬件/精度门槛不同 | 先验证 A3 INT8 支持；可先尝试共享分支与路由/通信重叠，再评估更大范围融合 |

**bound 的判断不能只看 Free 或一个 MAC ratio。** 低 Free + 长 MegaMoE 可以是 peer 等待；低 MAC 活跃也不自动等于 memory-bound。算力受限需有同 shape FLOPs 与 MAC 活跃对照，访存受限需实际 HBM/L2 字节与带宽，通信受限需传输量、peer 到达及等待分量，Host 下发受限需队列断粮与 Host self time。当前未采齐这些数据，不给出虚假的精确 Roofline 或固定优化收益百分比。

## 7. 交互时间线与原始行号

<p>默认 A3 中段有效主干；可切到 A5 DP0 本地第 3 轮。范围可选输入准备、40 层主干、43 次 Attention 前向或全采集。淡色任务属于其他前向/阶段，可能因异步执行与当前 Host 范围重叠。滚轮缩放，点击条形看完整 shape 和原始 CSV 行号。</p>
<div id="timeline-panel"><div class="controls"><label>Rank <select id="rank-select"></select></label><label>本地轮次 <select id="step-select"></select></label><label>范围 <select id="phase-select"><option value="main">主干 40 层</option><option value="prep">输入准备</option><option value="all">43 次 Attention 前向</option><option value="whole">整个采集</option></select></label><button id="reset-view">恢复范围</button><button id="first-layer">第 0 层</button><button id="tail-layer">第 21 层</button><button id="zoom-in">放大</button><button id="zoom-out">缩小</button></div><div id="window-caption"></div><canvas id="timeline" height="330"></canvas><pre id="inspector">点击条形查看原始证据。</pre></div>

## 8. A3 的后续实验与代码入口

1. **先跑准备阶段融合微基准。** 在当前 INT8、MRV2、DP4×TP4 下保持输入和缓存状态一致，分别做 visible_lens 去重与单 kernel，测调用数、准备关键路径和内存分配，不先更改并行拓扑。
2. **再做 Q-norm/quant 的 A3 分支。** 当前[DSA 预处理入口]({source}vllm_ascend/attention/dsa_v41.py#L401)可追到 q_norm 和后续量化；确认 Indexer 是否消费 BF16 归一化结果，保留必要双输出和 scale 精度。
3. **选取一个写 cache 的 epilogue。** 参考[MXFP 写入接口]({source}vllm_ascend/models/deepseek_v41/mixed_quant_attention.py#L322)的组织思路，但实现 A3 自己的 dtype、布局和 slot 写入合同。
4. **做全 EP 的同轮追踪。** 对齐完整共同轮次，分清必要的 dummy collective 与实际负载不均；不要把 profile 开始/停止边界中的长条拿来作性能目标。
5. **回到服务指标。** 同一镜像/模型 revision、输入/输出长度、并发、chunk、MTP、prefix/store 命中状态，做无 profiling 的 TTFT/ITL/TPOT/吞吐 A/B；短 trace 用于解释差异。当前报告没有运行新的 NPU benchmark。

旧 A3 ZIP 仍保留为历史参照：它确实只有 DP0 四个 P rank，另有 D 侧两个 rank；这与 P.rar 的完整 DP4×TP4 不冲突。历史样本中 370.62 ms Gloo 约有 365.16 ms 被前轮 NPU 任务覆盖，这条异步分析结论仍成立，但不再被当作 A3/A5 差异。

## 9. 证据与复现

<div id="evidence-links"></div>

CPU 范围包含嵌套调用，不能直接把各行耗时相加；硬件分类累计也可能重叠。C++ 编译后的长 kernel 名称在展示时缩写为 MQ-SMLA / QLI，点击时间线仍能看到原始名称。HIXL 异步事件没有可靠请求归属时不强行分给某个有效 step；它们仍在整段图中。

复现流程为 `extract_inputs.py → analyze.py → derive.py → build_report.py`。报告内嵌全部交互所需数据，断网可打开；附 CSV、压缩数据缓存和代码。原始 GB 级压缩包与完整数据库不重复提交。此次变更仅更新报告与分析工具，不修改推理服务功能。

<details><summary>原始压缩包 SHA256 与路径</summary><pre id="archive-hashes"></pre></details>
"""
    rendered = markdown.markdown(body, extensions=["tables", "fenced_code", "toc"])
    template = (HERE / "report_template.html").read_text()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    report = (
        template.replace("<!--CONTENT-->", rendered)
        .replace("/*DATA*/null", payload)
        .replace("/*SCRIPT*/", (HERE / "report.js").read_text())
    )
    (HERE / "report.html").write_text(report)
    (HERE / "narrative.md").write_text(body)
    print("wrote", HERE / "report.html", len(report.encode()))


if __name__ == "__main__":
    main()
