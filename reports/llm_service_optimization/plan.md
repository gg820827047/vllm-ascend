# LLM 服务请求全局视图与性能任务台账

更新：2026-11-27。以 A3 / DeepSeek / vLLM-CANN 为主要优化对象，A5用于机制对照。结合既有profiling、固定源码和用户提供的对方项目清单，整理可验证方向、版本差异、责任分工与收益口径；对方报告数字不作为本服务实测收益。 11月25日补充128k/64并发调度日志、预算门限与受控实验，以及MRV2/Prefill/Decode/DSpark和客户端TTFT边界；仍未新增性能收益实测。

## 当前主线：P侧Device对照与验证

全局简版见 [执行位置、成本与任务](../dsv41_a3_a5_prefill/stage_brief.html)，实现证据见 [Device模块与源码对照](../dsv41_a3_a5_prefill/device_module_comparison/report.html)。

- P0 / G04：待采集A3 2P2D、P侧32DP×TP1、128k输入profiling；部署/压测负责采集，性能分析负责复表。
- P1 / B03：Device-Metadata窗口/索引复用与融合，先核对等价组；Host耗时对照暂缓。
- P1 / B11：共享专家融合进MegaMoE，先核对A3 INT8/CANN支持；与B08共享专家多流独立。

采集与接口核对可并行；128k和TP1不保证逐step query/历史KV、实际EP和专家负载相同。A5旧样本未强制均衡，不可比行只保留实现差异；当前无新增性能收益实测。个人责任待认领。

## 全局位置、四类主线与边界观测

```text
S0 网关/API接入、排队、分词 → S1 引擎入队/等待、调度、缓存判定
                         ↓
P: S2 Host准备 → S3 Prefill模型层 → S4 采样/输出回传
                         ↓
S5 P/D KV交接（P2P） ↔ Store共享缓存读写旁路（可选）
                         ↓
D: S6 KV就绪/调度准备 → S7 Decode / draft / verify ↺
                         ↓ 每轮采样复用S4
S8 反分词/流式发送 → 客户端
X: DP/TP/EP/PCP/PP、资源与多流依赖横跨多个阶段
```

这是逻辑主路径，不表示所有事件严格串行，也不表示所有请求都经过Store。首token由P或D返回、D是否补算尾token/恢复窗口，以实际协议与代码为准。一个请求可跨多个step，一个step也可能合批多个请求；dummy不等于warmup。

| 用户事项 | 主要位置 | 核心边界 |
| --- | --- | --- |
| ① A3同步点 | S1/S2/S4/S6，及S3/S7跨流依赖 | 区分Host阻塞、设备流依赖、通信等待、输出线程wait；只优化暴露在关键路径上的可消除部分 |
| ② 算子/性能/引擎 | S2准备、S3目标层、S7生成层 | 模型侧优先复用/融合接口/布局/多流/合法切分；框架侧关注输入准备、Runner、下发与图；重型核内问题交接算子方 |
| ③ 快慢卡 | X，落到S2/S3/S5/S7 | 分开同工作量计算速度、晚到rank、通信链路与真实硬件异常 |
| ④ 切分配置 | S1/S2/S3/S5/S6/S7 | 固定资源预算与负载，比较DP/TP/EP、batch/chunk、P/D配比、缓存、MTP和执行路径 |
| ⑤ 网关排队/KV传输（P2） | S0/S1/S5/S6/S8 | F01分开网关真队列、API前处理和引擎等待；F02分开P/D交接与Store读写，联合请求级观测 |

## 各阶段看什么

### S0 网关/API接入与排队

记录网关接收、真实队列入/出、上游转发、API接收与引擎入队。网关排队、API前处理和引擎等待分别计时；worker NPU trace通常没有这些边界。

代码边界：网关 / API server / tokenizer；补请求级时间戳或span。。观测：请求ID/attempt、queue enter/leave、转发、API前处理、到达率与错误率。

### S1 引擎调度与缓存判定

本地 prefix cache 与外部 Store 是不同缓存域；先确定命中来源、实际未计算 token 与本轮预算。

代码边界：vLLM EngineCore / Scheduler / KVCacheManager / connector scheduler。观测：引擎入队到首调度、本地/外部命中、预算、抢占、外部加载等待；与网关队列分开。

### S2 P 端 Host 输入准备

请求状态、positions、slot mapping、SWA / DSA 元数据、H2D、DP 元数据协商。

代码边界：Ascend Model Runner V1/V2 / attention metadata builder。观测：Python scope、Torch op、CANN API、Gloo 与首层关联。

### S3 P 端模型计算

Norm / QKV / DSA-MLA / SWA / MoE；TP、EP 通信与多流依赖贯穿其中，不能把累计算子时间当成关键路径。

代码边界：DeepSeek 模型 → attention / MoE → torch_npu / CANN。观测：同 shape 算子、AI Core / Vector / AICPU、通信、空洞、依赖关系。

### S4 采样与结果回传

P 端可能为交接生成首 token；P/D 首 token 的归属和返回顺序以实际代理协议为准。D2H 也出现在 D 端每轮结果处理中。

代码边界：sampler / async output / executor 输出线程。观测：D2H 方向与字节数、完成事件、消费线程、输出处理。

### S5 KV 交接与外部缓存

P2P connector 负责 P/D 交接；Store 提供共享外部缓存。MultiConnector 按策略组合子 connector，不等于必须开启池化；不是每个请求都串行经过两者。

代码边界：MooncakeConnector / Hybrid / MultiConnector / AscendStore。观测：KV就绪/传输排队/建链/实际搬运/加载完成、字节数、缓存组；P2P与Store分开。

### S6 D 端调度与准备

等待 KV 可用、调度合批、准备位置与 block table；必要的尾 token / 窗口恢复工作取决于模型与 connector，不能直接假定为零。

代码边界：D 端 Scheduler / connector / Model Runner。观测：加载等待、输入准备、异步调度、有效/无效工作。

### S7 D 端重复生成

重复执行模型与采样，直到停止条件。投机接受率影响每轮有效输出数，step 不等于单个 token。

代码边界：D 端模型 / Attention / MoE / MTP / 图重放。观测：有效 tokens / step、step 延迟、接受率、KV 读取与通信。

### S8 流式返回

反分词、流式合包、网络发送；客户端 ITL 是输出事件间隔，不能无条件当作设备单 token 时间。

代码边界：detokenizer / API streaming / proxy。观测：客户端 TTFT / ITL / TPOT、服务端完成时间、取消/超时。

## 近期讨论：时序、多流与静态kernel缓存

2026-11-17归档。源码结论固定到官方0.32快照b12b6f7和CANN升级快照9a6a156；trace结论来自既有P.rar，不是新增性能实验。D侧启动日志、镜像中的npugraph_ex版本与约50个进度项的原文仍待补充。 2026-11-18补充DP2/TP2主线程/元数据/实际launch追踪，以及同名任务与时间零点复核；仍使用既有P.rar数据。 2026-11-19的多流阶段收尾与机制笔记见下方独立章节。 2026-11-20补充CPU验证方法、异步默认值/线程证据、真实与dummy逐轮区分，以及AG/A2A和MoE分块机制。

[11月27日：P侧Device模块与算子对照（Embedding与Engram分列）](../dsv41_a3_a5_prefill/device_module_comparison/report.html)：分别说明基础token Embedding和Engram参数查表/门控；TP通信按实际调用归属，草稿单列，提供真实step跨度、逐任务和源码依据。当前不同工作量不生成性能倍率。
[11月26日：A3裸模型执行全局视图与优化路线（简版）](../dsv41_a3_a5_prefill/stage_brief.html)：输入准备→40层目标模型→采样收尾→草稿→输出；补734.612ms整轮观测与655.317ms设备跨度，以中文区分执行阶段和优化事项，当前转向Device模块对照与受控样本。
[11月26日：Scheduler → Ascend → DeepSeekV4.1 代码地图](../llm_execution_code_walkthrough/report.html)：三张调用关系图、45处固定版本源码索引；区分启动绑定、跨进程提交、父类复用、目标模型与采样/草稿链。
[11月26日：模型执行范围的 A3 / A5 精确拆分](../dsv41_a3_a5_prefill/execution_scope/report.html)：核对40层、收尾、采样与草稿的Host/Device边界，提供Stream/Task与调用栈；附成本账本及A3验证计划，不将非受控差额当硬件收益。
[11月25日：128k / 64并发准入诊断与字段说明](index.html#mechanism-scheduler-admission)：包含四DP日志、0.8秒口径、两份预算、running=2与两组受控实验；未把估算写成TTFT实测。
[打开A3 DP2/TP2主线程与多流时序报告](../dsv41_a3_a5_prefill/host_issue_dp2tp2/report.html)：包含Task15092/15140对照、SESSION/共同坐标切换，以及DecoderLayer为何出现在Prefill调用栈。
[多流阶段结论与原理笔记](index.html#multistream-review)：展开metadata、Q/KV、通信、CSA2/CED、量化和并行切分对照；Host抖动保留为假设。

### A3一个有效prefill：先统一时间零点与结束边界

A3 DP0TP0，本地step1；以下均为相对worker execute_model进入的毫秒。该窗口不是完整请求TTFT，也不能代表所有rank/轮次。

| 边界 | 时间 | 解读 |
| --- | --- | --- |
| worker execute_model进入 | 0ms | CPU调用边界；设备此时仍可能执行上一轮任务。 |
| 上一轮设备结束 / 本轮首个准备kernel开始 | 78.622 / 79.294ms | 本轮准备launch很早就已提交，先被上一轮设备任务覆盖；不能把这段全部算成Host下发慢。 |
| Host进入首个HcPre / 实际launch结束 | 223.595 / 223.796ms | 首层之前已有约223.6ms Host准备，内部含DP同步和元数据构造；不是纯CPU计算。 |
| Device首个主流HcPre开始 | 235.027ms | 距真实launch结束约11.231ms；这段有约11.258ms已记录设备任务，不能作为Host-bound证据。 |
| Host execute_model返回 | 632.903ms | Host在设备尾部结束前返回，不代表全部Host开销已被隐藏。 |
| 目标层设备结束 / 目标层加draft结束 | 690.784 / 733.695ms | 461.233ms是目标层归属设备活动的包络；含draft时要用更晚的结束边界。 |
| 下一轮Host execute_model开始 | 710.069ms | 与本轮draft尾部已有跨step重叠；仍需区分有用准备与等待。 |

### 461ms主要花在哪里，如何判断Host-bound

既有trace数值已复核；局部窗口。位置：S2输入准备 → S3模型层 → S4/draft。

- 目标层设备包络461.233ms的互斥拆分：MegaMoe 299.701ms、SparseFlashMla 40.173ms、HcPre/Post 37.149ms、其他主流矩阵乘/量化/激活36.646ms、其他主流19.268ms、其他已记录流活动27.369ms、无记录任务0.925ms。
- 设备任务驻留覆盖高不等于算术忙；MegaMoE内部通信与等待仍占时间。首层16个EP rank到达差35.763ms，结束差仅0.031ms，应先查晚到路径。
- 首层前的prepare_attn约59.887ms、replay metadata约54.277ms值得继续拆；DP同步wrapper约88.347ms中已有约69.157ms与上一轮设备重叠。嵌套scope不得相加。
- launch到device的间隔长短都不能单独判Host-bound。要关联真实runtime launch、前序设备队列、stream/event依赖、Host有用self time和未覆盖等待。

边界：此窗口的元数据阶段有明显待分析空间；不是“整轮都Host-bound”的结论。无记录任务不自动等于Host造成空闲，Mega占比也不是可回收预算。

跟进：G03 / A04 / A06 / B05 / C03。证据：E13 / E14。

### 不传async参数不等于关闭，也不等于打开必然提速

运行栈与参考源码一致；开关A/B未做。位置：S1/S2调度准备 ↔ S3/S7设备执行。

- A3 trace出现async_output_busy_loop；参考core仅在async_scheduling为真时启动该线程，且已看到下一轮Host提前进入。应撤回“脚本没写参数，所以本次一定关闭”的前提。
- A3与A5样本都观察到部分跨step活动；不能将Host scope重叠全部算作有用准备重叠。
- 异步调度可尝试覆盖独立调度/输出处理和下一轮准备；DP同步、数据依赖及组内最后到达仍可能限制覆盖量。

边界：准确采集core SHA与最终配置仍待补。当前没有同硬件同负载显式ON/OFF实验，不能承诺隐藏223.6ms或给出加速百分比。

跟进：A01 / A03 / A07。证据：E08 / E13 / E14 / E20。

### MegaMoE与共享专家多流：官方和CANN均有互斥

两份固定源码已确认；实际收益待受控验证。位置：S3/S7 MoE层。

- 官方0.32与当前CANN中，用户enable_fused_mc2=2先归一化为内部1；内部值1会将multistream_overlap_shared_expert强制设为False。用户配置1的dispatch_ffn_combine路径也会触发该判断。
- 关闭的是框架层共享专家独立辅助流机制；其他多流项各自受控，Mega算子内部并行也需另外观测。共享专家可能表现为独立MLP，也可能被特定路径融合，必须按实际调用确认。
- A3既有分析未发现目标层共享专家MLP与Mega的框架级并行；已看到DSA预处理辅助流重叠。约0.108ms只是一个事件对的例子，代表窗口DSA活动重叠约6.538ms，均不能当成端到端提速。

边界：不能同时写两个true就认为两者生效，也不应直接删除互斥。对照若同时切换fused后端和多流，结果只能归于组合路径，不能声称是多流单项收益。

跟进：A08 / B05 / B08 / D07。证据：E13 / E14 / E15 / E16。

### A5 Prefill trace不能确定D侧static kernel配置

采集覆盖范围已明确；D配置待补。位置：S3 Prefill / S7 Decode。

- FULL_DECODE_ONLY限制图模式的覆盖范围，Prefill样本不能证明Decode的static kernel开关或实际命中。
- 没有捕获到static_kernel标识、编译日志或graph replay，不能据此确定全局开关关闭；开启开关也不证明每个算子都静态执行。
- 共享专家、量化格式、Runner及硬件不同，必须分别记录路径；不要把A3/A5差异都归因于静态kernel。

边界：补采D侧最终配置、实际算子和图执行证据后，才做相同负载下的稳态TPOT/ITL和吞吐对照。

跟进：B07 / D07。证据：E01 / E06 / E19。

### 每次启动约50项：先区分编译、缓存检查和图捕获

实现机制已核对；用户现场原因待定位。位置：启动前：实例就绪；稳态覆盖看S7。

- 三件事分别记录：静态kernel二进制编译、FX图代码编译、ACLGraph capture。当前路径中新进程仍需图捕获；静态kernel产物能否复用取决于实际npugraph_ex版本和缓存。
- 支持独立缓存的新版npugraph_ex默认disable_static_kernel_compile_cache=False；当前CANN未把它单独暴露为additional-config字段，不能直接塞入JSON当成通用开关。
- 底层静态产物与VLLM_CACHE_ROOT下的图代码缓存不同。以实际build directory定位；缓存索引与引用的产物均需保留，跨容器保持对应绝对路径并确保可读写。
- Starting static kernel compilation日志在命中查询前就可能打印，新建时间戳目录也不能证明重编译。没找到.run同样不能定论；先在实际D容器查镜像版本、日志路径、是否执行op_compiler或仅capture。
- 含Triton wrapper的FX图当前会跳过图代码保存；这与静态kernel二进制缓存是两个层次。减少capture档位属于另一个实验，需结合MTP、并发和padding代价选择。

边界：尚未拿到本次“50个”的日志原文，不能认定50个静态kernel都重编译，更不能承诺配置目录就能消除全部启动耗时。启动就绪时间与热服务TTFT/TPOT分别验收。

跟进：B07 / D08 / G01。证据：E17 / E18 / E19。

### DP2/TP2重叠少：独立分支未同时就绪

原始trace/Connection ID关联已核对；CPU底层原因待补。位置：S2元数据 → S3层内下发 / X跨rank。

- 相同输入shape和主要算子次数下，TP2元数据外层多25.567ms，40层Host推进多63.670ms；DP协调却更早结束。
- TP2计算流重叠0.088462ms，TP0为7.420534ms；第0/10/21层均观察到KV分支完成后，Q Norm才提交。
- 用户定位的Task15092为前一轮dummy第37层6 token；本次分析的Task15140为有效Prefill第0层1848 token。同名算子必须按Stream/Task/Connection ID与shape匹配。
- 单rank SESSION与本报告多卡共同坐标相差233.589ms；Insight若分别移动过Host/Device泳道，需先统一偏移才能量跨泳道延迟。
- DecoderLayer是模型层结构名，Prefill和Decode都调用它；不能据模块名把P侧前向判为D侧生成。

边界：Python self wall包含可能的CPU调度和采集开销，Event还被profiler包装；现有数据没有CPU调度/亲和/频率遥测，未确认物理慢卡或可回收收益。

跟进：A06 / A08 / C03。证据：E29 / E14。

### 异步默认值与MoE通信术语补齐

原始线程证据/固定代码复核；MoE外部收益未验证。位置：S2准备 / S3 MoE / X跨step。

- 未显式写async选项不能判关闭；A3原始TID4151已出现async_output_busy_loop/get_output。
- DP2/TP2有4个主干窗口、6次外层execute_model scope；只有编号1为真实Prefill。DP1有两个连续真实Prefill。
- 跨step存在部分准备/设备包络交集，但其中可能是Gloo等待；不能把区间交集直接当有效准备或收益。
- AG/A2A区别与Routing/GMM分块方案已归档；500~700us和无OOM是对方摘要，尚非本服务结果。

边界：采集镜像core SHA/最终配置与CPU调度轨迹仍缺；MoE分块规则、量化合同和收益范围待补，未新增上板实验。

跟进：A02 / A06 / A07 / B05 / C03。证据：E34 / E35。

### 128k / 64并发的直接准入门限与实验计划

远端日志摘录和固定源码；端到端归因/A-B仍待完成。位置：S1调度 / S3 Prefill / S5交接 / S6 D准备。

- 四个P侧DP的相邻窗口各有13次schedule和13次input_budget_exhausted，末轮均running=1、waiting=14。
- 函数均值6.6–8.1ms；约0.8秒是window_ms/steps，不是单次模型或Device耗时。
- 预算8192、DSpark K=5时，每请求额外记4个输入槽位；尾块和下一请求首块可拼成running=2、scheduled_tokens=8184。
- 分开做并发64→4诊断和固定64并发预算8192→16384的吞吐/TTFT实验，保持其他因素一致。

边界：仅有窗口日志，尚无每请求queue/TTFT闭环；不认定Host-bound、排除慢卡或宣称性能收益。P首组候选是否交D仍需在线connector证据。

跟进：G03 / A01 / D02 / F01 / F02。证据：E36 / E37 / E38。

## 多流专项：阶段收尾、证据边界与原理笔记

2026-11-19归档、2026-11-20补充。涵盖metadata、Q/KV、TP通信、CSA2/CED、并行切分，以及CPU绑核/抖动验证、异步默认值、逐轮遮盖和MoE通信分块。仅复核既有P.rar、固定代码与官方资料；MoE外部方案作为待验证线索，未修改推理功能或新增性能收益实测。 2026-11-25新增调度准入与请求阶段笔记，区分窗口级日志、机制示意与端到端测量。

多流机制与局部时序分析阶段完成；优化实施及端到端收益验证仍未完成。

Host准备/下发影响重叠是已观测机制；Host抖动是待验证假设。当前没有证据给抖动分配“大概率”，也未确认固定物理NPU慢卡。

### MegaMoE＋共享专家辅助流：限制已确认；兼容扩展后置

范围是特定共享专家框架辅助流与fused MC2互斥；原始历史说明涉及收益与通信顺序/资源风险，不等于所有MoE多流都不支持。

后续：保留互斥，跟踪可支持组合；需要独立兼容性设计与实验时再推进。。台账：B08。

### DSA Q/KV预处理多流：机制与局部低重叠定位完成

已有真实重叠实例。DP2/TP2的38/40层未观察到预处理交集，直接机制是Host没有及时使分支同时就绪。

后续：后续问题转入Host/metadata专项；A08保留收益验证，不标记性能优化完成。。台账：A08 / A05 / A06。

### Host抖动或固定开销：根因待验证

尚不能区分固定构造/包装成本、采集扰动和CPU运行态波动。

后续：先无profiler对照，再补CPU遥测；稳定复现后逐项验证环境或代码因素。。台账：A06 / C03。

### 服务性能收益：尚未验证

0.088/7.421ms重叠差和63.670ms Host差不是可直接回收的预算。

后续：固定负载做单变量精度、稳定性与TTFT/吞吐验收，记录无收益结果。。台账：A08 / A06。

### 下一轮验证

1. 先在同负载下比较无profiler压测与轻量采集，确认低重叠和晚到rank是否稳定复现。
2. 若复现，补CPU调度/绑核/NUMA/频率等观测，区分运行环境差异、随机抖动与固定Host构造开销。
3. A05/A06优先尝试同step/缓存组的合法元数据复用及下发开销消减；每次只改变一个因素。
4. B08的MegaMoE＋共享专家辅助流兼容扩展后置，不与Host实验混合；所有收益最终用无profiler服务指标确认。
5. 异步调度实验必须显式使用--async-scheduling / --no-async-scheduling，并保存镜像core SHA、最终值和线程证据；MoE分块先补实现与工作量口径，不套用外部500~700us。

### Metadata：搜哪些名称，哪些耗时不能相加

DP2/TP2主线程1601；表内时间统一使用DP0/TP0 SESSION共同零点。

在原始trace中查name字段，在Insight按节点名称搜索；metadata不是一个统一耗时字段。CPU/Python scope与设备算子需要分别查。

Metadata包含query边界、序列长度、block table、slot mapping、SWA可见窗口和Indexer/Attention执行计划。构造它不等于计算QK或生成KV内容。

Engram外层prepare_attn包含普通Attention与Replay元数据阶段。父子scope不能相加，Python持续时间也不是CPU实际运行时间。

### 可搜索名称与定位

| 位置 | 名称/路径 | 作用或时间 |
| --- | --- | --- |
| Python主线程 | model_states/default.py: prepare_attn | 2004.967920–2073.992690ms；69.024770ms |
| Python主线程 | decoder_replay_state.py: _build_metadata | 2076.023680–2143.750350ms；67.726670ms |
| Python主线程 | engram/model_state.py: prepare_attn | 外层2004.485700–2143.785850ms；139.300150ms |
| Python主线程 | build_replay_swa_visible_lens | 本窗口42次；每个query可见的SWA历史长度 |
| Ascend Hardware | SparseFlashMlaMetadata / IndexerV2Metadata | 设备侧执行计划/元数据；不等于Attention主计算 |

[原始主线程时序报告](../dsv41_a3_a5_prefill/host_issue_dp2tp2/report.html) · [元数据构造源码](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/models/deepseek_v41/decoder_replay_state.py#L346)

### Q/KV：生成向量、读取缓存与主Attention算子

首层1848-token示例；下面列的是Device共同坐标，先按Stream/Task/Connection ID匹配。

Q用于查询，K用于匹配打分，V提供加权汇总的信息。简化公式：P=softmax(QKᵀ/sqrt(d)+mask)，O=PV；这里省略了sinks和位置编码细节。不是只选最高分K再取一个V，而是按权重汇总多个可见位置的V。

当前模型这条算子路径使用共享KV：同一缓存向量承担K和V的角色，并由多个Q头读取。Q和KV使用不同模型权重生成。

首层H为[1848,5120]。Q_a投影得到[1848,1280]，经过RMSNorm/量化后，Q_b输出[1848,8192]，reshape为本rank的[1848,16,512]。KV投影输出[1848,512]，经归一化、RoPE后写入缓存。

前面的MatMulV3、MatMulV2和QuantBatchMatmulV3是在乘模型权重，不是在计算QKᵀ。QK打分、Softmax及PV汇总在SparseFlashMla内部以分块流水方式完成，因此不一定出现单独Softmax任务。

本例第0层没有压缩KV输入；后续带长程分支的层还会读取选中的全局KV。SparseFlashMla之后的逆RoPE与输出投影仍属于Attention模块。

```text
本层H ─┬→ Q_a投影 → Q归一化/量化 → Q_b投影 → Q RoPE ─┐
       └→ KV投影 → KV归一化/RoPE → KV缓存写入 ────────┤
                                                   └→ SparseFlashMla → 逆RoPE → wo_a/wo_b
这是数据依赖图，不是实际下发时间线。
```

### 原始CSV核对的13个计算/写缓存任务

| 算子 | Stream / Task | 开始ms | 作用 |
| --- | --- | --- | --- |
| MatMulV3 | 47 / 5905 | 2153.053187 | 生成Q_a；输入H乘模型权重Wq_a |
| MatMulV2 | 37 / 15140 | 2154.143529 | 生成共享KV；输入H乘模型权重Wkv |
| RmsNorm | 47 / 5907 | 2154.512096 | Q_a的RMSNorm |
| DynamicQuant | 47 / 5908 | 2154.716100 | Q中间表示动态量化，供Q_b投影 |
| RmsNorm | 37 / 15143 | 2155.344973 | KV的RMSNorm |
| InplacePartialRotaryMul | 37 / 15144 | 2155.561417 | KV位置编码RoPE |
| ScatterNdUpdateSk | 37 / 15145 | 2155.824862 | 按slot_mapping写SWA KV缓存 |
| QuantBatchMatmulV3 | 47 / 5911 | 2156.332692 | Q_b投影，得到本rank多头Q |
| InplacePartialRotaryMul | 47 / 5913 | 2156.819802 | Q位置编码RoPE |
| SparseFlashMla | 47 / 5914 | 2157.278271 | 融合Attention：QK打分、Softmax、加权V |
| InplacePartialRotaryMul | 47 / 5915 | 2157.427154 | Attention输出逆RoPE |
| TransposeBatchMatMul | 47 / 5916 | 2157.973205 | wo_a输出投影 |
| MatMulV3 | 47 / 5917 | 2158.347613 | wo_b输出投影 |

[完整shape、dtype和时刻CSV](../dsv41_a3_a5_prefill/host_issue_dp2tp2/qkv_first_layer_walkthrough.csv) · [投影层定义](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/models/deepseek_v41/model.py#L392) · [SparseFlashMla调用](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/attention/dsa_v41.py#L722)

### 多流：数学可并行、任务及时就绪、硬件资源允许是三个条件

DSA预处理子图；与共享专家多流分别判断。低重叠并非所有TP2或所有窗口共有。

当前代码先提交Q_a，再将KV矩阵乘提交到辅助流，然后向主流提交Q归一化/量化。CPU调用顺序串行，设备任务仍可异步重叠。Q Norm不需要KV投影结果；最终Attention消费缓存前才必须满足相关就绪条件。

实现通过event错开主要Cube矩阵乘，尝试让Q归一化/量化与KV矩阵乘重叠，再让Q_b矩阵乘与KV归一化/RoPE/写缓存重叠。算子也可能占用多种资源，不保证分流后净收益为正。

DP2/TP2第0层KV在2154.196610ms结束，Q Norm到2154.512096ms才在设备开始，已错过重叠窗口。TP0同层KV为2151.912405–2151.986627ms，Q Norm为2151.911425–2151.942366ms，实际交集约29.961µs。

DP2/TP2的40层中38层未观察到该预处理重叠，仅第5、18层有少量重叠。DP0/TP2、DP3/TP2有明显重叠；DP1/TP0另一有效窗口也出现低重叠。不能把现象绑定到TP2编号或固定物理设备。

日报7.420534ms是TP0目标层内全部计算流重叠；下面6.563496ms只统计DSA预处理子图，两者口径不同。A3/A5拓扑、Runner和量化路径不同，不能直接外推硬件优劣。

### 同DP2窗口的DSA预处理交集（40层）

| Rank | 重叠ms | 零重叠层数 |
| --- | --- | --- |
| dp2-tp0 | 6.563496 | 0/40 |
| dp2-tp1 | 6.605075 | 0/40 |
| dp2-tp2 | 0.088462 | 38/40 |
| dp2-tp3 | 6.515915 | 0/40 |

### 已确认的框架限制范围

| 路径 | 已确认 | 仍需验证 |
| --- | --- | --- |
| MegaMoE＋共享专家辅助流 | 官方0.32与当前CANN均有框架互斥；不等于所有MoE多流都不可能 | 已知跨rank/跨流通信顺序和资源竞争风险；扩展兼容性实验后置 |
| DSA Q/KV预处理 | 代码允许部分并行，TP0等rank已观察到真实交集 | 生产稳态是否复现TP2差异，以及改善下发能否带来服务收益 |

[DSA多流代码](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/attention/dsa_v41.py#L491) · [共享专家互斥历史提交](https://github.com/vllm-project/vllm-ascend/commit/2044419f8f3dec6288eb1b1b1f54245d4c2b0904) · [全部A3窗口统计来源](../dsv41_a3_a5_prefill/host_issue_dp2tp2/attention_mechanism_evidence.json)

### DP / TP / PP / CP / EP分别切什么，Attention何时通信

先区分纯并行概念，再看本例DP4×TP4 / EP16的组合；不能把所有集体通信都归给MoE。

切分通常以rank/设备为单位，不一定以服务器为单位。DP分不同请求；TP分每层的大矩阵/Attention头；PP分模型层；CP分同一请求的token位置和上下文。

Attention按TP组协作：每个DP分组的4个TP rank各算部分头和输出投影，再合并结果。MoE切换到EP组做专家分发/归并；同一进程同时有DP、TP和EP编号。开启EP后，不能认为每个DP分组都独立持有全部专家权重。

SparseFlashMla这次调用执行本卡Q/KV计算，但整个Attention模块有TP通信。原始CSV的wo_b之后出现AllReduce，CPU栈也有AscendRowParallelLinear_0 → tensor_model_parallel_all_reduce。

DP之间通常不为同一请求交换Attention的Q/KV，但框架仍可同步批量/调度信息；EP组也可能跨DP分组。受支持的PCP等上下文切分还会增加上下文数据组织通信。P→D的Mooncake KV传输属于connector层，应另计。

### 并行维度与权重/数据划分

| 方式 | 切分对象 | 权重布局及示例 |
| --- | --- | --- |
| DP | 不同请求/批次 | 纯DP复制模型；一份逻辑副本也可由TP/PP承载 |
| TP | 同一层内张量/头 | 每层部分权重分片，部分参数仍复制；通常有层内collective |
| PP | 不同层/模块 | 假设40层分四段，每段持有对应层；阶段间传激活 |
| CP | 同一请求的token/上下文 | 分摊上下文计算/KV负担；通常不以切权重为主要目标，具体结合TP/PP实现 |
| EP | MoE专家 | 按专家分权重，并按路由分发/归并token；不是EPLB自动开启 |

### 首层Attention与TP通信的实际记录

| 设备任务 | 开始ms | 持续µs | 含义 |
| --- | --- | --- | --- |
| SparseFlashMla / Task5914 | 2157.278271 | 98.382 | 本卡Attention主计算 |
| MatMulV3 / Task5917 | 2158.347613 | 157.803 | wo_b输出投影 |
| hcom_allReduce_ | 2158.655699 | 525.650 | TP输出合并；通信区间仍可能包含等待，不能当作可回收预算 |

[RowParallelLinear的TP合并](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/model_executor/layers/linear.py#L1762) · [当前PCP实现与支持条件](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/attention/context_parallel/dsa_v41_cp.py#L135) · [原始事件与dtype摘要](../dsv41_a3_a5_prefill/host_issue_dp2tp2/attention_mechanism_evidence.json)

### CSA2 / CED / 量化 / QuantMatmul不是同一个维度

V4.1官方模型设计与这份A3实际精度路径分别记录；架构术语不能仅由算子名推断。

CSA2是Compressed Sparse Attention 2，组织全局KV、Indexer与稀疏位置复用。其名称中的2与配置compress_ratio=2的二合一倍率不是同一概念。源码处理0/2/1等路径，具体层配置以checkpoint为准。

序列压缩改变KV条目数；稀疏选择改变本次关注哪些条目；跨层复用减少重复计算/存储；数值量化改变每个数的位宽。SparseFlashMla是执行Attention数学计算的算子接口。

Full生成全局KV并做索引；Reindex复用全局KV/Indexer K、重新选Top-K；Reuse复用全局KV及Top-K。每层仍保留自己的主Q和SWA KV，Reuse不代表整层Attention结果直接复用。

量化将数据表示为较低位宽格式及scale等辅助信息；QuantMatmul使用量化数据做矩阵乘。权重、激活、KV缓存均可有各自量化路径，不都对应QuantMatmul。

V4.1官方为CED：20层因果Encoder加20层Decoder。两段可使用同一DecoderLayer代码类；类名不能判定当前运行阶段。模型分段与部署P/D不是一一对应，bounded replay路径还需按实际开关/支持布局确认。此处修正此前仅用通用Decoder结构概括V4.1的不足。

### 这份A3采集的实际dtype

| 算子 / Task | 输入与输出 | 说明 |
| --- | --- | --- |
| DynamicQuant / 5908 | BF16 → INT8＋FLOAT scale | 量化转换 |
| QuantBatchMatmulV3 / 5911 | INT8激活/权重及scale等 → BF16 | 量化矩阵乘，生成Q_b |
| SparseFlashMla / 5914 | 主要数值输入Q/KV为BF16，输出BF16 | 旁边出现QuantMatmul不代表此Attention的QK/PV都是INT8 |

[官方模型说明（CED / CSA2 / FP4 KV）](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) · [官方技术报告§2.3](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) · [当前V4.1层复用规划](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/models/deepseek_v41/model.py#L522)

### 慢rank、物理慢设备与Host抖动：证据能支持到哪里

当前只完成带profiler的局部机制定位；没有新增NPU性能A/B。

已观察到DP2/TP2的Host准备和下发偏慢，并导致独立Q/KV分支错过重叠窗口。这可以称为本次采集中的慢rank/Host慢路径。

Host抖动尚未证实，更不能赋予“大概率”判断。现有数据无法区分重复构造/包装的固定成本、profiler额外开销，以及CPU调度、绑核、NUMA、频率或其他运行态争用。

当前CPU/NPU采集启用了大量Python记录，Event.record/wait还存在profiler包装。Python self wall不是CPU实际运行时间；_host_sys为空，缺少调度、亲和与频率轨迹。

称为固定物理NPU慢设备，还需在等工作量/精度/布局下重复观察设备计算差异，并用rank↔设备及CPU绑定对照判断异常跟随谁。collective内等待与逻辑晚到不能直接当作芯片计算慢。

enable_cpu_binding=true只表达绑核请求，仍需核对main/ACL/release实际线程亲和；绑核不会自动独占CPU，也不排除cgroup限额、NUMA/频率和其他线程竞争。当前A3路径先绑定worker线程，再对ACL/release线程分配对应CPU池，失败会warning后继续。

相同TP主路径/shape通常有相同的共同Host工作量，但实际CPU运行机会、依赖就绪时间和Device队列积压可能不同；整轮还有rank分支和EP专家负载差异。因而共同开销不能单独解释TP2额外变慢，等待最多的rank也不一定最先变慢。

区分固定Host开销与抖动：在相同输入、batch、缓存状态下重复观察；秒级pidstat先筛查，再用线程级perf sched将Host空洞分解为实际运行、就绪等待CPU和阻塞。只有调度等待与慢step关联，并随单变量环境调整一起变化，才形成较强归因。

```text
# 在代表性压测期间采集，替换为当前worker PID；perf在宿主机使用宿主机PID
WORKER_PID=12345
taskset -apc "$WORKER_PID"
ps -L -p "$WORKER_PID" -o pid,tid,psr,pcpu,comm
# 以下采集在独立终端运行
pidstat -u -w -t -p "$WORKER_PID" 1 30
mpstat -P ALL 1 30
sudo perf sched record -a -o host-sched.data -- sleep 15
sudo perf sched timehist -i host-sched.data -p "$WORKER_PID" > worker-sched.txt
```

### 证据等级与验证顺序

| 判断 | 当前状态 | 需要的下一步 |
| --- | --- | --- |
| DSA存在可重叠分支 | 已确认 | 保留数学依赖与正确性event |
| 本次TP2 Host影响设备重叠 | 已确认局部机制 | 无profiler重复压测及轻量采集确认稳态复现 |
| Host随机抖动为主因 | 待验证假设 | CPU调度/亲和/频率遥测；区分固定开销与随机波动 |
| 固定物理NPU慢卡 | 未确认 | 等工作量设备对照、rank与设备映射互换 |
| 多流优化已带来服务收益 | 未验证 | 单变量、精度/稳定性与无profiler TTFT/吞吐验收 |

### 如何读CPU观测，避免把所有Host慢都叫抖动

| 观测 | 能支持的判断 | 边界 |
| --- | --- | --- |
| pidstat线程%wait升高；perf sch delay增加 | 线程已就绪却等待CPU | 需与目标step对齐；不等于磁盘iowait |
| 线程实际运行时间长，调度等待少 | 检查Host代码、内存访问、频率与采集成本 | CPU满核也不能单独证明根因或随机抖动 |
| CPU利用率和就绪等待都低，但函数墙钟长 | 检查锁、事件、I/O或设备依赖 | perf wait time包含阻塞，不全是CPU争用 |
| 相同负载重复时固定某rank慢 | 检查固定绑核/NUMA/竞争及rank特有路径 | 不自动认定物理NPU慢卡 |
| 上下文切换多或整机CPU低 | 辅助线索 | 切换次数不是抖动证明，平均值可能掩盖单核繁忙 |

[主线程记录与方法限制](../dsv41_a3_a5_prefill/host_issue_dp2tp2/summary.json) · [官方TorchNPU Event采集包装](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/profiler/profiler_interface.py#L43-L60) · [CPU绑核实现（固定CANN快照）](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/cpu_binding.py#L679) · [pidstat字段说明](https://www.man7.org/linux/man-pages/man1/pidstat.1.html) · [Linux perf sched调度延迟定义](https://github.com/torvalds/linux/blob/v6.12/tools/perf/Documentation/perf-sched.txt)

### 异步调度、Runner V2与DP的Gloo同步

参考core 5df63a7；原始trace的异步线程证据与镜像配置的缺口分别标注。

必须撤回“启动脚本没写--async-scheduling，所以本次一定关闭”的前提。该core默认async_scheduling=None，经兼容性检查可自动设为True；显式--async-scheduling开启，--no-async-scheduling关闭。V2与该选项是两个维度，V1也可支持异步调度。

原始P.rar的DP2/TP2主线程TID1601执行execute_model；独立TID4151记录async_output_busy_loop及AsyncOutput.get_output。参考core只在use_async_scheduling=True时创建对应线程，因此强烈支持该采集已走异步调度对应输出路径；准确镜像core SHA与最终解析值仍待补，不能把参考代码当成镜像配置快照。

底层算子异步下发、模型内部Q/KV多流、异步输出拷贝和跨step调度流水需分开。V2会构造AsyncOutput；最终get_output需要copy_event.synchronize。关闭调度异步仍可能有底层异步；开启后把输出交给后台线程，并允许更多batch在途，但不会自动消除metadata成本或数学依赖。

NCCL是NVIDIA设备集合通信库，Ascend设备通信通常为HCCL；这里沿用上游字段名。disable_nccl_for_dp_synchronization=True表示相关DP元数据协调改走CPU Gloo，不是全局关闭设备通信，也不关闭TP/EP或P/D KV传输。

DP协调交换token数、图模式和microbatch等少量执行信息。通过设备AllReduce后CPU取值可能引入设备同步，Gloo让这类信息留在CPU处理；Gloo仍需等待其他rank，长区间可能来自晚到或CPU调度，不能直接当网络带宽差。

特别注意V2的sync_cudagraph_and_dp_padding直接构造CPU tensor并使用cpu_group，其Gloo调用不能独立证明async_scheduling开启；独立异步输出线程证据更直接。

### 四种异步及选项的作用范围

| 机制 | 作用 | 关闭async_scheduling后 |
| --- | --- | --- |
| 算子异步下发 | Host将工作提交到Device流 | 仍可能存在 |
| Q/KV等模型多流 | 独立分支通过stream/event执行 | 由对应模型多流逻辑控制 |
| V2异步输出拷贝 | 输出在copy stream搬回Host，取结果时等待事件 | 仍可异步发起，但输出等待可能在worker主线程 |
| 跨step调度/输出流水 | 后台处理输出，允许下一批在前一批输出收尾前推进 | 相关流水能力受限；PP=1参考实现的在途batch从2变1 |

### 采集中的异步输出定位（共同时间坐标）

| 线程TID | 调用 | 时间/含义 |
| --- | --- | --- |
| 1601 | execute_model | worker主执行线程，历史采集TID |
| 4151 | async_output_busy_loop | 独立输出线程；函数存续区间不等于一直占用CPU |
| 4151 | AsyncOutput.get_output | 2651.263660–2651.748230ms；使用E34的共同零点 |

[原始线程证据与逐轮数据 E34](runtime_coordination_evidence.json) · [async默认值及兼容性解析](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/vllm.py#L1585) · [输出线程创建条件](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/executor/multiproc_executor.py#L689) · [V2 CPU侧DP协调](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/dp_utils.py#L69)

### 采集有几轮：真实Prefill、dummy与跨step遮盖

P.rar共16个A3 rank，60个主模型前向窗口，其中20个为有效rank-step；TP副本不作为独立请求数。编号从0开始。

DP0/DP2/DP3每rank有4个主干窗口：dummy、真实Prefill、dummy、dummy；DP1每rank有3个：真实Prefill、真实Prefill、dummy。我们细看的DP2/TP2真实窗口是编号1，即采集中的第2个主干窗口，后面没有同rank的第二次真实Prefill可复验。dummy是空跑/协调参与，不能仅凭小shape就称为启动warmup。

4个主干窗口不等于所有execute_model函数调用数。原始DP2/TP2 Ascend外层scope共6次，其中2次短调用未匹配主干前向；上游与Ascend嵌套scope也不能重复计数。调用次数、主干前向、有效Prefill、请求数量是不同口径。

后续dummy也观察到TP2 Host推进偏慢：step2为359.576ms vsTP0的304.186ms；step3为363.293ms vs309.874ms。它支持差异并非只出现在那一次Prefill，但不能替代等负载、多次真实请求的稳态验证。

DP2/TP2 step0设备完整forward包络结束于1987.054670ms，step1准备为1906.859750–2151.831790ms；244.972040ms准备中有80.194920ms与上一轮设备包络相交，余164.777120ms未被上一轮覆盖。交集内约70.083694ms为Gloo区间；不能说80ms都是有效metadata计算。Attention metadata从2004.967920ms开始，已晚于上一轮结束。

DP1的两个连续真实Prefill也有部分rank跨轮交叠，见下表。交集使用上一轮包含target和draft的设备包络；包络交集不是CPU实际运行时间、精确有效遮盖或服务收益。无profiler基准、更多连续真实step与显式开关A/B仍是验收前提。

### DP2/TP2的主干前向窗口

| 编号 | 类型 | 主要token shape |
| --- | --- | --- |
| 0 | dummy | 6 |
| 1 | 真实Prefill | 1848；后段1024 |
| 2 | dummy | 6 |
| 3 | dummy | 6 |

### DP1真实Prefill step0→1的准备区间交集

| rank | 下一轮准备ms | 与上一轮设备包络交集ms |
| --- | --- | --- |
| TP0 | 123.367200 | 0 |
| TP1 | 145.282200 | 25.972077 |
| TP2 | 156.356190 | 39.065169 |
| TP3 | 122.215340 | 0 |

[全部16个rank的窗口、时间戳及口径 E34](runtime_coordination_evidence.json) · [原始逐轮统计](../dsv41_a3_a5_prefill/data/rank_steps.csv) · [时钟与同名任务校准](../dsv41_a3_a5_prefill/host_issue_dp2tp2/report.html)

### MoE术语入门：AG/RS、A2A、Init Routing与分段GMM

解释用户提供的48k/EP16/N256/K8方案；外部摘要与本服务已验证事实分开。

MoE一层可简化为：Gate/Top-K选择专家与权重 → 按专家整理并跨卡分发 → 专家FFN/GMM → 结果回传、恢复token顺序并加权合并。传输的是token隐藏向量等数据，专家权重仍在原卡。

AG（AllGather）把组内各卡数据收齐，让每张卡都有完整输入。A2A（All-to-All）让各卡分别交换不同数据，在MoE中按专家归属把token送到需要它的卡；Top-K下一个token仍可能发给多卡。RS（ReduceScatter）将部分结果归约并按分片返回。具体通信组取决于TP/EP/DP布局，不能只凭名称推定拓扑。

两卡例子：初始卡0有A/B、卡1有C/D。AG后两卡都拿A/B/C/D；若A/C属于卡0专家、B/D属于卡1专家，按需A2A后卡0拿A/C、卡1拿B/D。不是交换模型权重。

Init Routing是每次MoE计算前按专家对token展开/排序/搬运、统计数量并保存恢复映射的操作，不是服务启动初始化。它通常消费Gate/Top-K结果，有的接口能融合量化。GMM是Grouped MatMul，将多个专家各自的矩阵乘组织在一起，group_list描述各专家token分段，不是把所有专家合成一套权重。

48k暂按token规模理解，单卡/通信组总量、单请求/整batch口径缺失。N=256和K=8暂按总路由专家数与Top-8理解，均须原方案确认。若均匀EP16且无冗余专家，每rank为16个专家；48k×8约384k份token—专家任务，不代表生成384k个token。

AG/RS全量路径可能等整批AG后才做Routing，并同时保留大输入、重排输出与GMM中间结果，放大耗时和峰值显存。动态A2A常需把Device产生的收发计数交给Host，形成D2H/提交依赖；即使计数很小，等待也会留下空泡。当前CANN A2A _preprocess确有input_splits/output_splits转CPU的路径，但不能据此断言本次MegaMoE内部使用相同实现或已有同样瓶颈。

“GMM前优化500~700us”按摘要暂理解为计算启动前关键路径缩短0.5–0.7ms，未说明单层/单轮/整模型，不能乘层数推TTFT。无OOM也只对原作者测试条件成立。我们尚未对该组合做实现、精度或性能验证。

```text
初始：卡0 [A, B]；卡1 [C, D]
AG后：卡0 [A, B, C, D]；卡1 [A, B, C, D]

假设A/C需要卡0专家，B/D需要卡1专家：
A2A后：卡0 [A, C]；卡1 [B, D]
```

### 四项方案的可能机制与待确认点

| 方案原文 | 可理解的方向 | 不能直接下的结论 |
| --- | --- | --- |
| 前AG，后A2A | 可能是AG分发、A2A回收；也可能是前后token段切换 | 缺实现，不能确定作者是哪一种 |
| Init Routing与AG重叠 | 收到一块先整理一块，与后续通信形成流水 | 必须有就绪协议；不能读取尚未收齐的同块数据 |
| 分段GMM：固定＋可变 | 先算可提前确定的固定块，后算依赖实际路由的剩余块 | 分块规则未知；不是权重一部分固定、一部分变化 |
| A2A分两段，量化 | 分段通信隐藏准备等待；低精度激活减少字节数 | 量化对象/scale/精度未知；需要成本与精度验收 |

### 映射到本项目的验证事项

| 位置 | 需补证据 | 台账 |
| --- | --- | --- |
| Routing前准备与通信 | 真实后端、D2H计数等待、AG与Routing依赖、暴露时间 | A02 / B05 |
| GMM与中间缓冲 | 每专家token数、分块就绪、峰值显存、buffer复用 | B05 / A08 |
| 收益验收 | 补齐48k口径与对方实现，控制拓扑/精度/工作量，验证TTFT/吞吐和峰值显存 | G02 / B05 |

[用户方案原文与未确定口径 E35](moe_plan_evidence.json) · [官方Init Routing定义](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_moe_init_routing_v2.md) · [官方GMM定义](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md) · [当前CANN A2A计数预处理](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/ops/fused_moe/token_dispatcher.py#L645)

### 11月25日：128k / 64并发准入诊断、全部日志字段与0.8秒口径

新证据是用户提供的远端日志摘录；不是本机全量日志或新profiling。P配置明确为4DP×4TP、预算8192；先前误贴400已撤回。

直接证据：03:55:44–45附近四个DP窗口各有13次schedule和13次input_budget_exhausted；末轮running=1、waiting=14、scheduled_tokens=8188。长输入与高并发使输入预算连续无法满足新的准入，这是当前定位到的直接门限；每个请求几十秒的完整因果分解仍需request_id和入队/首调度时间。

0.8秒来自window_ms / steps / 1000；例如DP1是10437.648955 / 13 / 1000≈0.803秒。它是窗口摊到调度调用的墙钟时间，不是测得的单次execute_model、NPU计算或单请求服务时间。DP1函数自身均值86.106842 / 13≈6.624毫秒；其他工作可能发生在窗口剩余时间，但不能仅凭这个差值分配到Host/Device/通信。

8192限制的是一个DP在一轮中为所有请求新安排处理的token总量，不是max_model_len，也不是只允许Attention看最近8k。已有历史KV仍按模型注意力规则参与本轮计算；长上下文会影响同样8k新token的实际执行成本。

128k按128000/131072两种口径，无缓存时约需16/17个8188-token分块。64并发均匀分配到4个DP约16个请求/DP，4TP共同算同一DP批次，不能把8192再乘4。按当前窗口节奏粗算，前面少数完整长请求就有几十秒工作量；这不是测出的TTFT/排队值，也不证明前后窗口是同14个RequestID。

源码先安排RUNNING，再尝试WAITING。存在计算token_budget和输入input_budget两份预算；DSpark K=5按当前规则每个被调度请求额外扣4个输入槽位。8188计算token使剩余计算预算为4、输入容量为0，故waiting不能准入。即使step_limits为空，实际input_budget分支仍可触发；该快照检测没有覆盖所有输入预算条件。

running=2并不要求一轮算完两个128k请求。假设A恰有131072 token且前16轮各算8188，尾部剩64；下一轮可以算A的64与B的8120，再各预留4，输入总计8192、计算总计8184。这是与日志数字吻合的可行例子，不是已恢复出的实际逐请求分配。异步结果尚未回收时，请求也可能继续留在running。

两组实验分开做：①预算8192不变，并发64→4，诊断排队与低排队处理能力；②并发64不变，预算8192→16384，核对实际调度上限同步生效。固定实际输入、缓存命中、代码、Runner、拓扑、输出长度和总请求数，并记录实际到达过程与完成量。降低并发是换负载的诊断，不是64并发优化收益；扩大预算若同样放大轮次时间，也未必提高吞吐。

验收实际未命中Prompt tokens/s、请求queue、P侧Prefill和客户端TTFT分位数、goodput/错误以及抢占/OOM。仅把waiting等待转移到running内的Prefill，或把等待搬到网关，不算端到端改善。若业务必须承载64并发，需验证更高P吞吐、资源/切分调整或有效缓存，而非只降并发。

```text
计算预算 token_budget = 8192
输入预算 input_budget = 8192
草稿额外槽位 draft_slots = 5 - 1 = 4

给一个请求计算8188个token后：
token_budget -= 8188          # 剩4；branch样本打印这一项
input_budget -= 8188 + 4     # 剩0；这一项未直接打印

等待队列有请求，且 input_budget <= draft_slots：
    记录 input_budget_exhausted
    break  # 停止本轮新准入，后续轮次再尝试
```

### 相邻忙碌窗口的转录与计算（非严格同一时刻）

| DP | 窗口ms / 调用数 | 函数均值ms | 窗口秒/调用 | 末轮running/waiting | 实际分支次数 |
| --- | --- | --- | --- | --- | --- |
| 0 | 10418.299 / 13 | 7.483 | 0.801 | 1 / 14 | input_budget_exhausted: 13 |
| 1 | 10437.649 / 13 | 6.624 | 0.803 | 1 / 14 | input_budget_exhausted: 13 |
| 2 | 10426.520 / 13 | 7.332 | 0.802 | 1 / 14 | input_budget_exhausted: 13 |
| 3 | 10425.384 / 13 | 8.061 | 0.802 | 1 / 14 | input_budget_exhausted: 13 |

### 外层字段

| 字段 | 解释 |
| --- | --- |
| pid / dp_rank / scheduler | EngineCore PID、DP编号和实际调度器类；AsyncScheduler不证明Runner版本。 |
| timestamp | 本条汇总的Unix墙钟秒；branch_samples内timestamp则是事件发生时间。 |
| window_ms | 统计窗口墙钟毫秒；首窗包含初始化或空闲，后续也可能包含没有调度的间隔。 |
| steps | 窗口内完成的schedule调用次数，包含可能无token的调用；不等于请求数或有效execute_model次数。 |
| schedule_total_ms / schedule_max_ms | 函数调用累计/最大墙钟耗时；包括被测helper，不包括Worker/NPU执行与汇总快照/日志输出。不是CPU on-core时间。 |
| latest | 最后一次schedule返回的快照，不是窗口平均、最大值或所有请求的汇总时序。 |
| observations / samples / omitted_samples | 实际helper失败/未就绪结果计数、最多4条样本、未展开样本数；空对象只说明这些探针未记录相应结果。 |
| step_limits | 根据返回快照推导的限制计数，仅作提示；不等于实际触发的退出分支。 |
| branch_reasons / branch_sources | 实际等待路径停止/跳过的原因次数，以及执行这些分支的方法来源。次数不是不同请求数。 |
| branch_samples / omitted_branch_samples | 最多4条分支详情与未展开数；完整事件数仍在branch_reasons中。 |
| helper_timings | 辅助函数的calls、total_ms、max_ms；已包含在schedule耗时中，不能再与总量相加。 |

### latest快照字段

| 字段 | 解释 |
| --- | --- |
| running | 已准入且未结束/未完成结果回收的请求；可以仍在分块Prefill，不等于同时在NPU执行的请求数。 |
| waiting | 普通等待队列的请求数量。 |
| kv_holding_waiting | 持有KV block但仍等待的请求，例如某些异步加载路径。 |
| streaming_input_waiting | 等待后续流式输入并保留槽位的请求；不是普通流式输出数。 |
| active_limit / running_limit | 准入活跃请求上限 / running上限；本例均16，不要求必须达到16才会触发其他门限。 |
| token_budget | latest中是本轮配置的计算token上限8192；branch样本的同名字段则是当时剩余计算预算。二者口径不同。 |
| scheduled_tokens / scheduled_new_requests | 最后一轮分配的实际计算token总数 / 首次准入新请求数；running请求继续计算不算新准入。 |
| free_kv_blocks | 本地KV池空闲块数；不是token数或外部Store容量，也不单独证明所有缓存组都能分配。 |
| pause_state | UNPAUSED表示未暂停。 |
| balance_enabled / balance_running_counts | CANN特定调度均衡门限是否生效及其计数；不是所有请求分发均衡机制的总开关。 |

### helper与分支样本字段

| 字段 | 解释 |
| --- | --- |
| blocked_state | 等待请求就绪状态处理函数。 |
| local_cache_lookup | 本地前缀KV命中查询。 |
| cache_match | connector可匹配token数查询；不是KV数据搬运时间。 |
| allocate_slots | 本地KV空间分配；返回None会记录kv_allocation_failed。 |
| branch样本的reason / action / location | 实际原因、break/continue等动作与探针源文件行号；标签为诊断新增，原有准入条件未改。 |
| branch样本的scope / request_id / status / phase | 取出具体请求前scope为waiting_queue，ID/status可为null；phase为waiting。不是漏记了一个已识别请求。 |

[E36：日志选定字段、假设和两组实验](scheduler_admission_evidence.json) · [诊断字段与配套核心探针说明](https://github.com/gg820827047/vllm-ascend/blob/36b73a9c431d12e7b462f34c89de1da9b18b278d/docs/source/user_guide/feature_guide/scheduler_admission_diagnostics.md) · [诊断字段生成源码](https://github.com/gg820827047/vllm-ascend/blob/36b73a9c431d12e7b462f34c89de1da9b18b278d/vllm_ascend/core/scheduler_admission_diagnostics.py) · [固定核心schedule源码（运行日志行号含配套探针）](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py)

### 11月25日：AsyncScheduler、MRV2、分块与DSpark的token时序

按本次讨论的MRV2解释机制；SchedulerAdmission仅能确认AsyncScheduler，镜像Runner需独立核对。阶段按主模型输入/任务区分，不按节点名或是否看见草稿算子区分。

async_scheduling=True且没有自定义scheduler时选择AsyncScheduler；executor负责派发Worker任务，其类型并非由该参数切换。Runner V1/V2是Worker准备输入/执行模型的实现选择，与异步调度分属两层，两种Runner都可能使用异步调度。V2路径中sample_tokens同样调用speculator.propose。目录vllm/v1是引擎命名空间，不能据此判为Model Runner V1。

异步调度可在已有批次执行时准备/提交后续批次。调度器利用位置计数和占位信息提前规划，Worker使用实际token，结果返回后修正接受数/结束状态；真实自回归和KV依赖仍须满足。未写参数不代表关闭；选择异步也不证明所有Host准备已经被掩盖。

一个request可以跨很多step；一个step可以拼入不同request的片段，也可以只有一个request。chunk是单个请求的一段Prompt，batch才是一次调度组织的多请求集合。running既可能处于分块Prefill，也可能处于Decode或等待结果回收。

自回归模型在输入位置pN给出下一token的概率，因此完整Prompt的最后一次前向结束即可采样y1；下一次输入y1才计算它自己的KV并预测后续token。Prefill按处理Prompt定义，Decode按处理新生成token定义，首回答通常从最终Prefill结果产生。该模型计算边界不等于客户端TTFT结束。

DSpark默认anchor路径输入[y1,MASK,MASK,MASK,MASK]，5个query位置可以预测5个候选y2*–y6*，anchor本身也预测后继token。当前预算将一个基础query槽位计入已有容量，因此额外记K-1=4；不是首token已输出所以只剩4个候选，也不是所有投机算法都用相同规则。

后续验证时主模型可输入[y1,y2*,…,y6*]；假设5个候选全接受，可确认y2–y6并采样额外y7，再以y7为anchor起草y8*–y12*。候选并非必然接受，实际输出数可变。草稿query的5个输入位置与主模型验证的6个输入位置是两组计算，预留4不能用于推断forward或execute_model接口调用次数。

中间Prefill同一请求也会准备DSpark上下文：取主模型辅助hidden states，投影/写入草稿模型自身上下文KV；按分块维护，供后续草稿查询使用。统一propose路径还可能执行query块。当前输入kernel在num_sampled=0时显式用下一个已知Prompt token作anchor，而非把未读Prompt当作用户答案继续生成；不需要第二个请求来解释草稿网络出现。

因此，第一次实际草稿调用、第一次有效回答草稿、第一次主模型验证和客户端首输出是不同边界。中间分块的上下文准备有用途；其中固定query计算是否有可省部分，需要单独测量与检查状态/collective依赖，不能把整个草稿路径当无效工作。

```text
Prompt：p1 … pN；回答：y1, y2 …；*表示候选尚未确认。

Prefill中间：主模型处理Prompt分块 → DSpark上下文KV准备/统一query路径
Prefill末尾：主模型处理到pN → 采样得到y1
生成准备： DSpark输入[y1,MASK×4] → 候选y2*…y6*
Decode轮1：主模型验证[y1,y2*…y6*] → 假设全接受，确认到y7
轮1收尾： DSpark以y7为anchor → 候选y8*…y12*
Decode轮2：主模型验证[y7,y8*…y12*] → 假设全接受，确认到y13

本图是逻辑token依赖；P/D交接、缓存恢复与草稿状态传递另见下一笔记。
```

### 模型阶段、token与作用

| 阶段 | 主模型/草稿输入 | 有效结果 |
| --- | --- | --- |
| Prefill中间分块 | 主模型：本请求的Prompt子区间 | 积累上下文；中间分块的用户输出数为0 |
| Prefill最后分块 | 主模型：Prompt尾部至pN | 采样y1；已算出的y1还未作为主模型输入完成其自身KV |
| 第一组回答草稿准备 | DSpark：[y1,MASK×4] | 候选y2*–y6*；生成位置在P还是D、是否交接须核对协议 |
| Decode主模型验证 | 主模型：[已确定anchor,候选×5] | 接受/拒绝并采样；不重算整个128k Prompt |
| Decode后续起草 | DSpark：最新确定token及MASK位置 | 产生下一组候选，交由后续主模型验证 |

[异步调度器选择](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/scheduler.py) · [AsyncScheduler占位状态](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/async_scheduler.py) · [V2主模型采样与起草调用](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/model_runner.py) · [Ascend V2 DSpark入口](https://github.com/gg820827047/vllm-ascend/blob/36b73a9c431d12e7b462f34c89de1da9b18b278d/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py) · [分块上下文KV与next_prefill_tokens](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py) · [K-1草稿槽位预算](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/speculative.py)

### 11月25日：P算出首token、D首次处理与客户端TTFT的边界

固定源码中的示例代理只用于说明一种已查到的协议；未获得验证机在线proxy的完整版本和请求链路，不能断言所有P/D部署返回首token的方式相同。

客户端TTFT=收到首个有效输出token的时刻−发出请求的时刻；P主模型能够预测首token，不代表代理已把它交给客户端。以所查拉取式示例为例：P端stream=False且max_tokens/min_tokens为1，P响应用于提取kv_transfer_params；随后向D请求并把D响应流转发给客户端。

因此，该协议中的P排队/Prefill、交接、D排队/KV恢复/首次必要计算以及代理/网络返回都可能处于TTFT关键路径。D首次计算可能是Prompt尾token的重算或窗口恢复，不能将D节点上所有工作都叫持续Decode。后续第2、3…个客户端token当然不计入已经结束的TTFT。分层传输/并行准备有重叠时不能机械相加。

P/D没有改变模型next-token逻辑，但改变了物理资源、队列与KV位置，允许独立配置P/D算力并降低长Prefill对持续Decode的干扰，同时增加交接成本。不能从首token由Prefill结果产生推导出P/D分不分无关，或TTFT只看P。

必须区分主模型KV、草稿上下文KV与候选token列表。P端运行过DSpark不证明D将直接消费P的候选。所查示例代理没有显式转发独立候选列表；实际connector仍需检查隐藏状态/草稿KV/候选字段及D是否重建。当前没有在线协议证据支持“P的第一组候选必定移交D”。

逻辑上首组有效候选可以由P末尾准备并在支持的协议中交给D；也可由D恢复上下文、取得其首个确定token后自行起草。第二种情况下D得到该token的首次前向可能属于尾部Prefill，不应仅凭位于D就标成Decode。之后在D持续执行草稿→验证循环。

后续按request_id/attempt记录客户端发送、代理/API接收、P入队/首次调度/各chunk完成、KV就绪与加载完成、D入队/首次执行、首有效输出和客户端收到时间；跨机校时后定位暴露的关键路径。SchedulerAdmission是窗口级门限诊断，不替代这条请求级链路。

```text
所查示例（简化依赖，不表示所有活动都串行）：
客户端 → P排队/分块Prefill → P返回KV交接信息
       → D排队/恢复KV/必要尾部计算 → D响应流 → 客户端首token
                                                  ↑ TTFT结束

P或D上的草稿上下文准备 ≠ 用户有效输出
P生成过候选 ≠ connector已传递该候选列表
D节点的首次工作 ≠ 必然已是持续Decode
```

### 三类数据和两种首组候选路径

| 对象/路径 | 含义 | 本次证据边界 |
| --- | --- | --- |
| 主模型KV | 主模型复用上下文的缓存 | P/D需要恢复所需缓存；不是仅凭token计数就证明所有窗口均恢复 |
| 草稿上下文KV | DSpark按主模型辅助特征准备的上下文状态 | 与主模型KV和候选token列表区分 |
| 候选token列表 | 草稿模型预测、待主模型验证的token ID | 不能从P草稿kernel出现推定已传给D |
| P生成首组候选并交D | 需交接候选及必要状态并对齐验证位置 | 一种可实现协议；本次在线路径未确认 |
| D首次处理后生成首组候选 | D恢复状态，得到确定token再起草 | 不代表D重算整个Prompt；首次计算可能是尾部Prefill |

[示例P请求参数](https://github.com/gg820827047/vllm-ascend/blob/36b73a9c431d12e7b462f34c89de1da9b18b278d/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L933) · [示例P交接后选择D](https://github.com/gg820827047/vllm-ascend/blob/36b73a9c431d12e7b462f34c89de1da9b18b278d/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L1077) · [返回D侧输出流](https://github.com/gg820827047/vllm-ascend/blob/36b73a9c431d12e7b462f34c89de1da9b18b278d/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L1204) · [D接收KV后的尾部处理](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py) · [官方P/D设计](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/disaggregated_prefill.html)

## 对方21项优化：对A3 / vLLM-CANN的启发与适用边界

逐项映射到请求生命周期、现有profiling和固定代码快照。重点是模型调用/框架可控的实验；算子核内、外部调度组件和硬件引擎需要明确配套后协作。

所有“对方报告”数字均来自用户粘贴摘要，尚未独立核验。对方硬件、模型、框架/算子SHA、完整启动配置、Interactivity定义与芯片归一化分母待补。Motor暂按可能为MindIE Motor分析，未把它当作已确认事实。

- 先做低成本参数与供给验证：长Prefill的batch/chunk和阈值、实际D batch、P/D容量及有界会话亲和，比直接搬整套新算子更容易在A3形成可归因实验。
- 最强的本地对应是Host元数据与DP晚到：首层前223.6ms；prepare_attn约59.9ms、replay metadata约54.3ms；首层EP到达差35.8ms。这些指向准备复用、分发/工作量均衡与关键依赖分析，但还没有单项收益实测。
- MegaMoE约65%的任务驻留是计算/通信/等待的组合，不是可直接消掉65%。先确认晚到与负载，再考虑DSL/融合调用和算子方协作。
- 有两项明确适配差异：当前CANN的MegaMoE路径保留DP协调；DSV4.1显式设置FP32路由权重预转换。对方“MegaMoE绕过DP AllReduce”和“BF16 Gate”都不能按标题直接套用。
- D侧低时延项目必须补对应trace；当前主证据是P侧。多加P/D卡、达到不同SLO的吞吐和不同精度路径，必须分别报告成本、尾延迟与稳定性。

| 对方项目与原文结果 | 位置与优先级 | 我们的证据 | 下一步与任务 |
| --- | --- | --- | --- |
| P01 PCP8 + DRAM池化 + SWA bounded replay；对方报告：128并发组合256.87 tok/s/chip，较早期组合约+92.6%。 | S1/S2/S3/S5/S6；P1；组合方向可借鉴，收益必须拆分。 | 本地A3/A5均见后段尾窗缩小；已有本地APC/Store区别。尚无同资源PCP8或冷/热池化A/B。；证据 E21 / E01 / E28 | 固定P/D总卡数，分别验证PCP、池化命中和SWA；检查Runner/SP/connector/DSpark组合。；跟进 D01 / D04 / D05 |
| P02 Prefill batch与长Prefill阈值；对方报告：160并发下batch调优+4.64%，长Prefill阈值调优+10.57%。 | S1/S2/S3；P1 · 首批；最容易从现有A3环境开始受控验证。 | A3代表窗口首层前约223.6ms；小chunk可能放大准备/同步轮数，大chunk也可能拉长尾延迟。不能只按单kernel效率选。；证据 E21 / E14 / E25 / E26 | 固定DP4TP4基线逐项扫描token预算和阈值，记录真实chunk/step及P90/P99；核对当前scheduler实际语义。；跟进 D02 / A05 / A06 |
| P03 Motor + Prefill DP4×PCP2；对方报告：286.97→349.84 tok/s/chip，+21.91%。 | S0/S1/S2/S3；P1；分发层与拓扑共同变化，不能把21.91%全归给Motor。 | 16个EP rank首层到达差35.763ms、结束差0.031ms，存在分发/准备均衡的研究价值；尚未证明是路由造成。；证据 E21 / E14 / E27 / E28 | 先确认Motor名称/版本和请求路由，再分开变更分发策略与PCP；记录总rank和TP。；跟进 D01 / D09 / C03 |
| P04 Prefill Session Affinity；对方报告：349.84→355.04 tok/s/chip，+1.49%；较243.85累计+45.60%。 | S0/S1/S5；P1；适合Agent多轮；会话亲和与队列均衡存在权衡。 | 现有worker trace没有session→DP和cache-hit映射；本地APC收益不能由Store命中率替代。；证据 E21 / E11 / E12 | 比较有界亲和与按未处理token均衡；热点超阈值时允许迁移，冷/热缓存分开。；跟进 D09 / D04 / F01 |
| P05 DP均衡与48并发TTFT高、D负载不足；对方报告：对方拟结合dsinfer策略和看板定位；尚无量化收益。 | S0/S1/S2/S5/S6；P0分析 / P1实验；借鉴排查链路，不照搬dsinfer实现。 | 我们已见P侧晚到与元数据准备；尚缺D端有效batch、排队、KV就绪和真实产出速率，不能宣布本服务同样缺粮。；证据 E21 / E14 / E11 / E12 | 关联同一请求的P入队→产出→KV就绪→D首步；先判供给、路由还是传输，再选择均衡策略。；跟进 G03 / C03 / D03 / D09 / F01 / F02 |
| P06 EP均衡 / EPLB；对方报告：Decode压测出现“算子core”，Prefill多batch精度异常；对方仍在定位。 | S3/S7；P1准入；优先稳定性和精度，不直接列为已可用加速。 | MegaMoE包含跨rank等待，但没有每专家token/映射更新证据；不能将DP晚到直接当作专家映射不均。；证据 E21 / E04 / E14 | 区分DP请求负载与EP专家负载，补EPLB支持矩阵和故障最小复现，通过准入再压测。；跟进 B05 / C03 / R03 |
| P07 MegaMoE专家计算+cannbotdsl通信融合；对方报告：单卡吞吐+5.08%–50.64%，拟继续高并发寻优。 | S3/S7；P1归因 / P2适配；方向相关；由模型调用与算子团队共同判断可替换性。 | A3目标包络中Mega驻留299.701/461.233ms，含通信/等待；已有融合不能重复算成待开发收益。；证据 E21 / E14 / E22 | 先减少晚到并提供dtype/shape/EP组/专家负载，索取对方PR与算子ABI，确认A3支持再替换调用。；跟进 B05 / B06 / C03 |
| P08 DSL Attention/量化Indexer/分组复用/异步规划；对方报告：随MegaMoE组合验证，吞吐+5.08%–21.51%。 | S2/S3/S7；P1元数据 / P2 DSL；最贴近我们的准备热点；组合收益不能拆成单项。 | prepare_attn约59.887ms、replay metadata约54.277ms；A3准备任务1641 vs A5 221，42次RepeatInterleave与索引链可先查复用。；证据 E21 / E02 / E14 | 先同组/同step元数据去重复，再用双buffer/event提前独立规划；DSL核替换需另核硬件、量化、workspace。；跟进 A05 / A06 / B03 / B04 |
| P09 Decode DP AllReduce绕过；对方报告：MegaMoE+重算调度条件下吞吐+4.45%。 | S6/S7；P1条件分析；与当前CANN的MegaMoE合同有明确差异，不能直接配置复现。 | 当前should_skip_allreduce_across_dp_group在MegaMoE/FUSED_MC2等条件返回False，要求统一token数；P侧88ms同步也不是同一优化对象。；证据 E21 / E22 / E14 | 核对对方动态global_bs/算子及PR；拆D侧暴露同步，先证明协议允许，禁止直接删除guard。；跟进 A09 / A02 / D07 |
| P10 GateLinear BF16路由Gate；对方报告：BF16权重时减少输入FP32转换，保留FP32路由输出；未列单项收益。 | S3/S7；P2；通用能力部分已存在，先确认目标模型实际路径。 | AscendGateLinear有BF16输入权重→FP32输出分支；DSV4.1显式precast_fp32_weight且有internal router，不能机械替换。；证据 E21 / E23 / E24 | 查cast与FP32 hidden复用、实际matmul；对比路由TopK/权重与端到端精度后再决定。；跟进 B09 / B01 |
| P11 多Prefill供给：EP8的2P1D反例；对方报告：P16+D8、24卡：总吞吐−2.99%，单卡−35.33%，TTFT P90−13.06%；未采纳。 | S1/S3/S5/S6/S7；P1；很有价值的反例：更多P和更低TTFT不保证更高效率。 | 我们目前也偏P侧证据，不能靠多加P推断总吞吐增长；D容量、通信和缓存可能转成瓶颈。；证据 E21 / E01 / E12 | 同负载比较总吞吐、P+D卡效率与SLO goodput，明确对照卡数和D是否饱和。；跟进 D03 / D01 / F02 |
| P12 多Prefill供给：EP16的2P1D；对方报告：P16+D16、32卡；暂无性能数据。 | S1/S3/S5/S6/S7；P2；只作为待测容量方案，不能标为优化已完成。 | 尚无本环境的P/D容量曲线和该拓扑数据。；证据 E21 / E01 | 先完成基线/容量链路，再按成本预算测试是否值得增加P实例。；跟进 D03 / D01 |
| P13 1P1D高并发峰值与B300对照；对方报告：峰值1073.99 tok/s/chip，较10/27+55.63%；同P90 Interactivity达B300(vLLM)的25.15%。 | 全链路 / S0–S8；P0口径；借鉴固定SLO下比较，不能直接外推硬件差距或收益。 | 现有profile不是对应负载下的客户端结果，也没有B300同模型/精度/计费口径。；证据 E21 / E01 / E10 | 定义Interactivity与有效token、SLO和chip分母，记录完整完成量、错误、输出长度、MTP接受率和成本。；跟进 G01 / G02 / G03 |
| P14 低时延AllReduce走CCU；对方报告：对方称当前低时延场景CCU未生效，仍在定位。 | S3/S7通信；P2；先证明引擎实际命中，再讨论收益。 | 当前没有CCU执行证据和该D侧消息大小；不可假设换引擎必然适配A3小消息。；证据 E21 / E01 / E09 | 对齐硬件/工具链/拓扑，以小消息微基准和profiler确认backend/engine，向CANN团队提交证据。；跟进 C05 / B06 |
| P15 MoE适配TP / grouplist type 2与D侧5.4ms；对方报告：对方PD场景TPOT约5.4ms，拟查o_proj/slice顺序和Attention MatMul未使能NZ。 | S3/S7 MoE / Attention；P2；这是特定算子输入/布局合同，不是通用调优开关。 | 本地A3以P侧INT8和不同Runner为主；DSV4.1 wo_a消费者明确要求ND，不能据“未用NZ”判错。；证据 E21 / E24 / E01 | 取得D侧stack与grouplist语义/分组边界，核对实际矩阵和量化格式后再调整顺序或调用。；跟进 B10 / B05 / D07 |
| P16 Host消减不必要方法；对方报告：对方称有约1ms收益，待带stack profiling确认来源。 | S2/S6及下发；P1 · 首批；方法可直接借鉴，1ms需明确是每step、每层还是整个请求。 | 本地已有首层前223.6ms、prepare_attn与replay metadata热点；大量Host时间也包含等待，不能全算CPU优化。；证据 E21 / E14 | 针对热点短窗口采带stack trace，量化self time/分配/重复转换；用不带profiler的压测验收。；跟进 A05 / A06 / A02 |
| P17 Decode跨机EP16与P/D配比；对方报告：P8+D16、24卡；384并发，P90 Interactivity 102.08，578.61 tok/s/chip；对方优选EP16。 | S1/S5/S6/S7；P1容量 / P2拓扑；提供候选方案，不自动成为我们的最优拓扑。 | 本环境没有同负载D侧profiling；模型、卡型、Interactivity单位和D实际batch尚不齐。；证据 E21 / E01 / E10 | 先扫D实际seq/batch容量曲线，再在同SLO及资源口径下比较EP16配置。；跟进 D01 / D02 / D03 |
| P18 Decode EP32扩展与卡效率；对方报告：P8+D32、40卡；约131.5 Interactivity时，相对EP16：总吞吐+21.73%、单卡−26.96%、TTFT P90−78.02%；对方未继续扩大。 | S1/S5/S6/S7；P1比较口径 / P2扩展；不要只追总吞吐；需同时评估时延和用卡成本。 | 如果分母确含全部P+D卡，1.2173×24/40=0.73038，正好对应约−26.96%卡效率，算术上自洽但不是本环境实测。；证据 E21 / E10 | 在同SLO和精度下画总吞吐/卡效率/TTFT曲线，定义何时愿意用更多卡换尾延迟。；跟进 D01 / D03 / G02 |
| P19 Attention epilogue融合与ND布局；对方报告：逆RoPE+量化+输出投影融合；EP16+ND下16并发Interactivity+28.97%、吞吐+2.39%；≥120门槛下吞吐+172.98%。 | S3/S7 Attention尾部；P2 · 待D trace；可优先做已有融合调用替换；SLO收益不等于核内加速。 | 本地已见A5融合路径，A3量化/布局不同；部分DSA投影本已要求ND。尚缺D小batch尾部链。；证据 E21 / E02 / E24 | 画逆RoPE→量化→wo_a/wo_b→slice/copy链，核对已有接口覆盖和关键路径。；跟进 B10 / B02 / B04 |
| P20 MXFP8输出投影直接写回；对方报告：对方称减少临时拷贝、与ND融合分支互补；未给单项收益。 | S3/S7输出buffer；P2；调用侧有潜力，但MXFP8不能直接套到A3 INT8基线。 | 当前没有目标D侧copy字节数；直接写回还涉及别名、stride、图捕获和异步buffer寿命。；证据 E21 / E01 / E24 | 先确认A3支持和目标dtype的out接口，量化多余搬运，再做输出buffer复用与数值/竞态验证。；跟进 B10 / A08 |
| P21 Decode batch与并发承载；对方报告：seq2/3/4及部分seq8；SLO门槛≥130/165/180时较10/27吞吐分别+216.11%/+788.22%/+969.71%，有资源配比共同变化。 | S1/S6/S7；P1 · 首批需补D采集；可借鉴按SLO扫真实batch；大百分比不能等价为kernel提速。 | max_num_seqs只是上限，实际batch还受DP路由、MTP、KV与供给约束；我们现有P trace不能确定D最佳seq。；证据 E21 / E26 / E10 | 每step记录请求数、有效/填充token、接受率和KV占用；固定资源与负载做seq/准入曲线，再独立比较资源配比。；跟进 D02 / D03 / D06 / G02 |

### 收益比较口径

- 先补可复现manifest：对方与本地的模型/权重、芯片型号、两个仓库SHA、CANN/torch_npu/算子包、Runner、P/D拓扑、量化、SWA/图/缓存、输入输出、并发/到达率及MTP。
- tok/s必须说明是输出token还是含输入token、是否只统计成功请求；tok/s/chip必须注明只除D卡还是全部P+D卡，NPU device与物理芯片计数是否一致。
- Interactivity的单位、是单请求输出速率还是其他定义、分位数及样本聚合方式尚未知；不能直接等同TPOT/ITL毫秒。固定同一SLO再算goodput，并同时保留TTFT、TPOT、ITL。
- 组合+92.6%、+21.91%、+5.08%–21.51%及相对10/27的多项涨幅不能相加；Motor+拓扑、多流+后端、量化+布局、卡数+seq变更先做分组对照。
- SLO门槛附近，较小的时延改善也可能显著增加达标吞吐；+172.98%或+969.71%不是同幅kernel加速，更不是可迁移到本服务的保证。
- EP32例子的算术核对：如果都除全部P+D卡，40卡相对24卡总吞吐+21.73%，卡效率比为1.2173×24/40≈0.7304，即约−26.96%；数据归一化仍需原报告确认。
- 教学估算而非预测：在其余项固定的461.233ms目标包络内，把40.173ms Attention部分缩短20%只节约约8.035ms（约1.74%）；这不是TTFT预测，等待和其他流覆盖还可能变化。
- 带stack profiler用短窗口定位，性能收益用不带profiler的相同负载多轮交错对照；先通过精度、错误率和异常恢复，再发布吞吐。

### 首批实验顺序

1. G01/G02/G03先冻结A3基线和指标口径，补同请求P/D时间戳及D侧短trace；48并发或D缺粮案例若在本服务复现，将相关F01/F02观测提升优先级。
2. 固定现有A3 P侧资源和Runner，D02先逐项扫token预算、长Prefill阈值与实际batch；记录chunk数、Host准备和EP最后到达，避免同时切换PCP和分发。
3. A05/A06/B03针对prepare_attn、SWA replay元数据和重复visible_lens做复用/已有融合调查；异步规划要有双buffer、版本和消费事件，收益以暴露时间下降验收。
4. C03/D09再比较按剩余token/预计完成时间的均衡与有界会话亲和；随后D01在固定P卡预算下测试PCP拓扑，逐一检查SWA/SP、Runner、draft和connector组合。
5. 有D侧证据后再启动A09/B10/B09：分别确认DP同步合同、Attention尾部/写回热点和Gate精度路径；EPLB先经R03准入，DSL/CCU交接到明确的算子/硬件配套实验。

### 今天的进展

今天完成了对方21项优化项目与A3 / DeepSeek现有profiling、CANN代码的逐项对照，并更新了全局优化台账、责任角色和验收口径。

优先方向收敛为Prefill参数与P/D供给均衡、元数据复用/异步规划；同时确认MegaMoE下DP同步绕过和BF16 Gate存在模型/版本约束，不能直接照搬对方收益。

下一步先补D侧profiling和统一SLO压测，开展单变量验证；本次完成分析与实验设计，尚未新增NPU性能收益实测。

## 网关排队与KV传输：哪些需要补采？

worker Torch/NPU profiling主要描述被采进程与设备活动。完整请求链路还需要网关、API、引擎和connector的请求级日志/span；这些信息可与设备时间线关联。没有采到某段，表示未知，不代表该段耗时为0。

F01/F02初始列P2；若排队或KV加载等待主导TTFT、使D侧长期空等，或出现请求失败，应提升优先级。P2表示当前排期，不代表影响小。

| 位置 | 单独看worker profiling | 需要补采 | 计时口径 | 任务 |
| --- | --- | --- | --- | --- |
| 网关/API排队 · S0 | 单独的worker trace通常没有网关队列；NPU空闲不能反推请求在网关等。 | 网关queue enter/leave、转发、API接收、处理线程/分词以及request_id/attempt。 | 真队列出队−入队；API前处理与上游网络另计。 | F01 |
| 引擎等待 · S1 / S6 | 设备时间线不能独立区分引擎排队、KV未就绪或无请求；worker通常看不到全部scheduler状态。 | 引擎入队/首次调度、等待原因、抢占、版本支持的queue指标，关联实际请求。 | 首次调度−引擎入队；重调度/外部KV等待另列，核对版本定义。 | F01 / G03 |
| P/D KV交接 · S5 / S6 | 可能看到connector线程、同步调用、通信或Hixl/拷贝事件；可见范围取决于采集进程与后端，不保证含完整请求标识。 | P/D两端、Transfer Engine/connector日志、KV ready/提交/完成/可消费时间、cache group与字节数。 | KV请求→D侧可用，拆等待P产出、排队/建链、搬运、额外放置/确认；传输带宽单独算。 | F02 |
| Store池化读写 · S1 / S5 | 部分读写/拷贝可见；lookup、租约、SSD回读和后台保存不一定在NPU trace中。 | Store lookup/load/save、键/缓存组、命中来源、字节数、错误/超时和异步完成。 | P2P与Store分开；冷/热、本地/外部命中、读/写与重算分开；保存未必阻塞当前TTFT。 | F02 / D04 |

- 采集字段：request_id、remote_request_id、attempt、host/rank、阶段名、开始/结束、cache group/block、字节数、connector与结果状态；不用用户prompt正文来关联。
- 同一进程优先用单调时钟算局部耗时；跨机器拼时间线先校准时钟并记录误差，不能直接相减未对齐的日志时间。
- 真实KV ready/加载完成以数据可消费事件为准，异步API返回只代表提交；分层流水允许P计算、传输与D消费重叠。
- TTFT按同一请求的关键依赖解释，不把网关、引擎、KV和device窗口中相互重叠的部分重复累计。

## 已有工作和证据等级

代码集成、trace观察到、功能通过、性能收益通过分别记录；本台账没有新增任何NPU性能实验，也没有将既有观察改写成已兑现的加速百分比。

| 已有工作 | 请求位置 | 当前证据 | 范围与后续 |
| --- | --- | --- | --- |
| H01 版本与优化 PR 梳理 | S1 / S2 / S3 / S7 | 分析已完成 | 已形成30→31→32与DeepSeek优化清单；不是当前服务加速证明。；E06；后续 B01 / D07 |
| H02 CANN 基线升级到0.32 | 贯穿引擎、模型、connector | 代码已集成；用户功能反馈 | DeepSeek 1P1D 用户反馈通过；单项收益及全拓扑回归待补。；E07；后续 G01 / R01 |
| H03 SWA replay 与后段尾窗机制 | S1 / S2 / S3 / S5 / S6 | 已有集成；trace观察到后段缩小 | A3代表前向1848→1024，A5为11791→768；不能等同纯开关A/B，也不能取消SP等布局限制。；E01；后续 A05 / D05 |
| H04 A5 RMSNorm + MX 动态量化融合 | S3 / S7 | trace + 调用栈已观察到 | 代表前向40次；属于A5实际路径。A3 INT8消费格式、双输出和算子支持需单独确认。；E02；后续 B02 |
| H05 A3输入准备热点定位 | S2 | 部分分析完成；尚未实施优化 | visible_lens重复展开与索引链是候选；不能声称1641→221全部可在A3消除。；E02；后续 A05 / B03 |
| H06 D2H、流依赖、Host重叠分类 | S2 / S4 / S6 / S8 | 代表窗口已核对；待全量归档 | 已区分输出线程wait、采集结束同步和模型热路径；A5存在跨有效prefill的部分准备重叠。；E03；后续 A01 / A02 / A03 |
| H07 EP轮次与等待分析 | S3 / S7 / X | 部分分析完成 | 已对齐共同轮次并发现dummy等待；未证明固定物理慢卡。；E04；后续 C01 / C02 / C03 |
| H08 A3 Host/Device时序与异步路径复核 | S2 / S3 / S4 / S7 | 既有trace补充核对 | 已归档目标包络461.233ms拆分、首层前准备及launch关联；存在跨step重叠，仍无异步调度ON/OFF性能实验。；E13；后续 A01 / A06 / A07 / B05 / C03 |
| H09 共享专家多流互斥与静态缓存机制核对 | S3 / S7及启动前 | 源码机制确认；现场验证待补 | 官方0.32和CANN均有fused MC2互斥；启动编译、缓存检查和图捕获分开记录，不从50项进度或.run缺失推断重编译。；E15；后续 B07 / B08 / D08 |
| H10 对方优化项目适用性与实验计划 | S0–S8 / X | 21项对照已归档；NPU实验未执行 | 优先batch/chunk、DP供给与元数据；明确MegaMoE DP绕过及DSV4.1 Gate的适配差异，外部收益仅作线索。；E21；后续 A05 / A06 / A09 / B09 / B10 / D02 / D09 / R03 |
| H11 多流分析阶段收尾与机制学习笔记 | S2 / S3 / S7 / X | 阶段分析完成；收益未验证 | 确认共享专家辅助流互斥范围和DSA局部Host供给问题；Host抖动/物理慢设备待证实，后续移交Host与环境验证。；E30；后续 A05 / A06 / C03 / A08 / B08 |
| H12 CPU验证、异步与MoE通信机制补充 | S2 / S3 / X | 学习与证据归档；性能实验未执行 | 补齐A3异步线程和逐轮遮盖口径；AG/A2A/Init Routing/分段GMM说明保留外部方案歧义与收益边界。；E34；后续 A02 / A05 / A06 / A07 / B05 / C03 |
| H13 128k/64并发准入诊断与Prefill/Decode/DSpark学习归档 | S1 / S3 / S4 / S5 / S6 / S7 / S8 | 直接门限已定位；请求级闭环及受控性能实验待完成 | 四DP重复输入预算不足与长Prefill积压吻合；函数时间、窗口推进量级、用户TTFT分开。补充异步/Runner、首token和草稿交接边界。；E36；后续 G03 / A01 / D02 / F01 / F02 |

## 责任与预期收益口径

责任人按建议角色分配，具体人名均待认领，不代表人员承诺。分析/验证/归档任务本身的直接性能收益为0，产物是可复查证据和决策；优化候选的收益为假设，须实测。

仅可用关键路径中未被其他工作覆盖、且能够消除的时间估计时延空间；不能把所有同步时长、算子累加、Free占比或上游PR收益直接相加。吞吐、排队与缓存改变后，还需要重新测客户端指标。

教学示例（非实测）：100ms关键路径中某段10ms，若该段加速2倍、其余条件不变，则总延迟约95ms，减少5%；如果这段原本完全被其他工作覆盖，端到端收益可能为0。

## 子事项总表

P0先形成证据与回归基础；P1做受控实验和低风险候选；P2为后置专题、跨组件依赖或外部协同。F01/F02初始列P2，若成为TTFT主因则提升；点击HTML中的任务可展开实施步骤。

| ID/优先级 | 位置/类型 | 子事项 | 责任人（建议角色） | 状态与已有证据 | 预期收益 / 指标 | 前置 |
| --- | --- | --- | --- | --- | --- | --- |
| G01 / P0 | X / 分析 | 冻结真实基线与生效配置 | 待认领（部署 / 压测） | 待补齐；A3/A5的DP/TP、Runner、CANN和量化路径均不同；采集时完整脚本与最终core SHA不齐。 | 直接性能收益0；使后续结果可归因。 指标：环境/配置字段完整率；不同实验仅保留计划变量。 | — |
| G02 / P0 | S0/S8 / 分析 | 建立请求级指标与固定压测集 | 待认领（部署 / 压测） | 待补齐；已有单机SWA bench与profiling；当前缺少匹配每批trace的完整端到端结果。 | 直接性能收益0；避免把超时丢请求或缓存命中变化当成加速。 指标：TTFT/TPOT p50/p95/p99；客户端ITL；tokens/s；SLO内完成量。 | G01 |
| G03 / P0 | X / 分析 | 建立请求→step→rank→kernel关联与关键路径 | 待认领（性能分析 / 项目统筹） | 部分分析；已复核A3 DP0TP0 step1的Host/launch/device边界、目标461.233ms包络与draft尾部；仍不是完整请求链路。 已有11月25日窗口级准入证据，但缺少逐RequestID入队/首次调度与P/D首输出闭环，0.8秒不作为模型时长。 | 直接性能收益0；定位哪个阶段具有可释放时间。 指标：未覆盖设备空洞；CPU阻塞；通信/计算重叠；请求阶段等待。 | G01 / G02 |
| G04 / P0 | S2/S3/S4 / 采集与分析 | 采集A3 2P2D、P侧32DP×TP1、128k输入的对照profiling | 待认领（部署 / 压测） | 待采集；用户提出下一轮A3目标拓扑2P2D、P侧32DP×TP1、128k输入；尚未采集。对照现有A5 DP8×TP1样本；两侧实际EP及MoE负载可能仍不同。 | 直接性能收益0；消除TP切分这一项差异，缩小工作量混杂，为Device优化排序提供基线。不能仅凭新拓扑承诺A3/A5倍率。 指标：逐模块shape/dtype/layout、query/KV/请求数、EP组及专家token分布；Device阶段墙钟/任务并集；多step波动。 | G01 / G02 / G03 |
| A01 / P0 | S1/S2/S6 / 分析 | 确认异步调度和Runner实际路径 | 待认领（vLLM / Ascend 框架开发） | 部分分析；A3 MRV2 trace出现async_output_busy_loop及跨step重叠；与参考core的异步已开启条件一致。A5为MRV1；准确镜像core SHA与解析值待补。 11月25日日志实际类为AsyncScheduler；此字段不能单独确认MRV2或有用Host准备已重叠，需核对Worker Runner最终类。 | 直接性能收益0；为异步调度A/B提供前提。 指标：可用Host准备重叠量；未隐藏的调度时间。 | G01 |
| A02 / P0 | S2/S3/S4/S6/S7 / 分析 | 分类并追踪同步点的关键依赖 | 待认领（性能分析 / 项目统筹） | 部分分析；A3代表execute窗口有较多StreamWaitEvent，但未见阻塞式Synchronize；不能概括所有step。 | 直接性能收益0；只筛出值得修改的同步点。 指标：每类等待次数、墙钟、暴露时间；有因果关联的设备空洞。 | G03 / A01 |
| A03 / P0 | S4/S8 / 分析 | 核对D2H与异步输出是否挡住流水 | 待认领（vLLM / Ascend 框架开发） | 部分分析；A3/A5都观察到采样输出D2H；长Event同步可能在后台get_output。 | 分析阶段直接收益0；若有暴露等待，优化上限由可隐藏的ΔT决定。 指标：D2H实际传输、消费等待、下一轮启动及客户端输出延迟。 | A02 |
| A04 / P0 | S2/S6 / 分析 | 拆解DP元数据同步和Gloo等待 | 待认领（vLLM / Ascend 框架开发） | 部分分析；A3代表准备段Gloo约88.8ms，其中约69.2ms与上轮设备任务重叠。 | 分析阶段直接收益0；后续只能回收关键路径上可消除的等待。 指标：最后到达者；到达离散度；未覆盖等待；首层启动时间。 | G03 / C01 |
| A05 / P1 | S2 / 优化候选 | 复用step内SWA/DSA元数据与visible_lens | 待认领（模型适配开发） | 待实验；A3已有重复展开与索引链证据，尚未提交此项性能修改。 已定位两个metadata构造scope及42次visible_lens调用；每组输入是否等价仍须校验，不能直接假设42次可合并为一次。 | 预期减少重复元数据生成和准备开销；幅度待实测，不能套用A5的1641→221比例。 指标：准备关键路径Δms、launch/分配次数、TTFT、内存峰值。 | G02 / A02 / B01 |
| A06 / P1 | S2/S6 / 优化候选 | 优化Host下发与准备热路径 | 待认领（vLLM / Ascend 框架开发） | 部分分析；A3代表窗口首层前约223.6ms：prepare_attn 59.887ms、replay metadata 54.277ms；DP同步约88.347ms中部分已被前轮设备覆盖，尚需拆self time。 补充DP2/TP2对照：metadata139.300ms、目标40层Host447.453ms，比同DP TP0分别多25.567ms和63.670ms；先核对profiler扰动与CPU调度，再评估重复准备/stream-event包装。 Host抖动尚待证实；现有数据未能区分固定包装/重复构造成本、采集扰动与运行态CPU差异。 | 减少暴露的准备/下发时间；若已被设备覆盖，时延收益可能为0。 指标：Host self time、launch速率、分配数、设备空洞、TTFT/TPOT。 | G03 / A02 |
| A07 / P1 | S1/S2/S6/S7 / 验证 | 做异步调度受控A/B | 待认领（vLLM / Ascend 框架开发） | 待实验；参考core默认async_scheduling=None按兼容性自动启用；A3原始TID4151有async_output_busy_loop/get_output，与异步输出路径一致。脚本未写参数不能作为OFF基线；准确镜像core SHA和最终解析值待补。DP2/TP2真实窗口准备与前轮设备包络部分相交，DP1连续真实Prefill的交集因rank不同而异，尚无ON/OFF服务收益实验。 新日志已确认本次使用异步调度器，不能再将未传参数当OFF；没有相同负载ON/OFF收益结果。 | 有机会隐藏调度/输出处理开销；设备持续满载时收益可能有限。 指标：TTFT/TPOT p95、goodput、Host/device有效重叠、CPU占用。 | A01 / A02 / G02 |
| A08 / P2 | S2/S3/S6/S7 / 优化候选 | 模型多流与可切分计算的重叠评估 | 待认领（模型适配开发） | 阶段分析完成；收益待验证；A3已观察到DSA辅助流与主流任务重叠；共享专家框架多流被fused MC2互斥关闭。独立子图可否扩展及净收益仍待验证。 全目标窗口计算流重叠：DP2/TP2为0.088462ms、TP0为7.420534ms；第0/10/21层证实独立分支未同时就绪，重叠少不能直接当作物理卡慢或加速预算。 2026-11-19完成机制/局部时序阶段归档；Host根因转A06/C03，优先元数据/下发专项，不把提高重叠本身当作验收。 | 可能隐藏独立计算/通信；竞争或切分过细可能变慢，不预设加速。 指标：关键路径、重叠量、launch数、HBM峰值、TTFT/TPOT。 | B01 / A02 / G02 |
| A09 / P1 | S6/S7 / 分析 | 核对Decode DP同步绕过的版本、算子与一致性条件 | 待认领（vLLM / Ascend 框架开发） | 部分分析；对方称MegaMoE+重算调度下绕过DP AllReduce收益4.45%；当前CANN快照明确在MegaMoE/FUSED_MC2等路径保留协调，现有P侧88ms同步也不能套用D侧策略。 | 分析直接收益0；确认可安全绕过时才有机会减少每步暴露同步与padding，外部4.45%不作为本环境预算。 指标：D侧同步暴露ms、有效/填充tokens、TPOT/ITL、吞吐、collective一致性。 | G01 / A02 / B05 |
| B01 / P0 | S2/S3/S7 / 分析 | 建立同口径算子与执行引擎清单 | 待认领（性能分析 / 项目统筹） | 部分分析；已有代表窗口算子对比；A3/A5含Runner、CANN、量化和负载混杂。 | 直接性能收益0；定位能由模型替换、元数据或配置解决的部分。 指标：同shape单次分布、调用数、关键路径占比、实际执行引擎。 | G01 / G03 |
| B02 / P1 | S3/S7 / 优化候选 | 核对并替换可用Norm/量化融合路径 | 待认领（模型适配开发） | 待分析；A5的RMSNorm+MX动态量化已观察到；A3为不同量化消费路径。 | 可能减少中间读写、kernel下发和临时buffer；实际收益待实验。 指标：Norm+quant组合墙钟、HBM读写、launch数、TTFT/TPOT。 | B01 / G02 |
| B03 / P1 | S2/S6 / 优化候选 | Device-Metadata：复用与融合窗口/索引小算子 | 待认领（模型适配开发） | 差异已确认；等价性待核对；P侧输入准备的Device路径已有证据：RepeatInterleave42次累计7.470ms，伴随Index/IndexCheck；A5有专门metadata路径。这里研究Device重复工作和融合，Host准备优化暂缓。 | 减少重复展开/索引与临时张量读写；7.470ms是现有RepeatInterleave驻留成本，不是可全部回收的时间。实际净收益待A3 A/B。 指标：Device调用数、临时张量/读写量、模块任务并集、阶段关键路径Δms、TTFT及数值一致性。 | B01 |
| B04 / P1 | S3/S7 / 分析 | 定位Attention/Indexer/MLA的瓶颈与调用条件 | 待认领（模型适配开发） | 待分析；算子本体暂不作为团队主要开发范围，但调用方式和输入条件可控。 | 分析直接收益0；可能通过布局、接口或batch条件消除额外开销。 指标：同shape延迟、有效工作量、内存/计算指标、关键路径。 | B01 |
| B05 / P0 | S3/S7 / 分析 | 拆解MegaMoE的计算、搬运、通信与等待 | 待认领（性能分析 / 项目统筹） | 部分分析；代表目标层Mega驻留299.701/461.233ms；首层完整16个EP rank到达差35.763ms、结束差0.031ms，优先追踪晚到路径和内部等待。 新增用户48k/EP16/N256/K8分块方案作为线索：AG/RS全量Routing和A2A计数D2H分别排查；GMM前500~700us及无OOM尚非本服务实测。 | 分析直接收益0；优先减少输入晚到与不均衡，不能把Mega总占比全算可优化。 指标：到达离散、有效计算/等待、专家负载、层关键路径。 | B01 / C01 |
| B06 / P2 | S3/S7 / 分析 | 整理重型算子最小复现并交接算子团队 | 待认领（CANN / 算子对接） | 待分析；若公平对照后确认核内瓶颈，团队不默认自行重写Attention或MegaMoE。 | 分析直接收益0；解锁外部算子修复或已有实现替换，收益待返回版本验证。 指标：可复现率、核内延迟、服务回归收益与精度。 | B04 / B05 |
| B07 / P1 | S2/S3/S6/S7 / 分析 | 区分Eager、图重放和static kernel实际命中 | 待认领（vLLM / Ascend 框架开发） | 部分分析；A5 Prefill样本不能确定D侧static开关；已核对三层启动工作与缓存，实际D镜像版本和“约50项”日志仍缺。 | 分析直接收益0；图命中后可能降低decode下发成本，不能泛化到prefill。 指标：每step launch/Host时间、padding、图命中、TPOT p95。 | G01 / B01 |
| B08 / P2 | S3/S7 / 验证 | 核对共享专家多流互斥并验证支持路径的重叠收益 | 待认领（模型适配开发） | 限制已确认；实验后置；官方0.32和CANN均在内部fused_mc2=1时关闭共享专家overlap；A3观察到的其他多流不能用来证明共享专家并行。 互斥历史与风险范围已核对；不是普遍的MoE多流禁令。兼容扩展需另行设计，暂后置于Host验证。 | 验证本身直接收益0；支持路径有机会隐藏共享专家计算，收益受路由专家关键路径和资源竞争约束。 指标：共享/路由重叠ms、MoE层墙钟、TTFT/TPOT/ITL、吞吐及精度。 | G01 / G02 / B01 / A02 |
| B09 / P2 | S3/S7 / 优化候选 | 确认MoE Gate实际dtype路径并评估保精度的转换消减 | 待认领（模型适配开发） | 部分分析；AscendGateLinear已有BF16×BF16到FP32输出分支；DeepSeek-V4.1 gate却显式要求precast_fp32_weight，内部router还可能复用FP32 hidden。不能以类存在认定目标模型已走BF16路径。 | 候选收益取决于目标路由热路径的暴露cast/投影时间；不能从BF16输出声明推断精度等价或复制对方收益。 指标：Gate+cast时间/内存、专家选择一致性、TTFT/TPOT及端到端精度。 | G01 / B01 / G02 |
| B10 / P2 | S3/S7 / 优化候选 | 评估Attention尾部融合、投影布局与直接写回 | 待认领（模型适配开发） | 待采集；对方的逆RoPE/量化/o_proj融合、ND权重和MXFP8写回有方向价值；我们主证据是A3 Prefill INT8路径，D侧小batch与MXFP8收益尚无对应trace。DSV4.1 wo_a本身已要求ND。 | 有机会减少尾部launch、临时结果和内存搬运；外部SLO吞吐大幅改善不等于同幅kernel加速。 指标：epilogue墙钟、copy/format_cast字节数、launch数、HBM、TPOT/ITL与精度。 | G01 / B01 / B04 / G02 |
| B11 / P1 | S3 / 优化候选 | 将主干共享专家融合进MegaMoE：核对A3支持并验证 | 待认领（模型适配开发） | 差异已确认；接口待核对；现有A3主干共享专家是独立MLP：80次QuantBatchMatmulV3、40次DequantSwigluQuant、40次AllGather及40条通信helper，共200条任务累计15.344ms。A5主干将共享权重/scale传入MegaMoE；不能据此认定A3 INT8可直接打开融合。此项与B08共享专家多流独立。 | 目标是减少独立共享MLP的中间张量/调度与可能的边界开销，必要乘加仍存在。15.344ms含TP4通信/helper，不能全部算作收益；收益待测。 指标：独立共享MLP与通信调用数、融合命中、完整MoE关键路径Δms、内存峰值、精度与TTFT/吞吐。 | B01 / B05 |
| C01 / P0 | X / 分析 | 对齐通信组、共同轮次与物理卡映射 | 待认领（性能分析 / 项目统筹） | 部分分析；已有EP共同轮次对齐；仍需完整物理NPU、CPU、NUMA与链路映射。 | 直接性能收益0；避免把工作量不同或边界等待识别为慢卡。 指标：参与rank覆盖、事件对齐误差、缺失窗口。 | G01 / G03 |
| C02 / P0 | S3/S7 / 分析 | 做同工作量的计算卡间一致性分析 | 待认领（性能分析 / 项目统筹） | 部分分析；HcPre离散较小；A3某同shape MatMul可差约52%，原因未定。 | 直接性能收益0；识别持续退化、资源争用或调用条件差异。 指标：同条件中位数/p95比例、跨轮重复性、关键路径贡献。 | C01 / B01 |
| C03 / P0 | S2/S3/S6/S7 / 分析 | 查明晚到rank和EP拖尾的来源 | 待认领（vLLM / Ascend 框架开发） | 部分分析；A3代表首层到达差35.763ms而结束差0.031ms，结合dummy先到等待进一步查Host准备；该证据不足以确认物理慢卡。 补充DP2/TP2主线程的元数据与下发差异，尚缺CPU调度/绑核/频率证据；区分同名小shape dummy与真实Prefill，以及单rank/共同时间坐标。 阶段结论是慢rank/Host慢路径；需要固定负载复现、CPU遥测与rank↔设备映射对照，才能区分CPU侧问题和固定物理慢设备。 同一TP主路径相同不保证运行条件与就绪时刻相同；后续dummy中TP2也偏慢，需检查固定环境条件及连续真实负载，不能直接归随机抖动。 | 分析直接收益0；为减少全组拖尾提供依据。 指标：最后到达rank、跨rank到达差、暴露等待、层墙钟。 | C01 / G03 |
| C04 / P1 | X / 分析 | 核对频率、温度、CPU/NUMA与资源争用 | 待认领（NPU / CPU / 通信运维） | 待采集；当前静态NPU配置不足以证明运行时没有降频、带宽或CPU争用。 | 直接性能收益0；若确认降频/争用，修复部署可恢复损失性能。 指标：同条件算子分布、运行频率、CPU等待与内存/链路指标。 | C01 / C02 |
| C05 / P1 | S3/S7 / 分析 | 分析模型内HCCL集合通信与链路差异 | 待认领（NPU / CPU / 通信运维） | 待分析；本项聚焦TP/EP等模型内集合通信；P/D点到点与Store KV交接单独列F02（P2），避免混用口径。 | 分析直接收益0；若确认暴露的集合通信瓶颈，可评估通信配置或模型调用侧改进。 指标：collective p95、有效GB/s、到达差、计算通信重叠、失败率。 | C01 / G01 |
| C06 / P2 | X / 验证 | 通过绑核/换卡与独立微基准验证慢卡假设 | 待认领（NPU / CPU / 通信运维） | 待实验；尚未确认固定物理慢卡，不应先按硬件故障处理。 | 直接性能收益0；确认硬件/部署问题后可定向恢复损失。 指标：异常随卡/随rank迁移；跨轮稳定性；正常对照。 | C02 / C04 |
| D01 / P1 | S1/S3/S6/S7 / 验证 | 建立固定资源预算的DP/TP/EP切分矩阵 | 待认领（部署 / 压测） | 待实验；A3 DP4TP4/EP16与A5 DP8TP1/EP8并非同拓扑/卡数对照。 | 可能改善利用率与通信比例；更小TP不一定更快，也可能装不下。 指标：TTFT/TPOT p95、goodput、HBM峰值、通信、每卡有效tokens/s。 | G01 / G02 / C01 |
| D02 / P1 | S1/S2/S3/S6/S7 / 验证 | 扫描合批、token预算与prefill切块 | 待认领（部署 / 压测） | 部分分析；batch/shape决定下发、算子效率和排队；压缩包文件名不是完整配置。 128k/64并发、P预算8192的四DP窗口连续命中input_budget_exhausted，说明直接准入门限；完整请求排队归因与预算A/B尚未完成。 | 可能降低排队或摊薄下发成本；大batch也可能增加单请求延迟。 指标：队列时间、TTFT/TPOT、goodput、实际batch、抢占和OOM。 | G02 / D01 |
| D03 / P1 | S1/S5/S6/S7 / 分析 | 匹配P/D能力与KV交接容量 | 待认领（部署 / 压测） | 待采集；当前主证据偏P侧；排队归因由F01补采，KV交接分段由F02补采，再拼同一请求的P/D容量链路。 | 分析直接收益0；合理配比后可能减少排队、提升SLO内完成量。 指标：两侧利用率/排队、KV交接、TTFT/TPOT、goodput/卡。 | G02 / F01 / F02 |
| D04 / P1 | S1/S5/S6 / 验证 | 拆开本地prefix cache与Store收益 | 待认领（vLLM / Ascend 框架开发） | 待实验；MultiConnector不等于Store；本地APC与外部池化需要分别归因。 | 可能减少重复prefill并提高跨实例命中；冷请求可能多出查找/传输开销。 指标：分来源命中率、跳过token、TTFT、加载/保存时间、失败/重算。 | G01 / G02 / F02 |
| D05 / P1 | S1/S2/S3/S5/S6 / 验证 | 验证SWA replay相关机制的条件与净收益 | 待认领（模型适配开发） | 待实验；已有反合与tail shape证据；当前需要单机和P/D分别做受控开关实验。 | 长prefill可能减少后段工作量；元数据/恢复可能抵消收益，TPOT需单独测。 指标：TTFT、TPOT/ITL、目标层shape、准备时间、HBM、恢复/传输。 | G02 / A05 / D04 |
| D06 / P1 | S4/S7 / 验证 | 扫描MTP / draft配置与有效输出效率 | 待认领（模型适配开发） | 待实验；P侧也可能有draft调用；draft块次数不等于用户请求数。 | 接受率与开销合适时可降低TPOT；多draft也可能更慢。 指标：accepted tokens/step、draft耗时、TPOT、goodput、HBM。 | G02 / B07 |
| D07 / P2 | S2/S3/S6/S7 / 验证 | 分离Runner、量化、KV布局和通信开关影响 | 待认领（模型适配开发） | 待实验；A3/A5当前多因素同时不同，不能将差异都归给融合或硬件。 本次token时序按MRV2讨论，AsyncScheduler不等于Runner版本；草稿query、主模型验证与接口调用次数分开统计。 | 可能选择更适合的执行路径/精度/通信组合；收益与精度代价待测。 指标：TTFT/TPOT、goodput、HBM、精度、实际kernel/backend。 | G01 / G02 / B01 |
| D08 / P1 | X / 验证 | 拆分D侧冷启动与缓存复用，定位静态kernel重复编译 | 待认领（vLLM / Ascend 框架开发） | 待采集；用户反馈启动约50项、耗时长且没搜到.run；尚缺日志原文和运行时npugraph_ex版本，不能确认是二进制重编译还是图捕获。 | 若确有重复二进制编译，可缩短重启/扩容的实例就绪时间；不承诺消除图捕获，也不直接等价为热服务TTFT/TPOT改善。 指标：实例就绪秒数、各启动阶段耗时、真实编译次数/缓存命中、capture档位、稳态TPOT/ITL和HBM。 | G01 / B07 |
| D09 / P1 | S0/S1/S5/S6 / 验证 | 验证会话亲和与DP负载均衡的收益和冲突 | 待认领（vLLM / Ascend 框架开发） | 待采集；对方Motor+DP4×PCP2、Session Affinity有组合收益；本地只有P侧rank时序，尚未采到真实分发、会话命中或D缺粮链路。Motor组件含义待确认。 | 可能减少P组最后到达和D缺粮，并提高重复前缀复用；强亲和也可能加重热点，净收益待测。 指标：DP未处理token离散、缓存命中、P/D有效忙率、KV等待、TTFT/TPOT分位数和goodput。 | G01 / G02 / C03 / F01 / F02 |
| F01 / P2 | S0/S1/S6/S8 / 分析 | 分析网关/API排队与引擎等待时长 | 待认领（网关 / API 服务对接） | 待采集；现有worker NPU profiling缺少网关队列和完整入口时间戳；暂不能量化这部分TTFT，也不能把设备空档归因于网关排队。 11月25日已有引擎窗口级门限日志，尚无逐请求queue时长；不能将schedule函数6–8ms与请求几十秒等待混为一谈。 | 分析直接性能收益0；若网关/API/引擎排队是TTFT主因，后续可定向降低暴露等待。P2为初始排期，确认主导时提升优先级。 指标：网关queue时间/长度、API前处理、引擎等待、到达率、客户端TTFT、SLO内完成量、错误/超时。 | G01 / G02 |
| F02 / P2 | S1/S5/S6 / 分析 | 分段分析P/D KV传输与Store读写 | 待认领（NPU / CPU / 通信运维） | 待采集；worker trace可能含connector线程、拷贝/通信和Hixl事件，但现有采集缺少请求级P/D两端与加载完成关联；不能用一次kernel或API时长代表完整KV交接。 需记录客户端首输出来自P还是D、候选列表/草稿上下文是否实际交接；示例代理回传D流不能替代在线协议核对。 | 分析直接性能收益0；后续可减少KV就绪/加载的暴露等待、优化传输并发或布局。纯传输加速未必等比例降低TTFT，幅度待实测。 指标：KV请求到D侧可用p50/p95/p99、各段等待、实际传输字节/完成时间、有效GB/s、D空闲、TTFT、失败/重算率。 | G01 / G02 / G03 |
| R01 / P0 | X / 验证 | 做功能、精度、恢复与混合负载回归 | 待认领（部署 / 压测） | 待实验；用户已有部分功能验证；后续每项优化仍需对应场景验收。 | 直接性能收益0；防止局部加速换来精度、尾延迟或稳定性退化。 指标：精度、错误/超时、p95/p99、goodput、HBM与长稳。 | G02 |
| R02 / P0 | X / 归档 | 持续更新任务状态与可复现资产 | 待认领（性能分析 / 项目统筹） | 持续归档；已有学习地图、版本报告与profiling报告；本台账连接全局位置和责任。 | 直接性能收益0；减少重复试验，沉淀可交接知识。 指标：可复现案例、结论证据完整率、完成任务/待验证任务。 | — |
| R03 / P1 | S3/S7 / 验证 | 为EPLB收益实验补齐稳定性与路由精度准入 | 待认领（模型适配开发） | 待采集；对方记录Decode EPLB算子core和Prefill多batch精度异常；这只是风险线索，不能认定本服务存在同一故障或已从均衡获益。 | 验证直接性能收益0；通过准入后才评估减少EP热点的收益，不以错误请求减少伪造吞吐。 指标：异常/超时率、路由/输出精度、专家token分布、迁移开销和稳态goodput。 | G01 / B05 / C03 / R01 |

## 子事项实施与验收

### G01 冻结真实基线与生效配置

位置：X 跨阶段。责任人：待认领；建议角色：部署 / 压测；协同：vLLM / Ascend 框架开发 / NPU / CPU / 通信运维。

1. 记录镜像digest、两个仓库SHA、模型/量化版本、torch_npu/CANN、芯片及rank映射。
2. 保存P/D完整命令、最终解析配置、实际加载路径；列清async、SWA、SP、图、MTP、connector。
3. 输入/输出长度、并发、到达率、缓存状态和采集边界与trace一一对应。

交付物：可复用的基线清单、配置diff和采集manifest。

验收：第三人能用同一清单复现实验，未知字段明确标记而非猜测。

前置：—。证据：E01 / E08。

### G02 建立请求级指标与固定压测集

位置：S0 网关/API接入与排队 / S8 流式返回。责任人：待认领；建议角色：部署 / 压测；协同：性能分析 / 项目统筹。

1. 区分长短prompt、输出长度、单/多并发、饱和负载和Agent多轮；固定tokenizer与实际token数。
2. 将冷缓存、相同前缀命中、热身分开；随机输入seed可复现且不意外复用前缀。
3. 至少多次交错A/B；保留原始流式时间戳、错误、取消、超时与完成量。

交付物：统一压测数据集与TTFT/ITL/TPOT分位数、吞吐、goodput基线。

验收：负载、完成量和缓存口径相同；结论附自然波动与样本数。

前置：G01。证据：E10。

### G03 建立请求→step→rank→kernel关联与关键路径

位置：X 跨阶段。责任人：待认领；建议角色：性能分析 / 项目统筹；协同：vLLM / Ascend 框架开发 / 部署 / 压测。

1. 关联客户端/网关/API/P/D的request_id与attempt，再关联scheduled_tokens、step、进程/rank、Python scope和launch连接；网关排队和KV交接分别见F01/F02。
2. 标出有效prefill、dummy、draft/verify、采集首尾；跨机先核对时钟，勿用本地step序号硬对齐。
3. 分别计算阶段墙钟与多流区间并集，找客户端延迟的关键依赖。
4. 分别统计外层execute_model调用、嵌套scope、主干前向窗口、有效Prefill和请求数；将dummy与连续真实轮次分开，交叠量标明包络/任务并集/CPU运行口径。

交付物：一张带时间边界的请求关键路径图和可复查事件索引。

验收：不重复累计嵌套scope、多流任务和HCCL子任务；不把Free等同Host耗时。

前置：G01 / G02。证据：E01 / E04 / E13 / E14 / E34 / E36 / E38。

### G04 采集A3 2P2D、P侧32DP×TP1、128k输入的对照profiling

位置：S2 P 端 Host 输入准备 / S3 P 端模型计算 / S4 采样与结果回传。责任人：待认领；建议角色：部署 / 压测；协同：性能分析 / 项目统筹 / 模型适配开发 / NPU / CPU / 通信运维。

1. 冻结2P2D启动清单；核实P侧32DP指两台P合计还是单engine值，保存各engine DP/TP/实际EP组与所有rank映射；不要由DP直接推断EP。
2. 使用同一128k输入，保存tokenizer与实际token数、并发/到达率、max-num-batched-tokens、long-prefill阈值及缓存状态。按A5有效step的query/历史KV/请求数选择对照窗口，不能仅匹配总prompt长度。
3. 采集P侧完整EP组、连续多个稳定有效step，保留Device任务、shape/dtype/layout、stream/connection与低开销函数关联栈；用未采集时的请求级结果检查profiling扰动。
4. 记录实际专家数/top-k、每专家token数和均衡方式。A5旧样本未强制均衡；A3强制均衡只控制A3自身，不自动使两侧MoE计算量相同。
5. 记录SWA/尾窗裁剪、Engram、DSpark、量化、Runner、源代码SHA和CANN。复用Device模块归属脚本，分别输出主干/草稿、通信、任务并集；可比条件不足的行只列算子/功能差异，算子性能用匹配工作量的微基准补证。

交付物：新A3压缩包、采集manifest、全rank/多step事件索引及同版Device对照表（逐行可比性标注）。

验收：按同一功能与时间边界复算表格；主干/草稿分开，通信不漏算不重复算；不可比行禁用速度比，未知配置明确留空。

前置：G01 / G02 / G03。证据：E39 / E40。

### A01 确认异步调度和Runner实际路径

位置：S1 引擎调度与缓存判定 / S2 P 端 Host 输入准备 / S6 D 端调度与准备。责任人：待认领；建议角色：vLLM / Ascend 框架开发；协同：部署 / 压测。

1. 核对最终async_scheduling值、兼容性降级日志、executor与Runner类。
2. 用execute_model、输入准备、输出消费与上一轮设备任务识别真正重叠。
3. 把Host等待区间与有用CPU准备分开，不用scope重叠反推开关。

交付物：每份capture的执行路径卡片与开关真值表。

验收：实际配置和调用栈相符；采集时未知配置保持未知。

前置：G01。证据：E03 / E08 / E13 / E20 / E36 / E37。

### A02 分类并追踪同步点的关键依赖

位置：S2 P 端 Host 输入准备 / S3 P 端模型计算 / S4 采样与结果回传 / S6 D 端调度与准备 / S7 D 端重复生成。责任人：待认领；建议角色：性能分析 / 项目统筹；协同：vLLM / Ascend 框架开发 / 模型适配开发。

1. 逐处分类Host阻塞同步、设备流依赖、collective等待和采集器同步。
2. 记录线程、调用栈、被等待事件、是否阻塞下一次有效下发。
3. 统计未被计算覆盖的等待与实际device空洞，排除后台输出wait和profiler stop误归因。

交付物：同步点台账：位置/原因/正确性约束/关键路径贡献/能否优化。

验收：至少复现多个有效step；每个删除或后移建议有数据依赖证明。

前置：G03 / A01。证据：E03 / E09 / E34。

### A03 核对D2H与异步输出是否挡住流水

位置：S4 采样与结果回传 / S8 流式返回。责任人：待认领；建议角色：vLLM / Ascend 框架开发；协同：模型适配开发。

1. 记录D2H方向/字节数/同步方式/Host消费者，不把所有copy_或item都当D2H。
2. 区分采样结果、序列长度/接受token统计、调试输出与KV搬运。
3. 仅对关键路径候选评估批量回传、延后消费或保留设备状态；保留完成事件和buffer寿命。

交付物：D2H依赖图与可后移/必须等待列表。

验收：不读未完成结果；端到端p95和完成量共同验证，不能只报API耗时。

前置：A02。证据：E03 / E09。

### A04 拆解DP元数据同步和Gloo等待

位置：S2 P 端 Host 输入准备 / S6 D 端调度与准备。责任人：待认领；建议角色：vLLM / Ascend 框架开发；协同：NPU / CPU / 通信运维 / 模型适配开发。

1. 比较所有DP进入同步的时间、有效/dummy状态和各自输入准备。
2. 拆为已隐藏等待、上游晚到、通信本身与Host处理；不要把88.8ms全部算收益空间。
3. 先查晚到原因；仅在依赖允许时评估同步时机或元数据粒度。

交付物：跨DP到达/完成对齐图和暴露等待预算。

验收：同一共同轮次、完整参与rank；改动不破坏EP参与和token padding一致性。

前置：G03 / C01。证据：E01 / E04。

### A05 复用step内SWA/DSA元数据与visible_lens

位置：S2 P 端 Host 输入准备。责任人：待认领；建议角色：模型适配开发；协同：vLLM / Ascend 框架开发 / 部署 / 压测。

1. 确认各层输入、缓存组与窗口参数是否等价，先做同step安全复用。
2. 测RepeatInterleave、Index/Check等调用是否真实减少，检查分配与Python处理。
3. 复用缓存在step/请求状态改变时失效，覆盖chunk、抢占、prefix命中与窗口边界。
4. 借鉴分组复用/异步元数据规划：先复用相同输入和buffer，再将独立规划提前；为async建立版本化双buffer和消费event，避免上一轮读取被覆盖。

交付物：小范围patch、元数据微基准、正确性用例与端到端A/B。

验收：元数据逐项等价；多轮无陈旧状态；端到端收益超过自然波动。

前置：G02 / A02 / B01。证据：E02 / E21 / E30。

### A06 优化Host下发与准备热路径

位置：S2 P 端 Host 输入准备 / S6 D 端调度与准备。责任人：待认领；建议角色：vLLM / Ascend 框架开发；协同：模型适配开发 / NPU / CPU / 通信运维。

1. 先做无profiler与轻量采集对照，再补CPU调度/频率/亲和观测；稳定复现后才决定环境调整或代码实验，不预设Host抖动为主因。
2. 以首层前准备为起点，按未覆盖Host时间排序；结合真实launch、前序设备任务和event依赖，区分Python计算、runtime下发与等待。
3. 评估持久buffer、批量构造、移除无效转换和小操作复用；区分CPU张量与设备张量。
4. 核对NUMA/绑核和线程竞争，避免把运行环境问题改成复杂代码。
5. CPU绑核开启后核对各TID实际affinity，结合pidstat/perf sched区分CPU运行、就绪等待和阻塞；Host函数墙钟不能直接作为CPU运行时间。

交付物：Host热点清单及每次只改一个因素的patch/实验。

验收：有明确关键路径收益；输出、请求隔离和异步buffer寿命正确。

前置：G03 / A02。证据：E02 / E10 / E13 / E14 / E29 / E30 / E34。

### A07 做异步调度受控A/B

位置：S1 引擎调度与缓存判定 / S2 P 端 Host 输入准备 / S6 D 端调度与准备 / S7 D 端重复生成。责任人：待认领；建议角色：vLLM / Ascend 框架开发；协同：部署 / 压测 / 模型适配开发。

1. 两组分别显式传--async-scheduling与--no-async-scheduling，保存镜像/core SHA、最终配置和输出线程路径；区分底层异步、内部多流、D2H与调度异步。
2. 在版本支持的组合中显式ON/OFF，核对实际生效；禁止借切换Runner同时改变多个变量。
3. 覆盖长prefill、混合负载、decode与MTP；检查有用准备提前量。
4. 观察排队/吞吐、尾延迟与CPU占用，区分Host空档减少和collective等待转移。

交付物：异步调度兼容矩阵、前后时间线与服务级收益表。

验收：同工作量多次复测；无错误、输出状态错配和饥饿；给出无收益场景。

前置：A01 / A02 / G02。证据：E03 / E08 / E13 / E20 / E34 / E36 / E37。

### A08 模型多流与可切分计算的重叠评估

位置：S2 P 端 Host 输入准备 / S3 P 端模型计算 / S6 D 端调度与准备 / S7 D 端重复生成。责任人：待认领；建议角色：模型适配开发；协同：vLLM / Ascend 框架开发 / 部署 / 压测。

1. 先按B08核对各多流开关、最终值、实际stream和互斥；单个约0.108ms重叠例子不代表总重叠或收益。
2. 画出输入准备、独立投影、量化、通信与消费子图的数据依赖，选独立分支。
3. 评估torch_npu stream/event与生命周期；仅对允许的子图切分、流水，检查共享内存与核资源竞争。
4. 记录收益是否来自计算通信重叠，检查新增同步、HBM峰值与图捕获限制。

交付物：依赖图、最小多流/切分实验和正反例。

验收：数值等价、无数据竞争；支持配置边界明确，保留必要event。

前置：B01 / A02 / G02。证据：E09 / E13 / E14 / E15 / E16 / E29 / E30 / E31。

### A09 核对Decode DP同步绕过的版本、算子与一致性条件

位置：S6 D 端调度与准备 / S7 D 端重复生成。责任人：待认领；建议角色：vLLM / Ascend 框架开发；协同：CANN / 算子对接 / 模型适配开发 / 部署 / 压测。

1. 在D侧逐step关联DP元数据collective、padding、global_bs/mc2_mask、graph模式与MTP；区分TP/EP数据通信，先量化暴露同步而非累加API耗时。
2. 核对should_skip_allreduce_across_dp_group在V1/V2的调用及实际返回值，列清KV consumer、普通MC2、重算调度、hierarchy、MegaMoE的条件矩阵。
3. 取得对方具体PR/算子ABI和动态global_bs能力；确认dummy、SWA跨DP协调、投机、抢占与加载失败恢复的协议后再评估适配。
4. 先在现有支持路径做合法对照；不能删除MegaMoE guard来追求日志里少一个all_reduce。

交付物：D侧同步关键路径、版本条件矩阵和是否需要算子/框架协同的结论。

验收：分清P/D和V1/V2；不破坏统一输入/通信合同，支持边界经精度与异常恢复验证。

前置：G01 / A02 / B05。证据：E13 / E14 / E21 / E22。

### B01 建立同口径算子与执行引擎清单

位置：S2 P 端 Host 输入准备 / S3 P 端模型计算 / S7 D 端重复生成。责任人：待认领；建议角色：性能分析 / 项目统筹；协同：模型适配开发 / CANN / 算子对接。

1. 按阶段、目标/draft、真实kernel、完整shape、dtype、layout/stride和调用栈归档。
2. 标出AI Core/Cube、Vector、AICPU、SDMA与通信；区分API、CPU op和硬件任务。
3. 比较每次耗时、次数、并集关键路径及融合前后功能范围，避免同名不同工作量。

交付物：可筛选算子目录与公平比较的可比/不可比矩阵。

验收：明确已融合功能，不能以算子减少数量直接证明端到端收益。

前置：G01 / G03。证据：E01 / E02。

### B02 核对并替换可用Norm/量化融合路径

位置：S3 P 端模型计算 / S7 D 端重复生成。责任人：待认领；建议角色：模型适配开发；协同：CANN / 算子对接 / 部署 / 压测。

1. 核对A3可用融合算子及硬件支持，确认BF16归一化结果是否还有其他消费者。
2. 检查INT8/MXFP等格式、scale布局、双输出、精度与shape支持；不直接搬A5专用内核。
3. 先做等价替换微基准再测目标层和服务，缺少可用接口时转B06。

交付物：兼容性表、可用接口替换方案或明确不可复用的理由。

验收：消费者语义和量化精度一致；关注整体子图而非单个算子。

前置：B01 / G02。证据：E02。

### B03 Device-Metadata：复用与融合窗口/索引小算子

位置：S2 P 端 Host 输入准备 / S6 D 端调度与准备。责任人：待认领；建议角色：模型适配开发；协同：vLLM / Ascend 框架开发 / CANN / 算子对接。

1. 关联prepare_attn/build_attn_metadata到build_replay_swa_visible_lens及设备任务，按缓存组、窗口、query_start_loc/seq_lens、压缩与尾窗坐标核对输入等价性；42次不能直接合为一次。
2. 先在同step语义相同的组内复用visible_lens与buffer，再独立评估RepeatInterleave/Index/IndexCheck/边界计算融合；分别量化复用和融合的贡献。
3. 核对A3 INT8路径支持的aclnn或可维护Triton实现；保留必要依赖，避免改动引入D2H同步和跨step陈旧状态。
4. 覆盖变长、空query、dummy/padding、chunked prefill、prefix命中和尾窗裁剪；在新A3样本及A3单变量A/B中复算Device调用数/字节数、阶段墙钟及正确性。

交付物：Device-Metadata子图规格、复用/融合独立实验、微基准及A3模型执行A/B。

验收：语义逐项等价，无陈旧metadata和新增同步；同一A3配置单变量实验得到超出波动的收益。

前置：B01。证据：E02 / E39。

### B04 定位Attention/Indexer/MLA的瓶颈与调用条件

位置：S3 P 端模型计算 / S7 D 端重复生成。责任人：待认领；建议角色：模型适配开发；协同：CANN / 算子对接 / 性能分析 / 项目统筹。

1. 同shape同上下文比较，区分计算、HBM读写、元数据与稀疏选择工作量。
2. 核对布局/连续性、额外转置、量化转换和已有融合接口，确认实际走预期backend。
3. 结合PMU/内存指标与关键路径证据判断bound；单看耗时占比不能判断算力不足。
4. 将DSL Attention/量化Indexer作为现有算子调用替换候选，核对A3支持、quant/layout、workspace、cache索引和masked tail；其组合收益不能分拆归为Attention单项。

交付物：Attention子图瓶颈结论、可在调用侧处理的项及CANN问题单材料。

验收：不能优化时给出证据和边界，不把A3/A5原始耗时比当硬件倍率。

前置：B01。证据：E01 / E10 / E21。

### B05 拆解MegaMoE的计算、搬运、通信与等待

位置：S3 P 端模型计算 / S7 D 端重复生成。责任人：待认领；建议角色：性能分析 / 项目统筹；协同：模型适配开发 / NPU / CPU / 通信运维 / CANN / 算子对接。

1. 同一共同轮/层对齐所有EP rank，记录到达/结束、tokens和专家负载。
2. 结合MAC/向量/内存/通信数据拆分可计算部分和等待；内部不可见部分明确为未知。
3. 从调用侧评估输入格式、token分布、已有通信策略和粒度；核内瓶颈交B06。
4. 对方MegaMoE+cannbotdsl结果仅作融合方向；先交出shape、dtype、专家token分布、通信组与等待拆分，确认现有Mega已融合哪些阶段及对方依赖后再做调用替换。
5. 核对AG分发/A2A回收或token段切换的真实含义；分解Routing、计数D2H、GMM和回收，补齐固定/可变分块协议、量化精度及峰值显存，再设计单变量验证。

交付物：MegaMoE归因图和模型/框架/算子方各自可行动清单。

验收：包含完整EP组并排除采集边界；不能仅凭低MAC证明全部时间都在等。

前置：B01 / C01。证据：E04 / E13 / E14 / E15 / E21 / E22 / E35。

### B06 整理重型算子最小复现并交接算子团队

位置：S3 P 端模型计算 / S7 D 端重复生成。责任人：待认领；建议角色：CANN / 算子对接；协同：模型适配开发 / 部署 / 压测。

1. 导出shape/dtype/stride/layout、tiling条件、并发流、版本与可复现输入。
2. 提供带设备完成等待的微基准、预热、分位数、精度基线和原始trace。
3. 跟踪可用版本、融合接口、fallback或形状适配方案，回到服务验收。

交付物：可交接的算子性能问题包与责任接口。

验收：复现能在对应硬件运行；无数据时不指定compute/memory bound结论。

前置：B04 / B05。证据：E10。

### B07 区分Eager、图重放和static kernel实际命中

位置：S2 P 端 Host 输入准备 / S3 P 端模型计算 / S6 D 端调度与准备 / S7 D 端重复生成。责任人：待认领；建议角色：vLLM / Ascend 框架开发；协同：模型适配开发 / 部署 / 压测。

1. 标记prefill/decode/draft各路径的图捕获范围、回退原因和shape档位。
2. 比较图重放减少的Host下发与padding额外工作，区分SWA replay和graph replay。
3. 核对Runner、版本与算子支持后才做受控开关实验。
4. 冷启动编译/加载/捕获交D08单独计时；运行时以实际kernel、graph replay和回退证据判断命中。

交付物：执行模式覆盖表与可实验的shape/配置集合。

验收：相同工作负载和精度；记录未命中场景与显存代价。

前置：G01 / B01。证据：E01 / E06 / E17 / E18 / E19。

### B08 核对共享专家多流互斥并验证支持路径的重叠收益

位置：S3 P 端模型计算 / S7 D 端重复生成。责任人：待认领；建议角色：模型适配开发；协同：vLLM / Ascend 框架开发 / 部署 / 压测 / CANN / 算子对接。

1. 保存用户enable_fused_mc2与归一化后的值、Mega可用性/实际后端，以及multistream_overlap_shared_expert最终值和降级日志。
2. 关联共享专家MLP、路由专家、DSA准备的stream与event；逐层计算区间并集、关键路径及资源竞争。
3. 在支持共享专家overlap的相同后端上做OFF/ON单变量对照；若必须切换fused后端，将其登记为组合方案，单独比较。
4. 不直接删除互斥；若要扩展兼容性，先设计HCCL顺序、输入/输出就绪和图捕获约束，再独立评审实现。

交付物：配置互斥矩阵、逐层流依赖图、有效重叠和服务指标对照。

验收：区分单项与组合变量；证明实际后端和流已生效，端到端收益经重复压测且无数值或通信退化。

前置：G01 / G02 / B01 / A02。证据：E13 / E14 / E15 / E16 / E32。

### B09 确认MoE Gate实际dtype路径并评估保精度的转换消减

位置：S3 P 端模型计算 / S7 D 端重复生成。责任人：待认领；建议角色：模型适配开发；协同：vLLM / Ascend 框架开发 / CANN / 算子对接 / 部署 / 压测。

1. 逐层记录Gate权重/输入/输出dtype、internal router、hash层和实际cast/matmul，核对是否有可复用FP32中间值。
2. 区分A3 FP32预转换和A5融合cast的现有实现，再判断对方GateLinear改动是否已存在、是否适用该模型。
3. 若精度合同允许，单独验证减少输入转换/复用投影；保留FP32输出并对比router logits、TopK专家ID/权重、模型输出与Agent任务质量。

交付物：路由精度合同、cast热点与适用/不适用模型清单，必要时最小调用改动。

验收：实际命中和数值边界可审计；未经验证的强制FP32模型不改精度策略。

前置：G01 / B01 / G02。证据：E21 / E23 / E24。

### B10 评估Attention尾部融合、投影布局与直接写回

位置：S3 P 端模型计算 / S7 D 端重复生成。责任人：待认领；建议角色：模型适配开发；协同：CANN / 算子对接 / vLLM / Ascend 框架开发 / 部署 / 压测。

1. 先补D侧目标模型/draft的带stack短trace；逐个标注Attention输出、逆RoPE、量化、wo_a/wo_b、slice/format_cast/copy及其暴露时间。
2. 按具体矩阵消费者核对ND/NZ；不能从某Attention MatMul未用NZ推导全模型应统一NZ或ND。
3. 核对已有融合和out/output-buffer接口可否覆盖目标dtype/shape；以减少中间tensor、搬运、launch为主要开发点，核内问题交算子团队。
4. 直接写回需验证输出连续性/stride、别名、跨stream生存期和图捕获；MXFP8方案先确认A3硬件、量化和算子支持，保留不适用结论。

交付物：Attention尾部依赖/字节数图、已有融合覆盖表与可控调用替换实验。

验收：相同硬件/量化/负载下验证；写回无别名/竞态问题，区分Prefill和Decode收益。

前置：G01 / B01 / B04 / G02。证据：E01 / E14 / E21 / E24。

### B11 将主干共享专家融合进MegaMoE：核对A3支持并验证

位置：S3 P 端模型计算。责任人：待认领；建议角色：模型适配开发；协同：CANN / 算子对接 / 性能分析 / 项目统筹 / 部署 / 压测。

1. 按捕获栈核对共享专家调用与MegaMoE接口，逐项检查A3芯片、CANN原型、INT8权重/scale/layout、共享输入及EP/TP条件。固定参考源码的融合入口限制W4A8MXFP，还要求无dynamic EPLB/mix placement/冗余专家，不能直接删除这些保护。
2. 给出支持结论：已有兼容接口可适配、需要算子接口扩展，或当前硬件/量化不支持。前两种情况才安排实现；不以切换量化/后端的组合变化冒充融合单项收益。
3. 若支持，验证共享结果合并、scale与TP输出语义；保留回退路径，检查独立MLP是否消失、融合事件/参数是否真实命中，不能仅凭算子数下降判断正确。
4. 在同一A3拓扑/量化/专家负载下做OFF/ON，先微基准再模型有效step。比较MoE整体关键路径、显存、精度和端到端指标；新TP1样本的共享专家成本需重新量化。

交付物：A3支持矩阵与实施/延期结论；支持时交付适配patch、调用命中证据及A3 A/B。

验收：先确认支持再实现；融合功能与数值一致，实际命中且收益超过重复实验波动，无通信顺序/恢复退化。

前置：B01 / B05。证据：E39。

### C01 对齐通信组、共同轮次与物理卡映射

位置：X 跨阶段。责任人：待认领；建议角色：性能分析 / 项目统筹；协同：NPU / CPU / 通信运维 / 部署 / 压测。

1. 建立DP/TP/EP/PCP/PP到进程、host、物理device映射。
2. 同轮同层比较有效/dummy与到达/结束；不要用各rank本地step编号直接对齐。
3. 标注不完整首尾采集，跨机时间先校准。

交付物：拓扑图、rank映射和共同轮次索引。

验收：全参与组可复查；未知映射明确标记。

前置：G01 / G03。证据：E04。

### C02 做同工作量的计算卡间一致性分析

位置：S3 P 端模型计算 / S7 D 端重复生成。责任人：待认领；建议角色：性能分析 / 项目统筹；协同：NPU / CPU / 通信运维 / CANN / 算子对接。

1. 按kernel、shape、dtype、layout/stride、tiling、层位置与并发条件配对。
2. 跨多轮比较各卡分位数，覆盖MatMul、Norm等多类纯计算，独立于collective耗时。
3. 确认是否始终同一物理卡偏慢；记录局部异常而不武断判卡故障。

交付物：按物理卡的稳定性矩阵、异常样本与待排除因素。

验收：单一算子异常不作为固定慢卡结论；对异常用独立微基准复核。

前置：C01 / B01。证据：E05。

### C03 查明晚到rank和EP拖尾的来源

位置：S2 P 端 Host 输入准备 / S3 P 端模型计算 / S6 D 端调度与准备 / S7 D 端重复生成。责任人：待认领；建议角色：vLLM / Ascend 框架开发；协同：模型适配开发 / NPU / CPU / 通信运维。

1. 从每层collective/Mega反向追到Host准备、不同token量、KV加载和CPU调度。
2. 区分DP请求不均、专家token不均、通信慢和单卡计算慢。
3. 将等待归到实际晚到路径，评估均衡与调度方案而非删除dummy参与。
4. 分开DP请求/剩余token不均与EP专家token不均；记录每DP真实chunk/队列/缓存命中、每专家负载和最后到达rank，不把一组层0晚到当成已证明DP路由根因。

交付物：晚到根因排序和每类等待的可回收预算。

验收：同组共同轮次、输入工作量可解释；不破坏MoE collective一致性。

前置：C01 / G03。证据：E04 / E13 / E14 / E21 / E29 / E30 / E34。

### C04 核对频率、温度、CPU/NUMA与资源争用

位置：X 跨阶段。责任人：待认领；建议角色：NPU / CPU / 通信运维；协同：部署 / 压测 / 性能分析 / 项目统筹。

1. 同步采集运行频率/温度/功耗、HBM和链路状态，保留时间戳。
2. 核对CPU绑核、NUMA、进程负载、共享设备与后台任务，关联异常step。
3. 区分硬件限制、环境部署和模型实现，不只比较型号标称频率。

交付物：资源遥测与异常step对齐证据。

验收：与异常同一时段采集；保留正常对照，不用事后快照下结论。

前置：C01 / C02。证据：E05。

### C05 分析模型内HCCL集合通信与链路差异

位置：S3 P 端模型计算 / S7 D 端重复生成。责任人：待认领；建议角色：NPU / CPU / 通信运维；协同：vLLM / Ascend 框架开发 / 部署 / 压测。

1. 核对TP/EP等通信组、协议、接口、路由与同组rank到达差。
2. 记录同数据量下的集合通信延迟、带宽、建链与失败；冷/热连接分开。
3. 比较通信与计算重叠、未覆盖等待，不重复累加HCCL大算子及子任务；P/D与Store链路转F02。
4. 对CCU或新通信引擎先核对硬件/版本/拓扑、消息大小和实际引擎命中；本团队先做调用侧证据与复现，不能预设CCU在当前A3和小消息下可用或更快。

交付物：模型内集合通信的拓扑、基线与瓶颈归属。

验收：完整通信组、同工作量对照；不以集合通信耗时替代P/D KV传输时长。

前置：C01 / G01。证据：E04 / E07 / E21。

### C06 通过绑核/换卡与独立微基准验证慢卡假设

位置：X 跨阶段。责任人：待认领；建议角色：NPU / CPU / 通信运维；协同：CANN / 算子对接 / 部署 / 压测。

1. 隔离其他负载，固定shape/dtype，做多种算子的逐卡独立基准。
2. 在验证环境交换rank/物理卡或CPU绑定，观察异常跟卡还是跟任务。
3. 对通信异常使用相应链路测试，不拿计算微基准替代网络诊断。

交付物：慢卡假设的确认/排除报告及可复现步骤。

验收：没有稳定复现就保留未确认，不能以一次极值判坏卡。

前置：C02 / C04。证据：E05。

### D01 建立固定资源预算的DP/TP/EP切分矩阵

位置：S1 引擎调度与缓存判定 / S3 P 端模型计算 / S6 D 端调度与准备 / S7 D 端重复生成。责任人：待认领；建议角色：部署 / 压测；协同：模型适配开发 / vLLM / Ascend 框架开发。

1. 在同型号、固定物理卡预算内选择能装下模型的DP/TP/EP组合；先核对支持和HBM。
2. 保持模型/量化、数据集、到达率与缓存条件一致；每组记录本地token量和通信域。
3. 若评估PCP/PP/SP等，另开单变量实验并核对SWA、图与Runner限制。
4. 将对方PCP8与DP4×PCP2登记为拓扑候选；固定P侧总rank/芯片预算并记录TP、PCP与EP组，先过模型/Runner/SWA/SP/connector兼容检查。

交付物：各切分的容量/延迟/吞吐曲线与适用负载。

验收：同资源可比；同时报告精度、失败、显存和每请求代价。

前置：G01 / G02 / C01。证据：E01 / E21 / E28。

### D02 扫描合批、token预算与prefill切块

位置：S1 引擎调度与缓存判定 / S2 P 端 Host 输入准备 / S3 P 端模型计算 / S6 D 端调度与准备 / S7 D 端重复生成。责任人：待认领；建议角色：部署 / 压测；协同：vLLM / Ascend 框架开发 / 模型适配开发。

1. 分别扫描max_num_seqs、max_num_batched_tokens和chunk策略，区分P与D预算。
2. 固定输入长度分布和发送方式，测低负载与接近饱和场景。
3. 解释shape变化与准备/主干/通信成本，观察长prefill对交互尾延迟影响。
4. 结合对方batch与长Prefill阈值经验，在同一A3配置分别扫描max_num_batched_tokens、long_prefill_token_threshold和max_num_seqs；记录真实chunk、实际batch及尾窗重建次数。核对实际scheduler与参数归一化，不能把配置上限当有效batch。
5. 诊断组固定8192预算，并发64→4，核对每DP分发；保持128k实际输入与缓存条件，验证排队变化。该组负载变化不能当作64并发优化收益。
6. 性能组固定64并发，预算8192→16384，确认max_num_scheduled_tokens/实际scheduled_tokens生效；记录真实Prompt吞吐、queue、P Prefill、客户端TTFT及显存/抢占，多次交错对照。
7. 若线上要求64并发，容量不足时评估P侧算力/DP-TP-PCP配比与有效缓存；降低首调度等待但延长Prefill或把等待转移到网关均不算TTFT改进。

交付物：配置扫描表、SLO可承载点与推荐范围。

验收：相同SLO和完成量；不只选吞吐最大的点。

前置：G02 / D01。证据：E10 / E21 / E25 / E26 / E36 / E37。

### D03 匹配P/D能力与KV交接容量

位置：S1 引擎调度与缓存判定 / S5 KV 交接与外部缓存 / S6 D 端调度与准备 / S7 D 端重复生成。责任人：待认领；建议角色：部署 / 压测；协同：vLLM / Ascend 框架开发 / NPU / CPU / 通信运维。

1. 补采P/D和客户端，记录P计算完成、KV就绪、D可运行和首输出的关系。
2. 按输入/输出长度及到达率估计两侧服务能力，识别P排队、D排队或传输瓶颈。
3. 再比较P/D实例与卡数配比；不得只按日活数推算容量。
4. 将2P1D EP8负收益和EP16/EP32卡效率作为反例；先证明D缺粮来自P产出、路由还是KV等待，再增加P/D资源，并同时报告总吞吐与所有P+D芯片归一化效率。

交付物：P/D负载与容量模型、交接关键路径和配比实验计划。

验收：同一请求ID与可靠时钟；重叠路径不简单相加。

前置：G02 / F01 / F02。证据：E10 / E21。

### D04 拆开本地prefix cache与Store收益

位置：S1 引擎调度与缓存判定 / S5 KV 交接与外部缓存 / S6 D 端调度与准备。责任人：待认领；建议角色：vLLM / Ascend 框架开发；协同：模型适配开发 / 部署 / 压测。

1. 固定connector选择策略、consumer加载设置与命中来源，明确冷/本地热/Store热三个状态。
2. 比较实际跳过的token、KV字节、命中查询/加载/保存成本；SWA按缓存组确认窗口状态。
3. 覆盖加载失败、抢占重调度和recompute，检查异步完成后才消费。
4. 对方DRAM池化与Session Affinity需要拆本地APC/外部Store：用冷缓存、重复前缀和真实Agent会话分别观察命中、传输与排队，防止把缓存命中变化归给SWA或核融合。

交付物：缓存来源真值表与收益/代价、恢复正确性报告。

验收：命中token数不等于SWA窗口已恢复；不能用100%外部命中率单独证明正确。

前置：G01 / G02 / F02。证据：E07 / E10 / E21。

### D05 验证SWA replay相关机制的条件与净收益

位置：S1 引擎调度与缓存判定 / S2 P 端 Host 输入准备 / S3 P 端模型计算 / S5 KV 交接与外部缓存 / S6 D 端调度与准备。责任人：待认领；建议角色：模型适配开发；协同：vLLM / Ascend 框架开发 / 部署 / 压测。

1. 沿此前SWA/CED专题逐项确认实际代码路径，不将专题名当成一个独立开关或PR。
2. 保持Runner、SP/并行、量化、输入和缓存条件一致；检查后段每请求窗口对应的batch shape。
3. 分别核对前向计算、窗口恢复、额外元数据与传输代价；不为开启而删除布局正确性限制。

交付物：支持矩阵、精度结果、ON/OFF原始数据及净收益归因。

验收：功能与精度通过；P/D/Store分别验证，不以开关或启动日志代表必然生效。

前置：G02 / A05 / D04。证据：E01 / E07。

### D06 扫描MTP / draft配置与有效输出效率

位置：S4 采样与结果回传 / S7 D 端重复生成。责任人：待认领；建议角色：模型适配开发；协同：vLLM / Ascend 框架开发 / 部署 / 压测。

1. 确认P和D上的MTP角色、num_speculative_tokens、draft/verify执行方式。
2. 固定负载与sampling参数，比较关闭及支持的draft长度，记录接受率与实际输出token数。
3. 计算每个有效输出token的目标/draft/准备成本，观察图命中与KV开销。

交付物：投机配置的接受率、代价和服务收益表。

验收：以有效输出token归一，控制输出长度/EOS；不得只以接受率高低判断收益。

前置：G02 / B07。证据：E01 / E06。

### D07 分离Runner、量化、KV布局和通信开关影响

位置：S2 P 端 Host 输入准备 / S3 P 端模型计算 / S6 D 端调度与准备 / S7 D 端重复生成。责任人：待认领；建议角色：模型适配开发；协同：vLLM / Ascend 框架开发 / CANN / 算子对接 / 部署 / 压测。

1. 优先在同硬件固定一个可运行基线，分别检查MRV1/V2、支持的权重量化/KV dtype和布局。
2. 对FlashComm/MC2等按代码支持范围做单变量对照，检查实际backend与SP语义。
3. 跨硬件不可统一的格式或算子列为固有限制，不勉强声称纯硬件公平比。

交付物：受控差异矩阵、单项收益与组合兼容范围。

验收：每条结论注明硬件/版本/模型/量化；组合收益不可把单项百分比相加。

前置：G01 / G02 / B01。证据：E01 / E06 / E38。

### D08 拆分D侧冷启动与缓存复用，定位静态kernel重复编译

位置：X 跨阶段。责任人：待认领；建议角色：vLLM / Ascend 框架开发；协同：部署 / 压测 / CANN / 算子对接。

1. 在实际D容器记录镜像digest、CANN/torch_npu/npugraph_ex或torchair版本、完整启动命令、工作目录及实例就绪时间；保留“50项”的原始进度日志。
2. 分别计时权重加载、FX编译/加载、静态kernel采集/编译/安装与ACLGraph capture；以实际build directory寻找产物，不只搜VLLM_CACHE_ROOT。
3. 按安装版本确认是否有独立静态缓存支持；保留static_kernel_cache索引及其引用的全部产物，并核对绝对路径、权限、CANN/硬件/shape与Super Kernel选项。
4. 同容器同配置重复启动，并与新容器挂载相同缓存路径对照；用实际op_compiler执行、命中/安装日志和新增编译产物共同判断。没找到.run、出现Starting日志或新目录均不能单独判定。
5. 将FX图遇Triton跳过保存与静态二进制缓存分开；缓存命中后若capture仍主导，再独立扫描capture档位，核对MTP、padding、fallback、HBM与稳态吞吐。

交付物：冷启动/缓存重启分段表、产物与日志索引、可复现缓存复用方案和未支持项。

验收：相同配置多次重启可审计；冷/热与同/新容器分开，输出和稳态性能不退化，未支持的缓存机制明确记录。

前置：G01 / B07。证据：E17 / E18 / E19。

### D09 验证会话亲和与DP负载均衡的收益和冲突

位置：S0 网关/API接入与排队 / S1 引擎调度与缓存判定 / S5 KV 交接与外部缓存 / S6 D 端调度与准备。责任人：待认领；建议角色：vLLM / Ascend 框架开发；协同：网关 / API 服务对接 / 部署 / 压测 / 模型适配开发。

1. 明确Motor/网关/引擎各自职责，采集request/session/attempt→P/D DP的映射、预计未缓存token与队列长度；不记录prompt正文。
2. 在同资源/拓扑与缓存状态下比较现有分发、按待处理token/预计完成时间均衡、带队列上限的会话亲和；分别观察命中提升和热点DP。
3. 出现D利用率不足时，拆P未产出、P队列、KV传输/Store等待与D调度，不能一律归因DP策略；必要时提升F01/F02采集优先级。
4. 用冷/热缓存和真实Agent多轮分开验证；分发与PCP切分分两步变化，Motor是否有收益以本服务对照为准。

交付物：请求分发/缓存/队列看板、亲和与均衡策略对照及回退门槛。

验收：队列和缓存贡献可分离；相同到达率/完成量/总卡数，长短请求无饥饿。

前置：G01 / G02 / C03 / F01 / F02。证据：E04 / E12 / E14 / E21 / E27。

### F01 分析网关/API排队与引擎等待时长

位置：S0 网关/API接入与排队 / S1 引擎调度与缓存判定 / S6 D 端调度与准备 / S8 流式返回。责任人：待认领；建议角色：网关 / API 服务对接；协同：部署 / 压测 / vLLM / Ascend 框架开发 / NPU / CPU / 通信运维。

1. 按request_id/attempt记录网关接收、队列入/出、上游转发、API接收、引擎入队、首次调度和首输出；若网关没有队列也记录实际路径，缺少边界先补埋点。
2. 分别算网关真队列等待、API前处理/交接、引擎入队到首调度；分词、路由、重试和外部KV等待单独归档，不把API到引擎的整段都叫排队。
3. 结合发送率、队列长度、P/D利用率、取消/超时、错误和客户端TTFT分位数；先识别限流、调度/CPU处理还是实例服务能力不足，再设计配置实验。
4. 区分P输出token已在服务内部产生与客户端首次收到；按在线代理的D响应流与跨机KV交接补齐首输出边界。

交付物：请求入口分段时间表、queue p50/p95/p99与TTFT关键路径图，含各边界的采集字段和缺失项。

验收：真实队列入/出有证据；同一请求与attempt可关联；同发送率/完成量比较，不能把等待从网关搬到引擎或丢弃请求当收益；未测到时不填0。

前置：G01 / G02。证据：E10 / E11 / E36 / E38。

### F02 分段分析P/D KV传输与Store读写

位置：S1 引擎调度与缓存判定 / S5 KV 交接与外部缓存 / S6 D 端调度与准备。责任人：待认领；建议角色：NPU / CPU / 通信运维；协同：vLLM / Ascend 框架开发 / 部署 / 压测 / 网关 / API 服务对接。

1. 关联P/D request_id、远端请求ID、attempt、cache group/block、字节数、connector与读写来源；P2P交接和Store lookup/load/save分别建账。
2. 记录每批/层KV真实就绪、加载请求、传输排队、建链、数据搬运起止、D侧加载完成和可消费事件；区分等待P产出、查询/SSD回读、网络传输、额外拷贝/放置以及同步确认。
3. 联合P/D应用span、connector/Transfer Engine日志、传输字节/计数与设备trace；先对齐时钟，按请求关键依赖分析流水重叠、D端空等、缓存命中/失败/重算。

交付物：KV交接分段瀑布图、P2P/Store分来源时延与有效带宽表，以及对客户端TTFT的暴露等待预算。

验收：异步提交不等于传输完成；kernel/拷贝事件不直接等同网传时长；跨机时间有校准/误差，缺失时只报局部区间；分层传输与P计算重叠不简单相加；P2P/Store结果分开。

前置：G01 / G02 / G03。证据：E01 / E12 / E38。

### R01 做功能、精度、恢复与混合负载回归

位置：X 跨阶段。责任人：待认领；建议角色：部署 / 压测；协同：vLLM / Ascend 框架开发 / 模型适配开发。

1. 覆盖单机/P/D、冷/热缓存、长短输入、并发、MTP和支持的并行组合。
2. 除精度外测取消、超时、加载失败、抢占、请求状态与多流buffer复用。
3. 用固定baseline做交错回归；记录稳态、首次请求、显存峰值和长稳情况。

交付物：每项patch/配置的验收记录、回滚点和支持范围。

验收：所有待发布优化先通过本项；基线与变更原始数据可追溯。

前置：G02。证据：E07 / E10。

### R02 持续更新任务状态与可复现资产

位置：X 跨阶段。责任人：待认领；建议角色：性能分析 / 项目统筹；协同：部署 / 压测。

1. 每项任务补具体责任人、实验ID、版本、支持配置、关键证据和结论。
2. 分开标注分析完成、实现完成、功能通过、性能通过；无收益/否定结果也归档。
3. 将确认的分工和实测收益回写JSON，重建HTML/Markdown/CSV并提交。

交付物：团队性能案例库、版本化台账和面向新人的复现教程。

验收：不把上游PR、trace出现或功能通过自动升级为当前服务性能已验证。

前置：—。证据：E06 / E10。

### R03 为EPLB收益实验补齐稳定性与路由精度准入

位置：S3 P 端模型计算 / S7 D 端重复生成。责任人：待认领；建议角色：模型适配开发；协同：CANN / 算子对接 / vLLM / Ascend 框架开发 / 部署 / 压测。

1. 先确认当前模型/量化/冗余专家/动态EPLB支持矩阵及相关修复PR，保留对方core/精度问题的原始日志和环境。
2. 在不同实际batch、长短prompt、MTP、expert-map更新、压力和失败恢复中验证稳定性与输出；确认无问题后再扩大负载。
3. 用每专家token和每rank负载区分路由天然不均与映射不均；同时记录迁移代价、重分布频率和关键路径收益。

交付物：EPLB功能/精度/恢复准入矩阵及专家均衡收益表。

验收：先稳定和精度后性能；不能跳过异常case或仅凭平均专家负载宣称改进。

前置：G01 / B05 / C03 / R01。证据：E04 / E21。

## 建议启动顺序

1. G01/G02/G03 + A01/A02/B01/C01：补齐真实基线、请求级结果和关联，产出同步/算子/rank三张证据清单。
2. A05/A06 + C02/C03/B05：优先验证元数据复用与Host热点，同时查Mega等待和异常MatMul；A07异步开关只在支持组合中受控对照。
3. B02/B03/A08 + D01/D02/D03：再推进可用融合、多流/切分与拓扑/预算；每次只改变计划变量。
4. D04/D05/D06/D07 + R01/R02：缓存、SWA、MTP和执行路径分项验收；逐项回写实测与负结果，再评估组合。
5. 11月27日当前投入为G04采集与B03/B11核对并行，Host耗时对照暂缓；上面保留全项目通用顺序。多流阶段分析已归档，B08共享专家辅助流兼容扩展实验后置；D08独立记录冷启动、缓存重启与实例就绪时间，和热服务时延分开。
6. F01/F02作为P2补充专题，与主线复用request_id与时间戳采集；进入P/D容量或Store收益结论前，先补齐所依赖的排队与KV交接数据。

## 证据索引与适用范围

E03/E05是2026-11-13对同批原始数据库的补充查询摘要，原始数据路径/提取方法沿用A3/A5报告；本页没有附加新的全量同步统计或硬件遥测，不应理解为已排除所有同步/慢卡可能。

### E01 已提交的 A3/A5 主报告

P.rar=A3，DP4×TP4/EP16；bs12.zip=A5，DP8×TP1/EP8。另有历史 A3 ZIP，不能把两份 A3 当成 A3/A5。

[对应资料](../dsv41_a3_a5_prefill/report.html)

### E02 A3/A5 的输入准备与融合证据

代表有效前向：准备硬件任务 1641 vs 221；A3 42 次 RepeatInterleave、Index/IndexCheck 各176次。A5 40 次 RMSNorm+MX 动态量化融合。两组还有 Runner、CANN、量化与负载差异。

[对应资料](../dsv41_a3_a5_prefill/report.html)

### E03 D2H / 同步 / 重叠补充核对，2026-11-13

基于同批 SQLite/trace 的后续查询：A3 DP0/TP0 的两条明确 D2H 来自 sample_tokens→async_copy_to_np；约7.3ms Event wait 在后台输出线程。A5 DP0/TP0 两条 D2H 也来自采样输出。代表 A3 execute_model 内未见阻塞式 Synchronize；这不覆盖所有请求/所有隐式等待。

[对应资料](../dsv41_a3_a5_prefill/README.md)

### E04 跨 rank 的 MegaMoE 原始索引

A3 共同轮0/层0：DP0TP0 dummy 44.239ms，DP1TP0 prefill 0.491ms，结束差约5µs。A5 共同轮2/层0：DP1 dummy 104.821ms，DP2 prefill 5.581ms，结束差约0.464ms。指向内部等待，不能把最长 Mega 当成最慢计算卡。

[对应资料](../dsv41_a3_a5_prefill/data/mega_cross_rank.csv)

### E05 同 shape 卡间补充核对，2026-11-13

相同 kernel/完整输入shape、有效目标模型层内统计：A3 HcPre[1848]，12 rank 中位数352.648–357.117µs，差1.27%；A5 HcPre[128]，7 rank为17.022–17.566µs，差3.20%。A3某MatMul[1848,5120;512,5120]为51.571–78.362µs。尚未控制所有stride/tiling/并发资源因素，不能据此确认物理慢卡。

[对应资料](../dsv41_a3_a5_prefill/README.md)

### E06 DeepSeek 30→31→32 优化清单

已有 PR、代码依据、适用条件与代码量梳理；上游作者收益、当前代码已集成、trace出现、当前服务端到端收益是四种不同证据。

[对应资料](../deepseek_summary/report.html)

### E07 CANN 升级代码与验证反馈

本地升级分支核对到9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8。用户反馈 DeepSeek 1P1D 功能通过；缺少本次可审计的完整原始性能对照，不能据此量化单项收益或覆盖 GLM 2P2D。

[对应资料](https://github.com/vllm-project/vllm-ascend/pull/21521)

### E08 0.32 调度开关的固定源码

async_scheduling=None 时按兼容条件解析；未在命令中显式传入不代表关闭。采集镜像的实际 core SHA 和最终解析值仍需补齐。

[对应资料](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/vllm.py#L1585)

### E09 CANN 流依赖与 Host 同步语义

StreamWaitEvent 建立设备流依赖；SynchronizeEvent/Stream/Device 阻塞调用线程。接口耗时不等于设备等待量，后台线程 wait 不一定挡住模型下发。

[对应资料](https://www.hiascend.com/doc_center/source/zh/canncommercial/850/appdevg/acldevg/aclcppdevg_000013.html)

### E10 学习地图与实验模板

已有请求生命周期、代码入口、性能指标、Roofline、profiling 方法和实验模板，可复用为本台账的学习与复现入口。

[对应资料](../npu_inference_learning/index.html)

### E11 引擎queue指标与网关队列的口径边界

固定core代码包含引擎请求queue时间直方图；它不能代替网关真实队列。采集版本的名称与边界需用实际代码和/metrics核对，API前处理、重调度与KV等待要另行关联。

[对应资料](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/metrics/loggers.py)

### E12 KV connector与传输完成的代码边界

核对本地connector的start_load_kv、后台请求处理与batch_transfer_sync_read等路径；Store另有分层load/wait接口。源码仅提供埋点边界，不代表当前采集已有完整P/D或Store传输指标。

[对应资料](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py)

### E13 A3代表prefill的Host/Device交互时间线

对既有P.rar样本的后续复核；A3 DP0TP0本地step1，以worker execute_model进入为0。包含目标层、draft、Host准备、真实launch及EP首层对齐；不是新采集或开关A/B。

[对应资料](../dsv41_a3_a5_prefill/host_device_timing/report.html)

### E14 A3时间线数值与原始事件索引

附原始SQLite路径、rank目录、kernel CSV行号、launch关联与多流区间并集。目标设备包络461.233ms中MegaMoe驻留299.701ms；无记录任务0.925ms。该包络从归属目标层的AICPU活动起算；严格首个HcPre到目标结束为455.757ms。

[对应资料](../dsv41_a3_a5_prefill/host_device_timing/audit.json)

### E15 官方0.32：fused MC2与共享专家多流互斥

固定release快照b12b6f7：用户enable_fused_mc2=2在674–679行归一化为内部1；811–816行将multistream_overlap_shared_expert改为False。用户配置1的dispatch_ffn_combine路径也命中该互斥；仅配置2并不足以证明最终实际执行MegaMoE。

[对应资料](https://github.com/vllm-project/vllm-ascend/blob/b12b6f7f47bb0b87ae3268790f1b2503e95e0e01/vllm_ascend/ascend_config.py#L669-L820)

### E16 当前CANN分支保留相同共享专家互斥

固定升级快照9a6a156：729–731行将2归一化为1，863–868行关闭共享专家overlap。官方0.32已有相同判断，本次升级未新加这项限制；该判断没有统一关闭其他多流配置。

[对应资料](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/ascend_config.py#L721-L872)

### E17 npugraph_ex官方静态kernel编译与缓存说明

2026-11-17查阅的官方文档：支持独立静态kernel缓存的实现中，disable_static_kernel_compile_cache默认False。缓存要求CANN/硬件、算子属性及编译选项一致。该文档对应实现不能代替用户D镜像的版本确认；.run是其中一种编译包产物，图捕获本身不产生此包。

[对应资料](https://github.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/basic/static_kernel_compile.md)

### E18 npugraph_ex：静态编译日志、命中检查与产物路径

2026-11-17核对：Starting static kernel compilation在缓存查询前打印；命中仍可能新建时间戳目录、采集算子信息并安装旧包。safe_resolve_output_dir未指定build_dir时使用工作目录；索引保存run包绝对路径。实际容器行为须按安装版本复查，不能仅凭日志或新目录判定重编译。

[对应资料](https://github.com/Ascend/torchair/blob/master/npugraph_ex/npugraph_ex/_acl_concrete_graph/static_kernel.py)

### E19 CANN编译适配器：图代码缓存与Triton跳过规则

当前适配器设置force_eager=True，启用静态kernel并未额外暴露独立静态缓存开关；FX生成代码另走vLLM缓存。219–239行遇triton_kernel_wrapper会跳过保存图代码，因为索引只在进程内有效；该跳过不等于底层静态kernel二进制一定未命中。

[对应资料](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/compilation/compiler_interface.py#L143-L248)

### E20 core异步输出线程的启动条件

参考core快照中，WorkerAsyncOutputCopy/async_output_busy_loop仅在scheduler_config.async_scheduling为真时创建。A3样本观察到该运行栈及跨step重叠；与异步已启用一致，但采集时准确core SHA及解析配置仍缺，不能用源码参考冒充镜像配置证明。

[对应资料](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/executor/multiproc_executor.py#L681-L699)

### E21 用户提供的对方优化项目摘要

2026-11-17由用户粘贴；21项有实质内容的项目保存在peer_review.projects的reported字段。缺少原始报告、硬件/模型/框架SHA、完整负载及Interactivity定义，数字仅作对方报告引用，不是本环境复测或可叠加收益。Motor是否确指MindIE Motor也待确认。

[对应资料](tasks.json)

### E22 CANN DP同步绕过条件：MegaMoE明确受限

should_skip_allreduce_across_dp_group在hierarchy或MegaMoE可用/被选中、以及FUSED_MC2等条件下返回False；MoE可跳过路径还依赖KV consumer角色、普通MC2及相应图/重算调度条件。V1/V2均有调用入口。对方报告的MegaMoE+重算调度绕过方案与本快照并不等价，须核对算子动态global_bs与一致性协议。

[对应资料](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/utils.py#L1344-L1409)

### E23 现有Ascend GateLinear的BF16输入路径

BF16输入/权重、FP32输出且无bias时，已有torch.mm(...,out_dtype=torch.float32)路径。通用类存在不表示DeepSeek-V4.1实际命中；强制FP32/预转换权重模型有独立路由合同，必须保留路由精度验证。

[对应资料](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/ops/fused_moe/gate_linear.py#L28-L115)

### E24 DeepSeek-V4.1路由与输出投影的现有约束

模型gate设precast_fp32_weight=True；297–308行区分internal router与外部F.linear FP32输入路径。446–449行明确DSA wo_a消费者要求ND并跳过NZ转换。说明BF16 Gate、ND/NZ都必须按具体模型/矩阵/消费者判断，不能全局替换。

[对应资料](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/models/deepseek_v41/model.py#L191-L194)

### E25 长Prefill阈值与有效batch的调度语义

参考core在可调度请求超过一个时使用long_prefill_token_threshold；单请求时不应用该cap。阈值还受token预算和adaptive开关影响。实际Ascend scheduler可能覆盖此路径，实验必须记录最终scheduler类、真实chunk长度与每step有效请求数。

[对应资料](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py#L640-L655)

### E26 Ascend对部分调度参数的归一化

本快照将非1的max_num_partial_prefills重置为1并给出日志；不能将上游通用调参表逐项照搬。max_num_seqs也是容量上限，不等于每个decode step的实际batch。

[对应资料](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/vllm_ascend/platform.py#L966-L974)

### E27 官方MindIE Motor部署说明

官方文档将MindIE Motor定位为部署、调度、负载均衡和RAS层。若对方Motor指此组件，应在网关/调度与DP分发分析；不等价于单个模型kernel优化。对方Motor名称、版本和分发算法尚未确认。

[对应资料](https://github.com/vllm-project/vllm-ascend/blob/b12b6f7f47bb0b87ae3268790f1b2503e95e0e01/docs/source/user_guide/deployment_guide/using_mindie_motor.md)

### E28 CANN PCP资源与组合条件

本快照PCP引入额外rank，world_size_with_pcp=PCP×original_world_size。文档分别列DSA V4.1、图模式、DSpark、P/D connector和非layerwise Store组合；SWA仍须核对实际make_decoder_replay gate及SP/PP/DCP等限制。不能把PCP8直接加到原有16卡方案而忽略资源和路径变化。

[对应资料](https://github.com/gg820827047/vllm-ascend/blob/9a6a156fccccb7ad55c5d534c8948d3ec4efc3f8/docs/source/user_guide/feature_guide/context_parallel.md#L20-L75)

### E29 A3 DP2/TP2主线程、多流就绪与任务时钟复核

既有P.rar，同DP的TP0/TP2对照。TP2初始metadata139.300ms vs113.733ms，目标40层Host447.453ms vs383.783ms；多流计算重叠0.088462ms vs7.420534ms。原始connection关联确认Task15140第0层1848 token的Host准备/提交晚于KV完成；用户看到的Task15092属于前一轮_dummy_run第37层6 token。报告区分SESSION和共同坐标、profiler包装墙钟与CPU实际运行，未新增NPU收益实测或确认物理慢卡。

[对应资料](../dsv41_a3_a5_prefill/host_issue_dp2tp2/report.html)

### E30 首层Q/KV、Attention、TP通信和dtype的原始CSV复核

重新从P.rar提取同一DP2/TP2原始kernel_details.csv，核对13条Q/KV至输出投影任务，附2158.655699ms的TP AllReduce、dtype和原始CSV SHA256。另保存20个A3有效rank-step的多流统计，区分DSA预处理与全部计算流交集；未新增上板实验。

[对应资料](../dsv41_a3_a5_prefill/host_issue_dp2tp2/attention_mechanism_evidence.json)

### E31 DeepSeek官方V4.1模型说明与CSA2/CED架构

2026-11-19核对官方说明：CED为20层因果Encoder＋20层Decoder；CSA2有Full/Reindex/Reuse与分层Indexer；FP4 KV是配套精度设计。算法层与具体A3算子dtype分别归档，不把官方精度方案默认套到本次采集。

[对应资料](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)

### E32 共享专家多流互斥的历史来源

历史提交在0.32之前已引入相关互斥；说明涉及当时重叠收益与特定共享专家TP通信/主流MoE的跨rank跨流风险。它解释当前限制的来源，不证明所有后端/所有MoE多流都不可能，也不直接构成删除guard的依据。

[对应资料](https://github.com/vllm-project/vllm-ascend/commit/2044419f8f3dec6288eb1b1b1f54245d4c2b0904)

### E33 Attention输出投影为何有TP AllReduce

RowParallelLinear在reduce_results且TP>1时合并output_parallel。与原始CPU AscendRowParallelLinear_0 → tensor_model_parallel_all_reduce及设备hcom_allReduce记录相互印证；与MoE EP分发、DP批量同步、connector KV传输分开分析。

[对应资料](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/model_executor/layers/linear.py#L1762)

### E34 异步输出线程、逐轮类型与跨step准备交集复核

从P.rar原始DP2/TP2数据库复核TID4151的async_output_busy_loop/get_output，并保存16个A3 rank的60个前向窗口类型及跨轮包络交集。DP2/TP2有4个主干窗口但6次外层execute_model scope，只有1个真实Prefill窗口；DP1有两个连续真实Prefill。参考core的默认值None按兼容性自动启用；采集镜像准确SHA和解析值仍待补。未做ON/OFF收益实验。

[对应资料](runtime_coordination_evidence.json)

### E35 用户MoE分块通信方案、术语与收益口径

保留48k/EP16/N256/K8及AGRS、A2A、Init Routing、分段GMM、通信量化原文。N/K含义及前后分段规则按假设解释；500~700us和无OOM是对方摘要中的结论，未验证为本服务收益。

[对应资料](moe_plan_evidence.json)

### E36 11月25日远端调度日志选定字段、配置确认与实验计划

P侧4DP×4TP、128k/64并发、预算8192；保留四DP相邻窗口与计算公式。全文日志不在本机；0.8秒为窗口摊到调用，不是Device耗时。两组A/B未执行。

[对应资料](scheduler_admission_evidence.json)

### E37 调度探针、预算门限、AsyncScheduler与MRV2机制笔记

逐项解释诊断字段、两份预算、DSpark K-1、分支标签与快照差别；注明固定core/Ascend源码链接及Runner核对边界。

[对应资料](index.html#mechanism-scheduler-admission)

### E38 Prefill/Decode/DSpark token时序与P/D客户端首输出

区分P可预测首token、D首次处理与客户端TTFT。中间chunk使用next_prefill_tokens并准备草稿上下文，不等于两个请求或有效输出；P候选是否交D未确认。

[对应资料](index.html#mechanism-pd-ttft-handoff)

### E39 P侧Device模块归属：共享专家融合与元数据路径

固定现有A3/A5样本的任务归属、调用栈与参考源码。A3主干共享专家200条任务累计15.344ms，A5主干共享专家包含在MegaMoE内；A3准备阶段RepeatInterleave42次累计7.470ms。均为驻留成本，不能当作可回收时间。A5精确采集源码SHA仍未知。

[对应资料](../dsv41_a3_a5_prefill/device_module_comparison/report.html)

### E40 待执行：A3 2P2D、P侧32DP×TP1、128k输入采集计划

G04采集规格及待填写manifest，不是新profiling结果。核实P侧总DP与每engine DP、实际EP、每步query/历史KV和专家token分布。现有A5未强制均衡；新A3样本不能自动生成跨硬件速度比。

[对应资料](../dsv41_a3_a5_prefill/device_module_comparison/new_inputs.template.json)
