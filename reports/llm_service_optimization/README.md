# LLM 服务请求全局视图与性能任务台账

打开 [index.html](index.html)：从请求入口到 P/D 推理与流式输出定位任务，按阶段、主题、状态、责任角色和关键词筛选，导出当前筛选的 CSV。单个 HTML 可离线分享。

- [plan.md](plan.md)：完整 Markdown 版，含全局流程、已有工作、任务表和验收口径。
- [tasks.csv](tasks.csv)：可在表格软件中填写具体责任人并安排工作。
- [tasks.json](tasks.json)：阶段、证据、已有工作与任务的维护源。
- [validation.json](validation.json)：文档与浏览器交互检查结果。

“网关排队 / KV传输”专题：F01/F02初始优先级为P2，分别挂到S0/S1与S5/S6。HTML和Markdown均列出四类观测边界：网关/API排队、引擎等待、P/D KV交接、Store读写；注明worker profiling能看到的范围、需要补采的数据和计时口径。当前共42项任务。

2026-11-17新增“近期核对”入口，归档五项结论：A3目标层461.233ms与首层前Host准备的拆分、异步调度实际生效证据、官方0.32/CANN共有的共享专家多流互斥、A5 Prefill不能证明D侧static kernel配置，以及启动时静态二进制编译/FX图编译/ACLGraph capture的区别。新增B08共享专家多流验证、D08启动缓存排查，保留责任人、预期收益和验收口径。

附[交互时序图](../dsv41_a3_a5_prefill/host_device_timing/report.html)和[数值/事件索引](../dsv41_a3_a5_prefill/host_device_timing/audit.json)。它们是对既有`P.rar`的后续复核，非新采集或开关收益实验；D侧“约50项”与`.run`缺失的现场原因仍待日志和版本确认。

2026-11-18新增[DP2/TP2主线程与多流时序报告](../dsv41_a3_a5_prefill/host_issue_dp2tp2/report.html)。总览顶部导航、“近期核对”和E29均可进入，报告内可返回总览。包括metadata/实际launch对照、Task15092与15140的同名任务复核、时间坐标校准，以及DecoderLayer与Prefill/Decode的区别；对应A06/A08/C03，未新增性能收益实测。

2026-11-19新增[多流阶段结论与原理笔记](index.html#multistream-review)，归档metadata搜索、Q/KV生成与Attention主算子、DSA就绪时序、TP通信与DP/TP/PP/CP/EP、CSA2/CED、量化和QuantMatmul等六组可展开笔记。附13个任务的shape/dtype CSV和原始事件JSON。同步补正V4.1的CED分段与代码DecoderLayer命名的关系。

多流目前标记为“机制与局部时序分析阶段完成”，不是性能优化完成。Host准备/下发影响重叠已有局部证据，Host抖动保留为待验证假设。A05/A06/C03继续验证Host、元数据与运行环境；A08保留端到端收益验证；B08仅将MegaMoE＋共享专家辅助流兼容扩展实验后置。任务总数仍为42。

2026-11-20补充三组笔记：[异步调度与DP同步](index.html#mechanism-async-dp)、[逐轮类型与跨step遮盖](index.html#mechanism-step-overlap)、[AG/A2A与MoE分块通信](index.html#mechanism-moe-communication)。同时扩充[CPU绑核与Host抖动验证](index.html#mechanism-host-cause)，给出线程亲和、pidstat和perf sched的检查方法。

新增[运行态复核证据](runtime_coordination_evidence.json)：保存A3原始异步输出线程事件，以及16个rank的60个主干前向窗口。DP2/TP2的4个主干窗口不等于全部6次外层execute_model调用；其中只有编号1是真实Prefill。记录跨step准备与前轮设备包络的交集，并说明等待、有效CPU工作和收益的区别。参考core默认自动解析异步调度，未写参数不能判关闭；采集镜像准确core SHA和最终配置仍待补齐。

[MoE方案原文与口径](moe_plan_evidence.json)保留用户提供的48k/EP16/N256/K8、分段GMM、通信量化以及500～700us/无OOM表述。参数和分段规则按假设解释，外部结果不作为本服务实测。新增内容仍仅为文档与既有数据复核，未实施新优化或运行上板性能实验。

“对方项目借鉴”入口逐项分析用户提供的21项结果，映射到生命周期、现有trace、代码条件和跟进任务。优先验证batch/chunk、P/D供给与DP分发、元数据复用/异步规划；明确当前MegaMoE的DP同步限制、DSV4.1 Gate精度合同，以及Attention布局/写回和DSL的硬件量化前提。新增A09、B09、B10、D09、R03，原文结果保存在`tasks.json`的`peer_review.projects`，附首批实验顺序和可直接用于简报的三句话总结。

外部数字均标为“对方报告”，没有冒充本服务实测。Motor具体组件、Interactivity定义、模型/硬件/框架和chip分母待补；2P1D负收益与EP32卡效率反例同样保留。NPU性能实验尚未执行。

责任人暂按角色建议分工，具体人名均为“待认领”；这不是已确认的人员安排。分析任务的直接性能收益为零，收益体现在形成证据与决策。优化候选不承诺未经实验的加速百分比。

维护 `tasks.json`、`page_template.html` 或 `interactions.js` 后，在仓库根目录重建：

```bash
python reports/llm_service_optimization/build.py
```

生成 `index.html`、`plan.md` 与带 UTF-8 BOM 的 `tasks.csv`，仅使用 Python 标准库。CSV 中的人工改动不会自动写回 JSON；要更新团队台账，请将确认后的分工、状态和实测收益回写维护源再重建。

本次仅整理文档、证据与后续任务，不修改推理代码或运行配置。关联资料：[A3/A5 profiling 对比](../dsv41_a3_a5_prefill/report.html)、[30/31/32 DeepSeek 优化](../deepseek_summary/report.html)、[推理学习地图](../npu_inference_learning/index.html)。
