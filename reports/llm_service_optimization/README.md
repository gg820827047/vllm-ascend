# LLM 服务请求全局视图与性能任务台账

打开 [index.html](index.html)：从请求入口到 P/D 推理与流式输出定位任务，按阶段、主题、状态、责任角色和关键词筛选，导出当前筛选的 CSV。单个 HTML 可离线分享。

- [plan.md](plan.md)：完整 Markdown 版，含全局流程、已有工作、任务表和验收口径。
- [tasks.csv](tasks.csv)：可在表格软件中填写具体责任人并安排工作。
- [tasks.json](tasks.json)：阶段、证据、已有工作与任务的维护源。
- [validation.json](validation.json)：文档与浏览器交互检查结果。

新增“网关排队 / KV传输”专题：F01/F02初始优先级为P2，分别挂到S0/S1与S5/S6。HTML和Markdown均列出四类观测边界：网关/API排队、引擎等待、P/D KV交接、Store读写；注明worker profiling能看到的范围、需要补采的数据和计时口径。当前共35项任务。

责任人暂按角色建议分工，具体人名均为“待认领”；这不是已确认的人员安排。分析任务的直接性能收益为零，收益体现在形成证据与决策。优化候选不承诺未经实验的加速百分比。

维护 `tasks.json`、`page_template.html` 或 `interactions.js` 后，在仓库根目录重建：

```bash
python reports/llm_service_optimization/build.py
```

生成 `index.html`、`plan.md` 与带 UTF-8 BOM 的 `tasks.csv`，仅使用 Python 标准库。CSV 中的人工改动不会自动写回 JSON；要更新团队台账，请将确认后的分工、状态和实测收益回写维护源再重建。

本次仅整理文档、证据与后续任务，不修改推理代码或运行配置。关联资料：[A3/A5 profiling 对比](../dsv41_a3_a5_prefill/report.html)、[30/31/32 DeepSeek 优化](../deepseek_summary/report.html)、[推理学习地图](../npu_inference_learning/index.html)。
