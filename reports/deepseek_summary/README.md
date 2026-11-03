# DeepSeek 优化 PR 与代码量

- [HTML 专题](report.html)：直接优化、公共优化、性能使能及适用性复核。
- [PR 明细 CSV](prs.csv) · [逐文件行数 CSV](production_files.csv) · [完整数据 JSON](summary.json)。
- [人工选集](selection.json)：每个方向的 PR、DeepSeek 适用性补充、未计入项与原因。

范围沿用总览固定的 v0.30.0rc2 → v0.31.0rc → v0.32.0rc 快照，不代表最新 main。保守纳入 37 项直接优化 / 性能恢复，以及 20 项性能路径使能；另列 3 项部署条件待确认的候选，不计合计。

生产源码共历史新增 12800 行、删除 7953 行，变更量 20753 行；全部 PR 文本文件新增 29253 行、删除 9252 行，变更量 38505 行。每个唯一 PR 按 `git show --format= --numstat --find-renames <sha>` 校验并累加。生产口径只含 `vllm_ascend/`、`csrc/` 的 Python/C/C++/CUDA/CCE 源码，排除测试、文档、benchmark/golden。注释、空行仍计数；重复编辑按次计数，整 PR 中可能包含非 DeepSeek 部分。因此这是历史变更规模，不是最终净新增、最小移植量或人天估算。

筛选结合 PR 信息和关键生产代码，额外核对部分 DeepSeek 模型调用路径、算子封装及开关。没有对全部 PR 做完整逐行审计，没有新增 NPU 精度或性能复测。代码机制成立与实际性能收益分开表述，保留作者数据原条件、未证实收益与退化项。

在仓库内重建：

```bash
python reports/deepseek_summary/build_report.py
```

脚本从两阶段既有 `report_data.json` 加载证据，按人工选集分类，逐个用本地 Git 对象核验 numstat；输出自包含 HTML、JSON、CSV 和 SHA256 清单。重新选择 PR、改变源码口径或移动快照后需重新审阅选集。

本次校验：Git numstat、CSV 汇总、32 快照提交可达性、本地链接及哈希清单通过；Ruff 检查与格式检查通过；Chrome 在 1440/390 像素宽度下的筛选、搜索、布局及运行时错误检查通过，见 `data_validation.json`、`browser_validation.json` 和预览图。仓库全量 `bash format.sh ci` 在 markdownlint 的 nodeenv 依赖下载阶段被 SSL 证书链校验错误阻断，未完成全量检查。
