# vLLM Ascend：0.31 → 0.32 代码与性能演进

入口：[三版本总览](../index.html) · [本阶段 HTML](report.html) · [前一阶段 30 → 31](../v0.31.0rc_vs_v0.30.0rc2/report.html)。HTML 需下载后在浏览器打开，核心内容、筛选、分页和 CSV 导出可离线使用。

本报告锁定两个发布分支快照：

- 基线 `upstream/releases/v0.31.0rc`：`5e40bc723165bfc8d8b1ec245ce5908a45dfd9d2`。
- 目标 `upstream/releases/v0.32.0rc`：`d08fa9d4380684b64fd7a45a876fc080ecbd1d22`。
- 基线 SHA 与前一份 30 → 31 报告的目标 SHA 完全一致。

32 侧独有 **260** 个提交，31 侧独有 **3** 个；三条均有 patch-id 等价项，剩下 **257** 条未匹配等价的历史提交仍包括文档、CI、测试、重构及修复。净差异为 **1094** 文件、**+73862 / −44125** 行。

71 条性能与关联审阅记录：33 条直接优化或性能恢复、20 条能力使能、11 条正确性或运行栈约束、5 条退役或回退路径、2 条基线已有的重要缓存修复。另有一条基线等价测试修复保留在完整提交映射中。22 条显式 `[Performance]` 全部审阅，并补入 `[Perf]` 和无性能标签的相关代码。

主要演进包括：compiled embedding/RoPE/UVA 的主机与搬运开销、PCP/GDN metadata、GLM compile 与 RecoverSSM、QSFA/QLI、SP/DCP 通信布局、MiniMax 既有融合扩展，以及 Qwen3.8/PLE 模型、dispatch、并行和量化路径。性能数字均为明确注明来源的 PR 作者或原始验证评论数据；本次没有运行 NPU benchmark。

不要将回移前硬件数据当作 release checkout 的新测量，也不要将局部 commit A/B 与 RecoverSSM 整算法 OFF/ON 对照混用。报告保留短 history/小 batch 退化、decode 无收益及未完成验证的记录。

交付文件：

- `report.html`、`report_data.json`：自包含 HTML 与完整结构化分析。
- `commits.csv`、`baseline_only.csv`、`performance.csv`、`files.csv`：全部提交、反向等价映射、71 条审阅、文件净差异。
- `evidence/commits/`：263 份原始提交补丁。
- `evidence/prs/`：54 份公开 PR 作者说明与 1 份明确链接的 RecoverSSM 验证评论快照；其中两个 main 原始 PR 只作来源上下文，不另计新增。
- `annotations32.py`：人工机制、适用条件、收益边界与分类。
- `collect.py`：重新采集当时的分支历史，可选缓存性能 PR；重新采集后必须更新审阅结论及断言。
- `build_report.py`：固定 SHA 重建报告、JSON、CSV、校验清单。
- `manifest.json`：文件 SHA-256；原始补丁保留原有 whitespace，专用 Git 属性只豁免这些证据文件的 whitespace 检查。
- `validation.json`、`validate_report.cjs`、页面预览：生成与浏览器验证记录。

在当前目录运行 `python build_report.py` 可重建。需要当前 Git 仓库具备两个快照的历史，以及相邻第一份报告的 `build_report.py`（只提取字面 UI 模板，不执行其分析）。不需要安装 vLLM 或 NPU 依赖。

全局总览可在仓库根运行 `python reports/build_overview.py` 重建；它会核对共享 31 SHA、实际计算 30 → 32 的累计历史/净差异，并在三端源码定位关键实现符号。

验证：离线浏览器交互、桌面/移动端、Ruff、语法与数据校验通过。仓库全量 `bash format.sh ci` 在 Markdownlint 环境引导阶段失败：nodeenv 无法解析远端 Node 27 alpha 版本；完整 hooks 未完成，未报告为通过。
