# vLLM Ascend：30 → 31 → 32 演进报告

- [三版本 HTML 总览](index.html)：逐阶段统计、性能机制演进矩阵、关键代码符号追踪。
- [DeepSeek 优化专题](deepseek_summary/report.html)：37 项直接优化、20 项性能使能、适用条件及代码量；[PR 明细 CSV](deepseek_summary/prs.csv)。
- [30 → 31 报告说明](v0.31.0rc_vs_v0.30.0rc2/README.md) · [HTML](v0.31.0rc_vs_v0.30.0rc2/report.html)。
- [31 → 32 报告说明](v0.32.0rc_vs_v0.31.0rc/README.md) · [HTML](v0.32.0rc_vs_v0.31.0rc/report.html)。
- [演进矩阵 CSV](evolution_matrix.csv) · [总览数据 JSON](overview_data.json)。

HTML 报告需下载后在浏览器打开，支持离线筛选、分页和 CSV 导出。

三个快照：

| 发布分支 | 固定 SHA |
| --- | --- |
| v0.30.0rc2 | `7d825229c77025cfb185a010c25819c1214b9450` |
| v0.31.0rc | `5e40bc723165bfc8d8b1ec245ce5908a45dfd9d2` |
| v0.32.0rc | `d08fa9d4380684b64fd7a45a876fc080ecbd1d22` |

30 → 31：364 条新侧独有提交，44 条补丁等价、6 条同源适配；31 → 32：260 条新侧独有提交，3 条补丁等价。直接计算 30 → 32 得到 621 条新侧独有提交、1779 个净变更文件，不能把两个阶段的计数直接相加。

报告中的性能数字来自逐项注明条件的作者数据。本次未运行 NPU 性能测试，优化机制数量与 PR 数量不同，阶段百分比不能累加。

重建总览：在当前仓库执行 `python reports/build_overview.py`。重建阶段详报见各目录 README；重新采集移动后的分支需重新审阅人工分类。
