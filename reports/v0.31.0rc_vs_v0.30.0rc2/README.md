# vLLM Ascend 发布分支比对报告

下载后在浏览器打开 [report.html](report.html)。核心内容、筛选、分页和 CSV 导出不依赖网络；访问 GitHub 来源需要联网。单独分享 HTML 也可阅读，分享整个目录或 ZIP 可保留原始证据。

比对对象锁定于 2026-11-02 核对的远端提交：

- 旧：`upstream/releases/v0.30.0rc2`，`7d825229c77025cfb185a010c25819c1214b9450`。
- 新：`upstream/releases/v0.31.0rc`，`5e40bc723165bfc8d8b1ec245ce5908a45dfd9d2`。
- 共同祖先：`ac1b7bceb370265e8f794f38d32a88041b0c51c9`。

主要结论：新侧独有 364 个提交，旧侧独有 55 个。44 对补丁等价、6 对同源适配；未映射的 314 条不等于 314 项新能力。净差异为 1140 个文件、113380 行新增、61855 行删除。

66 条性能及关联审阅记录中：34 条直接优化或性能恢复、18 条能力使能、4 条基线已有、5 条关联适配、1 条后续替代的历史实现、4 条撤销或收缩的路径。提交数与独立优化机制数量不同。报告中的速度数据均注明其 PR 作者来源与场景，本次未运行 NPU 性能测试。

文件内容：

- `report.html`：可离线浏览的报告。
- `report_data.json`：完整提交、文件、映射和人工分析的结构化数据。
- `commits.csv`：364 条新侧独有提交。
- `baseline_only.csv`：55 条旧侧独有提交及映射。
- `performance.csv`：66 条性能和关联项分析。
- `files.csv`：1140 个净变更文件，包含准确的新旧路径、重命名状态和源码链接。
- `git_data.json`：原始 Git 采集结果。
- `evidence/commits/`：419 份原始提交补丁。
- `evidence/prs/`：52 份公开 PR 作者说明的 API 快照，记录抓取时间及合入 SHA。
- `annotations.py`：人工分类、机制、条件、证据与解释边界。
- `collect.py`：只读采集 Git 历史；`--github` 可缓存重点 PR 作者说明。
- `build_report.py`：从已有证据重建报告、数据与 CSV。
- `validate_report.cjs`、`validation.json`：浏览器验证脚本与结果。
- `preview-desktop.png`、`preview-mobile.png`：页面预览。
- `manifest.json`：证据及交付文件的 SHA-256 校验清单。

在当前目录执行 `python build_report.py` 可从已有数据重建。该命令使用当前 Git 仓库中的固定 SHA 解析文件重命名，不需要 NPU 或 vLLM Python 依赖。重建前需具备这两个 SHA 的完整历史；可从 upstream 拉取 `releases/v0.31.0rc` 和 `releases/v0.30.0rc2`。

`collect.py` 重新读取当时的发布分支 tip，属于一次新的快照采集。重新采集后需要更新人工分类和校验断言，不应将本报告结论直接用于移动后的分支。

浏览器验证使用 Node.js 22 自带 WebSocket 与本地 Chrome DevTools：需要先以调试端口 19223 启动 Chrome，再执行 `node validate_report.cjs`。验证包括离线加载、脚本异常、筛选、去重统计、分页、空结果、打印时展开当前匹配清单、CSV 数据与桌面/移动端布局。

推送前验证：报告重建、Ruff、Python/Node 语法与文件校验通过。原 HTML 已通过 Chrome 离线交互验证。仓库全量 `bash format.sh ci` 在 Markdownlint 的 Node 环境引导阶段失败：nodeenv 解析远端最新 Node 27 alpha 版本报错，完整格式检查未完成。
