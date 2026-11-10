# DeepSeek-V4.1：0.30 / 0.32 Profiling 验证与学习报告

直接打开 [report.html](report.html)。这是单文件离线报告，包含真实数据的可缩放时间线、SWA 层序列、算子对照、60 个优化候选的证据状态及完整教学章节。

建议第一次按这个顺序看：

1. 先看“数据与可比性”，理解版本、TP/EP 与 DP/PCP 推导的边界。
2. 看“两个已验证机制”：SWA 的 8188→128 行裁剪，Engram 的 6 次冗余清零消失。
3. 点击“看 17 秒等待”，观察 MegaMoE、Host Gloo、Peer DP0 Free 的时间关系；点击矩形查看原始 shape 和数据来源。
4. 看“系统学习路线”，再按练习取一个原始 trace 到 MindStudio Insight。
5. 用“60 个 PR 的证据”区分已验证、路径可见、配置不明和采集不覆盖。

**本次没有得到同条件的单 PR 提速比例或端到端 TTFT/TPOT。** 输入用例相同，不代表调度 batch 相同；0.30 缺 D 侧，采集里缺客户端请求链路和镜像精确 digest。不能把窗口级时长当成请求时延。

## 数据与结论

- 输入 ZIP 保持原样；分析直接从 ZIP 流式读取。
- 28 个外层 rank 导出，详细分析 6 个代表 trace；嵌套 ZIP 与外层覆盖相同 rank，不重复计数。
- TP/EP 组从 SQLite 的 `META_DATA.parallel_group_info` 获取。
- 第 21–39 层尾窗裁剪、Engram lookup 准备区间的 zeros 消失，有代码和 trace 证据。
- 第二批 P DP1 的 17.8706s MegaMoE 与 17.9575s Gloo 高度重叠；是需要解释的同步/采集停顿候选，不是已确认的 GEMM 性能问题。
- 多 stream、通信 op/task 父子记录和 CPU inclusive scope 不能直接累加成墙钟时延。

完整证据见 [data/evidence.json](data/evidence.json)，PR 矩阵见 [data/pr_verification.json](data/pr_verification.json)。

## 重建

建议 Python 3.11+，使用单独虚拟环境。本机本次使用 `/tmp/dsv41-profile-env`。

```bash
python -m venv /tmp/dsv41-profile-env
/tmp/dsv41-profile-env/bin/pip install -r requirements.txt
/tmp/dsv41-profile-env/bin/python analyze.py
/tmp/dsv41-profile-env/bin/python trace_analysis.py
/tmp/dsv41-profile-env/bin/python derive.py
/tmp/dsv41-profile-env/bin/python build_report.py
/tmp/dsv41-profile-env/bin/python -m unittest discover -s . -p 'test_*.py' -v
```

更换输入路径时，先运行：

```bash
python analyze.py --archives /path/to/20261026-DSv4.1-1P1D-profiler.zip /path/to/20261109-032.zip
```

`derive.py` 会读取工作区现有的 release 对比报告和固定 Git 快照。换机器时需同时保留这些代码依据，或者直接使用已经生成的 evidence.json 构建 HTML。

脚本会自动寻找本仓库、相邻 vLLM 仓库和既有 PR 报告，也可以明确指定：

```bash
python derive.py --ascend-repo /path/to/vllm-ascend \
  --vllm-repo /path/to/vllm --pr-reports /path/to/vllm-ascend/reports
```

vLLM 仓库需包含 `v0.32.0` tag；Ascend 仓库需包含本报告引用的固定发布提交。

CSV 统计覆盖全部 28 个 rank；流式 trace 分析覆盖每组的 TP0（第二批 P 的 DP0、DP1 分别处理）。图里按开始时间排列 HcPre 恢复三组 40 层序列，未把它们解释为请求数、输出 token 数或 ProfilerStep ID。

## 单 rank 导出给可视化工具

```bash
python extract_one.py --dataset v032_p_tp4 --dp 0 --tp 0 \
  --output /tmp/dsv41-p32-rank0-trace.json
```

这个命令只复制一个 trace，目标文件已存在时拒绝覆盖。默认不导出任何原始大文件。本机磁盘余量有限，不要将两个包全部展开。

## 产物

| 文件 | 用途 |
| --- | --- |
| report.html | 单文件离线交互报告 |
| tutorial.md | 可独立阅读的教学材料 |
| data/analysis.json | 全 rank shape、类型、覆盖、通信统计 |
| data/kernel_types.csv | 可导入表格工具的类型统计 |
| data/traces.json | 6 个代表 trace 的 scope 与原始事件证据 |
| data/db_metadata.json | TP/EP/MC2 通信组、会话时间与原始元数据 |
| data/evidence.json | 层序列、Engram 窗口、源码片段、精简时间线 |
| data/pr_verification.json | 60 项候选的验证结论及下一步 |
| validation.json | 输入校验、测试与方法限制 |
| browser_validation.json | Chrome 桌面/手机布局、筛选和时间线交互验证 |
| validate_report.cjs | 使用本地 Chrome DevTools 的界面验证脚本 |

浏览器验证需启动独立的本地 Chrome 调试实例（端口 19224），然后执行 `node validate_report.cjs`。报告本身不需要 Node、服务端或联网。

后续最有价值的补采是：0.30 D 侧、无 profiler 的原始请求结果、所有 EP rank 同步的稳定窗口、SWA 开关对照及实际配置/镜像版本。详见教学材料末尾。
