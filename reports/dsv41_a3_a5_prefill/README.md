# DSv4.1 Prefill 证据分析

直接用浏览器打开 `report.html`。页面无需网络，内嵌时间线及统计数据。

主对照为已确认的 `P.rar`（A3，DP4×TP4 / EP16）与 `vllm_profile_prefill-c192-bs12.zip`（A5 / Ascend950，DP8×TP1 / EP8）。旧的 `20261026-DSv4.1-1P1D-profiler.zip` 仅作 A3 历史参照。

两组实际还存在 Runner、CANN、量化格式和 workload 差异。报告区分已发生的融合、拓扑影响与等待，不能直接作为同配置硬件 A/B 的加速倍数。

附带 CSV：

- `data/rank_steps.csv`：28 个 rank 的 122 个本地前向窗口，区分有效 prefill 与 dummy。
- `data/operator_comparison.csv`：两个代表有效前向的主干算子数量与硬件累计时间。
- `data/mega_cross_rank.csv`：A3/A5 每层 MegaMoE 跨 rank 到达/结束时间、MAC ratio、原始 CSV 行号；按同层完成时间对齐共同轮次。
- `data/exact_shape_tail.csv`：Type、完整 shape、dtype 一致的 7 组尾窗算子比较。
- `data/report_data.json.gz`：重建 HTML 所需的压缩分析数据，已随报告提交。
- `data/manifest.json`：源压缩包 SHA256；报告未修改原始压缩包。

## 复现

分析脚本使用 Python 标准库；生成 HTML 额外需要 `Markdown==3.11.3`。读取 RAR5 使用 bsdtar/libarchive，系统旧 unrar-free 0.0.2 不支持。

默认输入文件位于 `/home/gengc/Downloads`，提取结果位于 `/tmp/dsv41-a3-a5-prefill-inputs`。若环境不同，修改 `extract_inputs.py` 的源路径与 `analyze.py` 中的 `RAW`。

```bash
python -m pip install -r requirements.txt
python build_report.py
python -m unittest test_analysis.py
```

上述命令在本目录执行，使用提交的压缩数据即可重建报告。若要从原始三个压缩包完整重新分析，再执行：

```bash
python extract_inputs.py
python analyze.py
python derive.py
python build_report.py
```

原始 trace/DB 和全量中间 JSON 没有重复提交或打入交付 ZIP。执行脚本会重新生成这些中间结果。硬件时间取整数纳秒后再转换为相对微秒；Host 优先通过 PyTorch connection ID 关联，绕过 PyTorch op 记录的 C++ kernel 则通过同一主线程的 CANN launch 关联。HCCL 子任务不与大算子重复累计。

报告主干区间含前 40 层，后面 3 次 draft 调用单独保留。主对照均有 execute_model Python scope；只有历史 A3 ZIP 缺该范围，使用重建锚点。不要把异步重叠的窗口相加当作 TTFT。A5 首尾共同轮次覆盖不完整，不能拿边界长等待推断稳态慢卡。
