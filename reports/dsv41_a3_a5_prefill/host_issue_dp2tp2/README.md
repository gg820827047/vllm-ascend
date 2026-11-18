# A3 DP2/TP2主线程与多流时序

打开 [report.html](report.html)，或从[服务优化总览](../../llm_service_optimization/index.html#recent)进入。报告提供返回总览的链接。

内容来自既有 `P.rar`，包含原始CPU提交、CANN launch和设备Task关联；不是新的NPU采集或性能收益实验。报告同时说明DecoderLayer与Prefill/Decode阶段的区别。

- [clock_audit.json](clock_audit.json)：用户选中的Task15092与分析Task15140的原始JSON、Connection ID和时间坐标对照。
- [inspection_points.csv](inspection_points.csv)：精确定位表。
- [summary.json](summary.json)：阶段耗时、原始零点与局限。
- [kernel_connections.json](kernel_connections.json)：42条CPU → runtime → device关联。

SESSION相对时间与多卡共同坐标分别标注。先按Stream/Task/Connection ID确认同一个任务，再校准显示时间；单锚点平移只适用于各泳道使用相同显示偏移的情况。

使用随附的两个rank元数据、压缩记录与关联JSON重建HTML，无需NPU：

```bash
python reports/dsv41_a3_a5_prefill/host_issue_dp2tp2/build_report.py
```

`inspect_host.py`和`correlate_launches.py`用于重新查询原始SQLite，另需父目录`data/report_data.json.gz`及其引用的原始压缩包提取目录。随附记录中的self wall是采集墙钟，不能等同CPU实际运行时间。
