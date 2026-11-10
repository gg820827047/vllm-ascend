# NPU 推理学习与性能实践

从 [index.html](index.html) 开始。单个 HTML 可离线阅读，含交互执行地图、六阶段学习进度、症状排查入口和教学 Roofline 图；外部文档链接需要网络。页面的演示数值不代表 A3 规格或当前服务的实测结果。

- [guide.md](guide.md)：完整手册，便于维护与 GitHub 阅读。
- [experiment_template.md](experiment_template.md)：实验、瓶颈案例与学习记录模板。
- [service_profile_template.json](service_profile_template.json)：2000+ 日活服务所需的负载与 SLO 清单。
- [manifest.json](manifest.json)：固定代码依据与结论范围。

主线：部署与指标 → 张量/模型 → 调度/KV/执行器 → profiling 定位 → 代码与算子 → 容量和服务运维。知识地图与性能方法一起学，每阶段留下可复现的实验资产。

生成页面（本次使用 `/tmp/dsv41-profile-env/bin/python`，其环境已有 `Markdown==3.11.3`）：

```bash
/tmp/dsv41-profile-env/bin/python build.py
```

仅生成文档，不加载模型、不执行基准，也不启动服务。浏览器进度保存在当前浏览器的 localStorage；可以重置，不会改动仓库或服务配置。

维护时修改 `guide.md`、`page_template.html`、`style.css`、`interactions.js` 或 `build.py`，再重新生成 `index.html`。生成页内嵌样式和交互脚本，分享单个 HTML 即可离线阅读。
