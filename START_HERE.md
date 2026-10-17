# LunarRisk · Coding Agent 交接包

**交接日**：2026-10-17  
**当下目标**：在已有 RX 7900 XT 20GB / Ubuntu / Docker / ROCm 7.0 环境中，完成真实数据与小模型**最小研究闭环**。不先开发大型模型，不先租云卡，不等周末四卡。

## 交接时的真实状态

- `scripts/` 已有 4 个 Python 脚手架：ROCm 基础训练 smoke、Hugging Face 仓库枚举/小样例下载、NetCDF 坡度统计、NAC crater 训练 smoke。
- 这些脚本以前在 CPU/模拟数据上做过基础检查；**没有证据证明已经在你的 7900 XT 上运行，或真实 NAC→3m 坡度训练成功**。
- 公开 SomBench 的 NAC_HighRes Viewer 目前仅有 6 条示例记录；目录中约百万 Tile 不代表完整 NetCDF 可公开下载。**不得用 6 张示例做科研泛化训练**。
- 坡度标签脚本目前未实现 NAC/DTM 地理配准；crater 模型只预测标注坑数量代理，不能代表真实着陆安全。

## 阅读顺序

1. `AGENTS.md` — 实施边界与停止条件（要求 Coding Agent 优先遵守）。
2. `docs/TASKS.md` — 本次 Sprint P0 工作单与验收标准。
3. `docs/RUNBOOK.md` — 可直接执行的命令及失败分流。
4. `docs/DATA_PROTOCOL.md` — 数据真值、泄漏防线和指标定义。
5. `docs/KNOWN_GAPS.md` — 当前代码缺什么、不要误以为已实现什么。
6. `references/PROJECT_CONTEXT.md` — 长期研究和异构 GPU 部署背景（不等于当下任务）。

## 本次最小验收

- `outputs/rocm_smoke.json`：实际 ROCm GPU 运算记录。
- `reports/data_audit.md`：真实可访问 NAC/DTM_3M/SLOPE_3M 数量、文件体积、单位、坐标/配准情况和组数。
- `outputs/slope_labels.csv`：如真实 slope NetCDF 可读，输出原始有效像素计数的坡度标签；必须标注 **GEO_JOIN_PENDING**（若未完成配准）。
- `outputs/crater_pilot/{metrics.json,risk_coverage.csv}`：若 crater 数据可下载，用作真实图像训练管线检查，不作为科学结论。
- `reports/first_run_report.md`：每个工单的 PASS/FAIL/BLOCKED、日志、耗时、是否允许进入 P1。

**Agent 先执行 P0，不在没有 G0 通过的情况下自动开始 P1。**
