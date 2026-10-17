# 可直接粘贴给 Coding Agent 的首条指令

请接手这个 LunarRisk 研究仓库，先阅读 `START_HERE.md`、`AGENTS.md`、`docs/TASKS.md`、`docs/RUNBOOK.md`、`docs/DATA_PROTOCOL.md` 和实际 `scripts/` 代码。**本次只执行 Sprint P0（P0-01 到 P0-06）**，在我的 RX 7900 XT 20GB / Ubuntu / ROCm 7.0 Docker 中进行环境 smoke、官方 SomBench 数据真实性审计、坡度 NetCDF 标签校验，并在主任务缺少充分真实监督时，使用官方 crater COCO 做真实图像训练管线验证。每项任务要留下可复现实验命令、输出文件和报告，最后明确 GO/BLOCKED_REAL_A1/NEEDS_MORE_DATA。

**不要**在 6 张公开样例上训练自称有效的月球地形模型；不要将未配准 NAC 和 slope 同像素索引对应；不要安装 CPU/CUDA torch 覆盖 ROCm；不要把撞击坑数量代理称为真实着陆安全；不要租云卡、占用周末四卡、做 Qwen/TiM/One-Pass 或大模型训练。遇到关键阻塞先记录真实证据与替代数据方案，不要虚构结果。
