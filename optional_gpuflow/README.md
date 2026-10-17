# LunarRisk GPU Flow — 低数据传输 / 多算力调度 MVP

**定位：** 读配置、检查资源、按策略选择机器、手动确认启动训练；不创建付费实例、不购买资源、不上传数据、不执行无提示的远程修改。**当前只适用于单台本地 AMD 机器 + 已获授权的周末 SSH 4090 服务器。** 实际训练脚本由你提供，`config.json` 中的 `train.py`、`train_ddp.py` 只是占位示例。

## 1. 开始

- Python ≥3.10，无第三方 Python 依赖；本地 Linux、WSL 或 macOS 都可作为控制端（本地 ROCm 任务必须在 ROCm Linux 上启动）。
- 将 `config.json` 的 `ssh_host`、两台机器的 `workdir`、`data_ready_marker` 改成真实值。
- 两台机器各自配置适合自身的 PyTorch 环境；不要复制 AMD 的 Python 虚拟环境到 NVIDIA 机器。
- 完成数据审计、确认文件与 Checksum 后才在目标机创建 `data/READY`。**READY 只代表用户确认数据就绪，代码未实现逐文件 checksum 校验。**
- 两台机器各自 `git clone` 相同提交的训练代码；训练数据留在各自存储。只同步 Git、JSON 配置、Manifest、checkpoint/指标。

```bash
python gpuflow.py list
python gpuflow.py plan smoke_test
python gpuflow.py plan smoke_test --probe
python gpuflow.py probe home_7900xt
python gpuflow.py run smoke_test --node home_7900xt             # 只打印，不启动
python gpuflow.py run smoke_test --node home_7900xt --execute --yes
python gpuflow.py status smoke_test --node home_7900xt
```

周末 SSH 机器：

```bash
ssh user@host 'nvidia-smi'
python gpuflow.py plan joint_seed42 --probe
python gpuflow.py run joint_seed42 --node lab_weekend_4090x4 --execute --yes
python gpuflow.py status joint_seed42 --node lab_weekend_4090x4
```

**启动必须显式传递 `--execute --yes`。** 运行持久化依赖 `nohup`，断开本地 SSH 不会使进程自动退出；建议在正式实验中用 `systemd-run`、Slurm 或 `tmux` 进一步增强进程与退出码管理。`PROCESS_PRESENT` 仅代表 PID 存在，不能替代实验完成标记。配置在 `workdir/runs/<job_id>/stdout.log` 中记录日志。

## 2. 执行策略

| Resource | Suitable workloads | Dataset requirements |
|---|---|---|
| RX 7900 XT, 20GB (ROCm 7.x) | 仿真后验、物理指标、冻结 LFM 特征、小 Batch/LoRA | 本地完整数据或小规模切片 |
| 周末 4×RTX 4090, 每卡24GB | 4 个独立 seeds/模型；必要时单机 DDP | 提前预置必要数据；不能每周大规模传输 |
| CompShare (可选) | 周末不可用时兜底 | 首先查询实时库存/价位，创建付费实例需要另行确认 |

GPU 小时计算：4 卡并行使用 8 小时 = 32 GPU-hours；不能把四卡显存简单合并为 96GB 单卡。

优先将实验划分为 **一 GPU 一个独立 seed/基线**，只对无法在单卡运行或单实验明显受吞吐限制的任务启用 `torchrun`。本例 `joint_4gpu_ddp` 仅支持**4 张 GPU 在同一主机**；若是四台机器，需另行配置多机地址、网络/NCCL 与共享代码和数据。

## 3. 数据零/低传输原则

1. 在本地 CPU/7900XT 上审计文件并生成 `manifest.parquet`、源 ID、每条记录所需模态及 checksum。
2. 需要原始图像端到端微调的机器，必须有**本地访问原始训练数据的能力**；仅复制 manifest 不够。
3. 周末服务器下载可用时从原始数据源就地选择性拉取，并缓存，之后不重复下载；如果对外网下载受限，仅把必要训练集压缩/打包后**一次性**同步，不进行每周重复传输。
4. 若无法预置原始数据，先在 AMD 上提取**冻结特征**：例：2,400 samples × 64 tokens × 768 dims × FP16 ≈ 236MB。这个缓存可以支持轻量决策头训练，但 **不能** 支持更新 LFM 编码器的端到端训练。
5. 日常同步 Git、配置、Manifest、指标和小 checkpoint；大型权重留在服务器且按周期备份。
6. 使用固定数据版本和脚本 Git commit，保证跨 CUDA/ROCm 实验公平。

可使用 Hugging Face `snapshot_download` 的 `allow_patterns` 或 `hf_hub_download` 选择文件，但前提是所需数据在仓库实际公开可用，不能把 Parquet 中的路径当作可直接下载的保证。

## 4. 只读云端库存查询

CompShare CLI 已有 `instance zones/search/create --dry-run`，创建真实资源前需要用户确认。MVP 只读查询地区列表：

```bash
pip install compshare-cli
compshare config --name default
python gpuflow.py cloud --execute-readonly
# 如需精确可用卡型，先查看：
compshare --json instance search --help
```

现阶段不实现未经你确认的云资源付费创建、停机或删除。官方 CLI 支持 `--json`、`instance create --dry-run`、`instance job submit` 等，后续可扩展为可预算限制的 provider 适配器。

官方文档：<https://github.com/compshare-cn/compshare-cli/blob/master/skills/compshare-cli/SKILL.md>

## 5. 当前限制与注意

- `weekend` 仅按北京时间周六/周日判定，不保证实验室机器当时空闲或你已获得授权；`--probe` 检查联网/GPU 数量，不检查单卡显存剩余和其他用户任务。正式调度应增加 GPU 利用率和显存阈值，最好配合实验室调度制度。
- 不自动下载/校验数据，不自动创建 PyTorch 环境，不集成 Slurm，不自动分派 4 个独立模型并行；需要在 `config.json` 中指定设备或使用已有作业管理器。
- 运行前请核实 `workdir` 和命令安全，任务命令来自用户可编辑配置。
- 不存储云端私钥，不保存 SSH 密码。建议 SSH key + Tailscale，或使用校园授权入口。
- 使用正式云付费实例时，避免把长时间下载/空闲误算为有效 GPU-hours。

## 6. 后续升级优先级

1. SSH `nvidia-smi` JSON/CSV 采样，检查每张 GPU 的空闲显存；显卡忙时不抢占。
2. 增加四路单卡 job queue 与最大使用时长限制、执行退出码、自动记录 git commit/数据版本。
3. 小文件与日志 rsync、checkpoint 自动备份和跨机恢复测试。
4. CompShare 有库存才生成带价格上限的 dry-run proposal；**需二次人工授权**才真实下单。
