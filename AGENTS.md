# 当前任务：P3.0 — LunarJEV 公开 Benchmark（2026-11-03）

**首先阅读并执行 [`docs/P3_0_PUBLIC_BENCHMARKS.md`](docs/P3_0_PUBLIC_BENCHMARKS.md)。**
机器可读范围在 [`protocols/p30_public_benchmark_plan.json`](protocols/p30_public_benchmark_plan.json)。这是新任务计划，不是已完成的实验或已冻结的评测协议。

本文件只保留当前有效指令，避免多个历史“ACTIVE SPRINT”相互冲突。此前 P0–P2.2 的完整交接仍保留在 [461316f 版本的 AGENTS.md](https://github.com/wbyuan030/lunarrisk/blob/461316fb50c10ba4570aa55ba56f96a056e6df32/AGENTS.md) 和对应 docs/reports 中，不再要求重复执行那些已完成的任务。

## 研究方向

用公开任务验证“月球预训练表征 + 任务/候选条件化决策”，不再将四属性联合高斯成功作为 LunarJEV 的前置。

- **先跑通：Moonstone 五类年代分类。** 沿用官方映射与未知类过滤；不是连续绝对年龄反演。
- **核心任务：Moonstone 49类地质分类。** 先普通头，再比较同表征、同模态、同预算下的候选条件头。
- SomBench 只在需要核验 NASA LFM 原生输入时做一个有边界的小探针；不同时展开所有任务。
- 本轮不做完整Qwen/OnePass语言主干微调、Latent TiM生成、主动路由、联合后验扩容或全球预训练。

## 现在立即实施：P30-00 / P30-01

1. 查看现有本地资产、磁盘与ROCm/PyTorch；只读列出官方数据/代码/模型版本、文件大小和许可，固定upstream commit/HF revision。
2. 核查真实公开权重。README的 `checkpoints/latest.pt` 只是示例，不算可用预训练权重；找不到则明确 BLOCKED，不从头预训练冒充复现。
3. **首要数据风险：Moonstone数据卡称benchmark可独立下载，但官方loader从全局mmap读像素。** 审计HDF5实际内容、归一化、通道和mask；替代loader必须有等价性证据，不能默默换数据处理。
4. 生成精确下载allowlist，默认dry-run。初次元数据/真实小样本总传输上限1GiB；不要自动拉取约74GB benchmark或约207GB pretraining，不允许用全量snapshot绕过预算。无法部分访问则提交最小完整下载与磁盘方案。
5. 实现离线loader/标签/split测试；已有真实数据且预算内时，读取Age小批次并跑普通头smoke。缺数据时区分 synthetic PASS 与 real I/O BLOCKED，不编造样本。
6. 首次交付 `reports/p3_0_benchmark_audit.md`、必要脚本/测试、asset manifest和 `reports/p3_0_progress.md`。不等待额外确认来替代已授权的免费审计/可逆本地开发。

后续按P30-02→03→04门槛推进。新脚本/报告路径只是计划，尚不存在时必须实际实现后再声称可运行。

## 科学与评测纪律

- 官方随机split、官方地理split、新校准/候选实验分轨记录；不得重分测试集或用Test选prompt/阈值/模型。
- 逐样本真实地质单元ID/名称、年代标签是target，不得作为输入；公开候选定义必须对所有样本统一。
- Age派生于地质标签，不当作独立确认；候选子集必须对比普通分类器masked-softmax。
- NASA LFM的NAC/WAC输入不等于Moonstone28通道；先核对物理含义/单位/尺度与缺失上下文，不能任意塞通道或伪造观测几何。
- 缓存特征只代表冻结encoder；端到端微调与完整延迟另测。训练/验证/校准/测试用途清楚，新增校准轨不得伪装成完全相同的官方复现。
- 用SHA、逐样本预测、可重载checkpoint、真实命令与日志支持结果；只归档可分享的脱敏证据。无证据不报告PASS或收益。

## 旧资产和资源边界保持不变

P2.2证据见 [`reports/p2_2_evidence_completion.md`](reports/p2_2_evidence_completion.md)：当前低秩路线 `STOP_OR_REFRAME`，H1/H2稳定增益 `NOT ESTABLISHED`。不删除负结果、不重新包装为成功。

**旧LROC外部协议 Gate 3仍NOT_FROZEN，Gate 4仍HOLD。** 不批准EDR→CDR变更、不覆盖任何 `protocols/heldout_*`、原始NAC/DTM/CDR、旧labels/cache、P2.2归档；公开benchmark路线不解锁旧Gate4。

主要使用已有RX7900XT/ROCm。不要安装CPU/CUDA torch覆盖现有HIP环境，不改实验室其他用户环境/路径。当前不使用四卡4090；其个人存储≤10GiB约束保留。¥500云预算不是购买授权，任何付费实例/充值仍需单独批准。

大文件/权重/缓存/凭据不入Git；新产物使用p30命名空间。清理空间不能删除他人文件或原始证据。

## 提交与验证

```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts src tests
```

历史444项测试（443通过、1跳过）只代表P2.2版本，不证明新benchmark代码。新增测试需真实执行，数据缺失skip必须说明。

每项完成后更新进度、代码、测试和脱敏证据，再按用户约定直接push `main`。写入前核对最新分支和工作树；遇到非快进/并发变更先协调，不force-push，不丢弃他人提交。超出下载/环境/付费边界则报告准确阻塞及最小方案。
