# 当前任务：P1.8 续篇（2026-10-17）

**请先阅读 [`docs/P1_8_CONTINUE.md`](docs/P1_8_CONTINUE.md)，按其 B→C→D 的 Gate 顺序直接开展工作。**

最近进展：P18-01 Resume、P18-02 原子 CSV、P18-03 统计协议检查已由 Coding Agent 实施，实测记录为 **148/148 PASS**；P18-05 严格 Group-OOD 测试集冻结返回 BLOCKED（当前 100k Manifest 的 272/272 个 Group 已被历史 760 Tile 触及）。留有 **64 Tile / 22 新 NAC 产品 / 4 个已使用 Group** 的产品级冻结候选，**不是严格 Group-OOD**。依据 [`reports/p1_8_progress.md`](reports/p1_8_progress.md) 与 [`reports/p1_8_heldout_protocol.md`](reports/p1_8_heldout_protocol.md)。

当前首要任务：**P18-04a fixed-mask 固定参数量的训练/消融实现及测试**；并行做官方/公开新立体 DTM/NAC 源数据索引、ID/地理独立性调查。然后用 7900 XT 做三种子、合法增强对照。**不要重复已完成 P18-01/02/03，不要改写 64 Tile 冻结清单，也不要冒充已有严格 OOD 评估**。

历史约束仍有效：旧 110 Tile 只能作为 `development_test`；不买云卡、不使用 4×4090、不修改原始 NAC/DTM、不覆盖既有 ROCm PyTorch、不训练大型 LFM / 联合后验。每项新增成果请附测试/真实证据并按用户既定要求推送 `main`。

---

## 历史任务指令（P1.8 初版和 P1.7 / P0，仅保留作追溯）

# 当前执行任务（P1.8，2026-10-17）

**立即阅读 [`docs/P1_8_NEXT_STEPS.md`](docs/P1_8_NEXT_STEPS.md) 并按 P18-01 → P18-07 的依赖顺序实施。** 本文件下面的 P1.7、P0 文字仅作历史溯源，**不再是当前执行任务**。最近一次已完成实测是 P1.7 的 12-run 消融，见 [`reports/p1_7_ablation.md`](reports/p1_7_ablation.md)。

本次用户授权 Coding Agent 完成以下范围：修复 `hydrate_tiles.py` 已知 Resume no-op/CSV schema 安全问题并回归测试；增强跨配置统计一致性与 NLL 口径；准备固定网络参数量的模态 masking 消融与合法空间增强对照；**先冻结新的独立 `heldout_v2` 测试协议**，再决定本地增量扩容到 3,000–5,000 train Tile。

**前两项必须立即动手**：P18-01 修复 `--resume` 已全部下载时的错误退出；P18-02 改造标签 CSV 原子合并与 schema 演进。旧 760 Tile 的标签、split、group、source 不能改变；在合成 fixture 和旧标签备份上完成验证前，禁止实际执行大规模追加下载。

科研底线：旧 110 Test 已被用于多轮模型分析，只能用作 `development_test`；不要把 P1.6 的 NAC 正增益或 P1.7 的无显著差异表述为最终因果结论。真实 stereo DTM 来源 ID 仍是 UNKNOWN，不能声称完全独立的 DTM 源级 OOD。

资源底线：本地 7900 XT 优先，保留既有 ROCm PyTorch；不得自行购买付费云卡、使用周末 4×4090、覆盖共享数据或训练大模型。每项代码修复完成后添加测试和 `reports/p1_8_progress.md` 的可复验证据，按用户偏好直接 push `main`；推送前先 `git pull --ff-only`，出现冲突及时停止。

---

# 当前任务覆盖（2026-10-17，P1.7）

**本仓库已完成 P0/P1/P1.5/P1.6；以下历史 P0 约束保留作溯源，不应再阻止当前 P1.7 任务。**
最新执行文档：[`docs/P1_7_NEXT_STEPS.md`](docs/P1_7_NEXT_STEPS.md)。
经用户授权的本轮工作是：修复指标/评估审计、运行 7900 XT 单卡受控消融、冻结新的未见过的测试区域协议；随后再决定是否扩至 3–5k 真实 Tile。

Agent 的本轮执行顺序：
1. `git pull --ff-only origin main`；单测及静态检查；核对现有 HIP/PyTorch，不得盲目覆盖环境。
2. P1.7 四组受控消融，均用 `--no-augment` 与种子 42/43/44；记录 `reports/p1_7_ablation.md`，不得复制 P1.6 指标冒充本轮结果。
3. 明确指出旧 110 Tile Test 已被多次查看，仅用于探索；冻结独立的 `heldout_v2` 地理/源组，检查 DTM provenance 缺口。
4. 再执行受限样本扩容和独立验证，逐步报告。不要提前动用四卡、云 GPU 或大模型。

**保护条款**：任何付费算力、覆盖现有环境、删除共享/原始数据必须先经用户单独批准。10GiB 四卡服务器存储限额仍有效。本次只需单卡 7900 XT。模型输出安全标签仅是工程代理，不是认证着陆安全结论。

---

# AGENTS.md — LunarRisk 单卡启动约束

> 给 Pi / Codex / Claude Code 的工作协议。本 Sprint 中 `AGENTS.md`、`docs/TASKS.md` 优先于旧版 LunarJEV v5.x 大型蓝图。

## 本次任务

首先验证本地 RX 7900 XT 20GB + ROCm 7.0 Docker 的训练环境，再核实真实月球 NAC/3m DTM/3m slope 资产是否存在且能获取，生成物理坡度标签；如 A1 被数据卡住，使用官方 NAC crater 数据跑通工程性训练与评估。提交可复现日志与明确 Go/No-Go 决定。

## 硬约束

1. 本地 AMD 是主要执行端。**不要重装/覆盖现有 ROCm PyTorch**，不要让 pip 装 CPU/CUDA torch 取代它。记录现有 `torch`, `torch.version.hip`, `/dev/kfd`, `/dev/dri`。
2. 用户周末可使用**同一台四卡 RTX 4090 服务器**，但个人存储至多 **10 GiB**，且本次 P0 **不使用这台机器**。
3. 云 GPU 现金预算上限 ¥500；**任何付费实例创建/充值/购买必须得到单独明确批准**。当前工作不使用付费资源。
4. 不删除或改写实验室其他用户路径、任务与共享环境；下载前审计磁盘和下载范围；保护凭据。
5. 真实物理标签只能由真实原生网格有效像素计算。可在本地模拟/随机数据上测试代码，但报告必须清晰区分 **synthetic test** 与 **real lunar measurement**。
6. 仅有公开 6 条示例记录，或者图像与地形未配准时，**禁止训练声称有效的 NAC→3m 安全筛查模型**；不通过编造/重复裁块来填样本数。
7. `crater_pilot.py` 的 target 是「人工标注坑数量大于训练集阈值」，只用于工程 smoke，**不是物理着陆安全真值**。
8. 不用测试集调阈值/模型、不把同源相邻影像当独立测试站点；留存 Git SHA、数据源 revision、split 和 seed。
9. 此 Sprint 不做 Qwen/One-Pass、TiM、联合高斯、四卡调度器改造、云卡自动购买。
10. 每完成一项在 `reports/first_run_report.md` 追加真实证据；阻塞就说明阻塞，不静默跳过。

## 已有代码与边界

| 文件 | 现有用途 | 必须注意 |
|---|---|---|
| `scripts/rocm_smoke.py` | 随机张量 + 小 CNN 的 60-step 训练 | 不代表 LFM 的实际吞吐 |
| `scripts/hf_audit.py` | repo 文件列表、过滤下载 | 不做完整 Manifest、资产真实性或源级联接 |
| `scripts/make_slope_labels.py` | 对独立 NetCDF slope 网格统计经验分布 | 不查单位、不做 NAC 几何 crop 配准 |
| `scripts/crater_pilot.py` | COCO/.npy 训练与 Brier/Risk-Coverage | 需要确认下载路径/standard 标注，科学用途受限 |
| `optional_gpuflow/` | 早期资源规划/SSH 启动 MVP | 尚无数据 staging/磁盘保护/安全四卡批量队列 |

## P0 完成定义

`P0-01..P0-06` 全部有 PASS/FAIL/BLOCKED/SKIPPED 状态和证据；产出 `reports/first_run_report.md`，明确 `GO_REAL_A1` / `BLOCKED_REAL_A1` / `NEEDS_MORE_DATA`。未经明确新指令不进入 P1。
