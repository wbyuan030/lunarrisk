# 当前执行任务（P2.0，2026-10-20）：CDR 多相机覆盖与独立配准 QA

**立刻阅读 [`docs/P2_0_MULTICAMERA_QA.md`](docs/P2_0_MULTICAMERA_QA.md) 并直接实施 P20-01 → P20-03 的探针工作。**
本文件后面的旧 P1.9 / P1.8 / P1.7 指令仅供历史追溯，不再是当前任务。

最新已完成的工作（见 [`reports/p1_9_optical_ortho_probe.md`](reports/p1_9_optical_ortho_probe.md)）：
`FRSHCRATER10 / M153639872LC` 真实 NAC CDR → `pds2isis` → 补仪器标签 →
显式 SPICE 内核 → `cam2map` 到与 DTM 相同的 14259×2389 / 2m 网格。
**光学覆盖仅占 DTM 有效像素 42.75%**；弱互相关的 1 像素偏移不等于独立地物配准 PASS。
232 项测试为上一轮 Coding Agent 的记录，新工作要重新运行。

**立即实施**：
- P20-01：优先复用已下载的 `M153639872RC`（右相机）和左相机，做双相机同格网正射、
  coverage-left/right/union/intersection、掩码与拼接 QA；如果联合覆盖不足，才查 README
  中其他采集，不要立刻批量下载 84 景。
- P20-02：独立影像地物配准误差（RMS/p50/p95、离群值）、光度残差与投影一致性；
  特别检查正射时是否使用了目标高分辨率 DTM 像元高度，避免未来测试的地形真值泄漏。
- P20-03：在 2–3 个 Pilot 场景复现（优先非冻结 Probe），统计可用 Tile、
  磁盘/GPU/ISIS 资源、真实失败原因，写报告和回归测试。
- 原预注册 [`protocols/p19_external_policy.json`](protocols/p19_external_policy.json) 规定 EDR 路线；
  新 [`protocols/p19_cdr_route_amendment_proposal.json`](protocols/p19_cdr_route_amendment_proposal.json)
  只是**待确认**的 CDR 协议提案，**不得静默改写原政策或声称新 CDR 已被正式批准**。
- Gate 3 目前仍为 **NOT FROZEN**，必须 QA 与协议审核后才可真正冻结
  `heldout_v2_external`；Gate 4 一次性模型评估**保持 HOLD**。

执行完每个阶段添加离线测试、记录真实命令/Hash/数据质量报告，
更新 `reports/p2_0_progress.md` 后按本仓库习惯 push `main`。
仅用既有 RX 7900 XT/ROCm 与免费公开数据，不购买云卡、不使用 4×4090、
不覆盖原始 NAC/DTM、760 张训练标签或 64 张已冻结产品级 V2。
**不要重复过时的 EDR URL Discovery / SPICE 初始化探索**，除非新数据暴露新的问题。

---

## 历史任务指令（P1.9 等，只供参考）

# 当前首要任务：P1.9 NAC EDR 官方 URL 反查（2026-10-18）

**立即阅读 [`docs/P1_9_EDR_URL_DISCOVERY.md`](docs/P1_9_EDR_URL_DISCOVERY.md) 并执行 Pilot。**
新的发现：LROC 官方分期观测 ZIP/DBF/CSV 自带 `PRODUCT_ID`、`PDS_VOLUME`、
`FILE_SPECI`、`URL` 字段。新增 `scripts/resolve_lroc_edr_urls.py`
可将已有 168 个观测号展开至左右 NAC 相机产品，反查**真实卷/日期路径**，并少量 HEAD 测试链接。
无需再盲猜 EDR 文件 URL。旧版 `docs/P1_9_EDR_ORTHO_RUNBOOK.md` 中的
`lrolc2isis` 也已经修为正确的 `lronac2isis`。

**Coding Agent 现在该做**：拉取 main → 运行全仓单测 → 先下载官方小型
`NAC_EQ_NOMINAL_MISSION_360.ZIP` 索引 → 用现有 `outputs/p19_edr_acquisition_list.csv`
运行 resolver → 分期增补 NP/SP/Science/Extended 观测索引 → 找至少一条本项目 EDR 的
`PDS_VOLUME/FILE_SPECI/URL`，再验证文件 HEAD + 少量 Range/对应 LBL。
在 `reports/p1_9_edr_url_validation.md` 记录每一步证据（真实响应、Git SHA、匹配率、
失败原因）；如 URL 已迁移，也先输出卷内真实路径，再针对卷根映射寻找镜像。

**当前重要事实**：本助手仅离线运行 resolver 的 5 项 synthetic 测试，全量 217 项历史测试
未重新运行；受容器 DNS 限制，**仍未取得经文件级验证的原始 EDR**。这不是 Gate 2 PASS。
严格 `heldout_v2_external` 仍保持 BLOCKED；不得下载整个 EDR 池、放宽独立性、
改写旧冻结集、提前跑 V2 模型或擅自投入付费/4×4090 算力。

---

## 历史 P1.9/P1.8 指令（下文仅供溯源）

# 当前任务：P1.9 外部 LROC 严格独立测试集（2026-10-17）

**必须首先阅读 [`docs/P1_9_HELDOUT_BUILD.md`](docs/P1_9_HELDOUT_BUILD.md) 并直接执行 Gate 0 → Gate 1。**
上文旧 P1.8/P1.7/P0 指令只作为历史备查，不是当前任务。Coding Agent 应先拉取 main，在原 ROCm
环境运行全套单测与 compileall，修复新增的 LROC 元数据筛选测试（如果有失败），随后开展真实
多边形/CRS/立体源 README 独立性审计。

**最新已完成（来源：仓库报告）**：P18-04 fixed-mask 统一 CNN 参数量 98,464、15 次本地训练，
163 项测试通过；P18-05b 找到 LROC NAC DTM 候选档案，但原 **143 景“严格 OOD”判定已被 Review 撤回**：
旧版用等面积圆半径代替 footprint 上界，而且未拦截空 stereo 源 ID。本轮代码只做更保守的
顶点元数据筛选，严格独立性必须验证双方真实 footprint、CRS、完整立体源，再冻结。
参阅 [`reports/p1_8_new_heldout_sources.md`](reports/p1_8_new_heldout_sources.md) 第 7 节勘误。

**务必遵守**：开发集的 110 Tile 已反复使用；旧 64 Tile 产品级冻结清单不得覆盖或重抽；
尚未完成严格 `heldout_v2_external`，不得基于外部候选运行模型预测或声明获得严格 OOD 成绩。
Gate 1 做只读元数据和 footprint 分析；Gate 2 仅在 QA 通过后下载有限 probe；Gate 3
冻结有哈希的正式外部测试集，再返回给用户核对，**不要提前执行 Gate 4 的最终模型评分**。

单卡 RX 7900 XT 和现有 ROCm 环境优先；不自行购买云卡、使用 4×4090、覆盖原始
NAC/DTM/旧 labels、训练 LFM/联合后验。每关必须有测试、实测证据和日志，再 push main。

---

## 历史执行指令（P1.8 等，仅供溯源）

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
