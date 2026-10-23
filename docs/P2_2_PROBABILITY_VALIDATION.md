# P2.2 — 概率评价正确性与 H1/H2 公平验证（2026-10-23）

> **Coding Agent 当前执行清单，尚非实验结果。** 起点 `main@6df35e6` 后的 P2.2 评分修复提交。先跑回归测试，再重跑开发集实验，**不得将历史 P2.1 指标冒充修正后的结果**。
>
> 总目标：判断在相同观测、相同场地和规则下，四属性条件概率后验是否改善事件 Brier/NLL 和固定放行覆盖率下 UAR；若 Joint 优于 Diagonal，厘清增益是否真来自 **跨属性低秩相关**，而不是不同 head、训练不稳定或校准方式。

## 先读的科学事实与方法限制

- 既有 P2.1 训练：760 Tile，其中 train/val/cal/development_test = **400/120/120/120**，开发集 25 Group，被反复查看；三个 seed 42/43/44。**不是外部 OOD**。
- P1.8/P2.1 部分此前报告使用约 110 vs 120 个开发 Tile，不能将不同样本口径的 MAE、Brier 直接放在一张表里判断优劣。
- 既有结论：校准后的 `g10>0.05` Point/Diagonal/Joint Brier 为 0.0811/0.0951/0.0796；Joint−Point 仅 −0.0015，无显著性证据。U=0 同 checkpoint 先前有改善（0.0792→0.0736），但**校准口径与实验运行配置必须修复并重新验证**。历史数字保留供追溯。
- `P1.8` 的 `coarse_ridge`、固定 `direct-event` 分类头目前尚未与 `P2.1` 公平配对；H1/H2 不能只和弱化的 Point 参照比较。
- 外部 DTM 分组 v2 的 41 Group / 15 Group、18 景候选属于新的未签核计划，`PENDING_SIGNOFF/NOT_FROZEN/HOLD`，**不得运行 Gate 4**。

## P22-00（P0）：核验三处已提交的核心修复

### 修复一：Point 温度应真实作用于采样

`scripts/train_posterior_p21.py::predict()` 的 `point_sigma`（训练残差 std）在温度拟合后，应变为
`point_sigma * T_point`。Point、Diagonal、Joint 都在 **cal** 上独立拟合自己的温度，不在 test 拟合。
分别测试 `T=1` 与 `T>1`：在相同模型、相同种子、相同噪声下 `z_T-mu = T(z_1-mu)`，
且 Point 的事件概率必须可因 T 而变化。错误或非有限 T 直接拒绝。

### 修复二：同 checkpoint U=0 的校准对照

- 同一个 Joint checkpoint 的 `mu` 和 `log_sigma` 及 encoder 完全不变；
- `U=0` 与原 Joint **共享同一个从 Joint Cal 拟合的 T** 的比较称为 `shared_T`；
  报告同 Seed、同蒙特卡罗随机数下的差，必须保证除 U 外全部相同；
- 可以另跑 `U=0` **重新从 Cal 拟合自己 T** 的公平配置，标为 `refit_T_on_cal`，
  严禁同一张表混用 `shared_T`、`refit_T_on_cal` 和 `uncalibrated`；
- 建议增加 `--ablate-u-calibration-mode {shared,refit}`、每种方案的 T、
  逐 seed 配对差/物理属性分布校准证据。
- 如果 U=0 同时使模型的 marginal variance 发生变化，说明它不是“只移除相关系数”：
  再加一种 `Sigma_diag = diag(diag(Sigma_joint))` 的 **matched-marginals** 消融，
  或保留联合模型全部边缘方差仅置零 off-diagonal covariance，才可针对“属性相关性本身”做归因。
  不能把只置零 U 的改善直接解释为“相关性有害”，因为 U=0 同时减少边缘方差。

### 修复三：Physical-space NLL 的变量变换恒等式

设 `a=f(z)`，在连续的内部点 `z=f^{-1}(a)`：

`-log p_A(a) = -log p_Z(z) + log |det Df(z)|`。

- 必须在**真实 z** 计算 log-Jacobian（不是网络预测 `mu`）；
- 使用**正号**而不是负号；
- Point 模型训练时的平方误差不是真正的概率密度 NLL；在用训练残差和 Cal 温度形成高斯后才可把它当分布评分，不能混为一谈；
- 若 `g10=0/1`、`g15=0` 等物理边界通过 epsilon clip，单纯连续变量变换不产生正确的边界点质量。必须明确说明评测只覆盖内部连续事件，或设计带端点质量的混合分布；当前只需保证公式和测试正确，**不要把边界被 clip 的物理 NLL 当最终论文指标**。
- 添加已知形状的 Jacobian 公式/符号测试、真值 z vs 均值 mu 的差异测试。

### 必须实测

```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
python3 -m unittest discover -s tests -p 'test_train_posterior_p21.py' -v
python3 -m unittest discover -s tests -p 'test_posterior.py' -v
```

出现任何测试失败先修；上一版 `412 tests PASS` 属于**历史版本**，不可当作新代码测试结果。
写 `reports/p2_2_scoring_fix_validation.md`，含命令、GPU/CPU/库版本、测试数、Git SHA 与真实输出。

## P22-01（P0）：建立可复现、公平的统一比较矩阵

1. **固定样本集合**：把 `coarse_ridge`、P1.8 `direct-event`、Point、Diagonal、Joint 映射到同一批
   `sample_id`、`group_id`、相同 train/val/cal/development_test；输出每模型可用 ID 集、
   并集/交集数、排除原因及 SHA256。若某基线只能使用另一个样本集，须重跑或另表注明，禁止混排。
2. **统一规则**：使用 `src/lunarrisk/rules/ast.py` 的同一 evaluator 生成 ground truth。
   预注册 `g10>0.05`、`g15>0.02`、含 roughness/relief 的 AND/OR，
   与 Train 确定的阈值网格；不能用 Test 挑“最有利的规则”。
3. **公平概率**：四属性 Point + Train 残差高斯（Cal T）、Diagonal + Cal T、
   Joint + Cal T、Joint U=0 shared_T 与独立 Cal refit_T、Joint matched-marginals；
   固定规则 direct-event 分数原样保留为不可迁移的强参考。Ridge 使用 Train/Cal
   的可审计概率映射，不要把回归概率直接当充分校准事件概率。
4. **checkpoint 真正落盘**：当前 `scripts/train_posterior_p21.py` 的 JSON 仅保留
   `weights_sha256`，丢弃了 `state_dict`；必须新增原始 PyTorch `*.pt` 可加载文件，
   保存标准化标量均值/方差、目标变换与 eps、训练种子与代码 SHA、全部 Cal T、
   训练/评估 ID、输入 cache provenance、网络结构与权重 SHA256。
   加 round-trip 测试，保证新进程重载后每个模型的事件概率与原输出一致。
5. **重复性**：ROCm 先检查 `torch.use_deterministic_algorithms`/MIOpen 的可行性，
   无法保证完全确定时报告限制；对最主要 P1/Joint 配置增加同 Seed 不同独立运行，
   将训练/校准/采样随机性分开。不能用一次重复的差直接定义统一的“噪声地板”。

## P22-02（P1）：H1/H2 的统计验收

- 保持 seed 42/43/44，所有配置在相同 Scene、规则上评估。保存每个 `sample_id`、
  `group_id`、GT 事件和概率，以便配对分析，不仅存聚合均值。
- 对每条规则的 **ΔBrier / ΔNLL**、给定 Cal 决策阈值下的 UAR/HMR 用 **Group paired bootstrap**，
  而不是对 Tile 或规则问句进行独立重采样。给出 95% CI、组数、危险基率、放行人数和危险放行人数。
- `coverage=10/20/40%` 同时提供后验排序的**oracle-style fixed-coverage 分析**和
  Cal 选定阈值的 **deployable risk threshold** 结果；不能用 Test 危险标签/分位数来校准部署阈值。
- 训练与测试内部分布变化已存在：`P(g10>0.05)` 为 train 0.568、val 0.350、
  cal 0.708、development_test 0.692。对不同危险率进行分层讨论，不夸大统计稳定性。
- 至少使用 256/1024 个样本检查蒙特卡罗误差，配对固定随机数；最终可再增加 QMC/Sobol 比较，但**不要把普通 MC 误写成 QMC**。
- 决策门槛：若 H1/H2 仍不能稳定超越 Ridge/direct-event baseline，
  报告 `STOP_OR_REFRAME`，不要为了正结果任意添加 LFM/One-Pass 或重复尝试 Test 阈值。

产出 `reports/p2_2_baseline_comparison.md`、`reports/p2_2_hypothesis_decision.md`，
附可复现命令、原始逐样本概率、参数量、GPU 时间、误差条、真实失败案例。

## P22-03（P1，并行小任务）：外部 Held-out v2 继续保持 HOLD

- v2 当前在 69 景之内经过 2346 对 polygon 距离计算为 41 个 Group，
  按报告跨组最小 40.05 km；SPARIM 合组问题已修。
- 剩余：确认更广泛 LROC 662 景中的桥接/同源风险，逐源 README provenance 和
  CDR→EDR 原预注册变更审批，确保 2m/3m 目标定义与独立影像正射不泄露目标 DTM 真值。
- 只执行不看模型风险结果的 QA，暂不正式签署 `heldout_v2_external_protocol_v2.json`，
  不在外部景上训练、调参或选模型；Gate 3 `NOT_FROZEN`，Gate 4 `HOLD`。

## Coding Agent 执行次序

**第一提交**：P22-00 修复验证与文档勘误；旧 P2.1 表格保留，但醒目标注来自修复前的历史代码。

**第二提交**：P22-01 基线的统一样本/概率口径、完整 checkpoint、同 checkpoint 的 shared/matched-marginals/refit 消融和单测。

**第三提交**：P22-02 三 Seed / 组级配对 CI / 动态 AST 规则实验和结论。

每次测试后按仓库惯例直接推 `main`。仅使用现有 RX 7900 XT / ROCm、已有 760 Tile、
已有数据缓存与公开免费元数据；不购买算力，不使用四卡，不扩大数据集；
**不要启动 Gate 4 或改写先前冻结文件**。
