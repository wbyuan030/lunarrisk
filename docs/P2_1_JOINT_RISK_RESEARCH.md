# P2.1 — 回到科学主线：联合属性后验 × 动态规则 × 风险受控筛查

> **Coding Agent 当前执行任务（2026-10-22）。这是计划与验收条件，尚非实验结果。**
>
> 当前已知基线：`main@016c0eba96dab82041f15e85e792202bdf5b63bc`，开发数据 `outputs/labels_hydrated.csv`（760 Tile）、P1.8 fixed-mask 模态消融、`110 Tile / 25 Group` 反复查看的 `development_test`。
> 数据与方法依据：`reports/p1_8_fixed_mask.md`、`reports/p2_0_registration_qa.md`、`protocols/heldout_v2_external_protocol.json`。
> 原始研究问题：**部分 NAC + 60m 粗地形 → 米级隐伏地形属性的联合条件分布 → 可变工程规则的危险概率 → 固定放行率下的错误放行风险**。P2.0 的数据工程是这条主线的基础，不是论文本身的主要创新。

## 同步执行的两条工作流

**A — H1/H2 最小方法闭环（主线，立即开发与跑小实验）**。沿用已经验证的 NAC/粗地形/观测几何输入和 train/val/cal/development_test。不要重新大规模下载、把旧 development_test 当真正 OOD，也不要因为想接 LFM 而延迟。
 
**B — Gate 3 的 Group 聚类修复（P0，高优先级并行）**。最新 `protocols/heldout_v2_group_pool.csv` 和 `protocols/heldout_v2_external_protocol.json` 的 15 组 / 18 景**仅为预注册草案、尚未签核或冻结**。组划分现只用最近邻边做 Union-Find，不等于所有距离小于 10km 的多边形对形成的真实 Single-linkage。当前 `SPARIM1` / `SPARIM2` 中心同为 (−51°, 171°)、nearest distance=0，却被分成两个选中 Group，必须核查真实 polygon 后重建所有 Group。**在 B 修复前不允许 Gate 3 FROZEN，更不得启动 Gate 4**。

## A0 — 输入与标签审计（0 GPU，最先做）

- 确认 `scripts/train_baseline.py` 中真正喂给预测器的通道仅来自推理可用的 NAC、`slope60m` 和几何标量；现有 `dtm3m_roughness_9m_m` 与 `dtm3m_relief_p95_p5_m` 是从高分辨率 DTM 派生的**监督目标，不得成为模型输入**。
- 从 760 行真实 label 中审计以下四个属性可用数、缺失率、单位、物理分布与组间差异：
  `slope3m_g10`、`slope3m_g15`、`dtm3m_roughness_9m_m`、`dtm3m_relief_p95_p5_m`。
  特别验证 `0 ≤ g15 ≤ g10 ≤ 1`、粗糙度/relief 为非负且计算范围与分辨率一致；数据缺失应 fail-closed 或使用显式 target mask，**不可把缺失目标当零**。
- 所有预处理的均值/方差、目标变换统计、AST 阈值范围和任何后验超参必须从 train 或 val 决定；cal 仅校准；旧 test 仅做**明确标注的探索性**评估，不能以结果反复挑规则。
- 训练-校准-评估各 split 的 `sample_id/group_id/source_id` 核对，保留 provenance/hash。对图像对齐与 target DTM 使用同一切片视域和有效像素 mask。
- 输出 `reports/p2_1_target_audit.md` + 必要脚本与离线合成测试；列出四个目标在每个 split 的真实非空数及分布。

## A1 — 真正的多属性条件后验（H1、H2 的前置）

新建小型模块（建议 `scripts/train_joint_posterior.py` 与 `src/lunarrisk/probabilistic/`，可按已有仓库习惯调整），**不要覆盖 `train_baseline.py` 或 P1.8 历史输出**。

至少四属性：`g10,g15,roughness_9m,relief_p95_p5`。注意两个坡度超限率天然有偏序：`g15≤g10`。建议用物理受限的可逆（内部开集）坐标：
- `z1 = logit(g10)`，`z2 = logit(g15 / g10)`（遇到边界用 train 固定 epsilon clip，必须记录边界处理），从而 `g10=sigmoid(z1)`，`g15=g10·sigmoid(z2)`；
- `z3=log1p(roughness)`，`z4=log1p(relief)`，必要时仅在 train 上标准化。
- **检查数值边界**：`g10=0`、`g15=g10`、粗糙度或 relief=0 时必须稳健处理；若模型在受限坐标中训练 NLL，应明确 NLL 的密度空间（含变量变换 Jacobian 或只报告 transformed-space NLL），不可混称物理单位 NLL。

同一个轻量视觉 encoder（建议重用现有 98,464 参数 fixed-mask 前端）：构造公平对照：
1. `P1 point`: 四属性点估计，从 encoder/条件后验均值导出，统一下游规则接口；包括原 `coarse_ridge` 与当前双事件分类头作为必要参照。
2. `P2 diagonal`: 条件异方差高斯后验 `z ~ N(mu, diag(sigma²))`。
3. `P3 joint`: `z ~ N(mu, diag(sigma²)+U Uᵀ)`，初始 rank=2；固定 encoder 结构、数据拆分、训练总预算与 seed 42/43/44。
4. 为隔离“相关性”而非额外参数收益，除 P2/P3 的 end-to-end 比较外，还要做 **同一 joint checkpoint 把 U 强制置零/相关性打乱** 的后验相关性消融（需保持 marginal variance 合理），再复查点估计与 diagonal 的独立训练结果。
5. 损失先使用明确口径的条件高斯 NLL，不假定这种后验一定优于简单 baseline；按 train-only/val-only 控制过拟合与轻量正则，记录 trainable 参数、显存、wall time。禁止在 development_test 上调先验、规则或网络配置。
6. 训练产物必须保存完整推理所需的 encoder、target transforms、scale parameters、random seeds、checkpoint SHA256。CPU synthetic smoke 后才使用 RX7900XT/ROCm 跑真实小规模。

**验收测试**：协方差对称半正定且 Cholesky 稳定，所有采样满足坡度概率序关系与属性非负；diag/joint 在零 U 时数学一致；mask/变换/序列化安全；任何高分 DTM 目标列不能出现在模型输入；GPU 失败可 CPU fallback（只做 smoke）。

## A2 — 确定性规则 AST 与模型共享的概率积分

统一 API：`rule.evaluate(attributes) -> bool`（向量化，包含明确单位）和
`prob_event(posterior, ast, draws, seed) -> P(unsafe)`。

- 节点首批仅 `Compare(attribute, threshold, op)`、`AND`、`OR`、`NOT`；
  AST 纯数据结构，JSON 可序列化、字段白名单+阈值/单位校验，任何非法节点/非有限阈值必须拒绝。
- 从真值计算布尔事件必须使用**同一个 AST evaluator**，不能一个地方写 `g10>c` 而另一个地方偷偷换条件；`safe` / `unsafe` 语义始终一致。
- 规则首批至少 `g10>0.05`、`g15>0.02` 与含粗糙度/relief 的 AND/OR 组合；多阈值扫描只按**train 上固定网格与物理单位**做。留出至少一个复合 AST 结构供结构泛化探索，旧开发测试结果仍不得称“未见规则独立泛化”。
- Sobol QMC / 普通 Monte Carlo 在四维后验做同一采样预算，例如 S=256/1024；P1/P2/P3 使用配对随机数减少采样噪声；输出事件概率而非抽样后单阈值分类。验证常数规则、互斥/恒真事件、De Morgan、AND/OR 边界、极端相关性、积分收敛和确定性可复现。
- **绝不把数据条件 `g10>0.05` 的直接概率头输出当作“可变规则的一次推理能力”**；保留 direct-event head 为固定规则参照即可。

交付：`src/lunarrisk/rules/` 或对应脚本、`tests/test_rules_ast.py`、`tests/test_posterior_sampling.py`，以及贯通测试。

## A3 — 第一轮研究实验（只使用历史开发数据，非独立 OOD）

固定三 seed（42/43/44），同组配对 Bootstrap（至少报告独立 Group 数与样本数），规则组合在运行前写入单独 versioned config。报告：
- **物理预测**：每属性 MAE/RMSE、分布校准、变换空间 NLL；对粗糙度和 relief 不要省略误差；
- **事件风险**：Brier、NLL、Reliability、ECE，以及在覆盖率 10/20/40% 的 UAR/HMR 和绝对危险/放行计数；
- **核心成对比较**：P1 vs P2（H1），P2 vs P3 + same-CKPT 相关性消融（H2），外加 `coarse_ridge` 和 direct-event fixed-rule head；
- **公平性与失败分析**：模型参数、训练/采样延迟、训练时间、相关性方向是否和属性残差相符；避免只挑“有利”的 AST/阈值。Bootstrap CI 不等于完全覆盖 seed/训练噪声，须独立报告。
- `Cal` 只用于单独校准映射及风险阈值选择；不能因“固定覆盖率”而用 Test 标签确定阈值；UAR–Coverage 报告须同时包含“曲线分析”（按排序扫 coverage）与“Cal 预定阈值的部署式决策”，两者不能混淆。
- 对全体规则与 scene/group 的结果使用成对设计；多重比较/少样本限制必须明确标注，不得宣称 P2/P3 已被证明显著优于基线。

交付：`reports/p2_1_joint_posterior.md`、`outputs/p21_*/` 的可重跑命令/配置与 Hash；如果后验不优于点模型，**先分析归因并启动止损 Gate，不能强行上 LFM/One-Pass**。

## B — P0 立刻修复组级地理独立性（与 A1 并行）

**已发现可复核的 Bug**：`scripts/audit_groups_p20.py::cluster()` 只沿每个 DTM 的 `nearest_candidate_DTM` 合并；但 Single-linkage 需要**所有真实多边形距离 <10km 的 pair edges**。最近邻还可能指向 69 景以外被剔除的景。现有协议选中了中心坐标同为 (−51,171) 且 `min_candidate_distance_km=0` 的 `SPARIM1/SPARIM2` 两个 Group。这是强烈数据污染警讯，**并不单凭坐标证明 footprint 重叠**。

1. 从原始 LROC DTM Shapefile 及已通过 Gate 1/2 的 69 景实际几何重新计算 `69×68/2=2346` 个 **polygon-to-polygon** 距离（注意月球半径/CRS/跨经线、极区与 multipart）；计算全部 `dist<10km` 边后 Union-Find；不要只用最近邻、不要只比较中心、不要沿用旧 `candidate_dtm_group`。
2. 明确输入的是现有可用候选集的几何全集，并审计与现有 probe / historical used / protected sets 的隔离。为了避免选样时的外部**桥接**污染，需要在完整数据全集（含潜在可用但当前无双相机的附近景）检查是否有同址几何重叠；定义并文档化不纳入桥接者的理由。
3. 交付每个 pair 的距离/edge 表、每组成员和组间最短距离。添加测试：
   - A 最近 B、B 最近 A、C 与 B 距离 <10km 但不是任何点的最近邻时，A/B/C 仍必须合组；
   - `SPARIM1/SPARIM2` 真实 polygon 的距离和实际 Group；
   - 同经纬中心/距离零、跨 ±180°、极区、空几何、两独立场景间 <10km → 必须 FAIL；
   - 全部 Group 间实际最小 polygon 距离 ≥10km（已批准的阈值）。
4. 重建 `protocols/heldout_v2_group_pool.csv` 和 **另一个新版本协议**（建议 `heldout_v2_external_protocol_v2.json`）。**不要静默覆盖**已有尚未签核的 v1 15 组 / 18 景草案，保留旧 hashes 供对照；v2 更新 Group 数、选样顺序、scene hashes、差异清单和独立性检查。
5. 本轮新增冻结前保守检查脚本 `scripts/check_group_pool_collision.py`：对旧池的同中心可疑分组生成 BLOCKED 审计报告，只是检测器，**不能用它代替完整两两 polygon 修复**。
6. 两个版本均维持 `PENDING_SIGNOFF`、`gate3_status=NOT_FROZEN`、`gate4_status=HOLD`，任何下载和最终模型预测依然等待研究人员明确批准。

## 当前任务次序与 STOP/GO

- **先做 B 的分组审计与 A0/A1 的代码框架**：不需要用户再次批准可逆、免费、离线的实验及回归修复。
- **再做 A2 和 A3 的小 GPU 实验**：复用现有 760 个 Tile、现有 ROCm，不重新下载，不训练大模型。
- **Gate 3 正式独立集冻结须单独审阅研究协议**，尤其要处理 `EDR → CDR` 的版本化方案、2m/3m 尺度和 provenance。
- **Gate 4 OOD 只许在冻结协议、文件 hash、模型候选/CKPT/规则预注册全部完成后一次性运行**，不能因开发集表现不好就在冻结集上继续选模型。
- 研究中仍称`preliminary lunar terrain-hazard screening`，不是“已验证安全着陆”或有形式保证的 OOD risk control。
- 原两条分工全部完成后才考虑 LFM frozen/LoRA 和领域强基线（ELunarDTMNet）接入；若 H1/H2 开发集没有合理增益信号，不优先转向 One-Pass/Qwen/JEPA。

### 交付节奏

1. **立即 push**：P2.1 本任务文件、AGENTS/README 新入口、保守 Group collision 检查器与离线测试；
2. **Coding Agent 第一次提交**：A0 标签 audit + B 的全 pair 距离表与真正重新分组；
3. **第二次提交**：A1/P1-P3 后验训练框架、损失与 CPU smoke、A2 AST/QMC + 单测；
4. **第三次提交**：A3 三 seed 小实验和报告，标注 exploratory，并另行审查真正 OOD 冻结条件。

每次必须 `python3 -m unittest discover -s tests -v` 和 `python3 -m compileall -q scripts tests`；结果不允许只引用过去 319 项 PASS 代替本次测试。
