# LunarRisk / LunarJEV：分档投稿与阶段止损实验协议（2026-10-16）

## 0. 决策摘要

**推荐研究主线**：部分观测的月球轨道 NAC + 粗地形 → 隐伏高分辨率地形属性的概率预测 → 动态约束 AST → 事件级风险概率 → 空间外推时的可靠筛查。

**先不强依赖** Qwen3.5-0.8B、Clef/OPD、Latent TiM、Moonstone 地质分类、水冰路由。它们保留为后续增益可检验的扩展，而不是主论文的前提。

**投稿策略**：先构建一套真实、独立、可复现的地形风险实验；实验强度决定最终投稿。不是用同一套数字重复发表三篇文章。

- A 档：联合后验/空间风险事件建模 + 动态规则泛化 + 跨区域 OOD + 可信风险，争取 TGRS / ISPRS JPRS。
- B 档：已有概率方法在月球部分观测风险筛查的显著实证改进、全面 OOD/校准对比，考虑 Acta Astronautica；J-STARS 需事先确认纯月球方向适配。
- C 档：聚焦单一高质量新方法或结论（例如单目 NAC 置信度建模/多光照不确定性），考虑 5 页短稿 GRSL。数据复现和实质贡献仍是必要条件，不能保证发表。

## 1. 研究假设（预注册，避免事后挑指标）

H1 — **概率而非点估计**：相同观测及工程规则下，相较物理属性点预测 + 阈值执行，条件联合属性分布 + 规则积分，能改善事件概率 Brier/NLL，并在给定安全放行覆盖率下降低误放行风险。

H2 — **联合而非独立边缘分布**：对于至少两个有相关预测误差的物理属性构成的 AND/OR 规则，联合建模较独立边缘抽样改善事件校准，重点检验边界区域。

H3 — **跨地域和光照**：在 DTM 源/站点隔离的地理 OOD 和同址多光照对照下，提升能否维持；不对任意 OOD 声称形式化统计保证。

H4（可选，突破一区的核心）— **空间结构的必要性**：相同坡度直方图不等于相同可用安全空间。引入低成本空间危险概率图/空间相关模型可提升最大可用连续平坦圆等空间组合事件的判断。

H5（可选）— **语言条件化的真实价值**：自然语言任务简报 → AST 规则编译 + 概率后验，是否已覆盖 end-to-end LLM 的能力；只有前者解决不了的规则泛化或延迟问题才训练 LunarJEV One-Pass。

## 2. 数据与 GT：唯一主战场 Task A1

### 输入及监督
- 输入：LRO NAC 光学（1m 级）、相应观测几何、在推理期可获得的粗 DTM/坡度（约 60m）；绝不把 3m DTM/坡度作为模型输入。
- 监督：与光学裁剪地理配准的 DTM_3M、SLOPE_3M。以**原生约 3m 栅格**生成物理属性真值。区分 DTM 生产误差与地形真实不确定性。
- 物理目标先设 3 项：坡度超越率曲线、9m/15m 局部平面去趋势粗糙度、高程 5–95% 极差。用其生成形式化可执行规则 `R(a;c)`。主任务名写成 **orbital terrain-hazard screening / preliminary landing-site assessment**，不要直接宣称已经判定真实着陆安全。
- 评测图像裁剪时遵守各模态 CRS、有效掩码、有效 DTM 原始覆盖范围，做地理重投影而非索引层对应；变更窗口必须同步重算 GT。
- A2: 408 张人工 NAC 坑框数据只做一个可控外推实验；按原官方 Box split，不能将框当作精确坑缘分割。

### G0 数据审计（0 GPU）
1. 下载官方 Parquet，枚举磁盘 / S3 上真实存在的 NAC、DTM_3M、SLOPE_3M、观测几何与粗地形；查看缺失率与字节大小，不先假设 2400 张或 3.8GB。
2. 根据 NAC 原始 EDR ID、DTM source ID、footprint WKT、站点/经纬块分组，消除父图重叠和源数据泄漏。
3. 记录每个候选地理足迹的配准质量、像素有效率、粗糙度和阴影/光照条件；检查标签过滤是否只留下平坦区域。
4. 输出 `manifest.parquet`, `split.json`, `audit.md`, `sample-gallery.png`, `checksums.txt`。
5. 建议门槛（工程启发阈值，非统计学定理）：至少 800 张高质量有效样本，至少 20 个空间独立数据块/DTM 源组，测试至少 5 个不同组；不足则缩小结论或换数据源，而不是靠增强填充数量。

### 切分和统计单位
- 固定 train / val_tune / cal / test 四个互斥的父源或空间组，按可用组数调整，例如 60/15/10/15 的组级比例；只在 train 做调参和规则采样。
- 所有阈值组、规则同义改写以及不同观测版本必须继承同一个地理组的 split。
- 同址多光照另建配对测试；不让任一同址图像泄漏到训练。
- 以**独立站点/DTM 源组**作 bootstrap 单位而不是裁剪出的几万条重复 rule-query；报告组数以及统计功效限制。

## 3. 规则库与标签（可复现代码计算）

- L1 阈值：`fraction_slope_gt(8°) < 5%` 等。训练、测试在阈值插值与外推上分别报告。
- L2 同切片属性逻辑：`(slope_exceed < a AND roughness < b) OR relief < c`；训练只见原子及简单双属性组合，测试留出若干 AST 结构。
- L3 空间规则（升级的价值）：从空间危险图计算 `max_clear_disk_radius >= r`，或最大连通低坡度安全区域面积，直接验证纯直方图模型的信息损失。
- L4 自然语言 AST：仅在数值/逻辑任务成立后添加；必须与确定性 AST 编译基线公平比较。
- 各类规则需要平衡正负样本和边界难度，但采样不改变原始 test 分布；报告所有预定义测试规则的完整分布，避免选择最有利阈值。

## 4. 最小且公平的模型矩阵

| 模型 | 用途 | 必要性 |
|---|---|---|
| B0: coarse-DTM rule | 最简物理先验 | 必须 |
| B1: LFM frozen/LoRA → point attributes → AST | 看深度模型点估计收益 | 必须 |
| B2: LFM → diagonal probabilistic attributes → QMC(AST) | 概率化但不建模相关性 | 必须 |
| B3: LFM → joint posterior (rank=2/4, 或可行的简单混合分布) → QMC(AST) | 相关后验的真实价值 | 必须 |
| B4: 领域地形重建网络（ELunarDTMNet 等） → attributes → AST | 地形重建直接竞争者 | 重要；若无代码，明确为复现版本，不能称官方结果 |
| B5: B3 + calibrated accept/abstain threshold | 给定放行覆盖率的风险控制 | 必须 |
| B6: B3 + 小型空间危险图/空间后验 | 空间拓扑规则是否需要新表示 | 仅在核心实验通过后 |
| B7: LLM parser + B3 与 LunarJEV/OPD | 语言与速度是否真正贡献 | 可选，不是基本论文的必要条件 |

训练时保持 LFM 前端、可见模态、划分、总训练预算一致；B1 从联合后验中导出点估计时要使用合适的 Bayes 后验均值或任务损失最佳决策，而不是故意采用差的潜空间均值反变换。

**注意**：区域坡度直方图是真实空间频率，不等于模型对直方图的不确定性。B3 应产生物理属性向量的联合条件分布；ALR/ILR 遇到零桶需要明确处理，仅用于建模变换，不可改变原始规则真值。

## 5. 评估指标与 Figure

**物理层**：坡度超限率 MAE / RMSE、粗糙度回归误差、后验 NLL、平均校准区间覆盖率与宽度（在条件满足时才宣称理论覆盖）；按观测几何/地形复杂度分组。

**决策层（核心）**：事件 Brier/NLL、Reliability Diagram、ECE、UAR = P(true hazard | accepted safe)、HMR = P(accepted safe | true hazard)、安全放行率 Coverage = P(accepted safe)。在 **匹配的 Coverage (例如 10%、20%、40%)** 下比较 UAR，并提供 UAR-Coverage 曲线与站点组 bootstrap 置信区间。若危险基率低，单报比率会不稳定，附原始危险数/放行数和区间；1% 作为压力测试而不是无条件安全保证。

**OOD**：空间留出、源 DTM 留出、同一地点不同光照配对；报告 ID / OOD 的绝对数值与性能下降，避免只报告相对百分比。

**空间拓扑压力测试**：寻找直方图近似相同但最大无障碍连续圆半径差异明显的真实或受控样例；若 B3 失败而空间方法成功，构成较强新研究问题。

**效率**：显存、batch=1/实际吞吐、GPU-hours、延迟，以及计算校准/采样的费用；如训练时用了 QMC 采样，报告 S=256/1024 的收敛与时延。

预计核心图：1 数据覆盖与切分地图；2 方法架构（点 vs 后验 vs 空间）；3 地形属性估计；4 事件可靠性；5 UAR–Coverage；6 地理/光照 OOD；7 空间拓扑反例；8 性能与计算成本。

## 6. 200 GPU-hours 阶段账本与门槛（估算，不是实测）

| 阶段 | GPU h 上限 | 累计 | 独立成果 / 止损标准 |
|---|---:|---:|---|
| P0 环境复现 + 真实 Profiling | 10 | 10 | LFM 能读真实地形影像，记录 seconds/step；若不可用先停 |
| P1 物理 GT + 基础对照 B0/B1 | 30 | 40 | 地理 OOD 数据划分 + 物理基线 + 规则求值，形成可复现数据基座 |
| P2 可靠概率模型 B2/B3 | 45 | 85 | 后验 NLL/事件 Brier/边界校准是否超过点估计与独立边缘 |
| P3 校准+风险曲线+OOD | 35 | 120 | 完整的应用型论文材料；检验提升是否在空间留出下保持 |
| P4 仅当 P3 有强信号时尝试 B6 空间后验或 B7 语言决策（优先二选一） | 45 | 165 | 形成一区可能需要的额外新方法和严格验证 |
| Buffer（仅用于失败重跑、额外种子、真实强基线） | 35 | 200 | 不预先消耗 |

预算不含 CPU 数据审计、存储/传输和投稿版面费。超时按实测吞吐量减少模型数/种子数/可选任务，不通过更改 test split 或丢弃失败案例凑结果。

### 门槛及转向
- **G0 fail**：高分辨率立体 DTM 足迹太少或组数不足。停大模型训练；另寻公开地形数据（不能把 408 张坑检测和 156 张水冰切片冒充独立的千样本地形真值）。
- **P1 fail**：LFM 连简单粗 DTM 基线都不稳胜。优先排查配准/标签/任务定义，不急于叠加 LoRA/LLM。若只有较窄的创新点，考虑短文路线；不保证可发表。
- **P2 fail**：联合后验不优于对角/点估计。保留诚实的负结果，不伪造创新；尝试事件直接概率或重新选重要任务，若无实质增益就停止。
- **P3 pass, P4 fail**：提交以可靠地形风险评估为主题的应用型论文，优先考虑适合月球空间工程的 2 区期刊或短文。
- **P4 pass**：如果方法创新、基线齐全且跨区域可靠性改善有稳定证据，争取一区。

## 7. 投稿阶梯、Scope 与钱

下列分区取所检索的 **2025 中科院大类分区资料**，正式投稿前需核对单位认可版本。不是 JCR Q1/Q2/Q3 的同义转换。

| 目标 | 示例期刊 | 特别注意 |
|---|---|---|
| 一区 | TGRS / ISPRS Journal of Photogrammetry and Remote Sensing | 有方法贡献 + 强地学实测；ISPRS 已发表月球统一 Transformer 工作 |
| 二区 | Acta Astronautica | 适合真实航天应用/场地风险；不应误用“安全着陆认证” |
| 二区（条件） | IEEE J-STARS | 官方 scope 更偏地球应用；必须事先邮件确认纯月球稿件适配；OA APC US$1800 |
| 三区 | IEEE GRSL | 官方明确包含月球和其他行星遥感；只允许 5 页，需聚焦单一重大方法/实验结论 |

J-STARS 官方 APC（2025-01-01 后投稿）US$1800；MDPI Remote Sensing 官网当前 APC CHF2700；不同学科和年份分区不同，勿仅看某一个 JCR Q1 标签。GRSL 对会员与非会员的页费政策不同，投稿前核官方。

GPU 现金成本用 `实际 4090 小时单价 × 实际 GPU-h` 计算。例如若为 ¥3/小时，全预算上限 ¥600；¥6/小时则上限 ¥1200，**仅为假设单价示例**，未含存储、数据传输与版面费。若使用自有显卡则现金支出较小，但仍应记录电费、设备占用和时间成本。

硬件前提：历次方案写 `NVIDIA RTX 4090 24GB`，但曾有实际项目资源为 `Radeon 7900XT 20GB + ROCm 7.0` 的情况；这会明显改变内核可用性与训练吞吐，请以本次实际 GPU 作为 G0 记录，不混用 4090 的估算数字。

## 8. 最近一周应该完成的可见成果

- Day 1–2: 下载并验证官方索引/影像，产出真实可用的 manifest + 漏洞/缺失清单；生成 20 例 NAC-DTM 对齐可视化。
- Day 3: 完成物理属性 GT + AST Rule Engine，跑 unit tests 和人为构造的同直方图不同空间排列反例。
- Day 4: 在真实显卡上运行 LFM frozen 的 200 steps profiling，得出吞吐量、内存占用、磁盘读入耗时。
- Day 5–7: 训练 B0/B1，出第一版 ID/OOD 坡度超越误差与 UAR–Coverage 曲线，完成 go/no-go 评议。

**第一周交付验收**：`manifest.parquet`、`splits.json`、`audit.md`、`baseline_results.csv`、`profile.json`、`fig_align.png`、`fig_risk_coverage.png`；凡是数据未下载/代码未运行，明确写 pending，绝不称实测。

## 9. 关键文献和官方资料

- NASA-IBM Lunar FM: https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model
- SomBench data: https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-pretraining-data
- ELunarDTMNet: https://doi.org/10.1109/TGRS.2024.3501153
- The Moon’s Many Faces: https://doi.org/10.1016/j.isprsjprs.2026.04.008
- Stochastic Terrain Maps for Lunar Landing: https://doi.org/10.1109/AERO66936.2026.11519921
- Conformal Risk Control: https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html
- GRSL scope: https://www.grss-ieee.org/publications/author-resources/scope-of-grsl/
- JSTARS APC/scope: https://www.grss-ieee.org/publications/journal-of-selected-topics-in-applied-earth-observations-and-remote-sensing/
