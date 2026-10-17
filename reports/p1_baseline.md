# P1-03/P1-04 — 首个真实物理基线（NAC → 3 m 坡度超出率）

> ### 2026-10-17 P1.6：第二轮 review 修复后的重跑（v3，**当前最新**）
> 本节取代 §0 中的 CNN 行（常数与 `coarse_ridge` 未受影响）；§5–§7 仍为更早的 pre-fix 记录。
> 完整数值：`outputs/baseline_v3_summary/{summary.json,summary.md,figures_seed42.png}`，
> 每次运行：`outputs/baseline_v3_seed{42,43,44}/`。**所有 v3 结论以本节为准。**

#### P1.6.1 review 逐条核实结果

| review 项 | 核实结论 | 处理 |
|---|---|---|
| §1 逐图 1%–99% 拉伸抹杀单目 NAC 对比度 | **机理成立**（回归测试复现：平坦切片 1e-4 噪声被 `robust_normalize` 拉到满量程） | 新增 `nac_contrast = (I−median)/median` 通道 + `log(p99−p1)`、`log(median)` 与观测几何标量进 head |
| §1 预测"修复后 cnn_nac 应显著变好" | **未被实验支持**：`cnn_nac` MAE 0.3238±0.0143（v2）→ **0.3115±0.0288**（v3），差异在噪声内 | 如实报告，不据此宣布 NAC 可用 |
| §2 缓冲区最近邻遮蔽 Bug | **成立**，已用 3 切片共线反例复现：train 被删而 val 距 test 仅 4.55 km 仍存活 | 改为按优先级对**存活集合**查询；新增 `buffer_contract` 断言与 6 项回归测试 |
| §2 对真实数据的影响 | **真实数据上改变 0 个 Tile**（split 前后逐一比对 90,786 行，`split_final` 零差异；drop 仍 123） | 记录为"真实缺陷但本轮无下游影响" |
| §3 缺少配对差值 Bootstrap CI 与导出列 | **不成立**（基于旧修订）：配对 CI 自 `af6a7af` 起在 `scripts/aggregate_seeds.py` 中实现，且 `constant_g15`/`coarse_g15`/`*_prob_<rule>` 均已导出 | 本轮仅补充新增两条规则的导出与聚合 |
| §4 60 m 粗坡度未按坐标重采样 | **成立** | 新增 `sample_grid_on_coordinates()`，NAC 与粗坡度共用同一坐标采样原语；`metrics.json.coarse_alignment` 报告偏移与 NaN 预填 |
| §5 缺少粗糙度 / 高程极差标签 | **成立**（E3/E4 前置） | 新增 `hydrate_tiles.py --enrich`，在本地已下载数据上补出（无需重下） |
| §6A `plot_baseline.py` 无用 `--reference` 读取 | **成立** | 删除 |
| §6B `--resume` 全量已下载时报错 | **成立** | 改为仅当既无新记录又无既有 labels 文件时才退出 |

#### P1.6.2 变更与验证

- NAC 3 通道：`nac_contrast`（保留绝对对比度）、`nac_texture`（局部结构）、`coarse_slope`（坐标采样到 3 m 网格）。
- head 标量：`log_contrast`、`log_median` 只给含 NAC 通道的变体；几何（incidence/emission/phase/方位 sin·cos）给所有变体，
  以保持"固定结构隔离光学信息"的对照有效。标量标准化仅在 train 上拟合。
- 新增预注册规则：`g10_gt_0.15`、复合 `g10>0.10 OR g15>0.03`（阈值只由 train 先验确定，未接触 test）。
- 切分：缓冲区修复后 `buffer_contract.satisfied = true`（存活集合最小距离：train–cal 10.02 km、train–val 49.63 km、
  train–test 20.14 km、cal–val 62.72 km、cal–test 74.13 km、val–test 622.60 km）。
- 单元测试 **81/81 PASS**（新增 15 项：对比度通道、坐标采样、退化 extent、标量表、平衡规则、规则打分的回归测试）。
- 队列未变：760 Tile、110 test / 25 组 / 10 zone；unsafe(>0.05) = 76/110。

#### P1.6.3 v3 结果（3 seeds，110 test Tile / 25 组，primary 概率）

| 模型 | MAE g10 | Brier g10>0.05 | Brier g10>0.15 | Brier 复合 | AUC(g10>0.05) |
|---|---|---|---|---|---|
| `constant` | 0.3554 | 0.2284 | 0.4273 | 0.3545 | 0.500 |
| `coarse_ridge` | **0.0504** (0.0000) | 0.0728 | 0.0779 | **0.0684** | 0.966 |
| `cnn_nac` | 0.3115 (0.0288) | 0.1891 (0.0118) | 0.2135 | 0.2028 | 0.729 |
| `cnn_coarse_grid` | 0.0687 (0.0168) | 0.0806 (0.0024) | 0.0815 | 0.1175 | 0.964 |
| `cnn_fusion` | 0.0606 (0.0057) | **0.0583** (0.0064) | **0.0506** (0.0030) | 0.0958 | **0.985** |

规则先验（test）：`g10>0.05` 0.691、`g15>0.02` 0.482、`g10>0.15` 0.573、复合 0.645。

**配对组级自助差异（1000 次重采样，3 seed 平均，`*` = 95% CI 排除 0）**

| 比较 | 规则 | ΔMAE [95% CI] | ΔBrier [95% CI] |
|---|---|---|---|
| fusion − coarse_grid | g10>0.05 | **−0.0081 [−0.0137, −0.0025]\*** | **−0.0223 [−0.0428, −0.0005]\*** |
| fusion − coarse_grid | g10>0.15 | **−0.0081 [−0.0137, −0.0025]\*** | **−0.0309 [−0.0596, −0.0019]\*** |
| fusion − coarse_grid | g15>0.02 | +0.0028 [−0.0083, +0.0144] | −0.0150 [−0.0377, +0.0121] |
| fusion − coarse_ridge | g10>0.05 | +0.0102 [−0.0013, +0.0220] | −0.0145 [−0.0390, +0.0092] |
| fusion − coarse_ridge | g15>0.02 | **+0.0311 [+0.0189, +0.0444]\***（更差） | +0.0180 [−0.0131, +0.0554] |
| coarse_grid − coarse_ridge | g10>0.05 | **+0.0183 [+0.0077, +0.0286]\***（更差） | +0.0078 [−0.0028, +0.0164] |

**固定 Coverage 的 UAR（3 seed 平均）**

| 模型 | g10>0.05 @0.2/0.4/0.6/0.8 | g10>0.15 @0.2/0.4/0.6/0.8 |
|---|---|---|
| `constant` | 0.818 / 0.818 / 0.742 / 0.716 | 0.773 / 0.705 / 0.606 / 0.568 |
| `coarse_ridge` | 0.136 / 0.250 / 0.485 / 0.614 | 0.000 / 0.068 / 0.318 / 0.477 |
| `cnn_nac` | 0.455 / 0.530 / 0.566 / 0.621 | 0.333 / 0.364 / 0.439 / 0.504 |
| `cnn_coarse_grid` | 0.106 / 0.242 / 0.485 / 0.614 | 0.000 / 0.098 / 0.359 / 0.492 |
| `cnn_fusion` | **0.015** / 0.227 / 0.485 / 0.614 | **0.000** / **0.061** / 0.318 / 0.466 |

#### P1.6.4 结论与限制

1. **单目 NAC 的增益首次变得可统计分辨**（fusion − coarse_grid 的 ΔMAE 与 ΔBrier 在 g10 两条规则上均排除 0），
   但量级很小（ΔMAE ≈ 0.008 g10 面积占比），且 **`cnn_fusion` 的 MAE 仍不优于 `coarse_ridge`**（g15 规则显著更差）。
2. **review 的归一化假设不成立**：修复对比度通道后 `cnn_nac` 仍远差于粗坡度（0.3115 vs 0.0504）。
   单目光度与地形的桥接瓶颈不是单次归一化，可能是分辨率/光照未建模（几何只进了标量而非空间变体）。
3. 平衡规则（`g10>0.15`，先验 0.573）确实给出更有判别力的曲线：`cnn_fusion` 在 20% coverage 下 UAR 0.015，
   40% 下 0.227；但 `coarse_ridge`/`cnn_coarse_grid` 在 20% 也是 0.000，优势仍不悬殊。
4. 新增的 DTM 多属性（`dtm3m_roughness_9m_m` 中位 0.0167 m、`dtm3m_relief_p95_p5_m` 中位 31.6 m）为 E3/E4 的
   联合属性审计准备就绪：relief 与 g10 相关 0.91，roughness 与 relief 仅 0.46（属性不共线），incidence 与 g10 相关 0.08。
5. 限制照旧：399 train Tile、128 px、单一架构、25 个测试组、`GEO_JOIN_PENDING` 仍未做 footprint 投影审计；
   v3 的 ΔMAE 结论依赖 3 个 seed 与 25 组，仍属小样本探索性结果。

---

> ### 2026-10-17 post-fix 重跑（§0，已被 P1.6 取代 CNN 行）
> 触发：上游提交 `4ba8511`/`29575da` 修复了风险排序反转、事件头共享、变体种子不配对、粗坡度未标定四个缺陷；
> `docs/P1_5_NEXT_STEPS.md` 要求重跑并给出配对 CI。§5 的旧数值仍保留，但仅作 pre-fix 记录，不得引用其 UAR/HMR。

## 0. Post-fix 重跑（3 seeds，110 test Tile / 25 地理组）

修复内容（上游，非本轮）：risk evaluator 改为按**最低 P(unsafe)** 优先放行（此前按 `1-P(safe)` 升序，等于优先放行最危险样本，
使 UAR 假性偏高）；每个 CNN 改为 4 输出 `[g10, g15, logit_unsafe_g10, logit_unsafe_g15]`，两条规则各有独立标签/BCE/标定/评估；
所有变体使用同一 `--seed`（此前按 `--variants` 顺序偏移）；`coarse_ridge` 事件概率同样用 `cal` 做 Platt 标定。
旧的 3 输出 checkpoint 不兼容，必须重训（本轮全部重训）。

环境：`AMD Radeon RX 7900 XT` 19.984 GiB，PyTorch **2.5.1+rocm6.2**，HIP **6.2.41133-dd7f95766**，Python 3.10.12。
耗时（wall，GPU）：seed 42 **44 s**（含重建 `cache_v2.npz` 34.5 MB）、seed 43 **21 s**、seed 44 **21 s**，合计 86 s。
数据：`outputs/labels_hydrated.csv`（1,087,235 B）、110 test Tile / 25 组 / 10 个 LTM zone、unsafe 76/110。
切分独立性复核：272 组中 **0** 组跨 split，**0** 个 PRODUCT 跨 split。单元测试 **66/66 PASS**（新增 8 项聚合与 fixture 测试）。

### 0.1 均值（种子间 std），primary 概率

| 模型 | 规则 | MAE | RMSE | Brier (primary) | Brier (raw) | AUC | NLL |
|---|---|---|---|---|---|---|---|
| `constant` | g10>0.05 | 0.3554 (0.0000) | 0.4355 (0.0000) | 0.2284 (0.0000) | 0.2284 | 0.5000 | 0.6498 |
| `coarse_ridge` | g10>0.05 | **0.0504 (0.0000)** | **0.0705** | 0.0728 (0.0000) | 0.0702 | 0.9659 | 0.2283 |
| `cnn_nac` | g10>0.05 | 0.3238 (0.0143) | 0.4473 (0.0224) | 0.1853 (0.0109) | 0.3421 (0.0405) | 0.7307 | 0.5504 |
| `cnn_coarse_grid` | g10>0.05 | 0.0593 (0.0059) | 0.0798 (0.0048) | 0.0783 (0.0031) | 0.0733 (0.0017) | 0.9663 | 0.2414 |
| `cnn_fusion` | g10>0.05 | 0.0573 (0.0037) | 0.0811 (0.0050) | **0.0612 (0.0069)** | 0.0778 (0.0144) | **0.9782** | **0.1873** |
| `coarse_ridge` | g15>0.02 | **0.0301 (0.0000)** | 0.0446 | 0.0751 (0.0000) | 0.0760 | 0.9374 | 0.2549 |
| `cnn_fusion` | g15>0.02 | 0.0501 (0.0038) | 0.0796 | 0.0759 (0.0015) | 0.0746 | **0.9521** | 0.2460 |

### 0.2 配对组级自助差异（1000 次重采样，单位 = 泄漏组，种子间先平均）

| 比较 | 规则 | ΔMAE [95% CI] | ΔBrier [95% CI] | 排除 0 |
|---|---|---|---|---|
| fusion − coarse_grid | g10>0.05 | −0.0020 [−0.0079, +0.0036] | **−0.0171 [−0.0331, −0.0019]** | 是 |
| fusion − coarse_grid | g15>0.02 | −0.0045 [−0.0135, +0.0045] | **−0.0277 [−0.0469, −0.0104]** | 是 |
| fusion − coarse_ridge | g10>0.05 | +0.0069 [−0.0014, +0.0154] | −0.0116 [−0.0264, +0.0031] | 否 |
| fusion − coarse_ridge | g15>0.02 | **+0.0201 [+0.0099, +0.0307]** | +0.0008 [−0.0141, +0.0161] | 是（MAE 更差） |
| coarse_grid − coarse_ridge | g15>0.02 | **+0.0245 [+0.0159, +0.0330]** | **+0.0285 [+0.0065, +0.0510]** | 是（CNN 更差） |

**结论修正**：在固定结构对照下，单目 NAC 通道对**事件概率**有小而可分辨的改善（两条规则的 ΔBrier 均排除 0），
但对**点估计 MAE 无改善**（CI 含 0）；`cnn_fusion` 相对 60 m 特征岭回归在 MAE 上不占优（g15 显著更差）。

### 0.3 Risk–coverage（3 seed 平均 UAR / HMR，primary 概率）

| 模型 | 规则 | cov 0.2 | cov 0.4 | cov 0.6 | cov 0.8 |
|---|---|---|---|---|---|
| `constant` | g10>0.05 | 0.818 / 0.237 | 0.818 / 0.474 | 0.742 / 0.645 | 0.716 / 0.829 |
| `coarse_ridge` | g10>0.05 | 0.136 / 0.039 | 0.250 / 0.145 | 0.485 / 0.421 | 0.614 / 0.711 |
| `cnn_nac` | g10>0.05 | 0.424 / 0.123 | 0.530 / 0.307 | 0.596 / 0.518 | 0.629 / 0.728 |
| `cnn_coarse_grid` | g10>0.05 | 0.076 / 0.022 | 0.250 / 0.145 | 0.485 / 0.421 | 0.614 / 0.711 |
| `cnn_fusion` | g10>0.05 | **0.061 / 0.018** | **0.227 / 0.132** | 0.485 / 0.421 | 0.614 / 0.711 |
| `cnn_fusion` | g15>0.02 | 0.045 / 0.019 | 0.068 / 0.057 | 0.146 / 0.182 | 0.352 / 0.585 |

无筛查基线（unsafe 率）g10 规则 0.691、g15 规则 0.482。修正后 20% coverage 下 60 m 基线 UAR 从假性的 1.000 降到 0.136。

### 0.4 标定审计（seed 42，raw → calibrated）

| 模型 | 规则 | Brier raw → cal | NLL raw → cal | AUC | Platt |
|---|---|---|---|---|---|
| `coarse_ridge` | g10>0.05 | 0.0702 → 0.0728 | 0.2240 → 0.2283 | 0.9659 | slope 1.05, b 0.15 |
| `cnn_coarse_grid` | g10>0.05 | 0.0714 → 0.0826 | 0.2268 → 0.2528 | 0.9656 | slope 0.98, b 0.76 |
| `cnn_fusion` | g10>0.05 | 0.0892 → 0.0705 | 0.2625 → 0.2094 | 0.9783 | slope 1.17, b −0.75 |
| `cnn_nac` | g10>0.05 | 0.3465 → 0.1791 | 0.9587 → 0.5367 | 0.7604 | slope 0.63, b 1.33 |

标定并非总是有益：`coarse_ridge` 与 `cnn_coarse_grid` 的 raw Brier 更好，`cnn_fusion` 与 `cnn_nac` 需要标定才有竞争力。
两规则分别报告，未在 test 上调参。

### 0.5 两个事件头是否退化

`|g10 − g15|` 均值 0.044–0.160，Pearson 0.88–0.99，三个种子均**非完全相同** → 两个头各自学习，未退化为同一输出。

### 0.6 Post-fix 结论

图（post-fix，seed 42）：`outputs/baseline_fixed_summary/figures_seed42.png`
（UAR–coverage 曲线已在 3 seed 平均下单调上升；20% coverage 时 `cnn_fusion` UAR 0.061、`coarse_ridge` 0.136、
`cnn_nac` 0.310、`constant` 0.815，无筛查基线 0.691）。

**GO small scale-up（有条件）**：修正后的风险曲线、双事件头、切分独立性与标定审计均通过；
但证据仍指向 **60 m 坡度层是主要信息源**，单目 NAC 只在事件概率上有边际增益。
下一步按协议：先冻结额外的 held-out 测试协议，再把训练扩到 3–5k Tile，并把**观测几何 + 光度归一化**作为受控消融。
LFM / 联合后验 / TiM / Qwen / 周末 4×4090 / 付费云仍 **HOLD**。

---

## 1. 任务与规则

> 数据版本：S3 `nasa-lunar-fm-bench` 100k release，索引 SHA256
> `f9c39d1b570693902bd5616d44f6330ea15a425db3c9d3eb9ceb2cdc9be1873e`。
> 脚本：`scripts/{hydrate_tiles,train_baseline,plot_baseline,aggregate_seeds}.py`。
> 单测：66/66 PASS（`tests/test_{slope_labels,build_manifest,finalize_split,train_baseline,aggregate_seeds}.py`）。

- 目标属性：`y1 = g(10°)`、`y2 = g(15°)`，即原生 3 m slope 网格面积占比（原始计数，未做平滑/伪计数）。
- 预先固定规则：`unsafe = 1[y1 > 0.05]`（工程代理，非认证着陆安全）。
- 第二条独立报告规则：`y2 > 0.02`，阈值未参与任何调参。
- 口径：`Coverage = accepted_n/test_n`；`UAR = unsafe_accepted_n/accepted_n`（accepted_n=0 未定义）；
  `HMR = unsafe_accepted_n/unsafe_total`；事件 Brier/NLL/AUC；物理属性 MAE/RMSE。

## 2. 数据与切分

| 项 | 值 |
|---|---|
| 已下载 Tile | 760（train 400 / cal 120 / val 120 / test 120），每 Tile 含 NAC + DTM_3m + SLOPE_3m + SLOPE_60m + metadata |
| 输入可用性过滤 | NAC 对齐后有限像素比例 ≥ 0.99：可用 749，排除 11（test 10、train 1） |
| 共享训练/测试 | 399 train / 110 test，25 个独立泄漏组 |
| 独立 zone | train 60、cal 25、val 8、test 10 个 LTM_CODE |
| 组级重叠 | train/cal/val/test 之间 groups = products = 5 km blocks = 0 |
| 10 km 缓冲 | 丢弃 123 个 train Tile（靠近 cal/val/test） |
| 组间最小中心距 | train–test 20.1 km，train–val 49.6 km，cal–val 62.7 km，val–test 622.6 km |

被排除的 11 个 Tile 的标签均值 g10 = 0.509、g15 = 0.322、`g10>0.05` 比例 0.636；
不是"全 NaN"（0 个），而是极区影像存在约 29% 无效像素。为公平比较，所有模型共用同一 110-Tile 测试集。

## 3. 地理对齐（GEO_JOIN_PENDING → 坐标重采样）

`align_nac_to_slope_grid()` 读取每个文件自带的 `x`/`y` 坐标向量，把 512×512 @1.000 m 的 NAC 用
`map_coordinates`（双线性）重采样到 171×171 @2.994152 m 的 slope 网格（比值恰为 512/171），
不使用"同名数组中心对齐"的像素索引假设。

实测跨全部 Tile 一致：

- 网格原点偏移 0.9971 m = 0.9971 个 NAC 像素（系统性约定差）；
- slope 文件无 `units` 属性；单位由官方 README 声明 + DTM 重算经验验证（12/12，corr ≥ 0.985，均值比 0.995–1.025）确认为 degrees；
- 60 m 层为 9×9 @56.889 m，覆盖同一 512 m 范围，来源 SLDEM2015（LOLA + Kaguya TC），与 3 m 层来源（USGS NAC 立体 DTM）不同仪器 → 作为基线输入不构成泄漏。

## 4. 模型（同一 train/test 子集、同一 seed、同一早停协议）

| 模型 | 输入 |
|---|---|
| `constant` | 无（train 均值 / train 先验） |
| `coarse_ridge` | SLDEM2015 60 m slope 的 7 个统计量（frac>10、frac>15、mean、q50、q90、q99、valid_fraction）→ Ridge + Logistic |
| `cnn_nac` | 坐标对齐后的单目 NAC（128×128，逐 Tile 鲁棒归一化），4 层 CNN |
| `cnn_coarse_grid` | 同一 CNN，通道改为 60 m 坡度网格（上采样到 128×128），**用于隔离 NAC 贡献** |
| `cnn_fusion` | 同一 CNN，两通道 = NAC + 60 m 坡度网格 |

CNN 输入尺寸 128、batch 16、Adam lr 1e-3、最多 60 epoch、patience 10、flip/rot90 增强、
按 val MAE 早停；事件概率用 `cal` 集做 Platt 标定。`cnn_nac` vs `cnn_coarse_grid` vs `cnn_fusion`
是唯一能分离"光学影像是否提供额外信息"的对照。

## 5. 结果（110 test Tile / 25 组）— **已被 §0 取代**

> ⚠️ 以下为 **pre-fix** 数值（commit `f55eb50`）：`risk_coverage` 排序方向错误，UAR/HMR 不可用；
> 且 CNN 只有一个事件头（g15 规则复用了 g10 的 logit），g15 事件 Brier 不可用。
> MAE/RMSE 与 g10 事件 Brier 的模型排序在 post-fix 后仍大体成立，但请以 §0 为准。

### 5.1 点估计

| 模型 | MAE g10 | RMSE g10 | Brier g10 规则（primary） | Brier 原始 | AUC | MAE g15 | Brier g15 规则 |
|---|---|---|---|---|---|---|---|
| `constant` | 0.3554 | 0.4355 | 0.2284 | 0.2284 | 0.5000 | 0.2724 | 0.2614 |
| `coarse_ridge` | **0.0504** | **0.0705** | 0.0702 | 0.0702 | 0.9659 | **0.0301** | **0.0760** |
| `cnn_nac` | 0.2943 | 0.4125 | 0.1746 | 0.2919 | 0.7628 | 0.1910 | 0.1981 |
| `cnn_coarse_grid` | 0.0595 | 0.0819 | 0.0798 | 0.0716 | 0.9659 | 0.0496 | 0.1221 |
| `cnn_fusion` | 0.0687 | 0.0883 | **0.0602** | **0.0567** | **0.9822** | 0.0568 | 0.1089 |

（常数基线：g10 规则 unsafe 先验 Brier = 0.2136；全量无筛选预测 Brier = 0.2284。）

### 5.2 组级自助 95% CI（1000 次重采样，重采样单位 = 泄漏组）

| 模型 | MAE g10 | Brier（g10 规则） |
|---|---|---|
| `constant` | 0.3548 [0.2911, 0.4134] | 0.2284 [0.2134, 0.2426] |
| `coarse_ridge` | 0.0504 [0.0405, 0.0614] | 0.0702 [0.0369, 0.1142] |
| `cnn_nac` | 0.2937 [0.2237, 0.3677] | 0.1742 [0.1388, 0.2032] |
| `cnn_coarse_grid` | 0.0592 [0.0481, 0.0697] | 0.0796 [0.0484, 0.1169] |
| `cnn_fusion` | 0.0687 [0.0567, 0.0819] | 0.0602 [0.0301, 0.0974] |

### 5.3 固定 Coverage 下的 UAR / HMR（primary 概率）

| 模型 | cov 0.2 | cov 0.4 | cov 0.6 | cov 0.8 |
|---|---|---|---|---|
| `constant` | UAR 0.818 / HMR 0.237 | 0.818 / 0.474 | 0.742 / 0.645 | 0.716 / 0.829 |
| `coarse_ridge` | 1.000 / 0.289 | 1.000 / 0.579 | 0.985 / 0.855 | 0.830 / 0.961 |
| `cnn_nac` | 0.909 / 0.263 | 0.886 / 0.513 | 0.848 / 0.737 | 0.761 / 0.882 |
| `cnn_coarse_grid` | 1.000 / 0.289 | 1.000 / 0.579 | 0.985 / 0.855 | 0.841 / 0.974 |
| `cnn_fusion` | 1.000 / 0.289 | 1.000 / 0.579 | 1.000 / 0.868 | 0.852 / 0.987 |

测试集 unsafe 总数 76 / 110（69.1%），无筛查基线 UAR = 0.691。

### 5.4 无筛选敏感性（`--min-nac-valid 0`，120 test Tile）

| 模型 | MAE g10 | Brier g10 |
|---|---|---|
| `constant` | 0.3637 | 0.2287 |
| `coarse_ridge` | 0.0518 | 0.0658 |
| `cnn_nac` | 0.3661 | 0.1940 |
| `cnn_coarse_grid` | 0.0519 | 0.0729 |
| `cnn_fusion` | 0.0561 | 0.0547 |

结论与 110-Tile 版本一致；被排除的 11 个极区 Tile 对结论无实质影响。

图：`outputs/baseline/figures.png`（标签分布、预测散点、UAR-coverage、逐组 MAE）。

## 6. 解释与限制

**可以得到支持的结论**

1. 独立的 60 m 坡度层信息量极大：MAE 0.0504、AUC 0.966，远优于常数基线（0.3554 / 0.500）。
2. 单目 NAC 单独使用时只有弱信号：AUC 0.763、标定后 Brier 0.175 优于常数 0.228，
   但 MAE 0.2943 远差于 60 m 基线；原始 Brier（未标定 0.2919）甚至差于常数基线。
3. 固定结构对照下，**没有证据表明单目 NAC 在 60 m 坡度之上提供额外可用信息**：
   `cnn_fusion` 与 `cnn_coarse_grid` 的 MAE 差为 −0.009（融合更差），Brier 差为 0.019（融合更好），
   95% CI 完全重叠。若要主张 NAC 增益，需要更大样本与多 seed，且不能只比较 point estimate。
4. 该代理规则在本队列上先验 69.1%，使得 UAR–Coverage 几乎不具判别力：常数模型在 20% coverage 下
   UAR 已达 0.818。把 UAR 当作主指标前必须先固定一个先验更均衡的规则。

**必须显式声明的限制**

- 训练集仅 399 个 Tile，输入 128×128，未使用 LFM 特征、未使用观测几何（incidence/emission/phase/方位角），
  未做光度归一化；NAC 的亮度本身受光照强烈混淆，这是弱结果最可能的来源之一。
- 每个变体单一 seed；本机 MIOpen 存在轻微运行间不确定性（重复运行 MAE 变动约 0.007），
  因此"融合 vs 仅粗坡度"的小差异不能作为结论。
- 测试组只有 25 个，自助 CI 较宽（Brier CI 宽度约 ±0.035）。
- 所有行仍标记 `GEO_JOIN_PENDING`：本轮实现的是坐标驱动的重采样，尚未做基于原始 footprint 的投影/裁切审计，
  也未验证极区投影畸变。
- 规则 `g(10°) > 0.05` 是工程代理，不是认证着陆安全；g15 规则同样只是第二条预先固定的报备规则。
- 未使用 crater 工程集，未使用 4×4090，未产生云支出。

## 7. 结论与下一步 — **pre-fix，见 §0.6 的修订**

> ⚠️ 本节的判定基于 pre-fix 数值。post-fix（§0）修订为：**GO small scale-up（有条件）**——
> 风险曲线已修正、双事件头与切分独立性通过，单目 NAC 对事件概率有边际但可分辨的增益，
> 但 60 m 坡度层仍是主要信息源，且 NAC 对点估计 MAE 无改善。

**判定：主假设在当前规模下未获支持（NAC-only 弱于独立 60 m 基线，融合无显著增益）。**

建议的最小后续动作（按优先级）：

1. **重新预注册规则**：在**未接触 test 集**的前提下，选取先验接近 30%–50% 的事件阈值
   （需在 train/cal 上固定并在报告中留痕），否则 UAR–Coverage 没有信息量。
2. **加入观测几何与光度归一化**：manifest 已含 `INCIDENCE_ANGLE`、`EMISSION_ANGLE`、`PHASE_ANGLE`、
   `SUB_SOLAR_*`，可作为输入通道或分区变量；这是"单目影像 → 地形"最常见的信息瓶颈。
3. **扩大样本与多 seed**：先补 train 到 3–5k Tile（S3 仍有 90,481 个 slope_3m Tile），
   再报告多 seed 均值与组级 CI。
4. **只在通过上述步骤后再考虑 LFM 特征**；当前不启动四卡、不租云。

## 8. 复现

```bash
# 单元测试（81）
python3 -m unittest discover -s tests -v

# 下载 + 物理标签（760 Tile，可 --resume 续跑）
python3 scripts/hydrate_tiles.py --train 400 --cal 120 --val 120 --test 120 --workers 12 --resume
# 在已下载数据上补几何 + DTM 多属性（粗糙度 / 高程极差），无需重新下载
python3 scripts/hydrate_tiles.py --enrich

# manifest（含观测几何）与修复后的切分（含 buffer_contract）
python3 scripts/build_manifest.py
python3 scripts/finalize_split.py

# P1.6 主实验（v3，3 seeds，4 条预注册规则、3 通道 + 几何标量）
for s in 42 43 44; do
  extra=""; [ "$s" = "42" ] && extra="--force-cache"
  start=$(date +%s)
  python3 scripts/train_baseline.py --seed $s --out outputs/baseline_v3_seed$s $extra
  echo $(( $(date +%s) - start )) > outputs/baseline_v3_seed$s/elapsed_sec.txt
done
python3 scripts/aggregate_seeds.py --out outputs/baseline_v3_summary
python3 scripts/plot_baseline.py --baseline outputs/baseline_v3_seed42 \
  --out outputs/baseline_v3_summary/figures_seed42.png
```

产物：`outputs/baseline_v3_seed{42,43,44}/{metrics.json,test_predictions.csv,risk_coverage.csv}`、
`outputs/baseline_v3_summary/{summary.json,summary.md,figures_seed42.png}`、
`outputs/{manifest_100k_final.parquet,split_summary.json,labels_hydrated.csv,cache_v3.npz}`。
（更早的 v2/pre-fix 产物：`outputs/baseline_fixed_seed*`、`outputs/baseline_alltiles`、`outputs/baseline`，仅作对照。）
