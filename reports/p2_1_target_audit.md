# P2.1 A0 — 目标标签审计（自动生成）

`labels_sha256 = eb2ff9326cd3d3180409355b53b2cf135c02ddd324cec87e583686abccdaafb5`

行数 760；split 分布 train=400, val=120, cal=120, test=120

## 四个目标的可用数与分布

| 目标 | 非空 | 缺失 | min | p50 | p99 | max | 缺失率 |
|---|---|---|---|---|---|---|---|
| `slope3m_g10` | 760 | 0 | 0 | 0.089737 | 1 | 1 | 0.000% |
| `slope3m_g15` | 760 | 0 | 0 | 0.0095756 | 0.999897 | 1 | 0.000% |
| `dtm3m_roughness_9m_m` | 760 | 0 | 0.00498623 | 0.0167501 | 0.116398 | 0.338987 | 0.000% |
| `dtm3m_relief_p95_p5_m` | 760 | 0 | 2.88501 | 31.6096 | 277.351 | 324.558 | 0.000% |

### 分 split 非空数

| 目标 | train | val | cal | test |
|---|---|---|---|---|
| `slope3m_g10` | 400 | 120 | 120 | 120 |
| `slope3m_g15` | 400 | 120 | 120 | 120 |
| `dtm3m_roughness_9m_m` | 400 | 120 | 120 | 120 |
| `dtm3m_relief_p95_p5_m` | 400 | 120 | 120 | 120 |

## 物理一致性

无违规：`0 ≤ g15 ≤ g10 ≤ 1` 与 roughness/relief 非负全部成立，且无 NaN 混入。

## split 完整性

- sample_id 唯一：True
- 组数 272，来源数 336
- 每 split 组数：train=195, val=26, cal=26, test=25
- 跨 split 的组：0
- 跨 split 的来源：0

## 单位与来源

- slope3m_variable: ['band_data']
- units_source: ['SomBench README: slope_3m/slope_60m degrees, range [0,90]']
- units_verified: ['True'] （760 行显式为真）

## 输入泄漏检查

`dtm3m_roughness_9m_m`、`dtm3m_relief_p95_p5_m`、`slope3m_g10`、`slope3m_g15` 均未出现在 `train_baseline.py` 声明的输入列表中。

## 人工解读（必须在本轮建模前读）

### 1. 数据完整性：干净

- 四个目标在 760 行上**全部非空**（缺失率 0），因此本轮不需要 target mask 分支；但代码仍必须
  用显式 mask 而不是"缺失填 0"——本轮 `scripts/audit_targets_p21.py` 的 `to_float()` 就是这样实现的，
  并有测试 `test_missing_targets_are_counted_per_split` 断言缺失不会被当成 0。
- 物理约束 `0 ≤ g15 ≤ g10 ≤ 1`、roughness/relief 非负、无 NaN：**0 项违规**。
- 单位：`slope3m_variable = band_data`，`units_source = "SomBench README: slope_3m/slope_60m
  degrees, range [0,90]"`，760/760 行 `units_verified = True`。即坡度为**角度制**，与 `g10/g15`
  的阈值语义一致。
- 输入泄漏：`dtm3m_roughness_9m_m`、`dtm3m_relief_p95_p5_m`、`slope3m_g10`、`slope3m_g15`
  均**不在** `train_baseline.py` 声明的输入列表（`COARSE_FEATURES`/`CHANNELS`/`GEOMETRY_SCALARS`/
  `NAC_SCALARS`）中；模型输入只有 NAC 两通道、60 m 粗坡度通道与视角标量。

### 2. split 之间有明显分布偏移（本轮最重要的约束）

| split | n | 组数 | mean g10 | P(g10>0.05) | P(g15>0.02) | roughness 均值 | relief 均值 (m) |
|---|---|---|---|---|---|---|---|
| train | 400 | 195 | 0.252 | **0.568** | 0.372 | 0.0204 | 52.5 |
| val | 120 | 26 | 0.152 | **0.350** | 0.283 | 0.0200 | 32.1 |
| cal | 120 | 26 | 0.357 | **0.708** | 0.500 | 0.0218 | 69.3 |
| test | 120 | 25 | 0.432 | **0.692** | 0.492 | 0.0344 | 82.9 |

含义与硬约束：

- **val 与 train/cal/test 不同分布**（危险率 0.35 vs 0.57/0.71/0.69）。val 上选出的规则阈值或
  校准参数会偏向"更安全"的子集，不能直接移植。
- **test 明显更陡、更粗糙**：roughness 均值 0.0344（train 0.0204），relief 82.9 m（train 52.5 m）。
  这与"旧 110 Tile Test 已被反复查看"叠加，意味着 test 上的绝对数值只能作**探索性**报告。
- 因此本轮所有预处理统计（均值/方差/目标变换/后验超参）**只能**在 train（或 train+val 且明确声明）
  上拟合；`cal` 仅用于校准映射与覆盖率阈值；`test` 不参与任何选择。
- 报告 UAR/HMR 时必须同时给出**实际放行数量与危险数量**，并且分 split 报告，不能只给合并数字。
- 组规模差异大（train 每 split 195 组 / 400 样本 ≈ 2.05 样本/组，val 26 组 / 120 ≈ 4.6 样本/组），
  组内配对 Bootstrap 要按组重采样，不能按样本重采样。

### 3. 本轮未做（明确）

- 未检查 `group_id` 的构成是否与 10 km 地理组一致（P1.8 的 `block_5km` 口径），也未与 P2.1 的 v2
  组池对齐；这会影响"独立组数"口径，后续需单独审计。
- 未检查四个目标在不同组之间的 ICC/组内相关，无法据此判断组级随机效应强度。
- 自动生成段落中的 per-split 分位数（p50/p99）已给出，本文不重复。
