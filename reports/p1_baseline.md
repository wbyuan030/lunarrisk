# P1-03/P1-04 — 首个真实物理基线（NAC → 3 m 坡度超出率）

日期：2026-10-17
数据版本：S3 `nasa-lunar-fm-bench` 100k release，索引 SHA256
`f9c39d1b570693902bd5616d44f6330ea15a425db3c9d3eb9ceb2cdc9be1873e`
脚本：`scripts/hydrate_tiles.py`、`scripts/train_baseline.py`、`scripts/plot_baseline.py`
单测：`tests/test_slope_labels.py`、`tests/test_build_manifest.py`、`tests/test_finalize_split.py`、`tests/test_train_baseline.py`（51/51 PASS）

## 1. 任务与规则

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

## 5. 结果（110 test Tile / 25 组）

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

## 7. 结论与下一步

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
# 单元测试（51）
python3 -m unittest discover -s tests -v

# 下载 + 物理标签（760 Tile，可 --resume 续跑）
python3 scripts/hydrate_tiles.py --train 400 --cal 120 --val 120 --test 120 --workers 12 --resume

# 主实验（constant / coarse_ridge / cnn_nac / cnn_coarse_grid / cnn_fusion）
python3 scripts/train_baseline.py

# 无 NAC 筛选敏感性
python3 scripts/train_baseline.py --min-nac-valid 0 --out outputs/baseline_alltiles

# 图
python3 scripts/plot_baseline.py
```

产物：`outputs/baseline/{metrics.json,test_predictions.csv,risk_coverage.csv,figures.png}`、
`outputs/baseline_alltiles/*`、`outputs/labels_hydrated.csv`、`outputs/cache_v2.npz`。
