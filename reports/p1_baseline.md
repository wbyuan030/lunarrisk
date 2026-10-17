# P1-03/P1-04 — 首个真实物理基线（NAC → 3 m 坡度超出率）

日期：2026-10-17
状态：**进行中**（本文件在真实训练完成后由脚本数值替换占位符）。

## 1. 任务与规则

- 目标属性：`y1 = g(10°)`、`y2 = g(15°)`，即原生 3 m slope 网格上的面积占比（原始计数，未平滑）。
- 预先固定规则：`unsafe = 1[y1 > 0.05]`（工程代理，非认证着陆安全）。
- 第二条独立报告规则：`y2 > 0.02`（阈值未参与任何调参）。
- 评估口径：`Coverage = accepted_n/test_n`；`UAR = unsafe_accepted_n/accepted_n`；
  `HMR = unsafe_accepted_n/unsafe_total`；事件 Brier/NLL；物理属性 MAE/RMSE。

## 2. 数据与切分

| 项 | 值 |
|---|---|
| 数据版本 | S3 100k release，索引 SHA256 `f9c39d1b570693902bd5616d44f6330ea15a425db3c9d3eb9ceb2cdc9be1873e` |
| 分组 | PRODUCT_ID × 5 km 投影块连通组 |
| split | train / cal / val / test，组级零重叠 |
| 缓冲 | 10 km 中心距，train 靠近 cal/val/test、cal/val 靠近评估集时丢弃 |
| 抽样 | 组内轮转抽样，seed 42 |
| 已下载 Tile | 760（train 400 / cal 120 / val 120 / test 120），NAC+DTM3m+SLOPE3m+SLOPE60m+metadata |
| 输入可用性 | 见 `outputs/baseline/metrics.json` 的 `nac_availability` |

## 3. 地理对齐（从 PENDING 升级为坐标重采样）

`align_nac_to_slope_grid()` 使用每个文件自带的 `x`/`y` 坐标向量，把 512×512 @1 m 的 NAC 用
`map_coordinates` 双线性重采样到 171×171 @2.994152 m 的 slope 网格上；不做“同名数组中心对齐”假设。

实测（`alignment_check`）：

- GSD：NAC 1.000 m，slope 2.994152 m（比值恰为 512/171）。
- 网格原点偏移：0.9971 m ≈ 0.9971 个 NAC 像素，跨 Tile 一致（系统性约定差，不是随机误差）。
- 重采样后 NAC 有效像素比例：多数 Tile = 1.0；极区 Tile 出现约 0.71（29% NaN）。
  低于阈值（默认 0.99）的 Tile 被排除，并单独报告被排除 Tile 的标签统计（避免隐藏选择偏差）。

## 4. 模型

| 模型 | 输入 | 训练集 | 说明 |
|---|---|---|---|
| `constant` | 无 | 同一可用 train 子集 | 预测 train 均值 / 先验 |
| `coarse_ridge` | SLDEM2015 60 m slope 统计（frac>10、frac>15、mean、q50、q90、q99、valid_fraction） | 同上 | Ridge 回归 + 逻辑回归事件概率 |
| `cnn` | 重采样后 NAC（128×128，逐 Tile 鲁棒归一化） | 同上 | 4 层 CNN，3 输出（g10、g15、事件 logit），flip/rot90 增强，按 val MAE 早停 |

三者使用**同一** train/test 子集，保证比较公平；`cal` 仅用于 CNN 事件概率的 Platt 标定。

## 5. 结果

（待填：真实数值。）

## 6. 限制

（待填。）

## 7. 复现

```bash
python3 -m unittest discover -s tests -v
python3 scripts/hydrate_tiles.py --train 400 --cal 120 --val 120 --test 120 --workers 12 --resume
python3 scripts/train_baseline.py
```
