# P0-04 物理审计 — 3 m slope 标签

日期：2026-10-17
输入：真实 SomBench NetCDF（HF 6 Tile + S3 100k 抽样的 12 Tile）
脚本：`scripts/make_slope_labels.py`；边界单测：`tests/test_slope_labels.py`（17/17 PASS，synthetic inputs）。

## 1. 网格与变量

| 项目 | 实测 |
|---|---|
| slope 数据变量 | `band_data`（每个文件唯一 2-D 变量） |
| 形状 | 171×171 |
| dtype | float32 |
| 值域（样例） | 0.038–43.807 |
| 有效像素 | 6/6 与 12/12 文件均无 NaN；valid_fraction = 1.0 |
| 负坡度 / >90° | 0 / 0 |
| GSD | 2.994152 m（`pix_res_x`；`pix_res_y` 为负，仅方向） |
| NAC 网格 | 512×512，1.0 m/px |
| CRS | `PROJCS["Moon (2015) - Sphere / Ocentric / Tranverse Mercator", ... central_meridian=-8, scale_factor=0.999, false_easting=250000, false_northing=2500000, UNIT=metre]` |
| NAC vs slope 边界 | dx=dy=0.997 m（HF 6/6）、1.000 m（S3 12/12） |
| 文件内 `units` 属性 | **缺失**（HF 与 S3 均如此） |

## 2. 单位验证（不是猜测）

官方 README 明确写 `slope_3m` 单位为 `degrees`、取值范围 `[0°, 90°]`。此外对 S3 12 个 Tile 做了独立经验验证：用同一 `dtm_3m` 网格和 GSD（2.994 m）重算

$$\text{slope}_{est}=\arctan\left(\sqrt{(\partial z/\partial x)^2+(\partial z/\partial y)^2}\right)\cdot\frac{180}{\pi}$$

| 检查 | 12 Tile 结果 |
|---|---|
| Pearson(slope_est, slope_file) | 0.9850–1.0000 |
| 均值比 slope_file / slope_est | 0.9947–1.0245 |
| 中位绝对差 | 0.011°–0.224° |

若文件单位是弧度，均值比应约为 57.3，实测约为 1.0；因此 **slope_3m 为 degrees，且与 `dtm_3m` 同网格同 GSD**。这同时构成网格配准的强证据（12/12）。

## 3. 标签定义（预先固定）

有效像素集合 V：有限值且 `0° ≤ slope ≤ 90°`；负值与 >90° 计为无效并单独记录。

$$g(\theta)=\frac{\#\{s\in V: s>\theta\}}{|V|}$$

- 阈值比较为**严格大于**：恰好 10.0° 不计入 g(10°)。
- 工程代理规则：`unsafe = 1[g(10°) > 0.05]`；恰好 0.05 判为安全（有单测覆盖）。
- 不做零值平滑、不对数比变换、不改伪计数。
- 11 个桶（10 个阈值 + 端点）：`bin_0=[0,1)`，`bin_1=[1,2)`，…，`bin_9=[40,60)`，`bin_10=[60,∞)`；桶使用左闭右开，因此 10.0° 属于 `bin_4=[10,15)`，但不计入 `g(10°)`。单测覆盖该差异。

## 4. HF 6 Tile 标签（真实测量）

| sample | split | valid | min° | max° | mean° | g(10°) | g(15°) | unsafe proxy |
|---|---|---|---|---|---|---|---|---|
| M1096572724R_r10462_c4900 | train | 29,241 | 0.039 | 21.947 | 2.286 | 0.0080 | 0.0024 | 0 |
| M1151623432_r19094_c6308 | train | 29,241 | 0.198 | 20.586 | 6.902 | 0.1804 | 0.0098 | 1 |
| M1160454221_r18070_c4922 | train | 29,241 | 0.038 | 13.721 | 2.274 | 0.0101 | 0.0000 | 0 |
| M1407958183_r4896_c1150 | train | 29,241 | 0.919 | 43.807 | 10.257 | 0.5090 | 0.0574 | 1 |
| M1471423196_r37214_c1924 | train | 29,241 | 0.048 | 37.848 | 3.520 | 0.0418 | 0.0210 | 0 |
| M189321511_r15560_c2424 | test | 29,241 | 0.145 | 21.408 | 5.629 | 0.0960 | 0.0054 | 1 |

6 个 Tile 的 g(10°) 覆盖 0.008–0.509，说明即使样例很小，标签也不是常数。

## 5. S3 12 Tile 标签（真实测量，12 个独立 PRODUCT / 12 个 zone）

| sample | split | g(10°) | g(15°) | unsafe proxy |
|---|---|---|---|---|
| M1112885551_r42696_c10644 | train | 0.4246 | 0.2855 | 1 |
| M1119207667_r18620_c18312 | train | 0.0177 | 0.0014 | 0 |
| M1123761564R_r15548_c10212 | val | 0.5733 | 0.0538 | 1 |
| M1141472655L_r31282_c2936 | train | 0.1007 | 0.0170 | 1 |
| M1148903478_r14998_c2986 | val | 0.0294 | 0.0110 | 0 |
| M1154293236L_r36108_c9950 | train | 0.0046 | 0.0000 | 0 |
| M1163402537_r18732_c4110 | test | 0.4384 | 0.1035 | 1 |
| M1192255206R_r1074_c2198 | train | 0.0950 | 0.0188 | 1 |
| M1241972774_r24814_c9570 | test | 0.4438 | 0.0506 | 1 |
| M1258981382_r49302_c9320 | train | 0.1646 | 0.0151 | 1 |
| M188071231R_r8754_c1274 | val | 0.0250 | 0.0082 | 0 |
| M189501657_r70244_c13276 | test | 0.9975 | 0.9604 | 1 |

g(10°) 覆盖 0.005–0.998，跨 12 个不同地理 zone；这是可用的真实监督分布证据。

## 6. 缺口与限制

- **GEO_JOIN_PENDING**：当前只验证了 NAC 与 3 m 网格的 CRS/边界一致，以及 slope 与 DTM 的逐像素一致性；尚未在代码中实现基于原始 footprint 的 NAC crop、重采样与裁切。`outputs/slope_labels*.csv` 不能直接当作 NAC→坡度监督对。
- 文件没有 `units` 属性；单位 provenance 来自官方 README + DTM 重算（12/12），P1 manifest 必须显式记录。
- 本批 18 个文件 valid_fraction 均为 1.0；极区/允许 NaN 的 Tile 尚未覆盖，脚本已处理 NaN、负值、>90° 并输出计数。
- `unsafe = g(10°) > 0.05` 只是预先固定的工程代理，不是认证的着陆安全标签。
- 未做地形平滑；所有计数直接来自原生网格。

## 7. 复现命令

```bash
# 17 个边界/无效值/变量选择单测（合成输入）
python3 -m unittest discover -s tests -v

# HF 6 Tile 标签
python3 scripts/make_slope_labels.py data/sample/NAC_HighRes/slope_3m \
  --units degrees \
  --units-source "SomBench README table: slope_3m unit=degrees, range [0,90]"

# S3 12 Tile 标签
python3 scripts/make_slope_labels.py data/s3/sample/slope_3m \
  --units degrees \
  --units-source "SomBench README tables: slope_3m unit=degrees, range [0,90]" \
  --out outputs/slope_labels_s3.csv --summary outputs/slope_labels_s3_summary.json
```

产物：`outputs/slope_labels.csv`、`outputs/slope_labels_summary.json`、`outputs/slope_labels_s3.csv`、`outputs/slope_labels_s3_summary.json`。
