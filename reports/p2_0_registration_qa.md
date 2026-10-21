# P2.0 P20-02：正射配准与网格一致性实测

**状态：网格/投影算术 PASS（40/40 精确匹配）；坑心 GCP 配准 ~1–2 px（2–5 m）PASS(2 px) / ≤1 px 未确立** · 2026-10-20
数据：`RANGER`（2 m DTM，14244×2315）、`PLANCKFLOOR`（2 m DTM，17556×3290）。两景均为 probe，不是冻结候选。

## 1. 结论摘要

| 检查 | 结果 | 证据 |
|---|---|---|
| DTM 格网 ↔ 正射像素算术一致性 | **PASS** | 40/40 精确匹配（最近邻 QA 立方体） |
| 太阳方向来自 SPICE 而非猜测 | **PASS** | RANGER az 100.0° / alt 42.6°；PLANCKFLOOR az 310.5° / alt 25.2° |
| 正射 vs DTM hillshade 相关（几何自检） | **不足** | RANGER 最佳相关 0.80（5/16 窗口），PLANCKFLOOR 最佳 0.23（0/16） |
| 左/右相机互检 | **不足** | RANGER 共同条带仅 85 px 宽，1 个窗口可用 |
| 坑心 GCP 配准（正射 vs DTM） | **PASS @2 px** | RANGER 主峰 (0,0) px / PLANCKFLOOR (1.4 px)；inlier p95 1.4 / 2.3 px |
| ≤1 px 严格结论 | **BLOCKED** | 左右相机互检存在 2–3 px 系统差，方法本身约 1 px 偏差 |

因此：**“正射与 DTM 是否同一格网”已用精确实验证明为是**；“正射内容与 DTM 地形是否落在同一位置”
用坑心 GCP 证明在 **1–2 px（2–5 m）** 内；**“≤1 px”仍不能声称**，因为左右相机之间本身就存在 2–3 px 系统差。

## 2. 网格一致性的精确实验（决定性）

思路：`cam2map` 若用最近邻重采样，则正射每个像素必然等于其映射到的 CDR 像素值（量化步长 3.05185e-05）。于是判据是**逐点相等/不等**，而不是模糊的相关系数。

流程（`scripts/grid_consistency_p20.py --direction ortho-to-camera`）：
1. 随机取正射像素中心 `(i+0.5, j+0.5)` → 用正射自身 Mapping 组反算 (lat, lon)；
2. `campt type=ground lat= lon=` → 相机 `(line, sample)`（往返实测精度 <1e-3 px）；
3. 取最近邻 CDR 像素值，与正射该像素值比较。

**关键发现**：默认 `cam2map` 用 `INTERP = CUBICCONVOLUTION`（见正射 History），因此普通正射的像素值是邻域卷积混合，**不能**用相等判据。专为 QA 重建 `L_nearest.cub`（`interp=nearest`，13 min）：

| 场景 | 点 | 精确匹配率 | 说明 |
|---|---|---|---|
| RANGER / `M144483945LC` | 40 | **1.000** | 包含 y=39、y=13319 边界点与 x=1296–2247 全宽 |

即：**正射格网与 DTM 格网（GeoTransform、CRS、像元中心、像素尺寸）逐像元一致**，无半像素或整像素系统偏移。该结论不依赖任何影像相似度。

## 3. hillshade 自检为什么不足（量化）

用 SPICE 提供的真实太阳方向（见第 4 节）渲染 DTM hillshade，与正射做 NCC，窗口按地形起伏排序，`--smooth-sigma 12`（24 m）：

| 场景/相机 | 最佳相关 | 可用窗口 (peak≥0.5) | 位移 p50 / p95 | ≤1 px 比例 |
|---|---|---|---|---|
| RANGER left | **0.802** | 5 / 16 | 64.2 m / 139.0 m | 0.20 |
| RANGER right | 0.268 | 0 / 16 | — | — |
| PLANCKFLOOR left | 0.229 | 0 / 16 | — | — |
| PLANCKFLOOR right | 0.203 | 0 / 16 | — | — |

失败原因（已实测排除其他解释）：
- 2 m NAC DTM 的**高频粗糙度**远高于光学 PSF：σ=0 时同一窗口 r=0.39，σ=12 px 时 0.65 → 必须多尺度平滑，但平滑后精度上限也只有 24 m；
- 影像以**反照率**为主（RANGER 有巨大亮斑/暗斑），与地形无关；
- 低照度场景（RANGER I/F p50 0.026）与 25° 太阳高度（PLANCKFLOOR）进一步压低对比度；
- 左右相机共同条带在 RANGER 上只有 **85 px 宽**（整帧 2315 px），互检样本极少。

**结论**：hillshade 相关性在本项目两景上无法支撑 1-px 结论，也不能反过来证明误差。它只能作为“粗配准是否明显错位”的筛查。

## 4. 太阳方向：从 SPICE 推导，而非搜索

`scripts/solar_azimuth_p20.py` 用 `campt` 的 `SubSolarLatitude/SubSolarLongitude` 做球面三角：

```
az = atan2( sin(Δlon)·cos(φs), cos(φ)·sin(φs) − sin(φ)·cos(φs)·cos(Δlon) )
elevation = 90° − incidence
```

| 场景 | 太阳方位 | 太阳高度 | 帧内方位离散度 | 与逐窗搜索对比 |
|---|---|---|---|---|
| RANGER | **100.01°** | 42.64° | 0.70° | 搜索给出 105°（步长 15°）→ 一致 |
| PLANCKFLOOR | **310.46°** | 25.23° | 0.18° | — |

这同时交叉验证了 SPICE 链（CK/SPK/FK/IK）与地表投影的一致性：两条完全独立的路径给出同一太阳方向。

## 5. 左/右相机互检

RANGER 的 `M144483945LC` 与 `M144483945RC` 共同条带仅 **85 px 宽**（覆盖 413 440 px），PLANCKFLOOR 的两 acquisition 左右并集重叠仅 **8.17%**。
因此“左/右互检”不能作为独立配准证据；跨 acquisition 的并集重叠（RANGER 32.98%）可用于粗查，但不构成 GCP 级验证。

## 5c. 坑心 GCP 匹配（决定性一步）

`scripts/crater_gcp_p20.py`：尺度归一化 LoG 检测圆形斑（双极性）→ 互最近邻匹配（KD-tree）→ 位移主峰聚类。

**关键设计**：早期版本在 hillshade 上检测坑，只能匹配 61 个、主峰仅 4 个内点 —— 因为阴影把坑沿切成
一亮一暗两半。改为 `--dtm-mode dem` 直接在**高程**上找“碗形”，与光照模型无关，匹配数立刻升到 160 / 81。

| 对比 | 匹配数 | 位移主峰 | 内点(占比) | 内点 p50 / p95 | 折合 |
|---|---|---|---|---|---|
| RANGER 正射 vs DTM 高程 | 160 | **(0, 0) px** | 56 (0.35) | 0.0 / 1.41 px | 0–2.8 m |
| PLANCKFLOOR 正射 vs DTM 高程 | 81 | (1.4, 1.4) px | 19 (0.23) | 1.41 / 2.30 px | 2.8–4.6 m |
| RANGER 左 vs 右相机（自检） | 260 | (−2, 0) px | 109 (0.42) | 2.0 / 3.16 px | 4–6.3 m |
| PLANCKFLOOR 左 vs 右相机（自检） | 664 | (3, 0) px | 292 (0.44) | 3.0 / 3.16 px | 6–6.3 m |

解读：
- 正射与 DTM 的坑心**主峰就在 (0,0)**（RANGER），说明投影位置正确、无系统偏移；
- 左/右相机之间有 **2–3 px（4–6 m）系统差**：这属于相机指向/几何定位残差（左右各自独立 SPICE），
  不是格网问题，也正好解释了第 3 节 hillshade 相关得到的“位移 p50 64 m”中相当一部分噪声；
- GCP 方法自身的质心偏差约 1 px（阴影/反照率不对称），故 1 px 内结论无法用本方法给出。

## 6. 仍未解决与下一步（BLOCKED 的具体含义）

要给出“≤1 px 绝对配准”，剩余工作：
- 用**圆形度筛选 + 加权质心**降低 GCP 质心偏差（当前约 1 px）；
- 用多景（≥3 acquisition）做**相对定向平差**，定量给出左右相机 2–3 px 系统差的来源（
  是指向、还是 DTM 自身质量）；
- 若许可，与 LOLA/已发布 GCP 做绝对比对。

在这些完成前，报告不得写“已达亚像素/1 px 配准”；可以写“与 DTM 在地形层面一致到 1–2 px（2–5 m）”。

## 7. 复现命令

```bash
# 太阳方向（容器内，需 ISIS）
python3 scripts/solar_azimuth_p20.py --cube M144483945LC_L.cub

# 网格算术精确实验（先建 nearest QA 立方体）
cam2map from=M144483945LC_L.cub map=dtm_map.cub to=L_nearest.cub \
        pixres=map defaultrange=map interp=nearest
isis2raw from=L_nearest.cub to=L_nearest.raw
isis2raw from=M144483945LC_L.cub to=M144483945LC_L.raw
python3 scripts/grid_consistency_p20.py --direction ortho-to-camera \
  --camera-cube M144483945LC_L.cub --camera-raw M144483945LC_L.raw \
  --ortho L_nearest.cub --ortho-raw L_nearest.raw --points 40

# hillshade 自检
python3 scripts/ortho_registration_p20.py --dir data/pilot_p20/RANGER \
  --dtm data/pilot_p20/RANGER/NAC_DTM_RANGER.TIF --lines 14244 --samples 2315 \
  --azimuth 100 --altitude 42.6 --limit 16 --min-peak 0.5 \
  --out outputs/p20_ranger_registration.json
```

输出：`outputs/p20_ranger_grid_nearest.json`、`outputs/p20_ranger_registration.json`、
`outputs/p20_planckfloor_registration.json`、`outputs/p20_ranger_solar.json`。
