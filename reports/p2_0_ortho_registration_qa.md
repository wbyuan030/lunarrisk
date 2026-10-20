# P2.0 P20-02：独立配准与辐射 QA

**状态：BLOCKED（不能宣称达到 1-pixel 配准）** · 2026-10-20 · 场景 `FRSHCRATER10 / M153639872`
输入沿用 P20-01 产物：`ortho_L.raw` / `ortho_R.raw`（14259×2389，2 m）与
`NAC_DTM_FRSHCRATER10.TIF`。新脚本 `scripts/qa_registration_p20.py`。

## 1. 投影/几何一致性核对（米为单位）

| 项 | DTM | 正射 |
|---|---|---|
| 投影 | EQUIRECTANGULAR（label `MAP_PROJECTION_TYPE`） | `equirectangular`（`maptemplate`） |
| 中心 | lat 48.0 / lon 180.0（label `CENTER_LATITUDE/LONGITUDE`） | 同 |
| 采样 | `MAP_RESOLUTION = 15161.67521207 pix/deg` | 同（`resopt=ppd`） |
| 天体半径 | `A_AXIS_RADIUS = 1737.4 km` | `eqradius=polradius=1737.4` |
| 网格 | 14259 × 2389 | 14259 × 2389（`getkey` 实测） |
| 纬度类型 | 行星中心（`KEYWORD_LATITUDE_TYPE`） | ISIS `camrange` 输出为 Planetocentric |

系数核对：2 m 后间距 = 1737400·π/180/15161.675 = 2.0000 m ✓；
经度方向 1 个 sample 在 lat 48.0 的地面宽 = 2.0·cos(48°) = 1.338 m（**配准残差报告用欧氏
sample 距离 × 2 m，属于保守上界**）。
`pixel-is-area`：两者均按像元面积栅格处理；DTM GeoTIFF 的 `ModelPixelScale=(2,2)`。

## 2. 特征点配准残差（独立地物，非强度互相关）

方法（`scripts/qa_registration_p20.py`）：在共享有效带内分别取正射 I/F 与 DTM 地形/hillshade
的边缘响应前 1% 点，按 12×12 像元分块配对（每块一次），把**正射相对 DTM 的偏移**以米记录；
用 ±1 sample 的受控位移做对照。DMS 分块配对不使用任何描述子阈值，只反映几何不一致。

| 特征对 | 匹配数 | 象限分布 | mean | RMS | p50 | p95 | max |
|---|---|---|---|---|---|---|---|
| relief（DTM 地形边缘） | 562 | Q3 417 / Q4 145 | 6.99 m | 7.71 m | 7.07 m | 12.20 m | 14.14 m |
| hillshade（DTM 山体阴影） | 653 | Q3 416 / Q4 236 | 7.76 m | 8.45 m | 8.25 m | 12.81 m | 14.14 m |

**受控位移对照（shift = 0 → 1 sample）：**

| 特征对 | shift 0 的 RMS | shift 1 sample 的 RMS | 差 |
|---|---|---|---|
| relief | 7.651 m | 7.712 m | +0.061 m |
| hillshade | 8.354 m | 8.450 m | +0.096 m |

**结论：该指标无法分辨 1 个像元（2 m）的系统性偏移**——残差 p50 ≈ 7.07 m（≈1.5 个像元的
欧氏距离）主要由特征真实错位与边缘响应不对称贡献，且受控位移只让 RMS 增加 6–10 cm。
因此**不能宣称配准精度达 1 pixel**。按任务要求记为 BLOCKED，不能只报 `campt` 一致。

象限分布也印证几何现实：左相机只覆盖 DTM 西经 243.775–243.924，Q1/Q2（东侧）本无数据。

## 3. 辐射统计（SALE 保留 I/F）

| 相机 | 填充哨兵（isis2raw） | 有效比 | p1 | p50 | p99 | max | 近饱和(≥0.3) | 暗(≤0.02) |
|---|---|---|---|---|---|---|---|---|
| L | 0.0132450（55.68%） | 0.4432 | 0.03067 | 0.06720 | 0.13016 | 0.14493 | 0.00000 | 0.00346 |
| R | 0.0321360（50.66%） | 0.4934 | 0.03632 | 0.07163 | 0.14508 | 0.15702 | 0.00000 | 0.00000 |

`SCALING_FACTOR = 3.05185094759972e-05`（CDR 标签）；`NULL = -32768`；两相机均无饱和像元，
暗像元比例 ≤0.35%。左右重叠区残差（P20-01 已报）：mean −0.001397、std 0.007634、p95 +0.008637。

## 4. 地形真值泄漏检查

**本链条未使用 DTM 逐像素高程**：`cam2map map=dtm_map.cub` 只用了 `maptemplate` 产出的
投影网格（投影类型/中心/采样/范围），DTM 仅作为输出网格的定义与 QA 基准，**没有**把 DTM
当作正射的输入 DEM。已确认：`maptemplate` 不含高程；`cam2map` 的 `map=` 是地图投影文件。
→ 记：*DTM used for map-projection geometry only; DTM elevations never entered the ortho.*

## 5. 需要提交版本化协议才能继续的部分

- 任务书的 1-pixel 门槛无法用当前不依赖调参的特征匹配达到；要么
  (a) 引入真正的控制点/精确配准（人工/半自动特征，跨 ROI ≥20 点），或
  (b) 提交“tile-level 覆盖率门槛 + 残差上限”的协议修订并经批准。
- 本报告未做：星星点源/阴影配准、不同光照下的线性亮度匹配、拼接 TIFF 落盘、多景复现（P20-03）。

## 结论

- 覆盖（P20-01）达标；辐射一致性良好；**投影语义与网格一致；无地形泄漏**。
- **配准 QA：BLOCKED**（现有指标无法把 2 m 的系统偏移从 ~7 m 的边缘定位噪声中分离）。
- Gate 3 继续 **NOT FROZEN**；Gate 4 **HOLD**。
