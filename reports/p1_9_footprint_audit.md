# P1.9 Gate 1 — 外部 LROC NAC DTM 候选的真实 footprint 与来源独立性审计

**状态：PASS（索引层）** · 日期 2026-10-17 · 代码 `main@fa0719a6cdf5b64caca3add487dec9db41cce63b`
脚本 `scripts/audit_lroc_footprints.py`，离线测试 `tests/test_audit_lroc_footprints.py`（23 项）
产物：

| 产物 | sha256 |
|---|---|
| `outputs/p19_lroc_footprint_audit.csv`（181 行，逐景） | `de88739797418a825b7a08e0f6f40a32e8ae7d08604ce05455fdbabcd1c1d620` |
| `outputs/p19_lroc_footprint_audit.json` | `a9f58e9aa4cce27d15296f0c3a144cf4468ef6a3702d599ce77d658af63739c6` |
| `outputs/p19_lroc_footprint_pass.csv`（128 行，两项 PASS 的派生清单，非冻结协议） | `39d2088d551f7e32eaf6010e41b0d5e7bfc304bc3ad418d136316dc2794e259e` |
| PDS README 缓存 `data/lroc_nac_dtm/readmes/*.txt`（181 文件） | 聚合 `ff6eed58a2fee3f741382dddb98125555ea788e50fdc4809b96763c98ac1df74` |

复现（容器 `rocm-lab`，`/data/lunarrisk`，需要出网代理访问 PDS）：

```bash
python3 scripts/audit_lroc_footprints.py          # 冷启动 173 s（抓 181 个 README），热缓存 11 s
python3 -m pytest tests/test_audit_lroc_footprints.py -q
```

## 结论

在 181 个「源产品与 SomBench 全不相交」的候选景中：

- **来源独立性**：174 景由官方 PDS README 的 `LROC NAC SOURCE IMAGES` 节确认源清单，且与 SomBench 832 产品交集为 0；7 景为 mosaic，README 该节为空 → **fail-closed 排除**。
- **几何**：180 景多边形有效（1 景 `NOBILE04` 在极区自相交 → 排除）。
- **空间独立性**：以官方 shapefile 多边形对 824 个已用/受保护 Tile 的真实 footprint 做 polygon-to-polygon 最小距离，**128 景 ≥10 km 且来源 PASS**，它们分布在 **91 个彼此间距 ≥10 km 的景级地理组**（最大组 13 景）。
- 因此本次 Gate 1 结论为 `CANDIDATES_PASS_SOURCE_AND_FOOTPRINT_QA`：有一批可进入 Gate 2 探针与 Gate 3 分层抽样的严格候选。
  **这不等于 `GO_STRICT_OOD`**：尚未下载任何 DEM/NAC 载荷，实际有效数据范围、原生分辨率与标签尺度未验证。

## 方法

### 1. CRS 核对（含一处真实索引缺陷）

- `NAC_DTMS_360.PRJ` = `GEOGCS["GCS_Moon_2000", DATUM D_Moon_2000, SPHEROID Moon_2000_IAU_IAG 1737400.0]`，
  pyproj 解析：`is_geographic=True`，`radius=1737400.0 m` ✓（与月球半径一致），DTM 顶点为**度**。
- 经度域：DBF 顶点与中心为 **0–360**（如 `303.74`），审计统一归一化到 `[-180, 180)`，并用**圆均值**作为展开中心处理跨 ±180° 多边形；极区不做平面近似。
- **索引 BOUNDS 不可当经纬度用（新发现，已写入 manifest v2）**：
  S3 索引只提供**一个** `crs` 字符串（`Moon (2015) - Sphere / Ocentric / Transverse Mercator`），但 `BOUNDS_XMIN/YMIN…` 米值是**按 LTM 分带**的局部投影坐标。对 90 786 张 Tile 做验证：用该单一 CRS 把 `CENTER_LATITUDE/LONGITUDE` 正投影后与 `BOUNDS` 中心比较，**仅 321 张（0.35%）在 5 m 内一致**；残差中位数 |dx| 1.41e6 m、|dy| 2.71e6 m，p99 |dx| 3.88e6 m、|dy| 7.90e6 m。分区统计显示 `LTM_*` 各带互不一致，`LPS_N`/`LPS_S` 为极区立体投影。
  → 结论：旧 manifest 的 `footprint_wkt`（由 BOUNDS 生成）**不是地理坐标**，不得用于空间独立性判断。
  → 处置：`scripts/build_manifest.py` 现额外记录 `upper_left_lon/lat`、`lower_right_lon/lat`（索引自带的地理角点）与显式标志 `bounds_crs_status = PER_ZONE_LTM_UNVERIFIED`；产物 `outputs/manifest_100k_final_v2.parquet`（sha256 `b7484fcc4b94d4f37ae2657082aa7967daee8df787a260557da43c43647de320`）。旧 100k manifest 与其冻结引用未被覆盖。

### 2. 双方真实边界

- **已用/受保护 Tile（824 个 = 760 hydrated + 64 冻结 product-level V2）**：用索引地理角点构造地理 bbox，跨反经线时拆成两段。独立校验：824 张 Tile 的投影边长恰为 512.00 m × 512.00 m，其地理角点对角弦长 **0.7230–0.7284 km**，与 512√2 = 0.7241 km 的比值 p1/p50/p99 = 0.999/1.001/1.005、最大 1.006 → 角点可信，误差 ≤0.6%。
- **候选 DTM**：官方 shapefile 各 part 构环；**内环（含 island-in-lake 型多部件）按“填充”处理**——只保留最外层 shell，面积只增不减（实测多部件样例 `final_area == Σ外层环面积`），因此净空判定只会更保守；
  自相交 shell 不修复、不猜补，保持 `INVALID_POLYGON` 并排除。
- **距离**：以候选中心建局部方位等距投影（aeqd，R=1 737 400 m），在多边形顶点级别计算 polygon-to-polygon 最小距离；对跨 ±180°、极区、狭长多边形、缺失 CRS、非法 ring/multipart 均有离线用例。

### 3. 来源审计

- 逐景抓取 `https://pds.mcp.nasa.gov/.../NAC_DTM/<NAME>/NAC_DTM_<NAME>_README.TXT`（缓存到 `data/lroc_nac_dtm/readmes/`）。
- 解析 `LROC NAC SOURCE IMAGES` 段：接受 `M\d{9,10}` 加可选 `L/R/LE/RE` 后缀；要求 ≥1 条源；遇无法完整解析的 token 一律 `UNKNOWN_SOURCE`。
- 归一化后与 **SomBench 832 产品 + hydrated cohort（manifest 内全部 `NAC_EDR_ID`）** 求交集，非空即排除。

## 结果

### 逐景判定（181 景）

| 来源 | 几何 | 净空 | 景数 |
|---|---|---|---|
| README OK | 有效 | ≥10 km | **128（PASS）** |
| README OK | 有效 | <10 km | 45 |
| README OK | 无效 | <10 km | 1（`NOBILE04` 极区自相交） |
| README 空节（mosaic） | 有效 | ≥10 km | 6 |
| README 空节（mosaic） | 有效 | <10 km | 1 |

- README 空节的 7 景：`MRINGENII`、`NRTHCRTRI`、`FRESHMELT`、`VIRTANEN`、`SPARIM`、`SULPICIUS`、`MARIUS`。
- 46 个「来源 OK 但净空不足」的分布：18 景与已用 Tile **接触**（0 km）、15 景 <5 km、12 景落在 [5, 10) km —— 后者是典型的「边界情形」，不因接近阈值而放宽。
- 128 个 PASS 的最小净空 10.62 km（`NECTRIMII2`），p10 16.72 km，中位 93.60 km。

### PASS 候选的构成（可进入 Gate 2/3 的严格池）

| 维度 | 分布 |
|---|---|
| 原生分辨率 | 2 m ×76、3 m ×8、4 m ×3、5 m ×35、6 m ×4、7 m ×2；**≤3 m 共 84 景** |
| 纬度带 | (−90,−60] 2 · (−60,−30] 25 · (−30,0] 15 · (0,30] 51 · (30,60] 27 · (60,90] 8 |
| 景级地理组（<10 km 连通合并） | **91 组**，最大组 13 景，单景组 77（其中 41 组 ≤3 m） |
| 标称覆盖（DBF `cov_sqkm` 求和） | 44 819 km²（多边形外接面积 46 594 km²） |

规模最大的几组（组号 + 组内最优分辨率景）：

| 组 | 景数 | 代表景 | 分辨率 | 中心 (lat, lon) | 组内最小净空 |
|---|---|---|---|---|---|
| CDTM_0055 | 13 | `LICHTENBER2` | 2 m | 32.0, −67.0 | 248.7 km |
| CDTM_0063 | 4 | `MARIUS2` | 2 m | 14.0, −56.0 | 20.0 km |
| CDTM_0010 | 4 | `ARISTARCHU2` | 2 m | 25.0, −49.0 | 39.8 km |
| CDTM_0108 | 4 | `SPARIM2` | 2 m | −51.0, 171.0 | 185.7 km |
| CDTM_0043 | 4 | `HSCARP1` | 2 m | −51.0, −68.0 | 11.1 km |
| CDTM_0075 | 4 | `MRINGENII3` | 2 m | −35.0, 164.0 | 28.6 km |
| CDTM_0109 | 3 | `SULPICIUS1` | 2 m | 20.0, 10.0 | 137.9 km |
| CDTM_0050 | 3 | `KINGCRATER2` | 2 m | 6.6, 120.0 | 87.6 km |
| CDTM_0100 | 2 | `RIMPARRY2` | 5 m | −7.5, −17.1 | 98.9 km |
| CDTM_0068 | 2 | `MNDLSHTMLS` | 2 m | 5.9, 161.4 | 262.3 km |
| CDTM_0070 | 2 | `MOOREF2` | 2 m | 37.4, −174.8 | 311.4 km |
| CDTM_0031 | 2 | `FRESHMELT1` | 2 m | 16.0, 177.0 | 110.0 km |

同组内相邻（`min_candidate_distance_km = 0`）的典型例子：`RIMPARRY1/RIMPARRY2`、`LICHTENBER1/LICHTENBER2`、`MARIUS*`、`KINGCRATER*` 等——它们只能作为**一个**地理单位进入冻结集合。

## 已知局限（必须在 Gate 2/3 解决）

- **多边形是外接边界，不是有效数据范围**：`polygon_area_sqkm / cov_sqkm` 比值范围 0.38–26.28（中位 1.00）。例如 `NEWCRATER12` 多边形 763 km² 而 DBF 标称覆盖 29 km²。以更大边界做净空判定只会更保守（可能误排除），不会误纳入；实际可用范围必须由 Gate 2 的有效像素掩膜确定。
- 多部件 mosaic 普遍：212 个 part 的景存在；本审计对含洞/嵌套情形采取保守填充，可能高估边界。
- 本关**没有下载**任何 DEM/NAC 载荷，也没有验证 2–7 m 原生分辨率与现有 3 m 标签尺度的等价性；不得据此宣称已获得 3m 真值。
- 本关不构成 `GO_STRICT_OOD`：冻结协议与一次性评估属于 Gate 3/4。

## 测试

- `tests/test_audit_lroc_footprints.py`：23 项，覆盖经度归一化/反经线展开、圆均值、非法 ring/重复点/越界纬度、内环填充、自相交报告、`.prj` 接受与拒绝（非月球椭球必须失败）、分离/重叠/反经线/极区距离、空 footprint、地理角点与反经线拆分、README 段解析（空节/缺节/畸形/suffix/去重）、判定逻辑、候选分组、以及一次端到端（合成 shapefile + manifest + labels，断言仅产品独立景入选、净空阈值生效、CSV/JSON 结构）。
- 全仓测试：`python3 -m unittest discover -s tests` → **198 项全部通过**（`python3 -m compileall -q scripts tests` 通过）。
