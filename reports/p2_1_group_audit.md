# P2.1 B — 组级地理独立性：全 pair 多边形重算与 v2 池

**状态：v1 分组缺陷已证实并修复；v2 池为 41 组，最小跨组距离 40.05 km。两个协议版本均保持
`PENDING_SIGNOFF` / `NOT_FROZEN` / `HOLD`。** 2026-10-22

## 1. 缺陷（可复核）

`scripts/audit_groups_p20.py::cluster()` 只沿每个 DTM 的 `nearest_candidate_DTM` 一条边做 Union-Find，
不是 true single-linkage：最近邻是在更大的候选集上预先算的，过滤后边会丢失；一个景即使与另一景
多边形距离 <10 km，也未必是任何点的最近邻。后果在池里可见：`SPARIM1`/`SPARIM2` 中心同为
(−51°, 171°)、原始记录 `min_candidate_distance_km = 0`，却被分到两个不同的拟选 Group。

## 2. 重算方法

`scripts/audit_groups_pairwise.py`：

- 从官方 DTM Shapefile（`NAC_DTMS_360.SHP`，`GCS_Moon_2000`，球半径 1737 400 m）读取全部多边形，
  保留 multipart；
- 对 **全部 2 346 = 69×68/2 个 pair**，把两个多边形投影到以二者中心中点为中心的**局部方位等距投影**
  （`+proj=aeqd +R=1737400`），取 shapely 多边形距离；经度先平移到与局部中心相差 ±180° 内，
  因此跨反经线与极区都能正确投影；
- 所有 `distance < 10 km` 的边一起进 Union-Find（不再只用最近邻边）；
- 对 `distance < 0.5 km` 的成对额外计算**相交面积**，区分"相邻共边"与"真正覆盖同一地面"。

## 3. 结果

| 量 | 值 |
|---|---|
| 可用景 | 69（几何齐全 69/69） |
| 评估 pair | **2 346** |
| <10 km 的边 | **81** |
| **独立组** | **41**（v1 最近邻口径给出 51） |
| 最大组 | **13 景**（LICHTENBER1/2/3/4/5/6/7/8/9/10/11/12/13） |
| **跨组最小多边形距离** | **40.05 km**（阈值 10 km，余量 4 倍） |
| 距离 ≈0 的成对 | 34 对：**31 对真实重叠**、3 对仅共边 |

多景组：LICHTENBERG(13)、ARISTARCHU2/3/4/5(4)、MRINGENII1/2/3/4(4)、SPARIM1/2/3/4(4)、
KINGCRATER2/3/4(3)、MARIUS2/3/4(3)、FRESHMELT1/2(2)、MOOREF1/2(2)、SULPICIUS1/3(2)。

### 3.1 SPARIM 验证

`SPARIM1/2/3/4` 现在同一组；`SPARIM1–SPARIM3` 重叠 **114.3 km²**、`SPARIM1–SPARIM4` 41.8 km²、
`SPARIM2–SPARIM3` 48.7 km²。v1 把它们拆成两个拟选组是明确的数据污染，v2 已合并。

### 3.2 重叠 ≠ 重复来源

31 对零距离成对**没有任何一对共享 stereo 源**：它们来自不同年份的独立 NAC 立体观测
（例如 `LICHTENBER2 = M125007614, M125014405` 与 `LICHTENBER7 = M140344876, M140351663`），
重叠面积可达 **112.7 km²**。含义：

- 它们是**对同一地面的独立观测**，因此在"来源独立性"上合格；
- 但对统计而言它们是**同一地点**，所以必须同组，且组内不应同时取两个重叠景当两个样本。

## 4. v2 池与选择

- `protocols/heldout_v2_group_pool_v2.csv` — 41 组，逐组记录成员、原生分辨率、stereo 源、
  组内最大重叠面积、中心、disjoint 标志、探针标记、`selection_rank`。
- `protocols/heldout_v2_group_pairs_v2.csv` — **2 346 行全 pair 距离表**（含重叠面积），
  SHA256 `46bc10e5092f6440831c33db140d49b5…`。
- `protocols/heldout_v2_external_protocol_v2.json` — v2 协议；
  `status = PENDING_SIGNOFF`、`gate3_status = NOT_FROZEN`、`gate4_status = HOLD`。
- 选择规则不变（`sha256(group_id)` 排序、组内 `sha256(scene)`），新增两条：
  1. 每组最多 2 景；
  2. **组内若与已选景重叠 >1 km² 则跳过**（本次因此跳过 `ARISTARCHU3`，它与 `ARISTARCHU5` 重叠）。
- 结果：**15 组 / 18 景**，≈ **46.8 GB**（单 acquisition）。
  `selection_hash`、`scene_hash` 写入协议，测试用池+pair 表重算并断言一致。
- v1 文件保持原样；差异：41 vs 51 组，36 个组 id 未变（均为单景组）。

## 5. 测试

`tests/test_audit_groups_pairwise.py`（11 项）：真值已知的间距、共边 vs 重叠面积、反经线、
极区、multipart 取最近部分、桥接景（A–B 8 km、B–C 8 km、A–C 17 km 必须并为一组）、
SPARIM 式同址零距离必并组、跨组最小距离断言。

`tests/test_heldout_v2_protocol.py`（7 项）：状态不得被自动改为 FROZEN、v1 未被覆盖、
选择可由池+规则重算、组内已选景两两重叠 ≤1 km²、**所有跨组 pair 距离 ≥10 km**、
pair 表覆盖数 = C(66,2)=2 145… → 实际断言 `pairs_evaluated == n(n-1)/2`、SPARIM 同组。

## 6. 未做（明确）

- 未下载任何新景、未运行任何模型、未读取任何 heldout 风险标签。
- 桥接污染检查（"附近但当前无双相机的景"是否会成为跨组桥梁）**尚未**纳入：本轮 pair 表只覆盖
  69 个可用景。P2.1 文档要求这一点，需在下一轮用全 662 个 DTM 多边形做一次扩展扫描，
  并把"是否纳入桥接者"的理由写入 v3。
- SPARIM1/2 的"0 距离"已由多边形面积证实为真重叠，但**没有**逐一核对这些产品的 README 立体源
  是否与历史训练源冲突（`source_disjoint` 沿用 Gate 1 结果）。
