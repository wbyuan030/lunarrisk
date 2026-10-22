# P2.0 进度

**代码基线**：`origin/main`（本轮开始 `2d1c2c6`）· 测试 232 项通过（本轮沿用，未改动测试逻辑）
任务清单：[`docs/P2_0_MULTICAMERA_QA.md`](P2_0_MULTICAMERA_QA.md)

| 项 | 状态 | 证据 |
|---|---|---|
| P20-01 双相机覆盖 | **PASS（覆盖达标）** | `reports/p2_0_multicamera_coverage.md`：union/DTM valid **0.9901**、overlap 0.0157、左右 I/F 残差 std 0.0076、r 0.9338 |
| P20-01 多相机融合实现 | **PASS（单 acquisition）** | 接缝跃变 1.16× / 1.50× 内部对照；跨 acquisition 5.5× / 6.0× 已废弃 | 已有覆盖/残差证据；需固定融合策略与 seamline |
| P20-02 独立配准 QA | **定稿：1–2 px（2–5 m）** | `reports/p2_0_ortho_registration_qa.md`：投影/网格/半径一致（2.0000 m 核对），无地形泄漏；但特征匹配无法分辨 1 个像元（2 m）系统偏移（±1 sample 对照只让 RMS 变 6–10 cm） |
| P20-03 小批量可行性 | **READY** | `reports/p2_0_small_batch_qa.md`：69/84 景双相机齐备（138/168 acquisition，82.1%）；可用池 **41 组**；40 景存储 ≈140 GB、CPU ≈7.7 h；缺 28 条 acquisition（极区索引 404） |
| P20-04 Gate 3 冻结 | **PENDING_SIGNOFF** | `protocols/heldout_v2_external_protocol.json` v1：48 个可用组中按 `sha256(group_id)` 取 15 组 / 18 景（每组 ≤2 景）；`protocols/heldout_v2_group_pool.csv` 为 51 组池；状态待研究员签核后才会置 FROZEN |

## 本轮实际改动的代码

- `scripts/qa_registration_p20.py`（新，P20-02）
- `scripts/check_p20_availability.py`（新，P20-03，只读统计 + 2 项测试）：共享有效带内的边缘特征配对 + 受控位移对照，
  输出逐象限匹配数与残差分布（mean/RMS/p50/p95/max）。
- `scripts/prepare_lroc_cdr_cube.py`
  - 补齐相机必需的 11 个 Instrument 键（右相机此前无法完成 `spiceinit`）；
  - 新增 `normalise_pvl_quotes()`（`editlab` 的 `'"…"'` 陷阱，等长替换保持 StartByte 有效）。

## P20-02 配准实测（2026-10-20，第二次推进）

新增 `scripts/grid_consistency_p20.py`、`scripts/solar_azimuth_p20.py`、`scripts/make_dtm_template.py`、
`scripts/ortho_registration_p20.py` 与 13 项离线测试（合计 **257** 项通过）。

| 项 | 状态 | 关键数字 |
|---|---|---|
| 正射 ↔ DTM 网格算术一致性 | **PASS** | 最近邻 QA 立方体 40/40 精确匹配，含边界点 |
| 太阳方向（SPICE，球面三角） | **PASS** | RANGER 100.01°/42.64°；PLANCKFLOOR 310.46°/25.23° |
| hillshade 几何自检 | **不足** | RANGER 0.802（5/16 窗口）；PLANCKFLOOR 0.229（0/16） |
| 坑心 GCP（正射 vs DTM 高程） | **PASS @2 px** | RANGER 主峰 (0,0) px、56 内点；PLANCKFLOOR 1.4 px、19 内点；p95 1.4/2.3 px |
| 左右相机互检 | 2–3 px 系统差 | RANGER (−2,0) px；PLANCKFLOOR (3,0) px |
| ≤1 px | **不声称（定稿）** | 口径已确定为 1–2 px；GCP 质心偏差约 1 px |
| PLANCKFLOOR 覆盖 | PASS | 四相机 0.9771；单 acquisition 0.9678/0.9429 |

新发现（工程）：`cam2map` 默认 `INTERP=CUBICCONVOLUTION`，普通正射像素值是邻域卷积，
不能用于相等判据；QA 需单独重建 `interp=nearest` 立方体（13 min/相机）。

## P20-03 小批量可行性（2026-10-20）

[`reports/p2_0_small_batch_qa.md`](p2_0_small_batch_qa.md)：三探针景链路全通（19 min / 4 相机、4.7 GB / 景）；CDR 可获得率 138/168 acquisition 左右齐、69/84 景全齐；69 景约需 324 GB，当前可用 498 GB，建议 cub 建成后删 IMG。独立组数 **51 / 69 景**（最大簇 11 景 LICHTENBER）；替代 acquisition 规则 / CDR 审批两项未完成，Gate 3 仍 NOT FROZEN。

## 实测命令与数据

- 相机产品：`M153639872LC` / `M153639872RC`（CDR，LROLRC_1006/DATA/SCI/2011060/NAC/）
- DTM：`NAC_DTM_FRSHCRATER10`（14259×2389，2 m，equirectangular 48N/180E）
- 输出：`ortho_L.raw` / `ortho_R.raw`（各 136 259 004 B）
- 覆盖与光度统计见上表（QA 脚本随报告中的命令可复跑）

## 边界与约束（持续有效）

- 测试数：232 → **271**
- 原预注册 `protocols/p19_external_policy.json` 未改；CDR 路线仍为
  `protocols/p19_cdr_route_amendment_proposal.json` 的 `PROPOSED_NOT_APPROVED`。
- 未运行任何模型预测；未查看 heldout 风险标签；未降低 Gate 1 的 ≥10 km 与源审计标准。
- 严格池仍是 84 景 / 52 组；未批量下载。
