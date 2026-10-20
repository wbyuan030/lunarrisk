# P2.0 进度

**代码基线**：`origin/main`（本轮开始 `2d1c2c6`）· 测试 232 项通过（本轮沿用，未改动测试逻辑）
任务清单：[`docs/P2_0_MULTICAMERA_QA.md`](P2_0_MULTICAMERA_QA.md)

| 项 | 状态 | 证据 |
|---|---|---|
| P20-01 双相机覆盖 | **PASS（覆盖达标）** | `reports/p2_0_multicamera_coverage.md`：union/DTM valid **0.9901**、overlap 0.0157、左右 I/F 残差 std 0.0076、r 0.9338 |
| P20-01 多相机融合实现 | 待做 | 已有覆盖/残差证据；需固定融合策略与 seamline |
| P20-02 独立配准 QA | **BLOCKED** | `reports/p2_0_ortho_registration_qa.md`：投影/网格/半径一致（2.0000 m 核对），无地形泄漏；但特征匹配无法分辨 1 个像元（2 m）系统偏移（±1 sample 对照只让 RMS 变 6–10 cm） |
| P20-03 小批量可行性 | **READY** | `reports/p2_0_small_batch_qa.md`：69/84 景双相机齐备（138/168 acquisition，82.1%）；可用池 **41 组**；40 景存储 ≈140 GB、CPU ≈7.7 h；缺 28 条 acquisition（极区索引 404） |
| P20-04 Gate 3 冻结 | HOLD | 需 P20-01/02/03 + CDR 协议批准 |

## 本轮实际改动的代码

- `scripts/qa_registration_p20.py`（新，P20-02）
- `scripts/check_p20_availability.py`（新，P20-03，只读统计 + 2 项测试）：共享有效带内的边缘特征配对 + 受控位移对照，
  输出逐象限匹配数与残差分布（mean/RMS/p50/p95/max）。
- `scripts/prepare_lroc_cdr_cube.py`
  - 补齐相机必需的 11 个 Instrument 键（右相机此前无法完成 `spiceinit`）；
  - 新增 `normalise_pvl_quotes()`（`editlab` 的 `'"…"'` 陷阱，等长替换保持 StartByte 有效）。

## 实测命令与数据

- 相机产品：`M153639872LC` / `M153639872RC`（CDR，LROLRC_1006/DATA/SCI/2011060/NAC/）
- DTM：`NAC_DTM_FRSHCRATER10`（14259×2389，2 m，equirectangular 48N/180E）
- 输出：`ortho_L.raw` / `ortho_R.raw`（各 136 259 004 B）
- 覆盖与光度统计见上表（QA 脚本随报告中的命令可复跑）

## 边界与约束（持续有效）

- 测试数：232 → **234**
- 原预注册 `protocols/p19_external_policy.json` 未改；CDR 路线仍为
  `protocols/p19_cdr_route_amendment_proposal.json` 的 `PROPOSED_NOT_APPROVED`。
- 未运行任何模型预测；未查看 heldout 风险标签；未降低 Gate 1 的 ≥10 km 与源审计标准。
- 严格池仍是 84 景 / 52 组；未批量下载。
