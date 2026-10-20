# P2.0 进度

**代码基线**：`origin/main`（本轮开始 `2d1c2c6`）· 测试 232 项通过（本轮沿用，未改动测试逻辑）
任务清单：[`docs/P2_0_MULTICAMERA_QA.md`](P2_0_MULTICAMERA_QA.md)

| 项 | 状态 | 证据 |
|---|---|---|
| P20-01 双相机覆盖 | **PASS（覆盖达标）** | `reports/p2_0_multicamera_coverage.md`：union/DTM valid **0.9901**、overlap 0.0157、左右 I/F 残差 std 0.0076、r 0.9338 |
| P20-01 多相机融合实现 | 待做 | 已有覆盖/残差证据；需固定融合策略与 seamline |
| P20-02 独立配准 QA | 待做 | 需独立控制点、投影一致性（米级容差）、地形泄漏检查 |
| P20-03 2–3 个 Pilot 场景 | 待做 | |
| P20-04 Gate 3 冻结 | HOLD | 需 P20-01/02/03 + CDR 协议批准 |

## 本轮实际改动的代码

- `scripts/prepare_lroc_cdr_cube.py`
  - 补齐相机必需的 11 个 Instrument 键（右相机此前无法完成 `spiceinit`）；
  - 新增 `normalise_pvl_quotes()`（`editlab` 的 `'"…"'` 陷阱，等长替换保持 StartByte 有效）。

## 实测命令与数据

- 相机产品：`M153639872LC` / `M153639872RC`（CDR，LROLRC_1006/DATA/SCI/2011060/NAC/）
- DTM：`NAC_DTM_FRSHCRATER10`（14259×2389，2 m，equirectangular 48N/180E）
- 输出：`ortho_L.raw` / `ortho_R.raw`（各 136 259 004 B）
- 覆盖与光度统计见上表（QA 脚本随报告中的命令可复跑）

## 边界与约束（持续有效）

- 原预注册 `protocols/p19_external_policy.json` 未改；CDR 路线仍为
  `protocols/p19_cdr_route_amendment_proposal.json` 的 `PROPOSED_NOT_APPROVED`。
- 未运行任何模型预测；未查看 heldout 风险标签；未降低 Gate 1 的 ≥10 km 与源审计标准。
- 严格池仍是 84 景 / 52 组；未批量下载。
