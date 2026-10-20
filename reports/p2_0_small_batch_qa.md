# P2.0 P20-03：小批量可行性（无新下载，仅可得性/容量核算）

**状态：READY（可得性达标）；协议修订仍 `PROPOSED_NOT_APPROVED`，Gate 3 继续 NOT FROZEN。**
2026-10-20 · 脚本 `scripts/check_p20_availability.py` · 产物 `outputs/p20_scene_availability.json`

本轮**没有下载任何新影像**；只把 P20-01/02 已经摸清的链路，与 Gate 1 严格池（84 景 / 52 组）
官方索引解析结果做联接，回答"真正可批量的是哪些景、代价多大"。

## 1. 168 acquisition → CDR 可得率

| 指标 | 值 |
|---|---|
| 严格池 acquisition 总数 | 168（84 景） |
| 左右相机**都**解析到 CDR | **138**（82.1%） |
| 仅单相机 | 2 |
| 未解析到（缺 NP/SP 极区索引等） | 28 |
| **全部 acquisition 双相机齐备的景** | **69 / 84（82.1%）** |

可得率按景统计而非按 acquisition：69 个景的所有立体源都能拿到双相机 CDR；
15 个景至少缺一条 acquisition（`ANTONIADI`、`GARDNER01`、`HSCARP3–5`、`MAUROLYCS`、`NEWCRATER1`、
`PYTHAGORAS1`、`REINER10/11/PHOT2`、`ROSSELAND2`、`SIMPELIUS`、`SULPICIUS2`），
其中部分位于极区（NP/SP 观测索引在该镜像 404），需要额外索引来源。

## 2. 可用池性质（69 景）

| 维度 | 值 |
|---|---|
| 景级地理组（Gate 1 分组） | **41 组** |
| 原生 post spacing | 2 m × 67、3 m × 2 |
| 与已用/受保护 footprint 的最小距离 | 最小 11.40 km、中位 114.3 km（≥10 km 门槛维持） |
| 纬度带 | (−60,−30] 17 · (−30,0] 5 · (0,30] 28 · (30,60] 19；两极各 0 |

**同一地理组内多景不得当作多个独立 Bootstrap Group**：本池 41 组 → 分层抽样的最大独立单位
是 41，不是 69。Gate 3 若按"每景 2–4 Tile"规划，40 景量级足够，但必须一组一景。

## 3. 存储预算（按实测文件，不按"每景 529 MB"估算）

| 项 | 实测 | 单景成本 | 40 景 |
|---|---|---|---|
| CDR `.IMG` | 528 929 736 B | ≈ 1.02 GB（双相机） | ≈ 41 GB |
| DTM `.TIF` | 136 373 747 B | ≈ 0.13 GB | ≈ 5 GB |
| ISIS `.cub` | ≈ 1 058 MB / 相机 | ≈ 2.1 GB | ≈ 84 GB |
| 正射导出（`isis2raw`） | 136 259 004 B | ≈ 0.26 GB | ≈ 10 GB |
| ISIS 内核 | `/opt/isis-env` 1.2 GB（内核子树） | 一次性 | 1.2 GB |

合计（40 景，保留 cub 与 raw）≈ **140 GB**；当前 `/` 已用 355 GB / 剩余 515 GB，可行。
若只保留 raw + IMG（可随时从 IMG 重建 cub）则 ≈ **57 GB**。

单景端到端耗时（本机实测）：`pds2isis` ≈ 10 s、`spiceinit` ≈ 3 s、`cam2map` ≈ 5.5 min/相机，
即**≈ 11.5 min/景（双相机）**，40 景 ≈ 7.7 h CPU。可并行 2–3 景（各约 1 GB IMG + 2 GB cub）。

## 4. 小批量 Pilot 建议（仍需你确认后才算正式启动）

按"先非冻结 probe 景"的原则，第一批判 3 景：
`FRSHCRATER10`（已完成，覆盖 0.9901）、`PLANCKFLOOR`、`RANGER`（严格池中净空最大、
纬度/地貌分散），用于验证不同分辨率/纬度下的链路稳定性，并测算真实失败模式。
Pilot 景**不进入**最终 heldout；随后只对候选做输入文件质量审核，**绝不查看模型性能**。

## 5. 仍缺口（不能算 PASS 的部分）

1. 28 条 acquisition 的影像缺失（极区索引 404 等）——需要 NP/SP/Extended 观测索引的另一来源。
2. P20-02 配准 BLOCKED 未解：1-pixel 门槛或需真正的控制点，或需协议修订。
3. `BLOCKED_NEW_HELDOUT` 触发条件：若 41 组中无法凑出足够通过 QA 的组，则降级输出该结论，
   保留 84/52 为候选池，不降低独立性标准。

## 6. 合规确认

- 原预注册 `protocols/p19_external_policy.json` 字节未动；CDR 提案仍 `PROPOSED_NOT_APPROVED`。
- 未下载、未运行模型、未查看 heldout 标签；≥10 km 与 README 源审计标准未放宽。
- 本轮新增仅一个只读统计脚本（不触网、不写数据目录）。
