# P2.0 P20-03：小批量 Pilot — RANGER 实跑结果

**状态：PASS（数据质量层面，4 相机齐、联合覆盖 0.9987）** · 2026-10-20
场景 `RANGER`（lat 9.0, lon 21.0，2 m DTM，净空 16.0 km，组 CDTM_0094）。
本景**不会进入最终 heldout**（FRSHCRATER10 之外的第二 probe 景；P20-01 已报告的 FRSHCRATER10 亦非冻结候选）。
新脚本 `scripts/run_p20_ortho.sh`、`scripts/coverage_qa_p20.py` + 2 项离线测试。

## 1. 链路实跑

| 阶段 | 结果 |
|---|---|
| 采集 | 4 个相机产品（2 acquisition × L/R）：`M144483945LC/RC`、`M144490730LC/RC`，各 528 929 736 B |
| DTM | `NAC_DTM_RANGER.TIF`（14244 × 2315，2 m，equirectangular 9N/180E）+ LBL |
| SPICE 内核 | CK `lrolc_2010304_2010335_v02.bc` + bus `moc42r_2010304_2010335_v03.bc`、SPK `fdf29r_2010305_2010335_v01.bsp`、FK v01 副本、IK/IAK/SCLK/LSK/PCK/TSPK |
| 相机模型 | 4/4 `spiceinit` 成功（L → `NaifIkCode=-85600`，R → −85610） |
| 正射 | 4/4 `cam2map` 输出 **14244 × 2315**，与 DTM 逐像元同网格 |

单个相机耗时：`pds2isis` ≈ 10 s、`spiceinit` ≈ 3 s、`cam2map` ≈ 5.5 min；4 相机合计约 19 min（与本轮 P20-03 估算的 11.5 min/景/相机口径一致）。磁盘：4×529 MB IMG + 4×529 MB cub + 4×126 MB raw ≈ 4.7 GB/景（保守口径）。

## 2. 覆盖（分母 = DTM 有效像元 31 996 276，0.9703）

| 相机/影像 | dtm_coverage | I/F p50 | I/F max |
|---|---|---|---|
| `M144483945LC` | 0.4597 | 0.0258 | 0.0446 |
| `M144490730LC` | 0.6541 | 0.0191 | 0.0314 |
| **左联合** | **0.7808** | — | — |
| `M144483945RC` | 0.4920 | 0.0266 | 0.0365 |
| `M144490730RC` | 0.2323 | 0.0189 | 0.0266 |
| **右联合** | **0.5477** | — | — |
| **四影像联合** | **0.9987**（整图 0.9985） | — | — |
| 左右重叠 | 0.3298（10 561 213 像元） | — | — |

> 复用同一脚本重算 FRSHCRATER10：union 0.9901、overlap 0.0157、残差 std 0.007634 I/F
> —— 与 P20-01 报告**数值一致**（脚本可复现性核对通过）。

**关键观察**：RANGER 需要 **4 条影像**（2 acquisition × 2 相机）才达到 0.9987；任何单相机（最佳 0.6541）或单 acquisition 的左右（联合 0.7808）都远低于 0.95。
因此"每个 acquisition 一景"的假设不成立，**必须按 acquisition 组合排产**。

## 3. 光度一致性（左右重叠 10.56 M 像元）

| 量 | 值 |
|---|---|
| I/F 残差均值 | −0.007078 |
| I/F 残差标准差 | **0.003157** |
| 比较基准 | FRSHCRATER10 为 0.007634；RANGER 更一致 |

## 4. 辐射水平（重要且此前未记录）

RANGER 属于**低照度采集**：原始 DN 中位数 961（FRSHCRATER10 为 2524），I/F p50 ≈ 0.026，
最大 ≈ 0.108；无饱和像元，有效像元率与 FRSHCRATER10 相同（0.9874）。
这不是解码错误（同一 IMG 内确有 DN 8047/32768 的高值像素；`pds2isis` 的
`Multiplier=3.05185e-05`、`Base=0` 与标签一致），而是采集条件差异。
**含义**：若未来做训练/评估，低 I/F 场景的对比度会被压缩，需要在输入端规范（P1.8 的
contrast/texture 约定）中显式处理，不能假定所有景有相同动态范围。

## 5. 本轮发现并修正的工程问题

1. `pilot_p20.sh` 初版把 DTM 标签键写成 `LINES/LINE_SAMPLES`，实际为 `LINE_LAST_PIXEL/SAMPLE_LAST_PIXEL`，
   且最小值/中心值需匹配负号；已改为解析出 `<scene>.grid.json`（projection/clat/clon/scale/radius/lines/samples/ppd/边界）。
2. `M144490730.IMG` 首轮下载只落盘 3.1 KB（截断），导致 `maptemplate`/`cam2map` 报"模板无 Dimensions"。
   driver 现在**先校验文件大小再处理**（`< 1 MB` 直接跳过并标记），截断文件需重下。
3. `coverage_qa_p20.py` 逐相机自动检测 `isis2raw` 的 NULL 哨兵（本例左 0.022126/0.015229、右 0.023133/0.015656），
   避免把填充值当有效像元。
4. 覆盖脚本对"多 acquisition × 多相机"输出 per-image / per-camera-union / 全 union / 左右重叠与残差，
   可直接用于后续批量。

## 5b. 第二景 PLANCKFLOOR（同链路，2 m DTM 17556×3290）

| 量 | 值 |
|---|---|
| DTM 有效像元 | 54 988 610（0.9520） |
| `M136648682` L+R | 0.9678 |
| `M136655466` L+R | 0.9429 |
| **四相机联合** | **0.9771**（整图 0.9715） |
| 左右重叠 | 0.0817（4 497 455 像元），残差 std 0.007483 |
| 太阳几何 | az 310.46°，alt 25.23° |

要点：单 acquisition 已达 0.9678，但第二 acquisition 只把覆盖推到 0.9771 —— 说明**两条 acquisition
的地面足迹高度重合**（左并集 0.482 vs 最佳单幅 0.477），与 RANGER 的互补情形相反。
排产必须依据**逐 acquisition 的联合覆盖**，不能假定“多一条 acquisition 就能补足缺口”。

## 6. 结论与边界

- **RANGER / PLANCKFLOOR：PASS（数据质量）**：联合覆盖 0.9987 / 0.9771 ≥ 0.95；
  左右残差 std 0.003157 / 0.007483 I/F；网格与 DTM 逐像元一致（精确实验 40/40，见
  [`p2_0_registration_qa.md`](p2_0_registration_qa.md)）。
- **P20-02 配准 BLOCKED 仍未解除**（1-pixel 门槛），因此**不得据此冻结任何 heldout**。
- CDR 路线仍为 `PROPOSED_NOT_APPROVED`；本报告仅数据质量，无任何模型指标。
- 存储/算力：按本景实测 4.7 GB、19 min（4 相机），40 景量级 ≈ 190 GB / 12.7 h，与 P20-03 估算（140 GB / 7.7 h）同量级。
