# P1.9 → P2.0：CDR 双相机覆盖、配准 QA 与外部测试集冻结

> **状态：下一轮 Coding Agent 执行清单（2026-10-20）。未实施的项均为 TODO，不是实验成果。**
>
> 起点 `main@7c26b3785122f49c509db4d69647df8984887081`。最新真实证据见
> [`reports/p1_9_optical_ortho_probe.md`](../reports/p1_9_optical_ortho_probe.md)：
> `FRSHCRATER10 / M153639872LC` 已完成 `CDR IMG → pds2isis → Instrument 补齐 → SPICE → cam2map`，
> 与 DTM 同网格 14259×2389，但左相机影像仅覆盖 **42.75% DTM 有效区域**；
> 2m 的边缘互相关峰不够尖锐，不构成独立地物配准证明。
> 已报告 232 项测试通过；新增工作必须重新实测。
>
> **目标**：把“单景成功”升级为可复现的“双相机/多影像 → DTM 同格网”流水线，量化**几何覆盖、辐射一致性与配准精度**，并在正式冻结任何 `heldout_v2_external` 之前修订数据协议。**绝不进行 Gate 4 模型预测**。

## 已确认事实与研究边界

- Gate 0/1 真实来源 + 空间审计：128 景候选 / 91 组；原生 DTM post spacing ≤3m、满足原尺度政策 A 的池为 **84 景 / 52 组**。这些是索引/几何层候选，**不是**已冻结测试集。
- 已从官方 NAC 观测索引找到 CDR：278 个相机产品 / 140 个观测号，138 个观测号有左右相机记录。`FRSHCRATER10` 实测下载 CDR 左右相机各约 529MB；CDR 像元为 scaled I/F，不能靠篡改 PDS3 `DATA_SET_ID` 假装 EDR 导入。
- 单景 `pds2isis` + 补齐 `Instrument` + 显式 `spiceinit` + `cam2map` 路线成功；**仅同尺寸/同投影不代表光学与地面真值物理精准配准**。
- 原始 `protocols/p19_external_policy.json` 预注册 `INSTITUTIONAL_EDR_ISIS`，**不能事后覆盖**以便声称原本就允许 CDR。新增 [`protocols/p19_cdr_route_amendment_proposal.json`](../protocols/p19_cdr_route_amendment_proposal.json) 是独立版本化提案；用户/研究人员正式确认之前，CDR **只可做 Pilot / QA，不可据此冻结或评分**。
- 不将任何 `NAC_DTM_*_SHADE/_SLOPE`、DTM 渲染 browse 图、地形派生纹理作为真正光学输入。光学 CDR 必须能溯源至通过 README 审计的同一 NAC acquisition / camera，且与旧 SomBench 训练源完全不交叠。
- 旧 110 Tile 仍只是 `development_test`；历史 64 Tile 产品级 V2 的冻结名单/哈希不能改写。

## P20-01（最高优先级）：双相机和多幅 NAC 覆盖验证

现有探针 `FRSHCRATER10` 已下载左右 CDR；**先复用现有数据**，不要直接下载 84 景。
1. 对 `M153639872RC` 运行已验证的 `prepare_lroc_cdr_cube.py`，检查对应相机 `FrameId=RIGHT` 和 ISIS `NaifIkCode=-85610`，为每条 CDR 单独记录显式 CK/FK/SPK/IK/IAK/SCLK/LSK/PCK/TSPK 与 SHA256。
2. 将左右 CDR 各自用 DTM 正式 label 中的 `CENTER_LATITUDE/LONGITUDE`、`MAP_RESOLUTION` 和输出边界进行 `cam2map`；**输出 GeoTransform、CRS、像元中心与 DTM 完全一致**，不仅阵列形状一致。
3. 以 DTM 非 NoData 像素为分母分别计算 `coverage_left`、`coverage_right`、`coverage_union`、`coverage_overlap`、`coverage_intersection / DTM_valid`；同时记录不按 DTM mask 归一的整图有效比例。当前单左光学覆盖 **42.75%** 不能直接 PASS 原 Runbook 建议的 95% 覆盖。
4. **左右相机可能是同一次观测的邻接条带，或者地面足迹有重叠；不得假设它们自动覆盖整个 DTM。** 若二者并集仍不足，再检查同 DTM README 源清单中另一 acquisition 的 L/R CDR；为每条 CDR 只选择来源被审计过的 ID。
5. 多幅正射镶嵌策略必须在观测测试成绩前固定：重叠区域优先使用注册误差较低且不饱和的 CDR，或使用明确权重；写出 seamline、遮挡/NoData 掩码及亮度匹配参数；光照过大变化不可简单线性融合。
6. **门槛**：先报告真实覆盖曲线，优先争取 DTM 有效区域 ≥95% 光学覆盖（原 Runbook 建议），若确实无法达到，优先限定在有足够联合有效像素、来源完整且满足最小 Tile 数的 ROI。**不能为了凑够场景直接降低全局门槛**；如调整成 “tile-level” QA，需先提出版本化协议修订并明确用户/研究人员批准。

**交付**：`scripts/ortho_mosaic_p19.py`（或等价的可重跑流水线），
`tests/test_ortho_mosaic_p19.py`（纯离线合成 mask、重叠融合与边界测试），
`reports/p2_0_multicamera_coverage.md`（左/右/联合/重叠统计、来源哈希与命令）。

## P20-02：严谨独立配准和辐射 QA

1. 记录并核对 DTM 与正射的 pixel-is-area / pixel-is-point、仿射变换、月球半径与 DTM label/GeoTIFF 投影；容忍误差以**米**为单位，而不是凭两个数组形状相同。
2. 用足够多、地理位置分散的**独立控制点或地貌特征**匹配（应至少跨不同 ROI），报告匹配数、RMS、p50/p95、最大残差，识别异常点；左右 CDR 互检/DTM relief edge 自检是补充，不能单凭弱互相关峰值断言精度达 1 pixel。
3. 真实 NAC CDR 像元保持 scaled I/F，记录 `SCALING_FACTOR`、`NULL`、饱和/阴影比例、左右相机重叠区光度残差；同 P1.8 网络特征约定（contrast/texture）需要**训练端规范一致**。
4. 正射时如使用输出 DTM 进行地形投影，必须在报告中标注 “DTM used for map-projection geometry”。**禁止把 DTM 中的真实高程/坡度通过归一化、几何角等方式直接泄漏给模型输入**。需要明确地理投影/正射所依赖的 DEM 来源、使用 DTM 投影 vs 使用其逐像素高程的区别，必要时做只依赖独立粗地形的正射对照。
5. 原始 `DTM` / `CDR` / `LBL` / `SPICE` / 正射 TIFF 的 SHA256 与操作版本、ISIS/ROCm 环境分开留档。合成测试覆盖极区、反经线、相机 R、NoData、错位 1 pixel 与左右亮度差。

**交付**：`reports/p2_0_ortho_registration_qa.md`，包含实际检查点数据和失败场景；达不到可信阈值则记 BLOCKED，不能仅报告 `campt` 一致。

## P20-03：非破坏式协议修订与小批量验证

先保持 [`protocols/p19_external_policy.json`](../protocols/p19_external_policy.json) **字节不动**。
新增版本化提案中的 CDR source equivalence、辐射量、导入链、拼接规则、模型输入独立性、覆盖阈值和 ROI 的选择口径，需明确审批状态。**本次仓库写入提案不是自动授权更改既有预注册选择**。

在 P20-01/02 探针 QA 达标，且用户确认 CDR 方案后：
- 选 **2–3 景**，最好先选明确**不会进入最终 heldout** 的 probe 景调通操作性，再对候选只做输入文件质量审核，绝不查看模型性能。
- 检查 168 acquisition 基准清单 vs CDR 的可获得率，明确缺失/部分左右相机、不同轨道、不同时间的补充影像是否在 README 立体源内。
- 小样本存储预算必须基于实测 `*.IMG / *.cub / *_ortho` 文件，不按每景一张 529MB 估算；ISIS CK/SPICE 已占磁盘也要计入。
- 同一地理连通组内若多景紧邻，不得作为多个独立 Bootstrap Group；持续保留 Gate 1 的 ≥10 km 来源/几何审计。
- 如果无法使足够站点达到冻结质量，输出 `BLOCKED_NEW_HELDOUT`，保留 84 景/52 组为候选池，不降低预注册独立性标准。

**交付**：`reports/p2_0_small_batch_qa.md`，通过/失败数量、各景性能**仅限数据质量**，不得含 CNN 风险评分/MAE/Brier/AUC。

## P20-04：Gate 3 正式冻结（以 Gate 0/1/2 全部 PASS 为前置）

- 在模型对任一候选输出预测值之前，记录批准后的 CDR 协议版本和冻结代码 Git SHA。
- 生成不可变 `protocols/heldout_v2_external_manifest.csv` 与
  `protocols/heldout_v2_external_protocol.json`，逐 Tile 记录完整 stereo 来源 ID、CDR 相机产品 ID、footprint、CRS、原生 GSD、有效 mask、拼接策略、光照特征、地形来源、哈希与 Group；按组分层冻结（而非按景随机散列）。
- 对其余未使用源、历史训练/标定/开发评估、受保护 64 Tile 做完整 source + polygon overlap 复核。
- 只在协议、QA 和足够新组成立时将 Gate 3 标为 `FROZEN`；**Gate 4 一次性模型评估继续 HOLD，等待单独授权**。

## 测试、代码审核与提交

```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
git status --short
```

每完成一个任务带上可运行代码、离线单测、真实命令与输出、数据来源 SHA256、
更新 `reports/p2_0_progress.md`，然后 push `main`。不要把未执行的单景/多景验证报告成 PASS。
**资源边界**：只使用已有 RX 7900 XT / ROCm 环境与免费公开数据，尽量复用现有左右 CDR；
不购买云卡，不使用 4×4090，不覆盖现有标签/冻结数据；不运行大型 LFM/模型测试。
