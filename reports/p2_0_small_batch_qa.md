# P2.0 P20-03：小批量可行性、可获得率与存储实测

**状态：探针景操作可行性 PASS；冻结仍未授权（CDR 路线 PROPOSED_NOT_APPROVED + 独立性审计未完成）**
**本报告只含数据质量与工程指标，不含任何模型评分。** 2026-10-20

## 1. 三个探针景（均不进入最终 heldout）

| 场景 | DTM | 相机产品 | cam2map | 联合覆盖 | 坑心 GCP（vs DTM 高程） |
|---|---|---|---|---|---|
| `FRSHCRATER10` | 2 m 14259×2389 | L+R（1 acq） | 同网格 | 0.9901 | 未做 |
| `RANGER` | 2 m 14244×2315 | L+R ×2 acq | 同网格 4/4 | 0.9987 | 主峰 (0,0) px，56 内点 |
| `PLANCKFLOOR` | 2 m 17556×3290 | L+R ×2 acq | 同网格 4/4 | 0.9771 | 主峰 (1.4,1.4) px，19 内点 |

细节见 [`p2_0_pilot_ranger.md`](p2_0_pilot_ranger.md) 与 [`p2_0_registration_qa.md`](p2_0_registration_qa.md)。

## 2. CDR 可获得率（168 acquisition 基准）

来自 `outputs/p20_scene_availability.json`（对已解析的 CDR 索引逐 acquisition 统计）：

| 量 | 值 |
|---|---|
| 基准 acquisition | 168 |
| 左右相机都可获得 | **138** |
| 只有一侧 / 部分 | 2 |
| 完全缺失 | 28 |
| 84 景候选池中"所有 acquisition 的 L+R 都齐" | **69** |

含义：单景要到达 0.95 覆盖，通常需要**同一 acquisition 的 L+R**，RANGER/PLANCKFLOOR 还需要第二
acquisition。因此 69 景是真正常可操作的子集；其余 15 景要么缺相机、要么需要替代 acquisition
（替代会引入不同时间的观测，必须在协议里写明，不能悄悄替换）。

## 3. 实测成本（单景 4 相机）

| 项 | 实测 |
|---|---|
| `pds2isis` | 10 s / 相机 |
| `spiceinit` | 3 s / 相机 |
| `cam2map`（cubic，默认） | 5.5 min / 相机 |
| `cam2map`（nearest，QA 用） | 13 min / 相机 |
| 4 相机全链路 | **19 min** |
| 磁盘：IMG 4×529 MB + cub 4×529 MB + raw 4×126 MB | **4.7 GB** |
| 另加 1 个 nearest QA cub | +0.66 GB（RANGER 目录合计 6.6 GB） |

排产含义：69 景 × 4.7 GB ≈ **324 GB**（若保留 QA 立方体则约 370 GB）。当前 `/` 可用 **498 GB**
（916 G 中已用 372 G），能放下但余量不大；建议策略是 cub 建成后删除 IMG（每景省 2.1 GB），
只在需要复核的景保留 nearest QA 立方体。

## 4. 未完成项（不得跳过）

- **来源独立性**：本报告的 69 景仍是"能做"的集合，未做组内 footprint 重叠与 10 km 间距复核；
  不能据此声称独立。
- **替代 acquisition 规则**：2 个只有单侧相机的 acquisition 需要协议明确处理方式。
- **CDR 路线审批**：`protocols/p19_cdr_route_amendment_proposal.json` 仍为提案状态。
- 因此：`Gate 3` 保持 **NOT FROZEN**，输出结论是"小批量可行"，不是"测试集已冻结"。
