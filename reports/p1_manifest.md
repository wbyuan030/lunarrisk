# P1-01 — 100k Manifest 与分组泄漏审计

日期：2026-10-17
范围：**只构建 manifest 与切分审计，不训练模型**。
输入：`data/s3/NAC_HighRes_100k.parquet`（SHA256 `f9c39d1b570693902bd5616d44f6330ea15a425db3c9d3eb9ceb2cdc9be1873e`，90,786 行）。
脚本：`scripts/build_manifest.py`；测试：`tests/test_build_manifest.py`（24/24 PASS）。

## 1. Manifest

产物：

- `outputs/manifest_100k.parquet`（90,786 行 × 40 列，12 MiB）
- `outputs/manifest_sample.csv`（前 2,000 行，便于人工检查）
- `outputs/manifest_summary.json`

字段覆盖 `docs/DATA_PROTOCOL.md` 的最小 Manifest：

| 字段 | 状态 |
|---|---|
| `sample_id` | `PRODUCT_ID_rROW_cCOL`，无重复 |
| `NAC_EDR_ID` | 来自 `PRODUCT_ID`，1,090 个 |
| `DTM_source_ID` | **Null**（官方索引未提供每 Tile source） |
| `image_path` / `dtm3m_path` / `slope3m_path` / `coarse_dtm_path` | S3 key，完整前缀 |
| `metadata_path` | S3 key |
| `LTM_CODE` | 80 个 |
| `footprint_wkt` | 投影坐标下的 512 m × 512 m 多边形 |
| `crs_wkt` | Moon 2015 Sphere / Transverse Mercator（551 字符） |
| `gsd_nac_m` / `gsd_3m_m` | 1.0 / 2.994152 |
| `split_official` | S3 索引原始 split |
| `split_group_clean` | 去泄漏后的 split |
| `group_id` | 产品 + 5 km 块的连通组 |
| `block_5km` | `LTM|floor(x/5000)|floor(y/5000)` |
| `has_dtm3m` / `has_slope3m` | 90,497 / 90,481 |
| `dtm3m_sha256` / `slope3m_sha256` / `nac_sha256` | 仅 12 个已抽样验证 Tile 有值，其余 Null |
| `modality_read_ok` / `valid_fraction` | 仅 12 个已抽样 Tile 有值，其余 Null |
| `co_registration_status` | `SAMPLED_VERIFIED`（12）或 `INDEX_BOUNDS_ONLY`（其余） |
| `geo_join_status` | 全部 `GEO_JOIN_PENDING` |

未下载的 90,774 行不伪造 SHA256/read 状态；需要时可分批 hydration 后补齐。

## 2. 分组策略

1. 节点 = `PRODUCT_ID`（父 NAC EDR）。
2. 同一 `LTM_CODE` 内落在同一个 5 km 投影块的产品合并；产品跨块/跨 zone 时，其所有块通过产品节点连通。
3. 连通分量 = `group_id`。因此**任何产品、任何 5 km 块都不会跨 split**。
4. 官方 split 只修正跨 split 的组：整组归入组内多数 split（平票归 train）。

## 3. 分组规模敏感性

| block | blocks | groups | 最大组 tiles | 最大组占比 | 中位组 tiles | 混合组 | 混合组内 tiles |
|---|---|---|---|---|---|---|---|
| 1 km | 52,640 | 297 | 8,190 | 9.02% | 138 | 5 | 857 |
| **5 km** | **8,962** | **272** | **8,190** | **9.02%** | **149** | **5** | **937** |
| 10 km | 3,359 | 269 | 8,190 | 9.02% | 149 | 5 | 980 |
| 25 km | 1,126 | 255 | 8,190 | 9.02% | 156 | 5 | 980 |

5 km 已足以把官方泄漏暴露出来；继续增大块只减少组数，不改变 5 个混合组。manifest 采用 **5 km**。

## 4. 官方 split 的泄漏

| 交叉 | 重叠 groups | 重叠 PRODUCT_ID | 重叠 5 km blocks |
|---|---|---|---|
| train ∩ val | **5** | **6** | 0 |
| train ∩ test | 0 | 0 | 0 |
| val ∩ test | 0 | 0 | 0 |

官方 test 在产品级完全隔离；泄漏集中在 train/val 的 6 个产品。0 个 5 km block 重叠说明这些产品是跨块的细长 EDR，产品级泄漏而非同块复制。

## 5. 修正后的 split

`split_group_clean`：

| split | tiles（全部） | tiles（有 slope_3m） | groups | products | blocks |
|---|---|---|---|---|---|
| train | 73,292 | 73,157 | 221 | 766 | 7,257 |
| val | 9,113 | 9,113 | 26 | 260 | 730 |
| test | 8,381 | 8,211 | 25 | 64 | 975 |

- 修正组：5 个；组内 tiles：937。
- 实际改变 split 的 tiles：**25**（train→val 18，val→train 7；test 未变）。
- 修正后三组交叉重叠：**groups 0 / products 0 / 5 km blocks 0**。

## 6. 验证覆盖

- 12 个已下载 Tile：NAC/DTM/slope/metadata 四个文件均有 SHA256；`modality_read_ok=true`；slope valid_fraction=1.0；单位经验验证 12/12。
- 其余 90,774 行：`sha256/read/valid_fraction = Null`，`co_registration_status=INDEX_BOUNDS_ONLY`。
- 空间 overlay 样例：`outputs/s3_sample_overlays.png`（12 组）。
- 全部行仍为 `GEO_JOIN_PENDING`：尚未实现 footprint 级 NAC crop/重采样。

## 7. 进入 P1-02/03 前仍需完成

1. **地理配准**：把 `INDEX_BOUNDS_ONLY` 行升级为真实 footprint 联接；当前 manifest 只适用于索引/统计，不应直接读 NAC 训练。
2. **空间缓冲**：5 km 块保证同块不跨 split，但相邻块仍可能跨 split；P1-02 应按连续区域/缓冲区再切一次（例如丢弃距异 split Tile < 1 km 的边缘 Tile），并报告损失量。
3. **DTM source**：保持 Null；分组以 `NAC_EDR_ID + block` 为准。
4. **Hydration**：训练前对选中的 split 批量下载并补齐 SHA256、可读性、valid_fraction。
5. **单位写入**：slope 文件本身无 `units` 属性，manifest 需在 hydration 时写入 `units=degrees` 及证据来源。

## 8. 复现

```bash
python3 -m unittest discover -s tests -v          # 24 tests
python3 scripts/build_manifest.py                 # 默认 5 km
python3 scripts/build_manifest.py --block-m 25000 # 敏感性
```

产物：`outputs/manifest_100k.parquet`、`outputs/manifest_summary.json`、`outputs/manifest_sample.csv`。
