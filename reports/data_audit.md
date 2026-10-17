# P0-03 数据审计 — SomBench 公开资产

日期：2026-10-17（Asia/Shanghai）
执行环境：`rocm-lab` 容器（`rocm/dev-ubuntu-22.04:7.1.1-complete`），见 `first_run_report.md`。
所有计数均来自实际 API 列举与落盘文件，不把 Parquet 索引行当作可下载文件。

## 1. 数据源 revision

| 源 | revision / 标识 | 许可 |
|---|---|---|
| HF `nasa-ibm-ai4science/Sombench-pretraining-data` | `e52a7f46090458227f970e7ea19896c179b6c371` | CC-BY-4.0 |
| HF `nasa-ibm-ai4science/Sombench-NAC-Crater-Detection` | `79419eb3486b2dd9ad09a283802e0bb8d75c767d` | CC-BY-4.0 |
| S3 `s3://nasa-lunar-fm-bench`（无签名公开读取） | 100k 索引 SHA256 `f9c39d1b570693902bd5616d44f6330ea15a425db3c9d3eb9ceb2cdc9be1873e` | 仓库 README 声明 CC-BY-4.0 |

S3 列举命令（容器内经宿主机代理 `127.0.0.1:7890`）：

```bash
curl -sS "https://nasa-lunar-fm-bench.s3.amazonaws.com/?list-type=2&delimiter=/&max-keys=100"
curl -sS -O "https://nasa-lunar-fm-bench.s3.amazonaws.com/pretraining_dataset/100k_release/NAC_HighRes/NAC_HighRes_100k.parquet"
```

## 2. HF 公开样例（6 Tile，仅作 I/O 集成测试）

| 指标 | 数值 |
|---|---|
| 仓库列出 | 83 文件 / 149,906,571 B（142.96 MiB） |
| 按 `NAC_HighRes/**` + README 选择 | 32 文件 / 6,965,774 B（6.64 MiB） |
| 实际落盘 | 32 payload 文件 / 6,965,774 B；0 字节文件 0 |
| Parquet 索引行 | **6**（训练 4 / 测试 2），不是约 100 万 |
| `nac` | 下载 6 / 可解析 6 |
| `dtm_3m` | 下载 6 / 可解析 6 |
| `slope_3m` | 下载 6 / 可解析 6 |
| `aspect_3m` | 下载 6 / 可解析 6 |
| 完整 NAC+DTM3m+Slope3m 三元组 | 6 |
| 独立 `PRODUCT_ID` | 6 |
| 每 Tile DTM source ID | 未提供（保持 Null） |
| CRS 一致（NAC vs slope） | 6/6 |
| 网格边界对齐（≤1.5×GSD） | 6/6，最大偏差 0.9971 m |

结论：6 条样例足以验证 NetCDF/HDF5 读取、变量、单位与网格几何，**不足以训练或做泛化声明**。

## 3. S3 公开 100k 发布（真实主数据来源）

索引 `pretraining_dataset/100k_release/NAC_HighRes/NAC_HighRes_100k.parquet`
（27,864,124 B，90,786 行，83 列）。

| 指标 | 数值 |
|---|---|
| 索引 Tile 行 | **90,786** |
| 独立 `PRODUCT_ID`（NAC EDR） | **1,090** |
| 独立 `LTM_CODE` | **80** |
| split 行数 | train 73,303 / val 9,102 / test 8,381 |
| `DTM_3M_TILE` 非空 | 90,497 |
| `SLOPE_3M_TILE` 非空 | **90,481** |
| 60 m 地形 | 90,786（全覆盖） |
| 1M release | 目录存在，索引 `NAC_HighRes.parquet` 198,851,147 B；本次未下载 |

抽样验证（`scripts/s3_sample.py --n 12`，固定种子 42）：12 个 Tile，**12 个不同 PRODUCT_ID、12 个不同 LTM zone**，6 train / 3 val / 3 test。

| 检查 | 结果 |
|---|---|
| 下载并读取 NAC/DTM/slope/metadata | 12/12 |
| CRS 一致（NAC vs DTM vs slope） | 12/12 |
| 网格边界对齐 | 12/12，dx=dy=1.00 m（GSD 2.994 m） |
| slope 值域 | 0.04°–37.06°，无负值/超 90° |
| slope 文件内 `units` 属性 | 0/12（缺失，见 physics_audit） |
| 由 DTM 重算坡度 vs 文件 slope | Pearson 0.985–1.000；均值比 0.995–1.025；中位绝对差 0.011°–0.224° |
| 12/12 经验证单位为 degrees | 是 |

可视化：`outputs/s3_sample_overlays.png`（12 组 NAC / slope / >10° overlay / 直方图）。overlay 中撞击坑边缘与高坡度区一致；定量上由 DTM 重算的坡度与发布 slope 几乎逐像素一致。

**split 泄漏审计（关键）**：

| 交叉 | 重叠 `PRODUCT_ID` 数 |
|---|---|
| train ∩ val | **6** |
| train ∩ test | 0 |
| val ∩ test | 0 |

test 与 train/val 在 PRODUCT_ID 级完全不相交；train/val 有 6 个 PRODUCT 重叠。P1 必须按 PRODUCT_ID/地理块重新分组或合并这 6 个产品，不能直接沿用官方 train/val 边界做模型选择。

## 4. 官方 NAC crater（工程备用集）

| 指标 | 数值 |
|---|---|
| 仓库列出 | 426 文件 / 104,997,349 B（100.13 MiB） |
| 选择并落盘 | 418 payload 文件 / 85,296,946 B（81.35 MiB） |
| `.npy` 图像 | 408（256×256，uint8） |
| 协议 | standard 317 / large 91 |
| annotations 文件 | `annotations_min5px_{train,val,test}.json` |
| train standard | 219 Tile，29,033 boxes，median 114/tile（min 15，max 447） |
| val standard | 52 Tile，5,830 boxes，median 97/tile |
| test standard | 46 Tile，7,061 boxes，median 141/tile |
| 标注引用但磁盘缺失的图像 | 0 |
| site 重叠 | train∩test 4，train∩val 4，val∩test 2 → **不是 site-OOD split** |

## 5. 汇总

| 数据 | listed | downloaded | readable | 几何对齐 | 独立源组 |
|---|---|---|---|---|---|
| HF sample | 83 文件 / 143 MiB | 32 / 6.64 MiB | 24/24 NetCDF | 6/6 | 6 PRODUCT_ID |
| S3 100k NAC | 90,786 索引行 | 12 Tile（48 文件） | 12/12 三元组 | 12/12 | 1,090 PRODUCT_ID / 80 zones |
| Crater | 426 / 100 MiB | 418 / 81 MiB | 408 npy | N/A | 4 个 test site（与 train 重叠） |

## 6. 必须保留的限制

- Parquet 是索引，不是文件本体；只有落盘计数才算数据。
- HF 6 Tile 不得用于训练或泛化声明。
- 每 Tile 的 `DTM_source_ID` 未在索引中提供；已知来源族为 USGS NAC-stereo DTM，manifest 中该字段保持 Null。
- slope 文件没有 `units` 属性；单位由官方 README 声明 + DTM 重算经验验证（12/12），仍需在 P1 的 manifest 中显式写入 provenance。
- 本审计只验证网格边界与 DTM 重算一致性，不替代 P1 的逐像素 footprint 重投影与视觉/几何配准流程。
