# 数据协议与评价定义

## 官方数据入口

- SomBench 预训练：https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-pretraining-data
- Crater 数据：https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-NAC-Crater-Detection
- LFM 实现：https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model

SomBench NAC 目录包含 `NAC_HighRes.parquet` 及每 Tile 的 `nac/`, `dtm_3m/`, `slope_3m/` 等 NetCDF（如可实际访问）；目录/Viewer 行数不能证明原始资源均可下载。官方 NAC 轨单张 512×512，拟裁剪为 256×256 时必须使用地理 footprint 映射，而不是切同名数组的中心。

Crater 数据按 35 个标注框分配官方 train/val/test，408 Tile，`standard` 标注较完整，`large` 仅 ≥10m。脚本当前只使用 standard 与 min5px 标注，应在真实数据运行时核实。官方面向检测的 split 不是站点 OOD split。

## 最小 Manifest 字段

`sample_id, repo_revision, image_path, dtm3m_path, slope3m_path, coarse_dtm_path, NAC_EDR_ID, DTM_source_ID, LTM_CODE, footprint_wkt, crs_wkt, gsd, sha256, modality_read_ok, valid_fraction, split_group, co_registration_status`

无法证实的字段应保持 Null，不能猜测。清单应分别统计索引行、可下载文件、可解析文件、同域有效三元组及**地理独立源组**。

## 主任务真实标签

对同一地理 ROI 上可用的原生 3m slope 有效像素集合 V：

- `g(theta) = count(slope > theta)/|V|`，初始阈值 θ=10°,15°；坡度必须核对为角度单位。
- `unsafe = 1[g(10°)>0.05]` 为**预先固定的代理工程规则**，并非认证的真实着陆危险标签。
- 现有脚本的 10 个坡度阈值形成 11 个经验分桶；进入多项分布/概率模型前必须固定维度与闭开区间。分桶经验统计可用于建模，但原始真值不做零值平滑、对数比平滑、伪计数更改。
- 有效像素率及来源 DTM 质量必须记录；若缺真实配准，则只作为 NetCDF 物理统计演示，不能作为监督训练对。

## 研究评估口径

`Coverage = accepted_n/test_n`；`UAR = unsafe_accepted_n/accepted_n`（accepted_n=0 时为未定义）；`HMR = unsafe_accepted_n/unsafe_total`（unsafe_total=0 时为未定义）；事件概率 Brier/NLL，物理属性 MAE。

比较 UAR 必须固定 Coverage，并写清实际放行数量和危险数量。地理块/DTM 源做 Bootstrap 的统计单位；不同规则评估样本不是额外独立地理样本。不能对未经验证的空间 OOD 宣称具有分布无关的形式化校准保证。
