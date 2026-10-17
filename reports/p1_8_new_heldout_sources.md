# P1.8 / P18-05b — 严格独立测试语料调研

日期：2026-10-17；代码状态 `a53937e` + 本轮 P18-04a。
脚本：`scripts/audit_heldout_sources.py`、`scripts/audit_lroc_dtm.py`（均只读索引/元数据，不下载影像、不读模型指标）。
产物：`outputs/p18_source_audit.json`、`outputs/p18_lroc_dtm_audit.json`。

## 1. 结论

| 候选路径 | 状态 | 证据 |
|---|---|---|
| SomBench 100k release | `PRODUCT_ONLY` | 272/272 泄漏组已被 760 Tile 触及（`reports/p1_8_heldout_protocol.md`） |
| **SomBench 1M release** | **`PRODUCT_ONLY`** | 999,808 个含 slope_3m 的 Tile、1,095 产品、**268 组**；**268/268 全部被触及**，仅比 100k 多 5 个产品 |
| **LROC NAC DTM 官方档案**（PDS） | **`GO_STRICT_OOD`** | 662 个 DTM 中 **188 个**仅使用 SomBench 未用过的 NAC 产品；其中 **143 个**同时满足空间隔离 |
| Kaguya TC DTM（USGS STAC） | 备选（不同标签定义） | 155,925 个 DTM 条目，约 10 m 分辨率 → 可用于方法级独立检验，不能生成 3 m 标签 |

**严格 OOD 的可行路线已找到**：LROC NAC DTM 档案（PDS，NASA 公共数据）中有 143 个 DTM 同时满足
"立体源 NAC 产品未参与 SomBench" + "与 760 个已用 Tile 空间隔离 ≥10 km（保守半径界）"。
该档案的 DBF 字段 `images` 直接给出每景 DTM 的立体源影像 ID，**正好补上本语料缺失的 `DTM_source_ID`**。

## 2. SomBench 1M release 审计（否定证据）

```bash
python3 scripts/audit_heldout_sources.py
```

| 项 | 值 |
|---|---|
| 1M 索引行（含 slope_3m） | 999,808 |
| NAC 产品 | 1,095（100k 为 1,090） |
| LTM zone | 80 |
| 按 5 km 块 + 产品并查集得到的组 | **268** |
| 被 760 个已用 Tile 触及的组 | **268（全部）** |
| 未被触及的组 / Tile | **0 / 0** |
| 100k 索引中存在于 1M 的 Tile | 90,481 / 90,786（305 个缺 slope_3m 的 Tile 不在 1M 过滤后集合中） |
| 1M 中未出现在 100k 的产品 | 5 |

结论：1M release 更大，但**没有提供任何未被触及的连通组**。原因：760 Tile 的组内轮转选样本就覆盖了
全部 272 组，而 1M 只是把同样的区域采样得更密（组结构反而合并为 268）。**不能通过重新划分或改名创造"新"组。**

## 3. LROC NAC DTM 档案审计（肯定证据）

来源（公开、无需认证）：

- 全量清单：`https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/SHAPEFILE/NAC_DTMS/NAC_DTMS_360.ZIP`
  （4.3 MB，shapefile + DBF + README；`NAC_DTMS_180.ZIP` 为 180° 经度域版本）
- 单景产品：`https://data.lroc.im-ldi.com/lroc/view_rdr/NAC_DTM_<NAME>`，
  PDS 云镜像 `https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/DATA/SDP/NAC_DTM/`
- README（单景）示例字段：`LROC NAC SOURCE IMAGES M1136306535 L/R` —— **立体源 ID 可直接获得**

DBF 字段：`DTM_NAME, sitename, dtm_vers, images, resolution, center_lon, center_lat, triang_rms,
completion, cov_sqkm, num_stereo, url, features` 等。

| 指标 | 值 |
|---|---|
| DTM 总数（2026-09 版清单） | **662**（2020 年 LPSC 摘要为 474，档案仍在增长） |
| 立体源产品完全未被 SomBench 使用 | **188** |
| 至少一个源产品未使用 | 661（仅 1 景完全落在 SomBench 产品内） |
| 未使用的 NAC 产品数（跨所有 DTM） | 792（SomBench 用了 1,090 个产品中的 420 个出现在 DTM 清单里） |
| 188 景的总覆盖 | **68,126 km²** |
| 分辨率 | 2–7 m（与 SomBench 3 m 网格同量级） |
| 完成时间 | 2010-07-26 – 2026-08-12 |
| 空间隔离后剩余 | **143 景**（保守判据：中心距 ≥ 覆盖半径 + 10 km，距 760 个已用 Tile） |

样例（按覆盖面积排序）：`NRTHCRTRI`（6 m，1,522.9 km²，2012）、`NRTHCRTRIII`（5 m，1,373.8 km²）、
`LAPLACEA`（5 m，1,313.0 km²）、`FRIGORIS`（7 m，1,306.0 km²）、`CHAPPELLT`（5 m，1,253.5 km²）。

`scripts/audit_lroc_dtm.py` 的判定：`verdict = GO_STRICT_OOD`。

## 4. Kaguya TC DTM（备选，不同标签）

- USGS STAC `kaguya_terrain_camera_usgs_dtms_v2`：**155,925** 个条目；`..._usgs_dtms`（v1）：130,799；
  立体观测 220,676。公开（AWS Open Data / USGS），资产含 `laz`、`orthoimage_left` 等。
- 分辨率约 10 m（TC），与 3 m NAC 任务不同：**不能**作为同一 3 m 标签的独立测试集；
  可作为"方法在独立仪器上的可迁移性"检验（标签需重新定义为 10 m 坡度超限率）。
- 若采用，必须在报告中明确：这是**不同标签定义**的独立验证，不是同一任务的 OOD 成绩。

## 5. 构建严格 `heldout_v2` 的下一步（尚未执行）

1. 从 143 景中按类别/纬度/分辨率分层选 **30–40 景**（建议每景 2–4 个 512 m Tile → 60–160 Tile）。
2. 下载每景 PDS 产品（DTM + 对应 NAC 影像，README 给出源影像 ID）；对 DTM 做固定 3 m 重采样并生成
   `g10/g15` 标签（记录重采样方法与原始分辨率）。
3. **先冻结** ID 清单、脚本 SHA256、每景的 `DTM_source_ID` 与立体源影像 ID，再读取任何模型指标。
4. footprint 级投影检查（PDS 产品使用 Moon 2015 / LROC 投影，需与 SomBench 的 Transverse Mercator 参数核对）。
5. 许可：NASA PDS / LROC 数据为公开数据，仍应在报告中引用 LROC/PDS 数据政策并记录获取日期与版本。
6. 冻结后**一次性**评估（候选模型/超参/规则先固定），不得据结果换站点。

## 6. 限制

- 本次只审计了索引/元数据（1M parquet 190 MB、DTM shapefile 4.3 MB），**未下载**任何 DTM 产品或 NAC 影像。
- 143 景的"空间隔离"用中心距 + 覆盖半径的保守界，不等于多边形级 footprint 检查。
- `DTM_source_ID` 在产品级已可确定，但**逐 Tile 的投影/配准**仍需人工核验。
- 这些 DTM 的原始分辨率 2–7 m 与 SomBench 3 m 不完全一致；标签定义需在冻结协议中写清楚。
- 未使用付费资源；未触碰已冻结的 `protocols/heldout_v2_product_level_*` 文件与其 SHA256。


## 7. Review 勘误：历史 143 景是元数据候选，不是已证明严格 OOD（2026-10-17）

本节是**追加勘误**，保留上方原始调查数字以便追踪，不覆写历史报告。旧版
`scripts/audit_lroc_dtm.py` 用 `sqrt(cov_sqkm / pi)` 作为 DTM 的所谓外接半径；
**等面积圆半径并非外接半径**，对狭长形状尤其可能低估占地范围。
因此原 `143` 景只是该近似筛选的候选数量，原 `GO_STRICT_OOD` **撤回**。

另一个代码缺陷是 `len(products_unseen) == len(products)` 会在 `products=[]`
时成立，导致缺少 DBF `images` 源 ID 的记录错误地被认作全部未见过。现在增加
非空、可解析数字源 ID 的必要条件，统一正规化左右相机后缀与 SomBench ID。

新的元数据审计使用 shapefile 所列多边形顶点到 DTM 中心的最大月心弦距代替等面积圆估算，
输出 `provisional_candidates_after_vertex_screen`，但**仍不构成完整 footprint
独立性证明**（边界曲线、CRS、已经使用的 Tile footprint、两景立体源完整性均待确认）。
输出 `strict_candidates_after_spatial=null` 且不会提前返回 `GO_STRICT_OOD`。
**新审计需要在 RX 7900 XT 机器使用原 Shapefile 与 labels 重新运行后，才有新的候选数量。**
不要沿用旧的 143 景作为严格合格数量。

执行门槛：逐景 PDS README 核查全部立体源产品，使用 DTM 与历史 Tile 的真实 footprint
在一致月球坐标参考系计算至少 10 km 隔离，确认原生 DTM 分辨率和输出坡度标签的物理可比性；
**冻结**候选清单及 SHA256 后，再进行一次性独立评估。
详见 `docs/P1_9_HELDOUT_BUILD.md`。旧 64 Tile 产品级冻结文件及哈希原封不动保留。
