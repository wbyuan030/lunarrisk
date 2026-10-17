# P1.9 Gate 0 — LROC NAC DTM 索引元数据审计（修复后重跑）

**状态：PASS** · 日期 2026-10-17 · 代码 `main@fa0719a6cdf5b64caca3add487dec9db41cce63b`
脚本 `scripts/audit_lroc_dtm.py`，测试 `tests/test_audit_lroc_dtm.py`（12 项）
产物 `outputs/p19_lroc_metadata_audit.json`
（sha256 `b39111570ea44778f37c013d447aa296863e633c99a99e8d6681e5ae0ae2583d`）

## 输入与哈希

| 输入 | 路径 | sha256 |
|---|---|---|
| DTM 形状索引 | `data/lroc_nac_dtm/shape/NAC_DTMS_360.SHP` | `0868eba67af55d6f24a0086c5f7b9a6b5e382d54a9924c6ac9347e8eb130b3db` |
| 属性表 | `.../NAC_DTMS_360.DBF` | `f469ff9311fc1935492c848be1ba6ab33e78124fdd320ee2091ff7d2d5114609` |
| 索引 | `.../NAC_DTMS_360.SHX` | `4d07d491a6c53d03d15f6c6581bf9fa6f0e03cfa7cd08a544e1a470b30d3ebb3` |
| 投影 | `.../NAC_DTMS_360.PRJ` | `a647814ae7ee7fe2d16a56483105e2a05e8bad386761a6caa2e61eea12aeaf2c` |
| 100k manifest（旧，冻结引用版本） | `outputs/manifest_100k_final.parquet` | `b2e3841be1dae4005e6977e27ffbec970691d62bc5eecfa2b202f22b6752f6cd` |
| 760 已用 Tile 标签 | `outputs/labels_hydrated.csv` | `eb2ff9326cd3d3180409355b53b2cf135c02ddd324cec87e583686abccdaafb5` |

重跑命令（容器 `rocm-lab`，工作目录 `/data/lunarrisk`）：

```bash
python3 scripts/audit_lroc_dtm.py --out outputs/p19_lroc_metadata_audit.json
```

## 结果

| 量 | 值 |
|---|---|
| DBF 记录总数 | 662 |
| `images` 字段宽度 | 254（定长；`len(raw) >= 254` 判为截断） |
| 源 ID 状态 | **OK 655 / DBF_TRUNCATED 7 / EMPTY 0 / MALFORMED_TOKEN 0** |
| fail-closed 排除记录 | 7（全部为定长截断，见下表） |
| SomBench 归一化产品集 | 832 个 |
| 全部源产品均未见过的 DTM | **181**（本次唯一进入候选集的类别） |
| 至少一个源未见 | 640 |
| 全部源均以见过 | 22 |
| 未见过产品并集 / 已见过产品并集 | 771 / 441 |
| 顶点筛选（provisional） | 113 |
| `strict_candidates_after_spatial` | `null`（不声称） |
| 候选分辨率范围 / 覆盖 | 2.0–7.0 m / 65 887.77 km² |
| 候选完成时间范围 | 2010-07-26 – 2026-08-12 |
| verdict | `CANDIDATES_REQUIRE_POLYGON_AND_SOURCE_QA`（元数据层结论，后续由 Gate 1 取代） |

fail-closed 的 7 条截断记录（原始 `images` 长度恰为 254，末条 ID 被切断）：

`GRUITHUI_2M`、`GRUITHUISEN`、`LICHTENBERG`、`HPONDS`、`RUMKERDOME`、`APOLLO17`、`GIORDNBRNO`

## 本轮修复（相对 P1.8 旧审计）

1. **9 位早期 NAC EDR ID 合法**（2009–2010 年产品，如 `M189321511`）。旧正则只接受 10 位，把 6 条记录误判为畸形。
2. **两字母相机/级别后缀合法**（`LE`/`RE`）。DBF 实际存在 `M1374407232LE, M1374421274LE`（`CHANGE501`）与 `M139939938LE, M139946735RE`（`MALAPERT01`）；旧正则只允许单个后缀字母，与其自身注释（L, R, LE, RE）矛盾，导致这 2 条被误排除。
   官方文档亦确认该约定：LROC NAC 处理指南写明「PDS 产品在影像 ID 末尾追加 `E`（EDR）或 `C`（CDR）」
   （`https://lroc.im-ldi.com/data/support/downloads/LROC_NAC_Processing_Guide.pdf`），因此 `…LE`/`…RE` = 左/右相机 + EDR。
   - 修正后 `MALFORMED_TOKEN` 由 2 → 0，`OK` 由 653 → 655。
   - 候选数仍为 181：这 2 条的源产品（`M1374407232`、`M1374421274`、`M139939938`、`M139946735`）都已出现在 SomBench 832 个产品中，因此落入「至少一个源已见过」而非「全部未见」。
   - 全量扫描 662 条记录的 1 717 个源 token：9 位 775 个、10 位 935 个；后缀分布 `""` 1711、`LE` 3、`R` 2、`RE` 1；7 个异常 token 全部来自上表截断记录的断尾。
3. **空/畸形源一律 fail-closed 排除**，不猜测补足；`images` 是定长字段，满宽即视为截断，不做「可能完整」的假设。

## 与历史「143」数字的差异

`docs/P1_9_HELDOUT_BUILD.md` 记录的历史调查为「188 个源 ID 候选、143 个圆形空间筛选结果」。差异来自三处，均为方法/规则修正，不是数据变化：

1. 源解析修正（9 位 ID + LE/RE 后缀）改变候选边界：产品级「全部未见」从含误判的集合收敛为 181。
2. 旧空间筛选用 `sqrt(area/pi)` 等面积圆半径当 footprint 上界，该假设不成立（DTM 多边形多为狭长/多部件），已由顶点外接圆筛选取代。
3. 顶点筛选（中心–已用 Tile 弦距 ≥ 最大列示顶点弦半径 + 10 km）得到 **113 个 provisional**，且明确不作为严格独立证明：`strict_candidates_after_spatial = null`。

## 交叉验证：DBF `images` vs PDS README（全量，非抽样）

Gate 1 对 181 个候选逐一抓取官方 PDS README（缓存 `data/lroc_nac_dtm/readmes/`，181 个文件，聚合 sha256 `ff6eed58a2fee3f741382dddb98125555ea788e50fdc4809b96763c98ac1df74`）：

- 174 条 README 含 `LROC NAC SOURCE IMAGES` 源清单；
- 这 174 条的 **归一化 README 源集合与 DBF 集合逐一完全相同**（0 例 README 增加、0 例减少），且与 SomBench 832 产品交集为 **0**；
- 其余 7 条为 mosaic 产品，README 该节为空 → 按 `UNKNOWN_SOURCE` 排除（见 `reports/p1_9_footprint_audit.md`）。

即：对全部可解析记录，DBF 初步源清单未被 README 推翻；被推翻的只是「定长截断仍可当完整清单」这一旧假设。

## 局限

- 本次只做索引/属性/README 审计，**未下载任何 DEM 或 NAC 影像**；候选的实际有效数据范围、原生分辨率与标签尺度仍需 Gate 2 的文件级探针。
- `images` 为定长 254 字符，长 mosaic 源清单必然截断；任何后续流程都必须以 README 为准（本仓库已如此实现）。
- 顶点筛选只是元数据层近似，不得当作几何独立性的证明；真正的多边形距离见 Gate 1。
