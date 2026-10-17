# P1.9 Gate 2 — 原生 DTM 文件级 I/O 与物理标签探针

**状态：部分 PASS**（DTM/label 侧 PASS；光学 NAC 供给 BLOCKED）· 日期 2026-10-17 · 代码 `main@fa0719a`（本报告随附 `scripts/probe_p19_io.py`）
脚本 `scripts/probe_p19_io.py`，离线测试 `tests/test_probe_p19_io.py`（13 项）
产物 `outputs/p19_io_probe.json`（sha256 `36766abc98668aeb5575120918d9ba226b2b46d652741a8fcf5f27110a1c138d`）
数据目录 `data/probe_p19/`（仅 3 景，合计 242 MB）

## 探针景的选择

从 Gate 1 的 47 个「来源已验证但与已用 Tile <10 km」的景中选 3 个：这些景**永远不能进入冻结 V2**，
所以可以安全地做文件级探针，不存在污染测试集的风险。

| 景 | 原生 post spacing | 投影 | 纬度带 | 下载字节 |
|---|---|---|---|---|
| `NAC_DTM_ESALL_SR12` | 2.0 m | POLARSTEREOGRAPHIC | −89.84 … −89.63 | 25 020 674 |
| `NAC_DTM_LASELMASIF2` | 2.0 m | EQUIRECTANGULAR | −15.48 … −14.54 | 126 708 791 |
| `NAC_DTM_NECTRIMII1` | 5.0 m | EQUIRECTANGULAR | −12.98 … −11.34 | 90 760 375 |

产品级 SHA256（`.TIF` / `.LBL`）：

| 景 | .TIF | .LBL |
|---|---|---|
| ESALL_SR12 | `52d35898a61991bb13bd463156bd95a8c01be6806247247e891752f56fbb425c` | `2852e4bc1bdbf7ee568cae6bc0a0869f51679cb2ed29e252aa6cacdc28bedb49` |
| LASELMASIF2 | `25d1bd7cf0a7621b69fe0dfc3c32e8de6af6f5a3580ed7cc49761d54c9b16511` | `89d10e8caa75e150dd097cef433f22d3e016d07d000294741842c77761e56a6d` |
| NECTRIMII1 | `18dbe9018dc17ab33f21c360180525fccac0cf1ed6079bacf11b9847a45f66c9` | `6be1cbc527905a2ba8965ac521b03e872a33a20a97ed37b49789f5fe5fc8431a` |

## I/O 一致性（真实文件核对，非假设）

| 检查 | 结果 |
|---|---|
| label `LINES`/`LINE_SAMPLES` vs GeoTIFF shape | 3/3 一致（2500²、14366×2203、9943×2280） |
| label `SAMPLE_TYPE=PC_REAL` / `SAMPLE_BITS=32` vs 文件 dtype | 3/3 一致（float32；classic TIFF） |
| `^IMAGE` 指针与文件字节布局 | **指针是 1 基**：`size − (offset−1)` 与 `lines×samples×4` 三种景全部**精确相等** |
| 原始 float32 视图 vs GeoTIFF 解码数组 | 3/3 **逐位相同**（`raw_offset_view_identical_to_tiff = true`） |
| 月球半径 / 投影 | label `A_AXIS_RADIUS = 1737.4 km`；GeoTIFF `GeogSemiMajorAxisGeoKey = 1737400` ✓ |
| 极区投影参数 | ESALL_SR12 GeoTIFF `ProjNatOriginLatGeoKey = −90`、`ProjScaleAtNatOriginGeoKey = 1.0` ✓ 与 label 一致 |
| 经度域 | label 经度 0–360（`350.87…351.02`、`180.82…226.62`），与 shapefile 的 0–360 一致（`−9.05 ≡ 350.95`） ✓ |

### 三个必须在后续流程中显式处理的坑

1. **PDS 指针 1 基**：按 0 基读取会整体偏移 1 字节，float32 全盘错位。已写进 `layout_check()` 并用真实文件逐位验证。
2. **真实 label 没有 `NoDATA` 关键字**：哨兵值只写在 `IMAGE.DESCRIPTION` 散文里（`The NoDATA value is -3.40282266e+038 with a missing constant of -1e+32`，跨 3 个物理行，需先做续行拼接）。探针从散文恢复（`nodata_source = IMAGE.DESCRIPTION`），并以 GeoTIFF `GDAL_NODATA` 作为回退。
3. **GDAL_NODATA 的 ASCII 标签对 float32 不可转换**：`tifffile` 解析该 tag 时会抛 `ValueError('… is not castable to float32')` 并降级为字符串；任何只信 GeoTIFF tag、不做回退的读取器都可能拿到 `None` 而静默把 −3.4e38 当数据。
4. **等距圆柱产品的 GeoKeys 不完整**：LASELMASIF2 / NECTRIMII1 的 GeoKey 只有 `GeographicTypeGeoKey=32767`（user-defined）+ 半径，没有 `ProjCoordTransGeoKey`；投影方法与像素尺度只能取自 PDS label（`MAP_PROJECTION_TYPE`、`MAP_SCALE`）。**不能用 GeoKeys 唯一确定 CRS。**

## 物理标签统计（原始面积比例，未做任何平滑/伪计数）

有效像元 = 排除 NoDATA(−3.40282266e38) 与 missing(−1e32) 后的像元；坡度用中心差分（Horn 之外的最简稳健式），只统计 3×3 邻域全有效的内部像元。

| 景 | 原生 spacing | 有效像元比例 | g10 | g15 | 坡度 p50 / p90 | `unsafe = 1[g10>0.05]` |
|---|---|---|---|---|---|---|
| ESALL_SR12 | 2.0 m | 0.6901 | 0.9042 | 0.6833 | 18.86° / 32.24° | true |
| LASELMASIF2 | 2.0 m | 0.9384 | 0.3695 | 0.2242 | 6.95° / 21.82° | true |
| NECTRIMII1 | 5.0 m | 0.7463 | 0.4801 | 0.2134 | 9.68° / 18.29° | true |

### 尺度敏感性（同一 DTM 的块平均，不是独立粗分辨率真值）

| 景 | 原生 | ×2 | ×4 | ×8 |
|---|---|---|---|---|
| ESALL_SR12 g10 | 0.9042 @2 m | 0.9065 @4 m | 0.9118 @8 m | 0.9239 @16 m |
| LASELMASIF2 g10 | 0.3695 @2 m | 0.3707 @4 m | 0.3677 @8 m | 0.3646 @16 m |
| NECTRIMII1 g10 | 0.4801 @5 m | 0.4787 @10 m | 0.4748 @20 m | 0.4686 @40 m |

- 这 3 景的 `g10` 在 2–8 倍块平均下漂移 ≤ ~2%（相对），`g15` 漂移 ≤ ~5%。
- **这不能当作「2–7 m 与 3 m 等价」的证据**：块平均只是同一观测的低通聚合，真实尺度差异来自不同立体像对/不同后处理。是否接受 (A) 仅原生 ≈3 m 景 或 (B) 重新定义共同尺度，仍必须由研究人员在观测冻结集成绩之前决定（`docs/P1_9_HELDOUT_BUILD.md` Gate 2 第 4 条）。
- 观察到的事实：即使 5 m 原生景（NECTRIMII1）`g10 ≈ 0.48` 也远超 0.05 阈值，即这些候选景大多是**粗糙地形**，`unsafe` 代理规则在此类地貌上几乎不退化为平衡分类问题。

## 光学 NAC 影像：BLOCKED（已量化）

**方法（含一次方法修正）**：光学文件不在 DTM 自身的 `view_rdr_product` 页面里，而是以**相对链接的兄弟产品**出现在场景页
`https://data.lroc.im-ldi.com/lroc/view_rdr/<DTM>`。第一版脚本只解析产品页，得到「0/128」的假阴性；
`scripts/audit_lroc_optical.py` 现已改为「场景页取兄弟产品 ID → 分类（optical/derived/quad）→ 对每个光学候选做 HEAD 验证文件真实存在」，
并对每个产品页也解析直接数据链接。产物：`outputs/p19_optical_audit_all.{csv,json}`
（sha256 `06b4eaaa029574767d721b2e7029b850bc304506b0928959df5f152dd7dcddfb` /
`c7ce3e22cb04a9aacc1de0fc6a0880e10658ebeb86829c05b65b9d74eb66957e`）。

**结果（181 景全量）**

- 181/181 产品页可访问、181/181 发布 DTM 产品；**8 景发布 HEAD 验证过的 NAC 正射镶嵌图**。
- **通过 Gate 1 双项筛选的 128 景中，0 景发布任何光学影像。**
- 8 景的构成是决定性的：

| 景 | 源清单 | 净空 | 光学文件（HEAD 实测字节） |
|---|---|---|---|
| `ALPHONSUS` | OK | **9.1601 km（未达 10 km 门槛）** | `MOSAIC_100CM` 842 MB、`MOSAIC_5M` 33 MB |
| `FRESHMELT` | README 空节 | 110.0 km | `MOSAIC_2M` 186 MB、`MOSAIC_60CM` 2.06 GB |
| `MARIUS` | README 空节 | 32.5 km | `MOSAIC_2M` 217 MB、`MOSAIC_50CM` 3.47 GB |
| `MRINGENII` | README 空节 | 28.6 km | `MOSAIC_2M` 400 MB、`MOSAIC_70CM` 3.26 GB |
| `NRTHCRTRI` | README 空节 | 232.5 km | `MOSAIC_200CM` 1.16 GB、`MOSAIC_6M` 128 MB |
| `SPARIM` | README 空节 | 185.7 km | `MOSAIC_2M` 337 MB、`MOSAIC_60CM` 3.75 GB |
| `SULPICIUS` | README 空节 | 137.9 km | `MOSAIC_2M` 155 MB、`MOSAIC_60CM` 1.73 GB |
| `VIRTANEN` | README 空节 | 0.0 km（重叠） | `MOSAIC_2M` 194 MB、`MOSAIC_70CM` 1.59 GB |

  即在可达公开镜像上，**「有光学影像」与「源级可验证 + 空间独立」几乎互斥**：唯一源可验证的光学景 `ALPHONSUS` 净空差 0.84 km；
  其余 7 景正是 Gate 1 中被 README 空源清单排除的 mosaic 产品。

**原始 NAC EDR 卷不可达（逐条记录）**

- JPL 卷索引（`https://pds-imaging.jpl.nasa.gov/volumes/lro.html`）给出两个权威入口：
  `https://pdsimage2.wr.usgs.gov/Lunar_Reconnaissance_Orbiter/LROC/EDR/LROLRC_0005/` 与
  `https://lroc.sese.asu.edu/data/LRO-L-LROC-2-EDR-V1.0/LROLRC_0005/`；两者在当前网络下分别返回 404 与 301→
  `pds.mcp.nasa.gov` 的 JS 目录浏览器（无静态列表、无可用 API），因此**无法构造出可下载的 EDR 文件 URL**。
- 另外尝试并失败的路径族：`pds.mcp.nasa.gov/.../pds4/lroc/lro-l-lroc-2-edr[-v1.0]/…`（含 `LROLRC_0001/0005/2001` 卷号与大小写变体）、
  `pds-geosciences.wustl.edu/lro/…`、`data.lroc.im-ldi.com/lroc/view_rdr/NAC_EDR_M139872173`。
- 明确禁止：不得用 `_SHADE.TIF` / `_SLOPE` / `*.PYR.TIF` 冒充独立光学输入（均由被测 DTM 派生）。

**对 Gate 3 的直接后果**：按计划「不足样本则返回 `BLOCKED_NEW_HELDOUT`，不降低独立条件填满数量」，
当前公开来源下**无法构造 30–40 景的严格外部冻结集**（可用的源可验证光学景数 = 0）。

## 独立粗地形（SLDEM2015）：可用性已确认，内容读取留到 Gate 3

- 入口 `https://imbrium.mit.edu/DATA/SLDEM2015/`，`TILES/{JP2,FLOAT_IMG}/` + README。
- 覆盖 ±60°，`SLDEM2015_512_<lat_band>_<lon_band>.JP2` 为 512 px/deg ≈ **59 m/px**（另有 256 px/deg ≈ 118 m/px 与带 label 的 `FLOAT_IMG`）。
- 覆盖 LASELMASIF2（lat −15.0, lon 350.9）的瓦片：`SLDEM2015_512_30S_00S_315_360.JP2`，`Content-Length = 175 131 646`，前 32 字节为合法 JP2 幻数（`jP ` + `ftypjp2`）✓，容器内 PIL 具备 JPEG2000 读取能力。
- 本探针**未下载**该瓦片（167 MB）；内容级读取与拼接留待 Gate 3 构造冻结包时进行——此处只声明「可获取」，不声明「已读通」。

## 可用样本数预估

**不予给出**：光学输入供给未解决、逐景有效像元掩膜未与冻结几何对齐、且标签尺度政策未定。
按现有证据只能说明：3 个探针景的原生有效像元比例在 0.69–0.94 之间，最大单景 14366×2203 像元（LASELMASIF2）。

## 复现

```bash
# 下载（需要出网代理；只取 .TIF 与 .LBL）
B=https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/DATA/SDP/NAC_DTM
for n in ESALL_SR12 LASELMASIF2 NECTRIMII1; do
  curl -s -o data/probe_p19/NAC_DTM_$n.TIF "$B/$n/NAC_DTM_$n.TIF"
  curl -s -o data/probe_p19/NAC_DTM_$n.LBL "$B/$n/NAC_DTM_$n.LBL"
done
python3 scripts/probe_p19_io.py --scales 1 2 4 8      # 6.4 s
python3 -m pytest tests/test_probe_p19_io.py -q       # 13 passed
```

## 下一步（需要明确决策）

按计划口径，Gate 3 当前为 **`BLOCKED_NEW_HELDOUT`**：可用的「源可验证 + 空间独立」光学景数为 0，不允许为凑数放宽独立性。可行路线：

1. **补齐光学来源（保持严格口径）**：在具备机构网络/白名单的环境访问 LROC EDR 卷（权威入口见上），
   或用 ISIS + SPICE 自行对 EDR 做正射校正。代价：NAC EDR ≈ 1 GB/景（左右相机成对）→ 30–40 景约 100–200 GB 下载与一条正射处理链。
2. **放宽「README 源清单」这一来源要求**：使用 7 个有光学镶嵌图的 mosaic 景（其 README 该节为空、DBF 清单被截断），
   改用其它 provenance 锚点（DTM label/PDS4 元数据）——**需要用户明确批准**，因为计划禁止降低独立条件。
3. **改变任务定义**：放弃 NAC 光学输入，改为「仅有粗地形 → 3m 坡度分布」的地形到地形任务——同样需要用户批准，且不与现有 SomBench 输入定义可比。
4. 与此独立的一项待决输入：尺度政策 (A) 仅原生 ≈3 m 景（84 景可用）/ (B) 定义共同可实现的尺度并在开发集重算标签。
