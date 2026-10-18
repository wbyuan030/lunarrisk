# P1.9 EDR → ISIS 正射校正运行手册（Gate 2 光学通道收尾）

> **2026-10-18 新的更直接入口**：请先使用 [P1.9 EDR URL Discovery](P1_9_EDR_URL_DISCOVERY.md) 和
> `scripts/resolve_lroc_edr_urls.py`。LROC 官方 NAC 分期观测 Shapefile 直接记录
> `PRODUCT_ID / PDS_VOLUME / FILE_SPECI / URL`，无需再盲猜卷目录。索引 URL 可能过期，
> 先核查少数 IMG HEAD 与文件字节，不要直接批量下载。本页第 1.0 节保留为早期失败路线的历史记录。
> **修正**：ISIS 的 NAC 导入程序是 `lronac2isis`（原误写 `lrolc2isis`）。
>

**适用决定**：`protocols/p19_external_policy.json` 中的 `optical_route = INSTITUTIONAL_EDR_ISIS`
（用户 2026-10-17 选定：在机构网络取原始 NAC EDR 卷，用 ISIS/SPICE 自行正射校正，保持严格来源与空间独立性）。

**为什么要这么做**：在本机可达的公开镜像上，181 个候选景中只有 8 景发布正射镶嵌图，且其中 7 景的 README 源清单为空；
**通过 Gate 1 双项筛选的 128 景里 0 景有光学影像**（证据见 `reports/p1_9_io_label_probe.md`）。
原始 NAC EDR 卷在本机网络不可达（权威入口 404 / 仅 JS 浏览器），因此这一步必须在机构网络内执行。

**不做的事**：不得使用任何 DTM 派生品（`_SHADE.TIF`、`_SLOPE.TIF`、`*.PYR.TIF`、`*_MOSAIC_*` 中由 DTM 渲染的图层）作为光学输入；
不得在看到任何冻结集成绩之后再挑选景或调参。

## 0. 输入清单

- 待取 EDR 清单：`outputs/p19_edr_acquisition_list.csv`（84 景 / **168 条立体 EDR**，逐条含左右相机需求）。
- 逐景 DTM 产品（已在本机取得或可再取）：`NAC_DTM_<NAME>.TIF` + `.LBL`，路径
  `https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/DATA/SDP/NAC_DTM/<NAME>/`。
- 每个 EDR 的官方 PDS label（`.LBL`，体积小）应与 `.IMG` 一同下载并计算 SHA256。

## 1. EDR 卷定位与下载（机构网络）

### 1.0 本机实测的入口矩阵（先看这里，别重跑探测）

结论：**EDR 数据是公开、免认证的；卡点是目录/URL 发现，不是权限。** 本机（headless + 代理）实测：

| 入口 | 实测 | 说明 |
|---|---|---|
| `pds.mcp.nasa.gov/.../pds4/lroc/lro-l-lroc-5-rdr/…` | **200** | RDR/5 bundle；3 景 DTM（25/127/91 MB）已成功下载 |
| 同站点 EDR bundle（约 25 种组合：`lro-l-lroc-2-edr[-v1.0]` × 卷号 `LROLRC_0004/0005/0006/2001` × `DATA/SDP/NAC` × 文件名带/不带 `LE/RE` 后缀） | 全部 404 | 该镜像大概率只放了 RDR/5 |
| `pds.lroc.im-ldi.com/data/LRO-L-LROC-2-EDR-V1.0/` | 302 | 跳到 MCP 的 JS 目录浏览器（无静态列表、无公开 JSON 端点） |
| `lroc.sese.asu.edu/data/LRO-L-LROC-2-EDR-V1.0/LROLRC_0005/` | 301 | 同上 |
| `pdsimage2.wr.usgs.gov/Lunar_Reconnaissance_Orbiter/LROC/EDR/LROLRC_000{5,6}/…` | 404 | 站点根 200；JPL 卷索引里的这条链接已失效 |
| `pds-geosciences.wustl.edu/lro/` | 200（Apache 目录索引） | 只有 LOLA/DLRE/LEND/MRF 等 GEO 节点数据，**无 LROC** |
| `oderest.rsl.wustl.edu/live2/?query=product&…` | 200 | REST API 活着，但 `productid=M139872173[LE|RE]` 返回 `Count 0`（未索引 LROC EDR，或需 `pdsid`/`odeid`） |
| `data.lroc.im-ldi.com/lroc/search`（含 EDR 级过滤字段：入射/出射/曝光/太阳位置/轨道） | 200 | POST 表单 + CSRF token；脚本提交后返回同一页 → 前端 JS 驱动，需浏览器或它的 JSON 端点 |
| `pds-imaging.jpl.nasa.gov/volumes/lro.html` | 200（193 KB） | 给出 `LROLRC_XXXX` 卷及日期范围（可用），但外链已失效 |
| `imbrium.mit.edu/DATA/SLDEM2015/…` | 200 | 粗地形可用（167 MB JP2，容器内 PIL 可读） |

**给执行人的最快路径（5–10 分钟）**：在浏览器打开 MCP 的 EDR 目录浏览器
`https://pds.mcp.nasa.gov/data/index.html#/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/lro-l-lroc-2-edr/`，
或 LROC 门户 `https://data.lroc.im-ldi.com/lroc/search` 检索一个已知 EDR（如 `M139872173LE`），
把**实际可下载的文件 URL** 记下来。有了一个可用 URL 模板，168 条 EDR 的清单即可脚本化批量执行。

### 1.1 步骤

1. **ID → 卷**：LROC EDR 卷按时间切分（如 `LROLRC_0001`…）。卷索引页
   `https://pds-imaging.jpl.nasa.gov/volumes/lro.html` 给出每个卷的起止日期与在线入口；
   下载后优先使用卷内 `INDEX/` 下的索引表把 `<EDR_ID>` 映射到具体文件路径。
   本机实测：该页可访问，但其指向的 ASU/USGS 下载入口在本网络返回 404 或重定向到无静态列表的 JS 浏览器。
2. **卷内路径约定**：`<VOLUME>/DATA/SDP/NAC/<EDR_ID>.IMG` 与同名 `.LBL`（左/右相机各一份）。
   首次下载后请**实测并记录**单文件字节数（用于预算复核，本机无法预知）。
3. **归档**：把 `.IMG`/`.LBL` 落到自有存储，逐文件 `sha256sum`，并记入 `outputs/p19_edr_sha256.csv`
   （列：edr_id, camera, volume, bytes, sha256, downloaded_utc）。

## 2. ISIS 摄取与指向（逐景）

```bash
# 每次运行记录：ISIS 版本、SPICE 内核集合与哈希
isis_version=$(isis_version)   # 或 cat $ISISROOT/version
lronac2isis from=<EDR_ID>.IMG to=<EDR_ID>.cub
spiceinit   from=<EDR_ID>.cub                     # 建议显式给出内核集，禁止 web 模式以便复现
# 记录内核来源：catlab <EDR_ID>.cub 中的 Kernels 组（写入 p19_isis_provenance.json）
```

- `spiceinit` 之后的 `Kernels` 组即 SPICE 溯源凭证：内核文件名 + 加载顺序必须逐景留档。
- 建议在机构环境预先镜像 NAIF 的 LRO 内核（`lro-l-spice-*`），避免运行时依赖外网。

## 3. 正射到 DTM 网格

关键：**输出网格必须与 DTM 一致**（同投影、同 post spacing、同月球半径 R = 1737.4 km），
这样光学像元与坡度标签才能逐像元对齐。DTM 标签已给出投影参数（`MAP_PROJECTION_TYPE`、`MAP_SCALE`、
`A_AXIS_RADIUS`），极区产品为 `POLARSTEREOGRAPHIC`（`CENTER_LATITUDE = ±90`），中低纬为 `EQUIRECTANGULAR`；
**注意等距圆柱产品的 GeoTIFF GeoKeys 不完整，必须以标签为准**（本机已实测）。

```bash
# a) 用标签参数建立 map 模板（示例：极区产品）
maptemplate map=dtm_map.cub projection=polarstereographic \
  clat=-90 clon=0  \
  scale=<MAP_SCALE 米/像元>  \
  targopt=user targetname=MOON targdef=user a=1737400 b=1737400 c=1737400 \
  minlat=<MINIMUM_LATITUDE> maxlat=<MAXIMUM_LATITUDE> minlon=<WESTERNMOST_LONGITUDE> maxlon=<EASTERNMOST_LONGITUDE> \
  linc=2500 sinc=2500

# b) 正射
cam2map from=<EDR_ID>.cub map=dtm_map.cub to=<EDR_ID>_ortho.cub \
  pixres=map defaultrange=map     # 采样到 map 模板的网格
# c) 光度角（供几何标量：入射/出射/相位/太阳方位）
phocube  from=<EDR_ID>.cub to=<EDR_ID>_pho.cub
# d) 导出
isis2std from=<EDR_ID>_ortho.cub to=<EDR_ID>_ortho.cub.tif format=GTiff
```

- **NoData**：DTM 哨兵为 `-3.40282266e+38`（另有 `-1e32` 缺失常数，仅在标签描述文本中给出）；
  正射影像自身用 ISIS `NULL`，融合前请显式统一。
- 同一景的多条 EDR（左右相机、多对立体）在正射后按权重或选择单条最优（如入射角接近 0、无饱和）**拼接**，
  并在清单里记录用了哪一条：绝不允许把 DTM 派生阴影混入。

## 4. QA 门槛（未通过则该景不得进入冻结集）

| 检查 | 门槛（建议，需在冻结协议中最终固定） |
|---|---|
| DTM footprint 覆盖率 | ≥ 95% 的 DTM 有效像元有对应正射像素 |
| 正射有效像素率 | ≥ 0.60（与 SomBench 同量级；本机 3 个探针景为 0.69–0.94 的*DTM*口径，正射影像需单独测） |
| 与 DTM 配准 | 检查点/互相关偏移 ≤ 1 个 DTM 像元；记录偏移量 |
| 光度范围 | 非饱和、动态范围正常；记录 min/max 与饱和比例 |
| 光源一致性 | `phocube` 的入射角中位数与 NAC 立体的光照条件一致；记录 |
| 溯源 | 每个 cube 的 SPICE 内核清单 + ISIS 版本 + 输入 EDR SHA256 齐全 |
| 产物哈希 | 每个 `*_ortho.cub.tif` 与 `*_pho.cub` 记录 SHA256 |

## 5. 预算（首次下载后复核）

- 下载：168 条 EDR；若单条 ≈ 0.3–1.5 GB，则总量 ≈ **50–250 GB**（本机 `/` 剩余 544 GB，仅作参考，
  实际在机构环境执行）。
- 计算：单景 2 条 EDR 的 `lronac2isis + spiceinit + cam2map + phocube` 约 10–20 min（单核，视内核缓存），
  84 景 ≈ **15–30 核时**；`cam2map` 对每条 5000×52000 幅面的大图是主要开销。
- 先做 **1 景端到端**（建议 `LICHTENBER2` 或 `RIMPARRY1` 组），确认 QA 门槛与体积，再批量。

## 6. 完成后的下一步

1. 生成逐 Tile 光学/几何清单（128×128 输入口径与 SomBench 一致），与 DTM/坡度标签按网格对齐。
2. 回到 `docs/P1_9_HELDOUT_BUILD.md` 的 Gate 3：从 **84 景 / 52 组**（政策 A）中分层抽样 30–40 景、
   每景 2–4 Tile，**先冻结协议与哈希**，之后才允许任何预测。
3. Gate 4 一次性评估；在此之前不得查看任何冻结集成绩。
