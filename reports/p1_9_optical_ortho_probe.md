# P1.9 Gate 2 光学通道 — 真实景正射校正闭环

**状态：PASS（单景）** · 2026-10-20 · 场景 `FRSHCRATER10` / acquisition `M153639872`（CDR，左相机）

本报告记录从下载到正射到 DTM 网格的完整链路，以及途中纠正的四个错误假设。
数据、内核与脚本均可复跑；Gate 3 冻结仍待用户/实验室侧决定批量范围。

## 一、链路与产物

| 步骤 | 工具/参数 | 结果 |
|---|---|---|
| 1. 取影像 | 官方索引解析得 `LRO-L-LROC-3-CDR-V1.0/LROLRC_1006/DATA/SCI/2011060/NAC/M153639872LC.IMG` | 528 929 736 B，sha256 `6cc14e77…` |
| 2. 导入 ISIS | **`pds2isis`**（CDR 正式导入器） | 像元正确：I/F p50 0.0679、max 0.3344、NULL 3 342 336 |
| 3. 补仪器元数据 | `editlab` 建 `Instrument` 组（从 PDS 标签取值）+ `Kernels/NaifIkCode=-85600` | 见第二节的坑 |
| 4. 相机初始化 | `spiceinit` **显式内核参数** | 成功（`Source = isis`，无 Error） |
| 5. 地面几何 | `campt line=26000 sample=2530` | 行星中心纬度 47.5571°，东经 243.8493° |
| 6. 影像范围 | `camrange` | lat 47.0851–48.0331，lon 243.7750–243.9238 |
| 7. 正射 | `maptemplate` + `cam2map` 到 DTM 网格 | **14259 × 2389，与 DTM 逐像元同网格** |
| 8. 统计 | `stats from=ortho3.cub` | mean 0.0683 / median 0.0671 / min 0.0035 / max 0.3097；valid 15 172 840、null 18 891 911 |

## 二、纠正的四个错误假设（每个都有实测证据）

1. **`lronac2isis` 不能读 CDR。** 把 `DATA_SET_ID` 改成 EDR 骗过导入器后，标签/时钟/温度全部正确，
   但**像元是坏的**：`stats` 显示 `Minimum = -2147483648`、`NullPixels = 0`。
   CDR 与 EDR 的像元编码不同，改名不等于转换。正确路径是 `pds2isis`。
2. **`pds2isis` 的 cube 缺少相机需要的 `Instrument` 组。** 它不是"可选元数据"：缺 `SpacecraftClock*Count`、
   `LineExposureDuration`、`SpatialSumming`、`SampleFirstPixel` 等键时 `spiceinit` 直接失败；
   缺 `SpacecraftClockPrerollCount` 时同样失败。做法是从同产品的 EDR 标签派生值逐键补齐（脚本已固化）。
3. **`editlab` 会把带空格的字符串写成 `'"…"'`**（PVL 字符串里再套引号），导致
   `MissionName2DataDir.trn` 翻译失败（`Unable to find a translation value for [MissionName, "…"]`）。
   修正：写完后把 `'"X"'` 规整为 `"X"`（等长替换）。
4. **USGS 数据区缺 `moc_kernels.????.db`。** `ck/kernels.0001.conf` 要求
   `moc_kernels`（航天器总线姿态）+ `lroc_kernels`（相机姿态），但下载器只给了后者，
   于是 `spiceinit` 能加载相机 CK 却拿不到 `LRO_SC_BUS` 姿态，报
   `the ck rotation from frame -85000 cannot be found`。
   补齐 `moc42r_2011031_2011060_v05.bc`（787 MB）后，纯 SPICE 的 `pxform(LRO_LROCNACL→J2000)` 立即通过。
   由于 ISIS 的内核数据库解析仍不认这个手工 db，最终改用 **`spiceinit` 的显式内核参数**（`LS/SCLK/TSPK/PCK/IK/IAK/FK/CK/SPK`）。

另外两处必须记住的细节：

- **FK 的 glob 是 `lro_frames_???????_v01.tf`**，而定义 `INS-85600` 的最新帧内核是 `_v02`；
  按 `_v01` 名字放一份副本 ISIS 才能选中（这也解释了此前 `frame -85600 is not supported`）。
- **`maptemplate` 的 `clat/clon` 必须取 DTM 标签的 `CENTER_LATITUDE/CENTER_LONGITUDE`（48/180）与
  `MAP_RESOLUTION`（15161.675 ppd）**；否则采样数会从 2389 变成 3571，网格与 DTM 不再逐像元对齐。

## 三、正射质量（OTF：与 DTM 同网格，无需重采样）

| 指标 | 值 |
|---|---|
| 网格 | 14259 × 2389，2 m，等距圆柱，R = 1737.4 km（与 DTM 完全一致） |
| 有效像元（正射） | 44.32%（填充值 0.013245，与 cube 报的 55.5% NULL 吻合） |
| 联合覆盖（正射 ∩ DTM 有效） | 网格的 35.53% = **DTM 有效footprint 的 42.75%**（受限于影像仅覆盖 DTM 西侧一半） |
| I/F 分位 | p1 0.0307 / p50 0.0672 / p99 0.1302（与原始 CDR 统计一致） |
| 配准（边缘互相关） | 峰值在 (line +0, sample −1) 即 **2 m 偏移**；峰值/零位移 = 1.067（曲线平坦，属"与零偏移一致"而非锐利判据） |
| 更硬的几何证据 | `campt` 在 CDR cube 与 EDR-标签 cube 上给出**完全相同**的地面点（47.5571°, 243.8493°） |

覆盖率 42.75% 是**几何原因**（单条 NAC 条带只覆盖 DTM 西半部），不是流程缺陷；
若需要整幅覆盖，须取同景的第二条立体影像或右相机拼接（Gate 3 的分层抽样会据此选择景）。

## 四、复跑命令（容器内）

```bash
# 0) 数据区（一次性）：官方下载器，只取内核子树
micromamba create -y -p /opt/isis-env -c usgs-astrogeology -c conda-forge isis=8.3.0
downloadIsisData lro /opt/isis-env/data        # 774 GB 全量；本次只保留 kernels/
# 0b) 补齐总线姿态内核（USGS 数据区缺失）
rclone copyto asc_s3:asc-isisdata/usgs_data/lro/kernels/ck/moc42r_2011031_2011060_v05.bc \
       /opt/isis-env/data/lro/kernels/ck/
# 0c) 帧内核按 v01 glob 命名
cp /opt/isis-env/data/lro/kernels/fk/lro_frames_2014049_v01.tf{,.bak}  # 已有 v01 即可

# 1) CDR -> cube（脚本固化仪器组补齐 + 显式内核）
python3 scripts/prepare_lroc_cdr_cube.py data/probe_frsh10/M153639872LC.IMG \
  --isisdir /opt/isis-env/data --out /tmp/cdr2.cub \
  --ck /opt/isis-env/data/lro/kernels/ck/lrolc_2011031_2011060_v02.bc \
        /opt/isis-env/data/lro/kernels/ck/moc42r_2011031_2011060_v05.bc \
  --frame-kernel /opt/isis-env/data/lro/kernels/fk/lro_frames_2014049_v01.tf \
  --spk /opt/isis-env/data/lro/kernels/spk/fdf29r_2011060_2011091_v01.bsp \
  --ik /opt/isis-env/data/lro/kernels/ik/lro_lroc_v20.ti \
  --iak /opt/isis-env/data/lro/kernels/iak/lro_instrumentAddendum_v05.ti \
  --sclk /opt/isis-env/data/lro/kernels/sclk/lro_clkcor_2026287_v00.tsc \
  --lsk /opt/isis-env/data/base/kernels/lsk/naif0012.tls \
  --pck /opt/isis-env/data/base/kernels/pck/pck00010.tpc \
  --tspk /opt/isis-env/data/lro/kernels/tspk/moon_pa_de421_1900-2050.bpc \
         /opt/isis-env/data/lro/kernels/tspk/de421.bsp

# 2) 与 DTM 同网格正射
maptemplate map=dtm_map.cub projection=equirectangular clat=48.0 clon=180.0 \
  targopt=user targetname=MOON eqradius=1737.4 polradius=1737.4 \
  rngopt=user minlat=47.0843749 maxlat=48.0248383 minlon=243.8046336 maxlon=244.0401158 \
  resopt=ppd resolution=15161.67521207
cam2map from=/tmp/cdr2.cub map=dtm_map.cub to=ortho3.cub pixres=map defaultrange=map
```

## 五、对 Gate 3 的含义

- 光学输入通道**已具备单景可复跑的实现**：任意已解析出 `FILE_SPECI` 的 CDR 产品都能走同一条链。
- 仍需在批量前确定：(a) 每景用左右哪相机或如何拼接以获得完整 footprint；(b) 单景约 0.5 GB 原始 +
  0.8 GB 内核（总线 CK 每月一份）的存储规划；(c) 逐景 QA 门槛（本报告给出的 42.75% 覆盖率是单条带的下限）。
- 未做：右相机、拼接、逐 Tile 128×128 切分、多景批量、Gate 3 冻结与任何模型评分。
