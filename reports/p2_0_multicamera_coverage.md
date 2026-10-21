# P2.0 P20-01：FRSHCRATER10 双相机覆盖与光度 QA

**状态：PASS（覆盖达标，配准 QA 待 P20-02）** · 2026-10-20 · 场景 `FRSHCRATER10` / `M153639872`
输入：已下载的 `M153639872LC` / `M153639872RC` CDR 各 528 929 736 B；DTM
`NAC_DTM_FRSHCRATER10.TIF`（14259×2389，2 m，等距圆柱，48N/180E）。

## 链路上的两处新增修正

1. `prepare_lroc_cdr_cube.py` 增加了 11 个相机必需键（`FrameId`、`SampleFirstPixel`、
   `PrerollTime`、`MissionPhaseName`、`InstrumentHost*`、`InstrumentName`、四个温度原始值），
   否则右相机 `spiceinit` 在翻译/键缺失上失败。脚本现在把左/右两条链都固化。
2. 脚本新增 `normalise_pvl_quotes()`：`editlab` 会把含空格的值写成 `'"…"'`，破坏
   `MissionName2DataDir.trn` 翻译。该函数把 `'"X"'` 等长规整为 `"X"`（保持文件长度，
   不破坏标签里的绝对 StartByte）。本次左/右各规整 4 处。
3. QA 侧发现 `isis2raw` 对左/右图像写出的 NULL 哨兵不同（左 0.0132450、右 0.0321360），
   因此覆盖统计必须**按相机各自检测填充值**，不能共用常量。

## 覆盖结果

| 指标 | 值 |
|---|---|
| DTM 有效像元 | 28 311 111（网格的 0.8311） |
| 左相机有效 | 15 096 940（网格 0.4432） |
| 右相机有效 | 16 809 077（网格 0.4934） |
| **coverage_left / DTM valid** | **0.4275** |
| **coverage_right / DTM valid** | **0.5783** |
| **coverage_union / DTM valid** | **0.9901** |
| **overlap / DTM valid** | **0.0157**（445 886 像元） |
| 联合占据（无 DTM 掩膜） | 网格的 0.9235 |

原 Runbook 建议的 ≥95% 覆盖门槛**双相机联合后达到 0.9901**；左右单相机分别只有 0.43 / 0.58，
因此**必须双相机**，不能假设单一相机覆盖整个 DTM。两相机地面范围：

- 左：lat 47.0851–48.0331，lon 243.7750–243.9238
- 右：lat 47.0861–48.0336，lon 243.8928–244.0390
- 重叠经度 243.8928–243.9238（≈0.031°，≈1.5% 的 DTM 宽）

## 光度与辐射一致性（重叠区 445 886 像元）

| 量 | 左 | 右 |
|---|---|---|
| I/F 均值 | 0.07150 | 0.07290 |
| I/F 标准差 | 0.02040 | 0.02129 |
| 残差均值 / 标准差 | — | −0.001397 / 0.007634 |
| 残差 p5 / p50 / p95 | — | −0.012085 / −0.001160 / +0.008637 |
| Pearson r（左 vs 右） | — | **0.9338** |

左右相机在重叠区高度相关（r = 0.93）且残差很小（std 0.0076 ≈ I/F 均值的 10%），
说明同一场景下两相机的辐射/几何一致性良好，具备融合基础。

## 复跑命令

```bash
# 相机 cube（左/右同一脚本；右需 FrameId=RIGHT -> NaifIkCode=-85610）
python3 scripts/prepare_lroc_cdr_cube.py <IMG> --isisdir $ISISDATA --out cdr.cub \
  --ck …/lrolc_2011031_2011060_v02.bc …/moc42r_2011031_2011060_v05.bc \
  --frame-kernel …/lro_frames_2014049_v01.tf --spk …/fdf29r_2011060_2011091_v01.bsp \
  --ik …/lro_lroc_v20.ti --iak …/lro_instrumentAddendum_v05.ti \
  --sclk …/lro_clkcor_2026287_v00.tsc --lsk …/naif0012.tls \
  --pck …/pck00010.tpc --tspk …/moon_pa_de421_1900-2050.bpc …/de421.bsp

# DTM 同网格正射
maptemplate map=dtm_map.cub projection=equirectangular clat=48.0 clon=180.0 \
  targopt=user targetname=MOON eqradius=1737.4 polradius=1737.4 \
  rngopt=user minlat=47.0843749 maxlat=48.0248383 minlon=243.8046336 maxlon=244.0401158 \
  resopt=ppd resolution=15161.67521207
cam2map from=cdr.cub map=dtm_map.cub to=ortho.cub pixres=map defaultrange=map
isis2raw from=ortho.cub to=ortho.raw storageorder=BSQ endian=LSB
```

## 拼接策略与实测（P20-01 第 5 项，2026-10-20）

脚本 `scripts/ortho_mosaic_p19.py`（12 项离线测试）。策略在见到任何成绩之前固定为**硬选择、绝不做混合**：

1. 只有非 NULL 像元参选；
2. 优先级按**字典序**比较：先比饱和比例（低者优先），再比清晰度（`mean|laplacian|` 高者优先），
   最后按输入顺序；把两者相乘会让"平坦但未饱和"的图像因为错误原因胜出；
3. 不跨图像做加权平均——两 acquisition 的太阳几何可能不同，线性混合会改变辐射量；
4. 每个输出像元记录来源编号，便于审计 seamline 与贡献比例。

新增**光照守卫**：由 `scripts/solar_azimuth_p20.py` 逐相机给出太阳方位/高度，若方位跨度 >10° 或
高度跨度 >5° 则直接拒绝拼接（要求拆景），避免在 seamline 上制造亮度台阶。

实测：

| 场景 | 有效像元 | 未覆盖 | seamline 像元 | 太阳方位/高度跨度 |
|---|---|---|---|---|
| RANGER | 0.9985 | 49 363 | 544 048（占有效 1.65%） | 0.32° / 0.96° |
| PLANCKFLOOR | 0.9715 | 1 646 479 | 507 565（占有效 1.14%） | 1.11° / 0.48° |

两景均无饱和像元（`saturation_share = 0`），因此优先级完全由清晰度决定；第二 acquisition 在
RANGER 只贡献 0.5%（左）/ 6.7%（右），PLANCKFLOOR 则反过来由 `M136655466` 主导（0.443 / 0.456）——
这正是"按 acquisition 联合覆盖排产"的量化体现。

### 拼接辐射台阶（实测，新增指标）

硬选择在几何上成立，但会留下**可测的辐射台阶**。`seam_step()` 比较"跨来源边界"的相邻像元差与
"同来源内部"的相邻像元差：

| 场景 | 接缝像元差 中位 | 内部差 中位 | 比值 | 单增益校正后 |
|---|---|---|---|---|
| RANGER | 0.00604 I/F | 0.00061 | 9.9× | **0.00202（3.3×）** |
| PLANCKFLOOR | 0.01062 I/F | 0.00134 | 7.9× | 0.01104（8.2×），几乎无效 |

单增益（`--equalize`，以最高优先级图像为参考，重叠区中位数比值）在 RANGER 上把台阶降到 1/3，
拟合增益例如 `M144490730LC = 1.34`（同一天、太阳高度差 <1°，因此是产品定标/曝光差而非光照）；
PLANCKFLOOR 的台阶不是全局曝光差，单增益无法消除。

**结论**：拼接结果目前只可用于几何/覆盖审计；**不得**作为训练/评估输入，除非接缝辐射问题
在冻结前解决（逐缝梯度校正、或改为按接缝带约束的分配）。该限制已写入协议提案的待批准项。

覆盖与拼接的一个新事实：PLANCKFLOOR 的 seamline 只占 1.14%，但 4 个相机彼此几乎不互补
（左并集 0.482 vs 最佳单幅 0.477），说明该景的覆盖来自"整幅互相重叠"而非"条带互补"。

## 结论与下一步

- **覆盖门槛达到**：union/DTM valid = 0.9901（≥0.95 的目标），重叠 1.57%。
- **多相机融合策略须在观测任何模型成绩前固定**：重叠区取残差更小且不饱和的相机；
  记录 seamline、遮挡掩膜；本次残差分布（mean −0.0014，p95 +0.0086）可作为取舍依据之一。
- 拼接已实现、已 TIFF 化（`scripts/mosaic_to_geotiff_p20.py` 复用 DTM 的 geo tags，读回校验一致），
  并有接缝台阶实测；仍未做：遮挡掩膜（几何遮挡判定）、接缝辐射校正。
- P20-02 已完成：网格算术 40/40 精确、坑心 GCP 1–2 px（见 `p2_0_registration_qa.md`）。
- Gate 3 仍 **NOT FROZEN**；Gate 4 **HOLD**。CDR 路线仍是
  `PROPOSED_NOT_APPROVED`，不得据此冻结或评分。
