# P3.0 loader 与真实 I/O 验证

2026-11-03；源码基线 `6dea815`。离线 contract 测试与真实 HDF5 部分读取 PASS；**官方 mmap 等价性 BLOCKED，接口协议为 HDF5_DERIVED_PILOT**。没有正式模型测试成绩。

## 已实现

- `scripts/p30_asset_plan.py`：从固定 HF listing 离线生成精确 allowlist；无网络，拒绝浮动 revision、超限小文件和提高初次上限。
- `src/lunarrisk/benchmarks/http_range.py`：校验 206/Content-Range/长度/编码、响应体预算、截断检测与有限 LRU，供 h5py seek/readinto。
- `scripts/p30_inspect_hdf5.py`：默认 dry-run，显式 `--apply` 才读 schema/metadata；不请求 patches dataset。
- `scripts/p30_extract_pilot.py`：固定 Train/Val 顺序，不借用 Test；校验通道后写 `.h5.partial`，成功才改名，逐 patch 留 hash。
- `scripts/p30_validate_metadata.py`：核对 ID、49 类 ID/name、网格中心，输出官方存储/纬度轨的计数、直方图、ID hash 与 footprint。
- `MoonstoneHDF5Pilot`：进程本地只读 h5py handle，pickle 清除 handle；输出 pixels、presence、target、sample_id，标签不拼进输入。

## 官方映射与 split

固定上游：Moonstone Git `cbd4720653eb05afd975d2a36280858c5c56a275`，HF `53099efd0f2db81eee96b28eae87e1efdd281310`。

Age 类序为 Pre-Nectarian、Nectarian、Imbrian、Eratosthenian、Copernican。大小写敏感的前缀匹配顺序：`pN→0, EI→2, N→1, I→2, E→3, C→4`，未知为 -1 并排除；不 strip/lower，也不用 preparation 中另一套七标签映射。Geology 为 `unit_id-1`；官方未过滤无效 ID，本地遇到无效资产直接阻塞，不悄悄删行。本次 16,200 行 ID/name 全部匹配发布的 49 类 map，Age 未知数为 0。

| 轨道 | Train | Val | Test | 排除 |
|---|---:|---:|---:|---:|
| 原始存储 split | 11,400 | 2,400 | 2,400 | 0 |
| 官方绝对中心纬度 split | 10,800 | 1,800 | 1,800 | 1,800 |

存储 split 原序保留，实际比例约 70.37/14.81/14.81，不重切为整数。生成器默认 80/10/10 与数据卡 70/15/15 的差异保留为来源差异。地理轨替换原 split：Train `|lat|≤60`，Val `(60,70]`，Test `(70,80]`；大于 80° 排除。按中心判断，不是整个 footprint 留出，也不证明预训练未见。

存储轨 Age 直方图：

| Split | Pre-N | N | I | E | C |
|---|---:|---:|---:|---:|---:|
| Train | 2120 | 3388 | 5037 | 669 | 186 |
| Val | 461 | 783 | 966 | 143 | 47 |
| Test | 503 | 772 | 991 | 105 | 29 |

地理轨 Age 直方图为 Train `[1479,3462,5162,527,170]`、Val `[388,531,627,208,46]`、Test `[437,524,662,131,46]`。存储轨出现的地质类别数为 46/40/42，未来仍保留 49 输出，不以本轮类别缺失悄悄改变任务；正式 Macro-F1 必须核对上游 absent-class 口径。

完整 49 类直方图、split 顺序 hash 见 [summary](evidence/p30_20261103/metadata_summary.json)，16,200 个 ID/footprint/标签派生记录见 `evidence/p30_20261103/metadata_rows.csv.gz`。本轮只审计 Test 监督元数据，没有读取 Test 像素或跑模型评分。

## 通道、归一化与 mask

28 通道顺序严格匹配七组 surface/thermal/spectral_m3/gravity/radar/hapke/composition；具体列表固定在 `moonstone.py` 和 HDF5 schema 中。网格为 128ppd，256px Tile；全球 mmap `(23040,46080)`，crop 起点 `row*256,col*256`，无随机经度回绕。

本地 derived 路径使用 HF 全局 `[mean,std]`，float32 z-score，原 nonfinite 归零，有限 raw zero 仍是数据。官方下游 presence 从**归一化后 finite 且 nonzero**重新计算，不采用 HDF5 has_channel；归一化后恰为 0 的像素按其代码算缺失。group presence 为 ANY 通道。这里忠实记录该语义，不将改进 mask 混入复现轨。

HF stats SHA256 `3e421d3a660a19cbd68c1941bf3b122da56777987de4848ac1382ebe3e39ab64`；Git stats SHA256 `355ae1b29720b5b5c20b392954e71ee0dbdd1495a74965f2c3a52abd1118f91e`。27/28 通道统计不同，只有 roughness 相同。通用 step15 与 `fix_minirf` 专用 raw-zero/log1p 顺序的发布 provenance 未解决，不能宣称 mmap 等价。

**预先写定的后续配对验收条件（本轮未执行）：** 固定本轮 8 ID 与全部 28 通道；官方 mmap crop 与 HDF5 derived 的 float32 像素比较 `rtol=1e-5, atol=1e-6`，mask/顺序/ID/标签/split 必须精确相同；逐通道报告最大误差与失败像素数。任何差异先解释来源，不能观测误差后放宽容差。通过 radar/统计来源审核前不称官方 loader PASS。

## 实际验证记录

最终隔离快照含 scripts/src/tests/protocols，既有 ROCm 环境执行：

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts src tests
```

**471 项，470 PASS、1 SKIP；compileall exit 0**。唯一跳过为旧 `labels_hydrated.csv` 未复制进源码快照的可选检查。新增 HDF5/Range 测试在已有 h5py 环境实际运行。覆盖前缀/边界/ID、stored split、target 隔离、通道/NoData/归一化、worker pickle、HTTP200 拒绝、错误 Range、缓存/预算/截断和真实 h5py 对 synthetic fileobj 的读取。日志见 [tests_471.log](evidence/p30_20261103/tests_471.log)。这些是离线 fixture，不替代以下真实读取。

此前第一次快照漏带 protocols，458 项中 12 个既有协议测试报缺文件；在新路径补全后重跑通过，没有忽略失败。初次宿主机 HDF5 测试因无 h5py 跳过；完整容器结果覆盖该缺口。

真实 HDF5 全 metadata 读取状态 `REAL_HDF5_METADATA_READ_NOT_LOADER_EQUIVALENCE`；8 patch 提取状态 `REAL_PIXEL_PILOT_EXTRACTED_NOT_EQUIVALENCE`。4 Train ID `0,1,2,3`，4 Val ID `20,21,22,23`；固定顺序来自最初有效行，北极相邻样本不能视作代表全月分布。pilot SHA256 为 `d6807ecc1a0d52f07fc46cc53f42146e3b66e9562ee9fd7357f5dfbf32d9bd0a`。实际累计响应体 39,016,735 字节，详见 [资产审计](p3_0_benchmark_audit.md)。

命令中本地路径以环境变量表达，网络命令仅记录已执行配置，不授权重复传输：

```bash
python3 -I scripts/p30_inspect_hdf5.py --url "$PINNED_HDF5_URL" --size 74449119347 --out "$NEW_METADATA_PATH" --prior-transfer-bytes 5309691 --apply
python3 -I scripts/p30_extract_pilot.py --metadata "$METADATA_PATH" --out "$NEW_PILOT_PATH" --per-split 4 --prior-transfer-bytes 27205487 --apply
python3 -I scripts/p30_validate_metadata.py --metadata "$METADATA_PATH" --class-map "$CLASS_MAP_PATH" --outdir "$NEW_AUDIT_PATH"
```

## 同日续推

[固定 mmap 行探针](p3_0_mmap_pair.md) 已核查原 8-ID 的 28 通道、3 行/裁片，共 172,032 像素，既定 HF 路径误差与 mask 差异均为 0。只覆盖探针行，未满足上述完整裁片/有效模态与 channel/group mask 验收条件，官方等价性仍不标 PASS。

新增 [五类 Age 工程 smoke](p3_0_age_fiveclass_smoke.md) 在读取前固定每类 1 Train/1 Val，真实提取 10 patch、seed42/2 步训练，独立进程回放 CSV 完全相同。支持该选择的最新代码通过 **478 tests（477 PASS/1 旧数据可选 SKIP）与 compileall**。累计实际响应体 78,502,175 字节，初次上限不变；旧 evidence 原样保留，新证据另设命名空间。

正式 Age/Geology benchmark、预训练对照、候选条件化和 Test 评分均未完成。
