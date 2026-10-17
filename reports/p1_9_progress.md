# P1.9 进度 — 外部 LROC DTM 独立测试集

**代码基线：`main@fa0719a6cdf5b64caca3add487dec9db41cce63b`**（2026-10-17）
计划文件：`docs/P1_9_HELDOUT_BUILD.md`

| 关卡 | 状态 | 证据 |
|---|---|---|
| Gate 0 — P19-00 审计修复回归 | **PASS** | `reports/p1_9_metadata_audit.md` · `outputs/p19_lroc_metadata_audit.json` |
| Gate 1 — P19-01 footprint/来源独立性 | **PASS（索引层）** | `reports/p1_9_footprint_audit.md` · `outputs/p19_lroc_footprint_audit.{csv,json}` |
| Gate 2 — P19-02 物理标签与 I/O 探针 | **部分 PASS** | `reports/p1_9_io_label_probe.md` · `outputs/p19_io_probe.json`（DTM/label 侧 PASS；光学 NAC 供给 BLOCKED） |
| Gate 3 — P19-03 冻结 `heldout_v2_external` | PENDING | 等 Gate 2；不得在预测后改动 |
| Gate 4 — P19-04 一次性评估 | HOLD | 未授权 |

## 验证命令与结果（容器 `rocm-lab`，`/data/lunarrisk`）

```bash
python3 -m unittest discover -s tests          # 198 tests OK
python3 -m compileall -q scripts tests         # OK
python3 scripts/audit_lroc_dtm.py --out outputs/p19_lroc_metadata_audit.json
python3 scripts/audit_lroc_footprints.py       # 冷 173 s / 热 11 s
```

## 本关结论

1. **元数据层**：662 条 DTM 记录 → 655 OK / 7 定长截断（fail-closed）；832 个 SomBench 归一化产品；
   **181 个产品级独立候选**；顶点筛选 113 provisional；`strict_candidates_after_spatial = null`。
2. **来源层**：181 景官方 PDS README 全量核对 → 174 景源清单与 DBF 完全一致且与 SomBench **交集为 0**；
   7 个 mosaic 空节排除。
3. **几何/空间层**：180 景多边形有效（`NOBILE04` 极区自相交排除）；
   **128 景同时满足来源 PASS 与 ≥10 km 净空**，分布在 **91 个景级地理组**（最大组 13 景），
   ≤3 m 原生分辨率 84 景。
4. 结论文本从 `CANDIDATES_REQUIRE_POLYGON_AND_SOURCE_QA` 推进为
   `CANDIDATES_PASS_SOURCE_AND_FOOTPRINT_QA`；**不是** `GO_STRICT_OOD`。

## 本轮修复与新增代码

- `scripts/audit_lroc_dtm.py`：接受 9 位早期 EDR ID 与两字母 `LE`/`RE` 后缀（后者此前被自身注释与实现矛盾地拒绝，
  影响 `CHANGE501`、`MALAPERT01`）；`MALFORMED_TOKEN` 2 → 0。
- `scripts/build_manifest.py`：记录地理角点 `upper_left_*/lower_right_*` 与
  `bounds_crs_status = PER_ZONE_LTM_UNVERIFIED`；发现索引单一 CRS 字符串无法描述按 LTM 分带的 `BOUNDS_*`
  （90 786 张 Tile 中仅 321 张与该 CRS 自洽）。
- `scripts/audit_lroc_footprints.py`（新）+ `tests/test_audit_lroc_footprints.py`（23 项）。
- `scripts/probe_p19_io.py`（新）+ `tests/test_probe_p19_io.py`（13 项）：PDS label 解析（含跨行续行拼接、
  `^IMAGE` 1 基指针、散文里的 NoDATA 恢复）、GeoTIFF GeoKey/NoData 核对、原生与块平均坡面标签统计。
- 测试总数 175 → **211**（`python3 -m unittest discover -s tests` 全绿；`compileall` 通过）。

## 下一步（Gate 2 剩余项 → Gate 3）

- **光学 NAC 供给未解决（当前唯一硬阻塞）**：RDR/5 bundle 只提供 DTM + `_SHADE.TIF`（DTM 派生，禁止当光学输入）；
  在本镜像上 12 种 EDR 路径组合与 LROC im-ldi EDR 页面均 404。下一步试 LROC ASU SDP 下载服务、
  USGS `pdsimage2` 镜像、PDS4 registry API 反查 EDR bundle 路径。
- 研究人员需在观测冻结集成绩之前选择尺度政策：(A) 仅采原生 ≈3 m 景 并保持 `g10/g15` 同尺度目标；
  (B) 定义共同可实现的尺度并在开发集先重算标签。探针已给出 3 景在 2–16 m 块平均下的 `g10` 漂移（≤2% 相对），
  但块平均**不等于**独立粗分辨率真值，不作为等价性证据。
- Gate 3 从 Gate 1 的 **128 景 / 91 组**中分层抽样（纬度/地貌/原生分辨率/净空），冻结前不得查看任何预测成绩。
- 磁盘：已用探针数据 242 MB（`data/probe_p19/`）；本机 `/` 剩余 544 GB。

## 约束遵守

- 只用本机 RX 7900 XT + 免费公开资源；未购买/创建任何付费实例，未使用 4×4090 节点。
- 新增纯 Python 读取库 `tifffile`（用于 float32 GeoTIFF 与 GeoKey），`torch`/ROCm 环境未改动。
- 未改动 ROCm/PyTorch 环境；仅新增纯 Python 辅助库（`pyshp`、`pyproj`、`shapely`、`netCDF4`、`h5py`），
  不触及 `torch`。
- 未覆盖既有 `protocols/heldout_v2_product_level_*` 与其 SHA256；新 manifest 写入 `_v2` 新文件。
- 未下载整批 DEM/NAC 大影像；抓取 181 个 README 文本（缓存可复现）+ 3 个 Gate 2 探针 DTM（242 MB，
  全部来自「永不入冻结集」的景）；SLDEM2015 粗地形仅做可用性 HEAD/魔数校验，未下载 167 MB 瓦片。
- 未读取/写入任何冻结集合的成绩，也未运行任何模型预测。
