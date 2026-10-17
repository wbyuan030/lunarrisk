# P1.9 进度 — 外部 LROC DTM 独立测试集

**代码基线：`main@fa0719a6cdf5b64caca3add487dec9db41cce63b`**（2026-10-17）
计划文件：`docs/P1_9_HELDOUT_BUILD.md`

| 关卡 | 状态 | 证据 |
|---|---|---|
| Gate 0 — P19-00 审计修复回归 | **PASS** | `reports/p1_9_metadata_audit.md` · `outputs/p19_lroc_metadata_audit.json` |
| Gate 1 — P19-01 footprint/来源独立性 | **PASS（索引层）** | `reports/p1_9_footprint_audit.md` · `outputs/p19_lroc_footprint_audit.{csv,json}` |
| Gate 2 — P19-02 物理标签与 I/O 探针 | **部分 PASS** | `reports/p1_9_io_label_probe.md` · `outputs/p19_io_probe.json`（DTM/label 侧 PASS；光学 NAC 供给 BLOCKED：**128 个 PASS 景中 0 景发布光学影像**） |
| Gate 3 — P19-03 冻结 `heldout_v2_external` | **BLOCKED**（等正射光学产物） | 政策与路线已预注册：`protocols/p19_external_policy.json`；执行手册 `docs/P1_9_EDR_ORTHO_RUNBOOK.md`；严格池 84 景 / 52 组 |
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
- `scripts/export_edr_acquisition.py`（新）：导出待取 EDR 清单（政策 A 池 → 84 景 / 168 条立体 EDR）。
- `scripts/audit_lroc_optical.py`（新）+ 6 项离线测试：场景页兄弟产品发现 + HEAD 实证光学文件存在；
  一次真实方法修正（第一版只解析产品页导致假阴性）已记录在 `reports/p1_9_io_label_probe.md`。
- `scripts/verify_clearance_spherical.py`（新）：球面无投影复算净空，与 aeqd 审计在 12 个最紧景一致到 0.3 m。
- `protocols/p19_external_policy.json`（新）：用户决策预注册（尺度政策 A、光学路线、候选池与规则）。
- `docs/P1_9_EDR_ORTHO_RUNBOOK.md`（新）：EDR→ISIS 正射手册（含 QA 门槛与预算）。
- `scripts/probe_p19_io.py`（新）+ `tests/test_probe_p19_io.py`（13 项）：PDS label 解析（含跨行续行拼接、
  `^IMAGE` 1 基指针、散文里的 NoDATA 恢复）、GeoTIFF GeoKey/NoData 核对、原生与块平均坡面标签统计。
- 测试总数 175 → **217**（`python3 -m unittest discover -s tests` 全绿；`compileall` 通过）。

## 下一步（Gate 2 剩余项 → Gate 3）

- **光学 NAC 供给未解决（当前唯一硬阻塞，已量化）**：181 景全量审计中 8 景发布 NAC 正射镶嵌图，
  其中 7 景是 README 源清单为空的 mosaic 产品，唯一源可验证者 `ALPHONSUS` 净空 9.16 km（差 0.84 km 未达门槛）；
  **通过 Gate 1 的 128 景里 0 景有光学影像**。原始 EDR 卷在本网络不可达（权威入口 404 / 仅 JS 浏览器）。
  三条出路见 `reports/p1_9_io_label_probe.md`「下一步（需要明确决策）」，其中两条需要用户批准放宽口径。
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
