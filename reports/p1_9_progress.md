# P1.9 进度 — 外部 LROC DTM 独立测试集

**代码基线：`main@fa0719a6cdf5b64caca3add487dec9db41cce63b`**（2026-10-17）
计划文件：`docs/P1_9_HELDOUT_BUILD.md`

| 关卡 | 状态 | 证据 |
|---|---|---|
| Gate 0 — P19-00 审计修复回归 | **PASS** | `reports/p1_9_metadata_audit.md` · `outputs/p19_lroc_metadata_audit.json` |
| Gate 1 — P19-01 footprint/来源独立性 | **PASS（索引层）** | `reports/p1_9_footprint_audit.md` · `outputs/p19_lroc_footprint_audit.{csv,json}` |
| Gate 2 — P19-02 物理标签与 I/O 探针 | PENDING | 未下载任何 DEM/NAC 载荷 |
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
- 测试总数 175 → 198（`unittest discover` 全绿）。

## 下一步（Gate 2，需用户确认后下载）

- 选 2–3 个**不会进入最终 V2 冻结**的单景产品做文件级探针：DTM/NAC/Label 实际下载、文件体积、
  格式、轴序、投影、NoData、水平/垂直单位、三角测量 RMS、有效像素比例。
- 明确原生 2–7 m 与现有 3 m 标签尺度不可自动等价；需在观测冻结集成绩之前选择 (A) 仅 3 m 原生景 或
  (B) 共同可实现的尺度并先重算开发集标签。
- 磁盘预算：先统计所需体积与本机剩余空间，仅下载明确需要的产品。

## 约束遵守

- 只用本机 RX 7900 XT + 免费公开资源；未购买/创建任何付费实例，未使用 4×4090 节点。
- 未改动 ROCm/PyTorch 环境；仅新增纯 Python 辅助库（`pyshp`、`pyproj`、`shapely`、`netCDF4`、`h5py`），
  不触及 `torch`。
- 未覆盖既有 `protocols/heldout_v2_product_level_*` 与其 SHA256；新 manifest 写入 `_v2` 新文件。
- 未下载整批 DEM/NAC 大影像；仅抓取 181 个 README 文本（缓存可复现）。
