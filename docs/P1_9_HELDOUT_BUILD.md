# P1.9 — 外部 LROC DTM 独立测试集：证据闭环与一次性冻结

**状态：Coding Agent 待执行（2026-10-17）。本文件是行动方案，不是已完成实验。**
起点：`main@cdafae4` 的 P1.8 fixed-mask 15 次训练与 163 项通过测试；随后针对外部 LROC 索引审计已修正“空源 ID 误判”和“等面积圆半径假冒 footprint 上界”的代码。详见 `reports/p1_8_fixed_mask.md` 与 `reports/p1_8_new_heldout_sources.md` 第 7 节勘误。

**当前最高优先级**：把已发现的 LROC NAC DTM *候选数据源* 变成可以审计的、真正源级和空间独立的 `heldout_v2_external`。不继续围绕旧 110 张 `development_test` 调参，不提前宣称 143 景严格合格，不上传/下载完整外部 DTM 数据池。

## 当前有证据支持的事实

- 旧 SomBench 100k 的 272/272 leakage Group 均已被此前 760 Tile 接触，1M 大目录并不能通过换分组名字提供全新组。已有 `protocols/heldout_v2_product_level_*` 的 64 Tile 是**产品级候选，4 个已使用 Group**；保护其 ID 和 SHA256，不得覆盖。
- 新 LROC NAC DTM 档案（PDS）的形状索引含 `images` 立体源 ID 与 DTM footprint；原始 662 景、188 个源 ID 候选、143 个圆形空间筛选结果属于**历史元数据调查**，不是已验证独立测试集。
- P1.8 fixed-mask 使三变体同为 98,464 参数；去 NAC 光度信息后开发集 MAE 有潜在改善，Brier 增量未显著。三种子/25 Group 的 Bootstrap 不涵盖全面训练随机性、地理 OOD 或多次模型选择。
- 现阶段可执行基础模型/数据检验；**不得提前运行外部 Held-out 的模型预测或根据其表现调整站点**。

## Gate 0 — P19-00 审计修复回归（最先做）

1. `git pull --ff-only origin main`，在当前已有的 ROCm 环境执行
   `python3 -m unittest discover -s tests -v` 与 `python3 -m compileall -q scripts tests`；
   列出新增 `tests/test_audit_lroc_dtm.py` 的实际结果，修复失败后再推进。
2. 使用**相同的原始 Shapefile + 100k manifest + 760 labels**重跑
   `python3 scripts/audit_lroc_dtm.py --out outputs/p19_lroc_metadata_audit.json`。
   明确报告空/畸形源数量、通过顶点 screen 的**新候选数**及历史 143 数字差异；
   不得把 `provisional_candidates_after_vertex_screen` 写成 `strict_candidates_after_spatial`。
3. 来源解析校验：抽样核对 DBF `images` 字段原文 vs `normalise_products()`、`source_id_metadata_valid()`；
   包含 L/R 尾缀、重复 ID、空/畸形、未知格式，遇到不能完整解析的原始源 ID 应标记
   `UNKNOWN_SOURCE` 并排除，不要猜测补足。
4. 审计结果写入 `reports/p1_9_metadata_audit.md`，列出原始索引版本、Hash、Git SHA、时间、异常样本。
   任何没有有效几何/源 ID 的记录 **fail-closed**。

## Gate 1 — P19-01 真正的几何/来源独立性验证（严格 OOD 的前置）

实现 **`scripts/audit_lroc_footprints.py` + 离线测试**，不得用 `sqrt(area/pi)`、简单等距圆或
仅比较中心点作为已验证 footprint 独立性的证明。

- **CRS**：核对 shapefile `.prj`、LROC/PDS 投影约定、月球坐标系/经度域
  (0–360 vs ±180)、极区定义、月球半径和 Tile 坐标单位。提供测试覆盖跨 ±180°、极区、
  狭长多边形、缺失 CRS、不合法 Ring / multipart polygon；转换不可证实则返回 UNKNOWN。
- **双方真实边界**：DTM 用官方 polygon（包括 multipart）；已使用 NAC/Slope Tile
  必须从可信的原始网格坐标和投影重建**每张 Tile 的 footprint**，不要把 local TM 米单位 WKT
  当作地理经纬度。按统一月球地理 CRS 计算 polygon-to-polygon 最小距离。
- **空间审计**：所有已使用 Train/Cal/Val/开发 Test + 已冻结的其他受保护站点都算排除对象；
  严格候选在原始 DTM polygon 与这些区域之间保持 ≥10 km，并验证候选彼此的间距。
  采用适当月球局部投影或严格球面几何，显式处理投影误差、跨经线与高纬度。
  建议将最小距离、精度方法、失效情形和保守 margin 写入每个候选记录。
- **来源审计**：对每个候选的 PDS README/label 实际列出的全部立体 NAC 源产品/摄像机 ID，
  与既有 SomBench + hydrated cohort + 其他冻结集合做标准化 ID 交集；不能只靠 DBF
  `images`，未知源/同源 DTM → 排除或 BLOCKED。提供文件路径/哈希/引用和精确 overlap。
- **输出**：`outputs/p19_lroc_footprint_audit.csv` 与 `reports/p1_9_footprint_audit.md`，
  每景 `DTM_name, all_stereo_EDR_IDs, geometry_valid, CRS, min_polygon_distance_km,
  source_disjoint, footprint_disjoint, reason`。
  **只有两项独立均明确 PASS** 才可入冻结候选；未知一律不入。
- 该阶段只做索引、PDS README/label 和所需少量边界/投影辅助文件的审计；
  不要下载整批 DEM 和 NAC 大影像。

## Gate 2 — P19-02 物理标签与 I/O 兼容性 Probe

在 Gate 1 通过后，从**不会进入最终冻结 V2 的少量单景产品**做 2–3 个外部数据探针：

1. 实际下载 PDS 原生 DTM、必要的 NAC 原始/正射影像与官方 PDS label；检查文件大小、格式、
   坐标轴顺序、投影参数、NoData、像素单位、竖直单位、质量参数（如 triangulation RMS）。
2. 确认能在相同物理 footprint 上获取光学影像和独立粗地形条件（SLDEM）。
   **禁止把被预测的 DTM 下采样后当作“独立 60m 输入”**。
3. 检查 NAC 分辨率/光照信息能否映射到现有网络 128×128 输入、模型的
   `nac_contrast` / `nac_texture` 预处理和几何标量。
4. **2–7m 原生 DTM 与现有 3m 定义不自动等价。** 不可把 7m DTM 插值成 3m
   就宣称得到 3m 地形真值。研究人员必须在观测 V2 成绩之前选择：
   (A) 仅采原生有效分辨率约 3m 的 DTM，并保持 `g10/g15` 同尺度目标；
   (B) 定义**共同可实现的尺度**，在开发集先重新计算标签/模型，明确这是新任务。
5. 只报告成功读取/物理 QA/样本容量/磁盘估算，不报告 Probe 的泛化成绩。

产物：`reports/p1_9_io_label_probe.md`，包含产品 ID、SHA256、影像/DTM/GSD/单位、
CRS 校验与最终可用样本数预估。

## Gate 3 — P19-03 冻结严格 `heldout_v2_external`

**必须在预测前冻结**：
- 按独立 DTM 景与光照/纬度/地貌/分辨率分层选样（例如 30–40 景，每景 2–4 Tile；
  实际数量由通过 Gate 1–2 的数据决定）；生成单景源列表与逐 Tile manifest。
- Manifest 必须包含稳定样本 ID、来源档案 revision、DTM 与全部 stereo NAC EDR ID、
  polygon/CRS、实际原生 GSD、单位、区域 Group、独立性最小距离、规则定义、NAC 预处理
  和来源 QA 哈希、生成 Git SHA。
- 仅对 Gate 1–2 全 PASS 的区域冻结；对不足样本则返回
  `BLOCKED_NEW_HELDOUT`，**不降低独立条件填满数量**。
- 输出 `protocols/heldout_v2_external_manifest.csv`、
  `protocols/heldout_v2_external_protocol.json`、
  `reports/p1_9_frozen_protocol.md`，记录哈希。
- **不能覆盖**既有 `protocols/heldout_v2_product_level_*` 及其 SHA256；
  不得在 V2 测试后修改、重新抽样或挑阈值。

## Gate 4 — P19-04 一次性评估（目前未授权开始，等前三关 PASS）

评估之前预注册候选（至少 Constant、60m Ridge、fixed-mask Coarse CNN、fixed-mask Fusion CNN）、
固定参数、训练规模、模型 Checkpoint SHA、随机种子、概率校准与 g10/g15 风险规则；
明确训练集/校准集不会包含独立测试源或其邻近区域。

- 一次性运行 `heldout_v2_external`；报告原始/校准 Brier、MAE、NLL、
  AUC、UAR/HMR–Coverage 与地理景级置信区间，并单列 DTM 测量不确定性和跨尺度转移风险。
- 严格声明当前事件规则是坡度面积比例的**工程代理**，不是官方着陆安全认证。
- 若冻结条件或原始数据质量不满足，保持 HOLD，不用开发集分数代替。

## Coding Agent 实施顺序和交付

1. **现在**实现/测试 Gate 0；随后 Gate 1 的 CRS+footprint+README 独立性审计；
   立即保存新增代码、测试和核查报告到 `main`。开发中可先报告元数据缺口。
2. Gate 1 PASS 后才进行 Gate 2 的少量公共数据探针；先查看文件体积/本机剩余磁盘，
   仅下载明确需要的产品，不触碰运行环境的 PyTorch/ROCm。
3. Gate 2 PASS 后构造冻结协议，准备下一轮用户 Review；**不要越过 Gate 3 直接跑 V2 成绩**。
4. 每关实际通过后更新 `reports/p1_9_progress.md`（PASS/FAIL/BLOCKED、命令、测试数、
   Git SHA、源索引/产物路径/Hash、可复现证据）。不许以计划代替实测。
5. 只用现有 RX 7900 XT 和免费公开资源；不擅自购买云卡、使用 4×4090、修改原始
   NAC/DTM、覆盖旧冻结集合、进行 LFM/Qwen 或大型联合后验训练。

验证命令：
```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
python3 scripts/audit_lroc_dtm.py --out outputs/p19_lroc_metadata_audit.json
```

**结论口径**：当前不是 `GO_STRICT_OOD`，而是
`CANDIDATES_REQUIRE_POLYGON_AND_SOURCE_QA`（若元数据筛选发现候选）；
获得正式严格独立验证能力，要以 Gate 1–3 的真实审计结果为准。
