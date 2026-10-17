# P1.8 — 数据扩容安全、可信消融与全新地理测试协议

> **状态：待 Coding Agent 执行（2026-10-17）。** 本文件是动作清单，不是已完成报告。
>
> 起点：`main@379fa37`（P1.7 的 12-run 消融和报告已完成）。请先 `git pull --ff-only origin main`，核实最新 SHA，以最新代码为准。研究证据：`reports/p1_7_ablation.md`。
>
> **本轮目标：** 在保持原始真实数据、既有测试集与实验记录不变的前提下，修复追加下载与统计协议的确定缺陷，完成容量控制的消融设计，冻结未参与模型选择的 `heldout_v2`；只有满足关卡后，才在本地 7900 XT 扩充 train 到 3–5k。避免为了「跑更多数据」先破坏现有的 760 Tile 标签。

## 本轮判断与科研边界

- P1.7 修正增强协议后，`p17_full` 的 NAC 光学增量 **ΔMAE −0.0043 [−0.0113, +0.0033]**、**ΔBrier −0.0131 [−0.0369, +0.0079]**，置信区间均跨零。**不能宣称 NAC 增量已得到稳定验证**。
- P1.6 的更好结果与增强开关变化相关，但**不能单凭这一点断言其完全由增强 Bug 造成**。合法的、物理一致的增强可以改善泛化，仍需在开发集做对照。
- P1.7 “删除特征”消融同时改变了第一层通道数和最后 head 输入维度，参数量不一致；`no_both` 的效果不能简单解释成两类物理信息均无效。
- 旧 110 Test 已多次用于分析与设计；**今后必须称其为 `development_test`，不得基于它挑选后声称最终无偏测试表现**。
- `g10>0.05`、`g15>0.02`、以及新增规则均是研究代理事件，**不是已认证的着陆安全定义**；高分辨率 DTM 的源立体像对 ID 仍缺，不能声称已完成 DTM 源级 OOD 审计。

## 任务顺序（必须按依赖执行）

| 任务 | 优先级 | 前置 | 产物 | 通过后才能 |
|---|---|---|---|---|
| **P18-01** 修复 Hydration 的 no-op Resume | **P0** | 无 | 脚本 + 测试 | 安全恢复断点 |
| **P18-02** 原子化、可演进的标签合并 | **P0** | P18-01 | CSV 合并测试 + 扩容 dry-run | 扩容下载 |
| **P18-03** 统计一致性检查与 NLL 口径 | P1 | 无 | `compare_ablations.py` + 测试 | 可信聚合 |
| **P18-04** 参数量固定的模态掩码消融 | P1 | P18-03 | 新的可配置训练入口和实验方案 | 物理贡献比较 |
| **P18-05** 冻结全新 `heldout_v2` | **P0（发表必需）** | P18-01/02 的数据治理可用 | 冻结清单 + HASH + 来源审计 | 任何宣称独立泛化 |
| **P18-06** 本地训练扩至 3k，再审查是否 5k | P2 | P18-01/02、P18-05 | 真实训练报告 | 一次性最终评估 |
| **P18-07** 预注册后执行 V2 一次性评估 | P2 | P18-04/05/06 | OOD 风险/误差报告 | 决定 LFM/联合后验 |

**允许 P18-03/04 与 P18-01/02 并行开发，但不允许在 P18-01/02 验收之前运行扩容下载，也不允许在 P18-05 冻结之前读取 V2 模型指标。**

## P18-01：Resume 的确定性死分支

**发现**：`scripts/hydrate_tiles.py` 在 `if not records: raise SystemExit("no tiles hydrated")` 后还有一个 `if not records:` 的正常 no-op 分支。全量下载完再执行 `--resume` 仍会提前退出。

**修改要求**
1. 已选全部 Tile 都存在时，`--resume` **退出码 0**，保留已有标签和源文件，不伪造新下载计数；正常读取并生成 summary。
2. `todo` 为空与 `todo` 中下载全部失败是**两种不同情形**；后者必须非零退出并记录逐项失败信息，不能宣称成功。
3. 记录 `selected_count`、`existing_count`、`download_attempted`、`new_records`、`failures`，并保留原始 S3 key / hash。
4. 对不一致的已存在 `sample_id` 的 split、group、`NAC_EDR_ID` 与样本来源做显式检查，禁止静默覆盖。

**单测（不能触网）**
- 已存在且全部完成：返回 0，`labels` 数据不丢失、旧文件不被改写。
- 存在一个新 Tile：只请求这一个，保持原有行。
- 有待下载但全部失败：非零退出，明确失败列表。
- 重复 `sample_id` 或 ID 对应 split/group/source 发生漂移：明确拒绝。

## P18-02：修复 CSV Schema 演进与原子追加

**发现**：`record_fieldnames()` 找到新列，但 `--resume` 的 `mode="a"` 沿用旧文件表头。新记录的字段数量如果增加，产生列数与 header 不一致的 CSV。新增字段尤其可能来自 `--enrich` / 几何属性。

**修改要求**
1. 由纯追加改为**读取既有记录→按 `sample_id` 合并并建立列并集→检查→写入临时文件→原子替换**。建议 CSV 保持兼容；不要在本轮无迁移计划时突然改 Parquet-only。
2. 相同 `sample_id` 只能出现一次；新增列对旧行填 `NaN`/缺省，不可发生字段移位；保留既有列顺序，再稳定追加新列。
3. **任何部分下载失败/写入中断不得破坏既有标签。** 使用临时路径、文件 flush/fsync，成功校验后再 replace；避免线程工作中写原始 CSV。
4. `--resume` 不因 train quota 增大而重新分配原有 val/cal/test Tile。必须对固定的旧 `sample_id` 集合做逐一比对，并拒绝跨 split 漂移。复用 `select_tiles` 时验证扩容选择满足“旧 cohort 是新 cohort 的子集”；如果不满足，先改成显式 ID 清单冻结/append-only 选择协议。
5. `--enrich` 对既有标签的字段补充也应原子写回，并验证 merge 是严格 `one_to_one`。

**验收**：用合成 CSV fixture 覆盖旧表头 2 列/新记录 3 列、重复 ID、下载部分失败、中断、重复执行、旧行全字段保存；用现有 `outputs/labels_hydrated.csv` 的**备份副本**做 400→401 train dry-run（可 stub S3），报告旧 760 Tile 的 ID/分组/真值完全不变。未通过前**严禁在正式文件上运行 `--train 3000`**。

## P18-03：统计和跨配置协议 Fail-fast

文件：`scripts/compare_ablations.py`、`scripts/aggregate_seeds.py`。

1. 对所有 config×seed 同时比较：`sample_id`、`group_id`、`y_g10`、`y_g15`、规则事件标签、同一 `LTM_CODE`。任意真值/排序/分组漂移都应立即失败；**不仅比较 cohort ID**。
2. 检查每个 config 的三个 seed 是否有相同 `protocol.ablation`、同样输入大小、augmentation 开关、epoch/patience/batch/lr、相同 Manifest / 标签版本。将不可用的 provenance 字段标成 UNKNOWN，而不是猜测。
3. 如果不同配置唯一允许变化是几何/光度开关，就强制验证其余训练超参数一致。若用于比较不同数据规模则另设统计脚本，不混进 P1.7 的受控消融。
4. **NLL 口径**：原 `cohort_table()` 从 `metrics_raw.nll` 取值，却与 calibrated Brier 并排展示为无标记 `NLL`。同时输出明确的 `NLL(raw)` / `NLL(primary 或 calibrated)`，并为每个输出字段加来源测试。
5. CI 是 **25 个地理 Group 的组级 Bootstrap，在 3 个固定 seed 上求平均**，不应写成可覆盖所有新地区或随机初始化不确定性的正式保证。注明多配置/多规则探索性比较。
6. 原始校准概率缺失、分数无穷/NaN、事件头退化、跨 split 泄漏必须拒绝聚合，不得悄悄忽略。

**验收**：人工交换 2 Tile 的 `y_g10`、单独修改某 seed 的消融 flag、调换 `group_id`、破坏 provenance、raw/cal NLL 不同的 fixture 均能正确检测；保留 P1.7 旧报告原始结果作对照，但不可用最新逻辑静默重写旧指标。

## P18-04：固定结构/固定参数量的消融 + 合法增强

目标：分离“输入信息量”与“模型容量/训练策略”。

**建议实现**：新实验模式（例如 `--ablation-mode fixed-mask`）让 `cnn_nac`、`cnn_coarse_grid`、`cnn_fusion` 均保留**相同 3 通道网络入口、相同完整标量 head 维度及网络参数量**。缺失模态以固定的中性值掩码处理，而不是删除通道或改变 `nn.Linear` 形状。

- 全量布局：`[nac_contrast, nac_texture, coarse_slope]` 加上统一命名的 NAC 光度标量与几何标量；对没有的部分填训练集标准化后的 0，不得以真实值进入被掩码模型。
- `coarse_only`：两个 NAC 图像通道和 NAC 光度标量掩码；`nac_only`：粗坡度通道掩码；`fusion`：全输入。
- `no_geometry`：所有变体的几何标量统一 mask；`no_photometry`：NAC 对比度通道与 NAC 光度标量统一 mask，保留 NAC 纹理。
- 记录每个模型实际可用的通道、掩码字段、参数总数与可训练参数数，并在比较时断言一致；数值标签、loss 权重、early stopping、train/cal/val/test 必须固定。
- 实验 seed 42/43/44 起步，开发集使用旧 `development_test` 且标记探索性。输出 Fusion−Coarse 的 MAE/Brier 配对 CI、NLL 和相同 Coverage 的 UAR。
- **合法增强单独成组**：至少比较 `no-augment` 与 `--no-geometry --augment`。如果使用几何标量+空间增强，需实现并测试 flip/rot90 后的坐标系方位变换（同时明确 phase/incidence/emission 哪些不变）；未经几何证明不能再启用。

不要把旧的删除通道消融的置信区间解释为固定参数量下的信息价值；原结果保留在 `reports/p1_7_ablation.md`。

## P18-05：在继续模型选择前冻结 `heldout_v2`，防评估污染

**不是继续从旧 110 Test 中随机划一部分。** 从 90,786 索引中选**从未参与训练、验证、标定、旧测试、多轮模型选择、或开发集试验**的全新来源/站点，优先按 NAC 产品与地理块做互斥，再做足够的 footprint/空间边界审计。

**清单产物**：`protocols/heldout_v2_manifest.csv`、`protocols/heldout_v2_protocol.json`、`reports/p1_8_heldout_protocol.md`。必须包含选样算法、seed、完整 `sample_id`、`NAC_EDR_ID`、`group_id`、`LTM_CODE`、投影/地理 footprint、分组计数、缓冲最小距离、manifest 的 SHA256、冻结时间、生成代码 Git SHA。

**硬要求**
1. 形成 V2 ID 清单后**冻结并提交 SHA256**；冻结前不读取这些地区上的模型指标。建议避免在 V2 上做任何“先验阈值挑选”。
2. V2 与旧 train/cal/val/development_test、以及后续扩容 train 做双向零重叠校验；空间距离检查使用实际保留集合。按 NAC 产品、Group、5km 块、footprint 检查。
3. **DTM_source_ID 缺失要保留为 UNKNOWN**；如果无法做严谨 DTM 立体源独立，明确声明该限制，而不是暗示来源层面完全 OOD。
4. 进入 V2 前，先冻结候选模型/超参数/规则阈值/校准算法/报告指标，一次性计算最终结果；不得依据 V2 结果反复调整、换站点、重划分。
5. 如果无足够的新独立组，可返回 `BLOCKED_NEW_HELDOUT` 并解释，不可假装完成。

## P18-06：受控扩容（依赖 P18-01/02/05 全通过）

- 使用**本地** RX 7900 XT，先扩大到约 3,000 train Tile，留意存储/下载体积、NAC 有效像素比例和组分布；复核后再考虑 5,000。
- `cal` / `val` / 旧 development_test / 新 heldout_v2 的 ID 与分组全部冻结，禁止挪作 train。确认采样是稳定 append-only。
- 每批输出 `download_manifest`、每模态可读文件数量、SHA256 校验、正确配准比例与失败重试；相同 `sample_id` 的标签值不可漂移。
- 本地 GPU 先用小 batch 重跑 Constant、Coarse Ridge、Coarse CNN、Fusion CNN（固定参数量版本）；不要未经证据直接使用 LFM。
- 验收 `reports/p1_8_scaleup.md`：3 seed、真实样本计数、Group/Zone、VRAM、耗时、MAE/Brier/NLL/AUC/UAR vs Coverage 和置信区间。
- **V2 只执行一次性独立评估**；报告原始预注册与探索性分析的界限。

## 测试、提交与进度纪律

每次完整改动后：

```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
git status --short
```

用不触网的合成数据完成确定性回归测试，再用已下载的真实 760 Tile 做小型 **CPU/ROCm smoke**，确认 `--resume`、数据写回、模型/标量形状没有意外变化。**不能把尚未运行的 P1.8 实测填写到 P1.7 报告里**。

**交付纪律**：每完成一个 P18-xx，提交代码 + 对应测试 + 小段实测记录，更新 `reports/p1_8_progress.md`（状态 PASS/FAIL/BLOCKED 和证据：Commit SHA、命令、测试数、产物路径）。可按用户原有习惯直接 push `main`，但必须 `git pull --ff-only`，避免覆盖别人的新提交；如果仓库发生非快进冲突，先停止并协调。

**本轮止损**：P18-01/02 出现真实标签损坏风险、P18-03 发现队列不一致、或 P18-05 无法冻结干净的新地理队列，都应停止后续依赖动作并报告。不要自行购买 GPU、使用 4×4090、覆盖 PyTorch/共享磁盘、改动原始 DTM/NAC、或训练 LFM / 联合后验。用户未要求付费算力。

## Coding Agent 开始执行时应如何回复

先给出简短的“现有 main SHA / 本轮要修改的文件 / 预计先完成哪两项”，然后**立即实施 P18-01/02**；运行回归测试、验证 no-op resume 和 CSV schema 扩容安全，再推进 P18-03/04 和 V2。不要只提供设计意见，不要以“等待用户确认”替代已有授权范围内的本地代码修复。


## 执行状态（2026-10-17，本机实测）

代码 `a53937e`，148/148 测试通过。

- **P18-01 PASS**：no-op resume 死分支修复；exit 0/2 语义与计数器到位；真实数据 400→401 dry-run
  网络禁用时 exit 2 且 labels 逐字节不变。
- **P18-02 PASS**：原子合并 + 列并集 + 去重；dry-run 后原 760 行所有列值完全一致，cal/val/test 选样未被重排。
- **P18-03 PASS**：真值/事件/分组/zone、协议/provenance、概率质量 fail-fast；NLL 拆为 primary/raw 并交叉校验；
  在 P1.7 的 12 次运行上复检通过（缺失字段标 UNKNOWN，未覆盖旧结果）。
- **P18-05 BLOCKED（严格）**：272/272 泄漏组已全部被 760 Tile 覆盖 → `BLOCKED_NEW_HELDOUT`；
  另冻结 `protocols/heldout_v2_product_level_manifest.csv`（64 Tile / 22 未用产品 / 4 个已用 group，
  缓冲 ≥10.078 km，SHA256 `33f0898f…`），明确标注非组级独立。
- **P18-04 / P18-06 / P18-07 未开始**：04 的 fixed-mask 设计见 `reports/p1_8_progress.md`；
  06/07 依赖 04/05，且 V2 的严格定义需要先决定是否引入语料外 NAC 产品。

详见 `reports/p1_8_progress.md` 与 `reports/p1_8_heldout_protocol.md`。
