# LunarRisk P1.8 → P1.9：下一轮执行交接（2026-10-17）

> **执行状态**：本文件是 Coding Agent 的动作清单，不是已完成的实测报告。基线为 `main@8628340`（P18-01/02/03 PASS；P18-05 严格模式 BLOCKED），包含之后的冻结元数据修复提交。历史结果见 `reports/p1_8_progress.md`、`reports/p1_8_heldout_protocol.md` 与 `reports/p1_7_ablation.md`。
>
> **现在直接做 P18-04 和独立数据来源可行性调查**，无需重复已经通过的 P18-01/02/03。此阶段不要求用户再次授权非付费、本地实验。不要把过去 760 张 Tile 或其源数据改写、删除、移动，也不要新建付费算力。

## A. 已核验状态与研究判断

| 项 | 当前状态 | 重要限制 |
|---|---|---|
| P18-01/02 | PASS（Coding Agent 报告 148/148 测试，含真实标签备份的 400→401 dry-run） | 不等于已经完成 3k 下载 |
| P18-03 | PASS（更严格的跨 seed/config 协议、真值/概率检查和 NLL 口径） | 历史 P1.7 部分 provenance 为 UNKNOWN |
| P18-04 | **TODO** | 当前删除通道消融改变模型参数数量 |
| P18-05 严格 Group heldout | **BLOCKED_NEW_HELDOUT** | 100k 发布目录的 272/272 个泄漏 Group 都被此前 760 Tile 触及 |
| P18-05 产品级候选 | **FROZEN_PRODUCT_LEVEL_NOT_GROUP_DISJOINT** | 64 Tile、22 个未使用 NAC 产品、4 个**已使用 Group**，最近中心距 10.078km；只支持有限的产品级迁移声明 |
| P18-06/07 | **NOT STARTED** | 不得冒充已通过独立地区/DTM 来源泛化 |

**不要将 P1.6 更好的结果武断归因为「增强伪影」**：已知的是它在关闭增强后没有稳定复现。合法增强可能提供真正的正则化收益；必须单独对照。

## B. 最高优先级：P18-04 固定模型容量的模态消融

### B1. 工程实现

保持已有 `legacy` 模式和既有 P1.7 结果不变，另加入 `fixed-mask` 模式（建议参数 `--ablation-mode legacy|fixed-mask`）：

- 每个 CNN 统一 3 通道空间输入 `[nac_contrast, nac_texture, coarse_slope]`、统一完整标量特征维度，`TinyCNN(in_channels=3, n_scalars=FULL_SCALAR_COUNT)`；**所有变体参数量必须相同**。
- `cnn_nac`：粗坡度网格通道置 0；`cnn_coarse_grid`：两路 NAC 通道及 NAC 光度标量置 0；`cnn_fusion`：全量；全局 `no_geometry`、`no_photometry` 在相同 tensor 形状中置 0，保留 NAC 纹理。
- 标量特征先仅在 Train 上 `StandardScaler.fit`，再用已标准化的 0 作为中性值；不能通过 NaN、缓存缺失或不一致的零值偷偷泄漏被屏蔽的信息。不要在标准化前把原始物理量直接置 0 后误认为「均值」。
- Mask 必须应用在 `train` / `val` / `cal` / `test` 所有路径；反向传播和推理一致，不要只在训练阶段掩码。
- `metrics.json.protocol` 需记录：`ablation_mode`、各变体可见/屏蔽通道、mask 的标量字段、总参数与可训练参数数、网络架构配置、输入 hash/provenance。
- 若当期只能做原架构删通道消融，可以保留为 `legacy`，但禁止作为「固定容量」证据。

### B2. 必须新增的离线测试

1. 三个变体的完整形状一致且参数总数严格相同（对所有消融配置）。
2. 改动被屏蔽通道/标量的输入数值，对某模型推理输出**无影响**；改变可见通道应可能影响输出。
3. 不同 Split 的掩码配置一致；序列化/加载后掩码仍生效。
4. `--no-nac-photometry` 只屏蔽绝对对比度 + 光度标量，保留纹理；`--no-geometry` 屏蔽几何标量。
5. 新 `fixed-mask` 的聚合协议严格拒绝混入 `legacy`、不同参数量、不同 mask 方案或不同数据/Seed。
6. 小规模 1 Epoch CPU synthetic smoke；其结果只能作工程验证，不能标记为月球地形物理实测。

### B3. ROCm 实测（不扩容、不看 64 Tile V2 指标）

先用已有 760 个 hydrated Tile、既有 Train/Val/Cal/开发 Test，种子 42/43/44，跑固定容量的 full / no_geometry / no_photometry / no_both 四组；必要时可减少组合为最有解释力的 pair，以确保对照清楚。

- 记录按规则的 `g10/g15` MAE、Brier(raw/primary)、NLL(raw/primary)、AUC、UAR/HMR@20/40%，Fusion−Coarse 的配对地理 Group Bootstrap CI。
- 对常数基线的同分概率使用并列随机放行的**期望**，不是 CSV 行序；保持统计定义一致。
- 报告三种子分散度、模型参数量、GPU 时间、数据/源路径与 SHA256。
- 把结果写到 `reports/p1_8_fixed_mask.md`，旧 110 Test 明确标记 `development_test`，结果**仅作探索性**。
- **对照合法增强**：使用 `--no-geometry` 下的空间增强开/关两组配对实验。若拟对带太阳方位的输入增强，必须先实现并测试方向矢量/图像坐标的同步变换，不能直接忽略方位几何。

## C. 并行推进：严格独立测试语料调研（P18-05 的解除条件）

当前“272/272 Group 被触及”是**基于现有 100k Manifest 与现行 Group 定义**的结论，不是「整个 SomBench、NASA 所有影像都没有新区域」的结论。

- 优先审计官方或可合法获取的更完整 NAC/SomBench 数据发布，以及 USGS/LROC 公开立体 DTM 产品；**先获取索引/元数据、文件大小与许可证，再决定是否需要下载**。
- 输出候选来源的真实可访问链接、索引行数、NAC 产品 ID、DTM stereo 产品 ID（若有）、月面 footprint、地理分布、许可/下载上限、估算磁盘。
- 与现有 760 个历史已用 Tile + 100k manifest 已用来源比对，选择从未用于模型训练、验证、校准、开发评估、模型选择的 Group/产品/地理区域；严格记录双向重叠与缓冲距离。
- **不要通过重新划分 272 个已经用过的 Group、或把它们改名来创造「新」Test。** 如果只能做到新 NAC 产品、中心距 10km，而 Group 仍重合，必须称「产品/位置 holdout，非 Group 独立」。
- 现有 `protocols/heldout_v2_product_level_manifest.csv` 与协议原始 SHA256 **必须保持不变**；这是备用的受限迁移集。不得重跑覆盖这些冻结文件，不得拿其指标迭代模型。
- 固定协议产品级模式的 `algorithm.exclusions` 现在仅列实际排除项（sample ID、NAC Product），不会错误声称排除了 used Group；但仓库中**之前生成的冻结 JSON 是历史快照**，如果其中 `git_sha: UNKNOWN`、误写 group exclusion，应保留原件并写补充审计说明，不得偷偷改写冻结文件。
- DTM_source_ID / stereo 立体像对 provenance 仍然 UNKNOWN 时，任何评估都不能表述为 DTM 来源完全独立。
- 报告路径：`reports/p1_8_new_heldout_sources.md`。结论只能是 `GO_STRICT_OOD`（含可执行数据与冻结脚本）、`PRODUCT_ONLY` 或 `BLOCKED`，每项有证据。

## D. 扩容 Gate：先能解释、再跑大规模

在 **P18-04 的固定参数量消融可复现**、以及 C 的新测试协议得到可审计方案前，**不启动 3–5k 的正式训练下载**。

准备可以先做：磁盘审计、下载容量估算、候选 train IDs 的 stable append-only 选择演练、已有 Val/Cal/Test 和冻结产品级 V2 ID 的排除清单、ROI 配准与 valid fraction 分布分析。

若严格独立测试来源暂不可得：
- 允许单独做**开发性质的**训练规模敏感性分析（例如先 1k，再 3k），但必须明确其不能回答 Group-OOD 泛化问题，且下载应在磁盘安全和 ID 审核通过后启动；
- 绝不把产品级 64 Tile 当成严格独立 OOD 的替代品，也不宣称获得着陆安全认证。

## E. 下一次交付/提交

每个阶段独立提交并更新 `reports/p1_8_progress.md`（真实证据，不是预期结果）：

1. `P18-04a`：fixed-mask 功能 + 单测 + 1 Epoch smoke；提交至 `main`。
2. `P18-04b`：7900 XT 受控 3 Seed 消融与合法增强对照，提交 `reports/p1_8_fixed_mask.md`。
3. `P18-05b`：新语料候选调查、真实来源与 Group/DTM provenance 审计，提交 `reports/p1_8_new_heldout_sources.md`。
4. 再依据结果决定 `P18-06/07`（数据扩容与最终测试）；若 BLOCKED，清楚说明缺什么，不通过调整旧 Test 按钮「解锁」。

完整测试：
```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
git status --short
```

**资源/安全**：本地 AMD 7900 XT，使用现有可工作的 ROCm PyTorch；不得未经单独同意覆盖运行环境/共享数据、购买云卡、使用周末 4×4090、做 LFM 大模型训练、重命名/替换已经冻结的 V2 文件。

**Coding Agent 请直接动手**：拉取最新 main，先实现 `P18-04a`，同时调研 C 的公开索引，不要重复执行已经 PASS 的数据标签修复；完成后测试、记录实际结果并按仓库既有流程推送 `main`。
