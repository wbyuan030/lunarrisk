# P3.0 执行进度

日期：2026-11-03。任务基线 `6dea815`；P2.2 负结果与旧 heldout 政策保持不变。

| 阶段 | 实际状态 | 证据与剩余门槛 |
|---|---|---|
| P30-00 | PASS（审计交付） | [固定来源、许可、磁盘、allowlist、传输与权重缺口](p3_0_benchmark_audit.md)；MG-MAE 权重 BLOCKED；NASA 输入/环境/大权重接入 BLOCKED |
| P30-01 | PASS（离线 contract 与真实部分 I/O）；完整官方等价性 BLOCKED | [loader 验证](p3_0_loader_validation.md)；[28 通道固定行配对](p3_0_mmap_pair.md) 的 172,032 像素误差/mask 差异均为 0；仍是 HDF5_DERIVED_PILOT |
| P30-02 | PASS（真实五类工程 smoke）；正式基线 BLOCKED | 首批单类 smoke 保留；[新增 5 Train/5 Val 五类 smoke](p3_0_age_fiveclass_smoke.md)，seed42/2 次更新，checkpoint 独立进程回放 CSV 字节相同，无正式分类性能结论 |
| P30-03 | PASS（同特征工程链路）；正式矩阵 HOLD | [49 输出 Geology 两头](p3_0_geology_baselines.md) 在既有 5 Train/5 Val 上完成，20 行/980 概率独立回放精确相同；不是正式基线或预训练收益 |
| P30-04 | HOLD | 尚无锁定扩展实验协议 |

首次交付离线验证：471 tests，470 PASS、1 SKIP（隔离源码快照缺旧 labels 可选数据）；compileall exit 0。此前漏带 protocols 的快照报 12 个缺文件错误，修正后在全新目录重跑。真实网络首次超时为 0 新响应体，转发已有 proxy 后成功。所有日志使用本轮真实结果；详见 [验证记录](evidence/p30_20261103/validation.json)。

最新累计实际响应体 **78,502,175 字节**（约 74.9 MiB），保守计数 **83,468,143 字节**，连续承接初次 1 GiB 上限。未下载完整 HDF5/mmap/权重，未请求 Test 像素。源码/manifest/脱敏日志与元数据派生表入 Git；像素、checkpoint、缓存保留本地。首次证据 [p30_20261103](evidence/p30_20261103/) 保持原样；新增证据为 [mmap 探针](evidence/p30_mmap_20261103/) 与 [五类 smoke](evidence/p30_fiveclass_20261103/)。

运行源码 commit `2f064fd6b83e923dca993bbc90f63dbce2d3d58e`：8 patch Age CNN 工程 smoke 与独立进程回放完成，8 行预测/40 个概率精确相同，最大差 0。checkpoint 和逐样本概率均有 SHA256；不读取 Test。已从该提交快照再次验证 471 tests（470 PASS/1 SKIP）与 compileall exit 0，见 `evidence/p30_20261103/age_smoke/`。checkpoint 保留本地、不入 Git。

本次续推：`bfe37b3` 先锁定 mmap 探针再运行，168 次请求/672 KiB、28 通道三行配对精确相同，但未冒充完整等价性。`1e32a7b` 锁定五类 ID/训练配置后运行 10-patch 工程 smoke，10 行/50 个概率回放最大差 0；checkpoint 与输入/源码 hash 已核对。该阶段完整回归 **478 tests，477 PASS/1 旧数据可选 SKIP，compileall exit 0**。

本轮继续至讨论门槛：提交 `baac7dc` 锁定并实现同特征 Geology 普通 MLP / candidate-ID scorer，复用已知来源的 Age scratch CNN 冻结特征，无网络新增。两头 49 输出、各 2 次更新，20 行/980 概率独立进程回放字节相同；候选排列最大差 3.73e-9，普通头完整 mask 最大差 0。最新回归 **482 tests，481 PASS/1 旧标签可选 SKIP，compileall exit 0**。证据在 [p30_geology_20261103](evidence/p30_geology_20261103/)。

当前已到需要讨论的节点，详见 [下一阶段方案](p3_0_next_decision.md) 和 [未批准的数据获取提案](../protocols/p30_next_acquisition_proposal.json)：是否先批准唯一固定 HDF5 文件的 74,449,119,347 字节下载、开展数据充足的 scratch/shared-encoder 基线，或优先取得可核实 MG-MAE 权重后再正式比较。现有本机磁盘约 479.6 GiB 可用，不需要现在购买 GPU/磁盘。完整配对、正式实验/候选语义协议仍需锁定；不能继续用十张图 smoke 代替科研证据。初次 1 GiB 范围未修改，提案无自动下载/审批效果。

旧 LROC Gate 3 **NOT_FROZEN**、Gate 4 **HOLD**；P2.2 **STOP_OR_REFRAME / H1/H2 NOT ESTABLISHED**。没有付费资源、四卡、环境替换或旧资产修改。
