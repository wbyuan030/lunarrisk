# P3.0 执行进度

日期：2026-11-03。任务基线 `6dea815`；P2.2 负结果与旧 heldout 政策保持不变。

| 阶段 | 实际状态 | 证据与剩余门槛 |
|---|---|---|
| P30-00 | PASS（审计交付） | [固定来源、许可、磁盘、allowlist、传输与权重缺口](p3_0_benchmark_audit.md)；MG-MAE 权重 BLOCKED；NASA 输入/环境/大权重接入 BLOCKED |
| P30-01 | PASS（离线 contract 与真实部分 I/O）；完整官方等价性 BLOCKED | [loader 验证](p3_0_loader_validation.md)；[28 通道固定行配对](p3_0_mmap_pair.md) 的 172,032 像素误差/mask 差异均为 0；仍是 HDF5_DERIVED_PILOT |
| P30-02 | PASS（真实五类工程 smoke）；正式基线 BLOCKED | 首批单类 smoke 保留；[新增 5 Train/5 Val 五类 smoke](p3_0_age_fiveclass_smoke.md)，seed42/2 次更新，checkpoint 独立进程回放 CSV 字节相同，无正式分类性能结论 |
| P30-03 | HOLD | 正式数据/普通头基线与公平配置未就绪，未训练 Geology/候选头 |
| P30-04 | HOLD | 尚无锁定扩展实验协议 |

首次交付离线验证：471 tests，470 PASS、1 SKIP（隔离源码快照缺旧 labels 可选数据）；compileall exit 0。此前漏带 protocols 的快照报 12 个缺文件错误，修正后在全新目录重跑。真实网络首次超时为 0 新响应体，转发已有 proxy 后成功。所有日志使用本轮真实结果；详见 [验证记录](evidence/p30_20261103/validation.json)。

最新累计实际响应体 **78,502,175 字节**（约 74.9 MiB），保守计数 **83,468,143 字节**，连续承接初次 1 GiB 上限。未下载完整 HDF5/mmap/权重，未请求 Test 像素。源码/manifest/脱敏日志与元数据派生表入 Git；像素、checkpoint、缓存保留本地。首次证据 [p30_20261103](evidence/p30_20261103/) 保持原样；新增证据为 [mmap 探针](evidence/p30_mmap_20261103/) 与 [五类 smoke](evidence/p30_fiveclass_20261103/)。

运行源码 commit `2f064fd6b83e923dca993bbc90f63dbce2d3d58e`：8 patch Age CNN 工程 smoke 与独立进程回放完成，8 行预测/40 个概率精确相同，最大差 0。checkpoint 和逐样本概率均有 SHA256；不读取 Test。已从该提交快照再次验证 471 tests（470 PASS/1 SKIP）与 compileall exit 0，见 `evidence/p30_20261103/age_smoke/`。checkpoint 保留本地、不入 Git。

本次续推：`bfe37b3` 先锁定 mmap 探针再运行，168 次请求/672 KiB、28 通道三行配对精确相同，但未冒充完整等价性。`1e32a7b` 锁定五类 ID/训练配置后运行 10-patch 工程 smoke，10 行/50 个概率回放最大差 0；checkpoint 与输入/源码 hash 已核对。最新完整回归 **478 tests，477 PASS/1 旧数据可选 SKIP，compileall exit 0**。

下一门槛：覆盖有效模态的完整裁片配对，以及适当规模、多地质类别 Train/Val 的预先固定数据/选择协议；正式 Age/Geology、公平候选条件化矩阵与预训练轨继续 BLOCKED/HOLD。当前五类工程 batch 不能替代正式基线。完整下载方案已写入资产审计，尚未执行。

旧 LROC Gate 3 **NOT_FROZEN**、Gate 4 **HOLD**；P2.2 **STOP_OR_REFRAME / H1/H2 NOT ESTABLISHED**。没有付费资源、四卡、环境替换或旧资产修改。
