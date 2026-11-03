# P3.0 执行进度

日期：2026-11-03。任务基线 `6dea815`；P2.2 负结果与旧 heldout 政策保持不变。

| 阶段 | 实际状态 | 证据与剩余门槛 |
|---|---|---|
| P30-00 | PASS（审计交付） | [固定来源、许可、磁盘、allowlist、传输与权重缺口](p3_0_benchmark_audit.md)；MG-MAE 权重 BLOCKED；NASA 输入/环境/大权重接入 BLOCKED |
| P30-01 | PASS（离线 contract 与真实部分 I/O）；官方等价性 BLOCKED | [loader 验证](p3_0_loader_validation.md)；16,200 行真实 metadata 与 8 个 Train/Val patch；HDF5_DERIVED_PILOT |
| P30-02 | READY_FOR_ENGINEERING_SMOKE | 已有 4 Train/4 Val 像素，有限普通 CNN runner；尚未运行，不能报告训练成功或 benchmark 成绩 |
| P30-03 | HOLD | 正式数据/普通头基线与公平配置未就绪，未训练 Geology/候选头 |
| P30-04 | HOLD | 尚无锁定扩展实验协议 |

最终离线验证：471 tests，470 PASS、1 SKIP（隔离源码快照缺旧 labels 可选数据）；compileall exit 0。此前漏带 protocols 的快照报 12 个缺文件错误，修正后在全新目录重跑。真实网络首次超时为 0 新响应体，转发已有 proxy 后成功。所有日志使用本轮真实结果；详见 [验证记录](evidence/p30_20261103/validation.json)。

累计实际响应体 **39,016,735 字节**，保守计数 **43,982,703 字节**，初次上限 1 GiB。未下载完整 HDF5/mmap/权重，未请求 Test 像素。源码/manifest/脱敏日志与元数据派生表入 Git；像素、checkpoint、缓存保留本地。完整执行证据在 [p30_20261103](evidence/p30_20261103/)。

下一项已授权动作：以当前固定 8 patch、HF stats、seed42 做 2 次普通 Age CNN 更新，保存可重载 checkpoint 和逐样本概率，再用独立进程回放。该 smoke 只验证工程链路；官方 mmap 等价性、预训练权重与正式矩阵仍需解决各自门槛。

旧 LROC Gate 3 **NOT_FROZEN**、Gate 4 **HOLD**；P2.2 **STOP_OR_REFRAME / H1/H2 NOT ESTABLISHED**。没有付费资源、四卡、环境替换或旧资产修改。
