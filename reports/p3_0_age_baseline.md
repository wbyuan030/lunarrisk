# P3.0 Age 普通头：真实像素工程 smoke

2026-11-03；运行源码为提交 **`2f064fd6b83e923dca993bbc90f63dbce2d3d58e`** 的全新快照。状态 **REAL_PIXEL_ENGINEERING_SMOKE_ONLY / HDF5_DERIVED_PILOT**。训练、保存和独立进程 checkpoint 回放 PASS；正式 Age benchmark 仍未完成。

## 固定数据与训练

固定 HF revision、全局统计和 class map 见 [资产审计](p3_0_benchmark_audit.md)。使用已提取真实 patch，不追加网络传输，不读取 Test 像素。

| 项目 | 实际配置/结果 |
|---|---|
| Train ID | `0,1,2,3` |
| Val ID | `20,21,22,23` |
| 两个 split 的五类计数 | 均为 `[0,4,0,0,0]` |
| 输入 | 28×256×256，固定 HF 全局 z-score，标签只作为 target |
| 模型 | 上游 ScratchAgeClassifier 的普通 CNN 架构，28→64→128→256→5；459,077 参数 |
| 训练 | seed42、deterministic algorithms、AdamW lr=0.001、batch4、2 次更新 |
| Train cross-entropy | `1.6874988079071045 → 1.4170664548873901` |
| 训练/预测记录窗口 | 2.9383 秒 |
| Torch 峰值 allocated/reserved | 117,457,920 / 140,509,184 字节 |
| 硬件/运行时 | RX 7900 XT；torch 2.5.1+rocm6.2；HIP 6.2.41133-dd7f95766 |

采样规则在预测前固定为“每个存储 Train/Val split 最前四个有效 Age 行”，没有按模型分数挑样本。这些行恰在北极相邻位置且全部为 Nectarian；不能代表五类任务或独立区域。loss 降低只验证梯度/优化器能工作，不报告本批 Accuracy/Macro-F1 为 benchmark 成绩，不用于架构/超参选择。

时间从模型/优化器构建之后计起，包含本轮批次读取、训练和预测；不含 Python 冷启动、依赖导入和网络获取。显存为 PyTorch allocator 计数，不等于整个 GPU 进程/驱动占用，不据此推断完整训练吞吐或大模型显存。

运行时打印 hipBLASLt 不支持当前架构并回退 hipblas 的 PyTorch warning；两个进程都完成且结果一致，原文保留在脱敏日志中。没有改环境、安装 torch、使用预训练权重或四卡/云资源。

## checkpoint 与回放证据

checkpoint 保存模型与 optimizer state、更新数、seed、CPU/GPU RNG、输入文件 hash、完整 stats/map、通道顺序、样本 ID、源码 commit 与 runner/loader SHA256。当前 runner 支持 checkpoint 加载与预测回放；没有声称已测试中断后继续训练的轨迹等价性。

- checkpoint SHA256：`05d57fe20ed038aa1ae31b1cf178eedaebc19ca61fb1cd340436b1f9d481c6d9`
- pilot SHA256：`d6807ecc1a0d52f07fc46cc53f42146e3b66e9562ee9fd7357f5dfbf32d9bd0a`
- 原训练与独立 Python 进程回放的 8 行 CSV **字节相同**，sample/split/target 顺序相同，40 个概率的最大绝对差 **0.0**。
- CSV SHA256：`4b72c374e57e0cc7b931564ea1b899bcbafd77979a6a6df0de7c96e3ef0591f0`

公开证据：[训练 report](evidence/p30_20261103/age_smoke/train_report.json)、[逐样本概率](evidence/p30_20261103/age_smoke/train_predictions.csv)、[回放验证](evidence/p30_20261103/age_smoke/replay_verification.json)、[真实训练日志](evidence/p30_20261103/age_smoke/train.log)、[回放日志](evidence/p30_20261103/age_smoke/replay.log)。checkpoint/原始像素不入 Git；以本地 `P30_OUTPUT_ROOT` 表示输出根，保留 `age_pilot_01.h5`、`age_smoke_01/checkpoint.pt`、`age_replay_01/`。

实际命令（以变量替换本机位置；从提交快照执行，无需网络）：

```bash
export LUNARRISK_GIT_SHA=2f064fd6b83e923dca993bbc90f63dbce2d3d58e
export OMP_NUM_THREADS=1
python3 -I scripts/p30_age_smoke.py --pilot "$P30_OUTPUT_ROOT/age_pilot_01.h5" --stats reports/evidence/p30_20261103/hf_channel_stats.json --class-map reports/evidence/p30_20261103/geologic_units.json --outdir "$P30_OUTPUT_ROOT/age_smoke_01" --seed 42 --steps 2 --device cuda
# 必须另起 Python 进程与新输出目录：
python3 -I scripts/p30_age_smoke.py --pilot "$P30_OUTPUT_ROOT/age_pilot_01.h5" --stats reports/evidence/p30_20261103/hf_channel_stats.json --class-map reports/evidence/p30_20261103/geologic_units.json --outdir "$P30_OUTPUT_ROOT/age_replay_01" --seed 42 --steps 2 --device cuda --replay "$P30_OUTPUT_ROOT/age_smoke_01/checkpoint.pt"
```

从同一提交快照重新执行全套 unittest 和 compileall：**471 tests / 470 PASS / 1 SKIP，compileall exit 0**。唯一 SKIP 为旧标签可选数据未复制入源码快照；新增 HDF5 tests 均实际运行。见 [提交源码验证](evidence/p30_20261103/age_smoke/validation.json) 与 [完整日志](evidence/p30_20261103/age_smoke/tests_committed_471.log)。公开证据内容 hash 清单为 [checksums.json](evidence/p30_20261103/checksums.json)。

## 下一阶段门槛

P30-02 的最小真实工程 smoke 已完成，正式基线仍需足够且预先固定的多类 Train/Val 数据、训练预算与选择规则，以及 HDF5/mmap 配对归一化证据或明确另设 derived 协议。预训练轨仍被 MG-MAE 权重缺口、NASA 输入/运行时/大权重获取门槛阻塞；不以 scratch 训练冒充领域预训练。

Geology 普通头与候选条件头未训练；本批单一类别数据不支持正式公平比较。下一步先解决 loader 的固定 ID 配对与数据获取/协议门槛，再锁定同 encoder/输入/ID/预算的普通头、候选头与 masked-softmax 对照。没有模型 Test 评分，旧 LROC Gate 4 继续 HOLD。
