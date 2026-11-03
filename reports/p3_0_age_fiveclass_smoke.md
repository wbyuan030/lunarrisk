# P3.0 五类 Age 工程 smoke

2026-11-03。运行源码锁定为 `1e32a7bc3a555d64135381c52ec2d882b5ad36be`；输入协议为 **HDF5_DERIVED_PILOT**。10 个真实 patch 的提取、五类 CNN 工程 smoke 与独立进程 checkpoint 回放均 PASS；不报告正式 benchmark 性能。

## 读取前固定的协议

[预先提交的样本与训练计划](../protocols/p30_age_fiveclass_pilot_plan.json) 按官方 Age 顺序，在存储 Train、Val 内分别取每类第一个有效原始索引。Train ID 为 `[65,0,81,245,1333]`，Val ID 为 `[130,20,85,560,2289]`；每个 split 的目标序列 `[0,1,2,3,4]`，不借用 Test。ID 0/20 与第一轮 pilot 重叠，不当作新增独立样本。

抽样使用监督标签仅为固定工程测试批次，不进入模型输入；该分层小样本不代表完整官方分布。样本仍主要位于北半球，不声称空间独立或泛化。现有 8-ID mmap 三行探针完全匹配支持保留既定 HF stats，但不证明新 10-ID 的完整 mmap 等价性。

沿用原 459,077 参数 Age scratch CNN、seed42、deterministic algorithms、AdamW lr=0.001、2 次更新，固定后不因 Val 结果增加训练或改模型。无预训练权重；输出逐样本概率和可加载 checkpoint，独立进程只回放、不再优化。无正式 Accuracy/Macro-F1、模型收益、Cal 或 Test 评分。

本次提取承接保守累计 44,670,831 字节；累计 cap 268,435,456，低于整个初轮 1 GiB 上限。没有重置账本，原始像素/checkpoint 不入 Git。

## 验证状态

支持五类选择的完整回归为 **478 tests：477 PASS、1 SKIP**，compileall exit 0。跳过仍为隔离源码快照缺旧 labels 的可选检查。见 [验证日志](evidence/p30_fiveclass_20261103/validation.json)。

## 真实运行结果

| 项目 | 实测 |
|---|---|
| Train / Val 五类计数 | 两个 split 均为 `[1,1,1,1,1]` |
| Train cross-entropy（2 次更新） | `1.6167749166488647 → 0.7659187316894531` |
| 模型参数 | 459,077，与首轮相同 |
| 本地训练/预测窗口 | 1.9590 秒，不含导入/模型构建/网络获取 |
| Torch 峰值 allocated / reserved | 148,259,840 / 182,452,224 字节，不等于驱动全进程占用 |
| checkpoint 独立进程回放 | 10 行 sample/split/target 与 50 个概率相同；CSV 字节相同，最大差 0 |
| 新增响应体 | 38,797,312 字节；无自动重试 |
| 初轮累计实际 / 保守响应体 | 78,502,175 / 83,468,143 字节（约 74.9 / 79.6 MiB），低于 1 GiB |

这些样本只有每类每 split 一张，没有独立性能估计意义。训练 loss 下降只验证多类优化链路，不用于调参或声明准确率收益。运行时沿用既有 hipBLASLt 不支持架构后回退 hipblas 的 warning，两个进程正常完成；无环境替换。

pilot SHA256：`cf106933772b7c08ba973cf9c7cfdfdfbcb9e2d038fb87d34f7e05c4cc2b8a35`；checkpoint SHA256：`1b463c3b459f4e1ed6012ab03f263d96d8fcc04cd6a58ea735de04e535a7de1f`；预测 CSV SHA256：`0029925fd03303bee48c08b1c662e8345a1d6ab02c8c8a58e3421b367c44573b`。重叠 ID 0/20 的 raw patch hash 与第一轮逐字节相同，没有改写旧 pilot。

公开证据：[提取审计](evidence/p30_fiveclass_20261103/extraction.json)、[训练 report](evidence/p30_fiveclass_20261103/train_report.json)、[逐样本概率](evidence/p30_fiveclass_20261103/train_predictions.csv)、[回放核验](evidence/p30_fiveclass_20261103/replay_verification.json)、[累计账本](evidence/p30_fiveclass_20261103/transfer_ledger.json)。原始像素与 checkpoint 只保留本地；`P30_OUTPUT_ROOT` 下为 `age_fiveclass_pilot_01.h5`、`age_fiveclass_smoke_01/checkpoint.pt`、`age_fiveclass_replay_01/`。

实际提取参数为 `--selection first_per_age_class --per-split 5 --prior-transfer-bytes 44670831 --transfer-cap 268435456 --apply`，来源 metadata 与 HF revision 不变。随后使用原 `p30_age_smoke.py`、上述 pilot、同一 HF stats/class map，设置 `LUNARRISK_GIT_SHA=1e32a7bc3a555d64135381c52ec2d882b5ad36be`、`OMP_NUM_THREADS=1`、`--seed 42 --steps 2 --device cuda`；另起 Python 进程加 `--replay` 指向 checkpoint 和新输出目录。源码文件 SHA256 与提交内容已核验，完整命令参数见锁定协议与脱敏日志。

## 后续门槛

真实输入、五类目标、训练和恢复链路已具备，但当前样本/等价性证据不足以启动正式研究矩阵。下一步应预先固定覆盖有效模态与多地质类别的 Train/Val 规模、完整配对方法、预算与选择规则，再比较 Geology 普通头/候选头/masked-softmax；不把本轮抽样升级为官方全量复现。MG-MAE 权重和 NASA 输入/运行时/大权重门槛继续阻塞预训练轨。旧 LROC Gate 4 HOLD。
