# P3.0 Geology 同特征工程验证

2026-11-03。本阶段只验证 49 输出、同特征普通头/候选 ID 头、缓存与恢复接口，**不是正式 Geology benchmark、月球预训练实验或候选头收益结论**。输入仍为 HDF5_DERIVED_PILOT。

## 运行前固定的配置

[锁定计划](../protocols/p30_geology_engineering_plan.json) 复用已有 5 Train/5 Val，无网络、无新增数据。两 split 的 Geology target 序列均为 `[47,43,32,4,0]`，输出仍完整 49 类；只出现 5 类不足以估计主任务表现。样本 ID 与五类 Age pilot 不变，不重切 split、不读 Test。

共同前端为上一轮 Age scratch CNN 的卷积/池化特征，已在同一 5 张 Train 的 Age 标签上训练 2 次；这里 eval、冻结、去除 5 类输出层，只计算一次 256-D 特征缓存。它不是 MG-MAE/NASA 领域预训练。Age 与 Geology 真值相关，必须记录该额外监督来源；两头共享同一份缓存与来源，Val 不更新前端。

| 头 | 结构 | 参数 |
|---|---|---:|
| ordinary | 256→40→49 MLP | 12,289 |
| candidate_id | 16-D 公共类 ID embedding，与特征拼接后共享 272→42→1 scorer | 12,293 |

参数差为 4，不能声称严格完全相同参数。每张图均输入相同完整 49 个候选 ID，没有逐样本真实单位名/ID作为特征，没有只给正确选项的提示。candidate_id 独立打分，不使用候选之间的交互或自然语言类描述；它只验证最小候选 ID 接口，不证明语义迁移或未见类别泛化。

两头固定 seed42、AdamW lr=0.001、同一 Train batch5、各 2 次更新；不按 Val 选模型。固定排列检验阈值 1e-6；普通头完整候选 mask 需等于 softmax。masked-softmax 子集控制与候选排列/子集 API 有 synthetic 测试，但不运行候选子集性能扩展矩阵。

## 验证状态

新增 4 个 synthetic 测试覆盖候选排列/子集语义、普通分类器 mask 后归一化、空/重复/非法 ID、参数预算与 49 输出。完整 ROCm 环境回归 **482 tests，481 PASS、1 旧标签可选检查 SKIP**；compileall exit 0。

## 真实运行与回放

从提交 `baac7dc6f6e7750957cc185156de35e787f422c6` 的全新快照运行。两个头读取同一份冻结特征 cache，各做 2 次 Train 更新；Val 仅输出概率，不参与选择。所有 loss、梯度和概率有限。

| 检查 | 实测 |
|---|---|
| ordinary Train CE | `3.992940902709961 → 3.975635051727295` |
| candidate_id Train CE | `3.917956590652466 → 3.8966808319091797` |
| 候选逆序并对齐后的概率最大差 | `3.725290298461914e-09`，低于事前 1e-6 |
| ordinary 全候选 masked-softmax 与 softmax 最大差 | `0.0` |
| 独立进程恢复两头、同 cache | 20 行、980 个概率最大差 `0.0`；CSV 字节相同 |
| 训练/特征提取/预测记录窗口 | 1.4527 秒，不含进程导入，不代表正式全量吞吐 |
| Torch 峰值 allocated / reserved | 142,609,920 / 157,286,400 字节，不等于全进程 GPU 占用 |
| 新增网络响应体 | 0 字节；初轮累计仍为 78,502,175 实际 / 83,468,143 保守字节 |

Train CE 不可作为架构收益比较：初始化、参数形式不同，本批只有 5 个类、5 个 Train，且前端此前使用相关的 Age 监督。当前不计算/选择 Val 精度，也不从较低 loss 推断候选头更好。

feature cache SHA256：`28a4d4c26d3c12c8fdc32fd9fd0537d2c54051ab54a6a4bbc2273fb0b007eb63`；checkpoint SHA256：`0400fc820728ce9bb21c44876ca9ca8fc836357f9fe49adaa26147cf5f826953`；CSV SHA256：`183022074305d471ef739afdb0414d7245214a0bc762c00aea4227bb9dd45133`。原 cache 与 checkpoint 留在本地 `P30_OUTPUT_ROOT/geology_smoke_01/`，回放输出在 `geology_replay_01/`。这里测试的是恢复预测，不声称中断后续训轨迹等价。

公开 [训练 report](evidence/p30_geology_20261103/train_report.json)、[逐样本概率](evidence/p30_geology_20261103/train_predictions.csv)、[独立回放核验](evidence/p30_geology_20261103/replay_verification.json)、[测试/脱敏日志](evidence/p30_geology_20261103/validation.json) 保留源码与输入 hash。运行记录代码缩写 `baac7dc` 已解析为上述完整 commit 并逐文件核对。

实际命令从固定源码快照、既有 ROCm 环境运行：

```bash
export OMP_NUM_THREADS=1
export LUNARRISK_GIT_SHA=baac7dc
python3 -I scripts/p30_geology_smoke.py --pilot "$P30_OUTPUT_ROOT/age_fiveclass_pilot_01.h5" --encoder "$P30_OUTPUT_ROOT/age_fiveclass_smoke_01/checkpoint.pt" --stats reports/evidence/p30_20261103/hf_channel_stats.json --class-map reports/evidence/p30_20261103/geologic_units.json --outdir "$P30_OUTPUT_ROOT/geology_smoke_01"
# 独立进程、新输出目录，使用同一 feature cache：
python3 -I scripts/p30_geology_smoke.py --pilot "$P30_OUTPUT_ROOT/age_fiveclass_pilot_01.h5" --encoder "$P30_OUTPUT_ROOT/age_fiveclass_smoke_01/checkpoint.pt" --stats reports/evidence/p30_20261103/hf_channel_stats.json --class-map reports/evidence/p30_20261103/geologic_units.json --outdir "$P30_OUTPUT_ROOT/geology_replay_01" --replay "$P30_OUTPUT_ROOT/geology_smoke_01/checkpoint.pt" --feature-cache "$P30_OUTPUT_ROOT/geology_smoke_01/feature_cache.pt"
```

正式 Geology 普通头/候选条件收益仍未检验；预训练表征假设也未检验。进入正式实验需 [下一阶段决策与资源方案](p3_0_next_decision.md)，不是继续延长十张图 smoke。
