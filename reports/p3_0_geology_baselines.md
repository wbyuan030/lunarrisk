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

新增 4 个 synthetic 测试覆盖候选排列/子集语义、普通分类器 mask 后归一化、空/重复/非法 ID、参数预算与 49 输出。完整 ROCm 环境回归 **482 tests，481 PASS、1 旧标签可选检查 SKIP**；compileall exit 0。真实训练、同 cache 回放、排列与资源实测将在运行后记录，尚不声称成功。
