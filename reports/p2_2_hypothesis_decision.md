# P2.2 — H1/H2 结论口径修订（2026-11-03）

**工程决定仍为 `STOP_OR_REFRAME`；H1/H2 的普遍增益尚未得到证实。**
本页替代 `60dd5ac` 中过强且内部矛盾的判定，不新增模型或测试集选型。
完整更新结果、实际计数、日志和证据文件已在[本轮证据报告](p2_2_evidence_completion.md)列出。

源码 `ac93b10` 的真实六模型/三 seed 重跑中，两个主规则平均 Brier：
coarse-logistic 为 **0.0728 / 0.0751**，Point 为 **0.0748 / 0.0924**，
Diagonal 为 **0.0897 / 0.1149**，Joint 为 **0.0857 / 0.1197**。
Joint−Diagonal 在 seed42/44 改善、seed43 变差；Joint 对恢复的 coarse-logistic
在两个主事件的三个 seed 均没有更低的 Brier/NLL。部署风险结果还须结合不同实际覆盖率理解。
因此保持停止扩展的工程决定，不把局部改善解释成普遍收益。

本轮全仓测试为444项（443通过、1项可选本地标签路径检查跳过），9个后验 checkpoint 概率重放完全一致，
9个基线新进程重放在既定数值容差内；Point/Joint seed42各两次独立重训均与原运行的权重、温度和概率一致。

## 撤回的判定与归因

1. 原报告把“至少 2/3 seed 显著且同向”称作 P2.2 预注册标准。实际
   `docs/P2_2_PROBABILITY_VALIDATION.md` 没有这一计数门槛。它不能事后充当预注册决策规则。
   原报告又在 joint−diagonal 的 g10/g15 表中列出两个 seed 改善，却声称不满足 2/3，逻辑矛盾。
2. “未能证明 H1/H2 稳定优于对照”不等于统计上否决整个假设。
   原来的 **H1 REJECTED** 改为 **NOT ESTABLISHED**。组级区间是给定 checkpoint、
   给定 Cal 阈值下的开发组不确定性；3 个 seed 共用相同 25 个组，不能当作三次独立确认。
3. 原报告 g10 diagonal−point“2 个显著”与自身星号计数不符；g15 joint−point
   “3/3 显著更差”与自身数值表的 2/3 不符。原手写显著性汇总不作为证据。
4. `diag(diag(Σ_z))` 固定的是**内部链潜变量**的边缘方差。
   `g15 = sigmoid(z1) * sigmoid(z2)`，移除 Cov(z1,z2) 会改变物理 g15 的均值、方差和事件概率。
   因此此前“保留所有物理边缘后，证实跨属性相关性有害”的说法撤回。
   正确描述是“同 checkpoint 的潜变量协方差干预，效果可以由物理 g15 边缘变化产生”。
   g10 抽样严格不变这一结构事实仍成立。
5. 增加 Ridge 回归目标不保证事件概率更强；历史的独立物理高斯还可能违反 g15≤g10。
   要保留该失配参照的明确标签，并恢复原 coarse-logistic 事件参照，不能仅与被替换的 Ridge 比较。
6. 仅凭规则分数优劣，不能确定 roughness/relief 的误差是由尾部假设、均值偏差、优化还是其他因素导致。
   这类解释只是待验证的失败机制，不是已证实的原因。

## 保持的研究边界

- 原始 H1：Diagonal 相对 Point 的概率预测增益；原始 H2：Joint 相对 Diagonal 的增益。
  Point 已用 Train 残差和 Cal-T 形成分布。所有比较必须与强事件参照、同样本/规则配对。
- 已反复查看的 110 Tile / 25 Group 只记为 `development_test`。本轮补充的 Train 分位数规则
  只作完整保留的事后诊断，不能包装为预注册确认，不能从中挑有利规则再重训。
- 组级置信区间不含重新抽取 Train/Cal 的不确定性，也未解决多规则比较问题。
  同 seed 重复用于检查执行可复现性，不代表新的独立地理验证。
- 没有可审计的稳定优势时保持 `STOP_OR_REFRAME`，不继续添加 LFM、One-Pass、JEPA 或新架构。
- 外部协议仍是 `PENDING_SIGNOFF`，Gate 3 `NOT_FROZEN`、Gate 4 `HOLD`。
  662 景桥接/同源审计、源 provenance、EDR/CDR 变更签核仍是外部集的独立前置条件。

## 结果出处

- 历史原始数值保留在 [旧比较报告](p2_2_baseline_comparison.md)，页首已标勘误。
- [2026-11-03 证据报告](p2_2_evidence_completion.md)记录实际代码 SHA、源码哈希、完整日志和新运行目录；
  [执行清单](evidence/p22_20261103/logs/execution.json)可逐条核对命令及退出码。
