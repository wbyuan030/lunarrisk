# P3.1 — 从公开 Benchmark 工程 smoke 转向正式基线

**日期：2026-11-03；执行状态更新：2026-11-04。** 原交接的权限判断保留在下文作为历史。P31-01/02工程验收及ID结构控制已完成，当前见 [进度](../reports/p3_1_progress.md)。用户于2026-11-04明确授权“允许大文件下载 推进基线”；新增权限以 [单文件授权记录](../protocols/p31_acquisition_authorization.json) 为准。现在执行固定HDF5单副本下载与已冻结的Age/Geology Train/Val基线，不再等待重复确认；全mmap/额外权重/付费资源不在本次范围，Test评分仍HOLD。

评审基线：`main@25c8ac93e6ea94a1f402c3daecf547bba1990278`。本次用户要求将评审与下一步动作推送到 main。本文件承接 [P3.0](P3_0_PUBLIC_BENCHMARKS.md)，当前入口为 [AGENTS.md](../AGENTS.md)，机器可读任务在 [p31_execution_plan.json](../protocols/p31_execution_plan.json)。旧 P3.0 计划、实验和资产提案不覆盖。

## 1. 评审结论：工程通过，研究效果尚未验证

来源：[进度](../reports/p3_0_progress.md)、[资产审计](../reports/p3_0_benchmark_audit.md)、[loader 审计](../reports/p3_0_loader_validation.md)、[mmap 配对](../reports/p3_0_mmap_pair.md)、[Geology smoke](../reports/p3_0_geology_baselines.md)、[后续获取提案](../protocols/p30_next_acquisition_proposal.json)。以下是这些证据支持的状态，不是本轮新实测：

- Moonstone 来源、标签、split、部分 HDF5 真实像素读取和独立进程回放已经完成工程验证。最新归档 482 项测试中 481 通过、1 项旧数据路径检查跳过。
- Age 主 pilot 只有 5 Train + 5 Val；Geology 两头复用这些切片和训练两步的 Age scratch CNN 特征，只覆盖 5 个地质类别。它们不是正式 benchmark 或预训练收益证据。
- mmap 固定行探针共比较 172,032 像素，HF 处理的数值与 mask 差异为零；但原八张北极 patch 中 17/28 通道全缺失。不能由零值一致性推断这些模态的有效数据处理已验证。
- 官方 stored split 实测为 11,400 / 2,400 / 2,400。Train 出现 46/49 类；Train 缺失的三个类在 Val/Test 中也无样本，因此当前划分不构成新类别泛化测试。见归档 metadata_summary.json；不要重新切分掩盖这一事实。
- MG-MAE 权重仍未在已审计来源核实；NASA 文件可达但物理输入映射、上下文和环境兼容性未通过。不能从头训练冒充月球预训练。

**工程决定：结束反复延长十张图 smoke；立即开发可扩展正式 runner、设计有覆盖的完整裁片验收、锁定普通基线协议。** 科研上优先形成数据充足的 Age/Geology scratch 参照；预训练支线独立推进，不是普通基线必须等待的前置。

## 2. P31-01 — 正式训练入口（P0；现在即可离线开发）

不要仅放宽 `p30_age_smoke.py` 的步数/样本上限，或让 `p30_geology_smoke.py` 处理全量数据。旧脚本保留为 smoke 与证据回放入口。

评审发现的扩容风险：两个 smoke 脚本的 `digest()` 用 `Path.read_bytes()` 整文件哈希；`dataset_batch()` 将全部样本堆成 NumPy 数组并一次放入 GPU。它们在十张图上没有问题，但直接用于 74.45GB HDF5 或 11,400 张训练图会造成大内存需求。训练像素按 11,400×28×256×256×4 字节计算约 77.9GiB，尚不含任何副本或激活。

新建正式 runner（建议 `scripts/p31_train_classifier.py`，路径是待实现接口），必须具备：

1. **流式哈希**：固定 chunk 大小迭代 SHA256；完整资产成功校验后缓存 manifest，按版本/大小/哈希管理，不在每个 epoch 重扫整个文件。不能仅用 mtime 冒充完整内容校验。
2. **真正的 mini-batch**：只读、worker-local HDF5 handle；DataLoader 批量加载，按 batch 转 GPU；验证和特征导出也分批，不拼装全部数据进内存。初始 `num_workers=0`，其他设置经现有 ROCm 实测再调整。
3. **固定数据语义**：复用已审计通道顺序、Age/Geology 映射、stored/geographic split 和归一化来源；不能借正式化改标签、修订缺失值处理或把答案元数据送入模型。
4. **检查点与日志**：保存架构、训练状态、scaler/统计 hash、数据 revision/hash、split ID hash、seed、优化器、模型选择标准、源码与环境。分清预测重放与续训可复现，两者单独测试。
5. **资源证据**：报告 batch、峰值进程 RSS、GPU allocated/reserved、训练窗口时间与样本吞吐；峰值 allocated 不能当作整卡/整进程显存。不要提前承诺全量训练时间。
6. **断点与覆盖安全**：新输出命名空间，拒绝意外覆盖旧 run；不删除原始数据或 P2.2/P3.0 证据。

离线测试至少覆盖：流式 hash 与已知文件一致、超过 smoke 上限仍按 batch 读取、最后一个不足 batch 的批次、worker handle、重复/错 split/无效 target 拒绝、检查点恢复概率一致、训练和验证不读取 Test 样本。随后在已有真实 pilot 上做有限回归，仅证明新入口正常，不新增科研性能结论。

产出：`reports/p3_1_runner_validation.md`、代码/测试和实际资源记录。不要用历史 482 项 PASS 代替本轮测试。

## 3. P31-02 — 有代表性的完整裁片配对（P0，与 runner 并行）

使用 Train/Val、按数据覆盖而非模型表现选择少量固定切片，使每个模态组有实际有效值，并覆盖部分缺失、有限 raw zero 和 radar。已有北极八张图不满足全部有效模态覆盖条件。

- 网络前保存 ID、选择依据、固定 upstream revision、每通道范围、请求数/字节预算、数值容差与 mask 判据。沿用 P3.0 的 `rtol=1e-5, atol=1e-6`；不看误差后放宽。
- 比较完整 256×256 裁片、全部通道、channel/group presence、ID/标签/split；不能继续只比较三行就宣布完整等价。
- 固定 HF 统计为被验证路线；Git 统计差异保留为来源诊断，不事后换最接近的统计并称预定路径通过。
- 报告每个通道真正比较了多少有效/缺失像素，不以全缺失通道零一致性验证有效数据处理。
- Range 请求不得悄悄读取未选择的 Test 或额外场景；合并请求时记录范围与额外字节。运行前估计延迟/请求量，不承诺完成时间。
- 通过后称为“已验证样本覆盖范围内的 loader 数值一致性”；不能扩写成全球逐像素验证。仍有来源差异时保留明确的 HDF5-derived 轨，不冒充 exact official reproduction。

这应是一轮有完成标准的验收，不继续无限追加同类微探针。只能使用既有累计传输范围或后来明确批准的范围；不足则提交精确缺口，不自行重置 1GiB 计数。

产出：`protocols/p31_loader_pair_plan.json`（执行前锁定）、`reports/p3_1_loader_pair_validation.md`、逐通道结果与 hash。

## 4. P31-03 — 数据充足的普通基线（runner、数据与协议具备后）

### 先固定设计，再看成绩

- **Age**：先检查正式训练可学、收敛与资源，避免将其变为独立大型调参项目；不是绝对年龄回归。
- **Geology**：核心主任务，保留官方49输出和类别映射。
- 官方 stored split 为第一条可比轨；geographic 与任何新校准轨分别命名。所有超参、早停、prompt/描述选择仅使用 Train/Val；正式 Test 评分需先锁定模型与分析配置。
- 类别先验/多数类参照只从 Train 拟合。固定 Accuracy、Macro-F1 的 absent-class/zero_division 口径；核对上游实现并同时提供完整49类、实际支持类与每类计数，不能挑有利的 macro 定义。
- Age 与 Geology 标签相关；若复用 Age-supervised encoder，所有头接受相同额外监督，并明确标注。不能把它称作无监督预训练或纯 Geology scratch。
- 一次正式单种子运行先验证吞吐和训练链路，再按预先锁定的预算跑 seeds 42/43/44。候选配置数量和 Val 选择标准事先限制，不循环试 Test。
- 普通基线至少包括 Train prior 与 scratch CNN；预训练权重可得时另列其 frozen/有限适配轨，不让权重缺口阻塞已有数据上的普通基线。

正式 runner 允许正常的 Train/Val 学习曲线、分类报告和失败诊断；不需要额外批准每次免费本地验证。下载批准与实验配置冻结是不同门槛，不能把它们无限拆成等待节点。

产出：`protocols/p31_baseline_eval_plan.json`（运行前锁定）、`reports/p3_1_age_baseline.md`、`reports/p3_1_geology_baseline.md`、逐样本概率、checkpoint 与源码/数据清单。源定义未验证的口径必须标明差异。

## 5. P31-04 — 把候选 ID 对照与语义假设分开

当前 CandidateIDHead 对各候选独立打分：`s_c=f(x,embedding(c))`。对共享已知候选集 S，`softmax(s[S])` 与 full-class 概率遮罩再归一化相等。这是当前结构的数学性质，不是新实验结果；并不意味着它与 ordinary MLP 学成相同函数。

因此：

- 现有头保留为 **candidate_id_control**（新报告名称；不破坏旧 checkpoint/API）。它用于结构/参数共享对照，不代表自然语言理解、候选间推理或未见类别泛化。
- 现有排列测试验证接口正确，但排列稳定与子集可调用不能独立作为科研增益。新增子集调用与其自身 full-softmax mask 恒等式回归测试。
- 普通头与 ID 头参数量接近但计算量不等价；ID scorer 对每个候选重复 MLP。报告实际 full/subset head 延迟和完整前端延迟，不能仅凭参数数目声称同预算或 OnePass 更快。
- 下一种语义头仅在**单独设计锁定后**实现：公共地质类别描述/固定文本表示，对所有样本统一；绝不能输入该切片真实 `dominant_unit_name`、真实年龄前缀或只含正确答案的候选集合。
- 最低公平对照为 ordinary + masked-softmax、candidate_id_control、semantic scorer，使用相同 encoder/模态/feature cache/Train-Val-Test。类别描述来源、文本模型版本/额外预训练知识与推理成本应公开。
- 候选顺序、子集、同义描述、未知类与新任务是不同能力。当前 stored split 没有训练缺失而 Test 出现的类别，不能用它声称 zero-shot 新类别。另造类别留出属于新协议，不能替代官方分数。
- 候选子集若按真值保证含正确类别，必须对全部模型一致，并明确这是条件选择任务、不是现实任意候选路由；真值缺席时另定义拒判，不静默保证总有答案。

优先交付设计与小型离线控制测试；在正式普通基线尚未跑通时，不恢复 Qwen/完整 OnePass 联训或复杂重采样桥。

## 6. 数据获取与预训练支线

评审支持的推荐是**单一固定 HDF5 文件的数据先行方案**，不是全仓 snapshot。原提案 [p30_next_acquisition_proposal.json](../protocols/p30_next_acquisition_proposal.json) 保持 `PENDING_USER_DECISION_NOT_AUTHORIZED`；“把建议推远端”不自动等于批准执行74GB传输。

若后续明确批准：固定 HF revision `53099efd0f2db81eee96b28eae87e1efdd281310` 的 `benchmark/lunar_patches_v4.h5`，74,449,119,347字节；下载到新目录的单一 `.partial` 文件，分块校验完整大小与远端声明 SHA256 后原子改名。不重复保存 snapshot/cache 双份。原提案80GiB空间预留不是已测峰值，执行前重新检查磁盘、配额与已有部分文件。

全 HDF5 包含官方 Test 像素，存储整个文件不等于获准训练/查看模型测试结果；runner 必须仍隔离 Test 访问和最终评分。没有下载批准时，立即完成 P31-01代码、已有资产回归和协议设计，不再为了显得推进而重复十张图训练。

MG-MAE 官方权重核实与普通基线并行；有限来源搜索找不到时保留准确 BLOCKED，不声称互联网不存在。联系作者或发公开 issue 需按用户对外通信权限处理，不因本交接自动发信。NASA LFM 的 Python/物理通道/几何上下文适配是独立研究支线，不随意补零伪造测量，不为填缺口重做全球预训练。

## 7. 提交顺序与保护边界

1. 第一提交：P31-01 流式正式入口、离线测试、已有真实数据回归；锁定完整裁片验证与基线设计。
2. 第二提交：在有权使用的数据范围内完成 P31-02；数据充足且协议固定后执行 P31-03，更新真正的 Train/Val 结果与资源证据。
3. 第三提交：基于同一编码器的候选结构对照；语义扩展另立协议、避免把 ID 分类替代 LunarJEV 科学问题。

每次更新 `reports/p3_1_progress.md` 的 PASS/FAIL/BLOCKED、命令、commit、数据 hash 与证据；历史482项通过不算新测试。建议验证 `python3 -m unittest discover -s tests -v` 与 `python3 -m compileall -q scripts src tests`。

旧 P2.2 负结果与所有 evidence 保留；LROC Gate3 NOT_FROZEN、Gate4 HOLD，不覆盖任何 heldout/provenance/原始数据。仅用现有 RX7900XT/ROCm，不替换torch、不购买云、不启动四卡、不上传大文件/凭据。按用户既有习惯直接push main，不force-push、不覆盖并发提交。
