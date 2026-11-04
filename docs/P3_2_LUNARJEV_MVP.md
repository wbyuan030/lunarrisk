# P3.2 — LunarJEV 最小模型：任务与候选语义查询月球观测

日期：2026-11-04。评审基线：`main@edffc0a7c274efb4c1cee03e6c0f3203d87ebba7`。

**执行状态更新2026-11-05：** 三个冻结空间cache、四头×三seed及固定描述置换共18run均已实现、完成并通过独立进程精确预测回放；[最终报告](../reports/p3_2_lunarjev_mvp.md)、[阶段决策](../reports/p3_2_decision.md)、[实际manifest](../protocols/p32_head_run_manifest_v2.json)。当前taxonomy J未胜过B0/B2，语义/空间收益未建立。用户最新授权必要模型/数据直连下载、不走代理；本tier未新增下载。下文保留实施前设计及当时权限判断，未实施/全面暂停措辞不是最新执行指令，不要重复已完成矩阵。

**当前决定：开始 LunarJEV 设计与本地最小实现，不再重复训练 P3.1 普通 CNN。** 用户本轮指示为：“推一版远端吧，我觉得现有实验能说明特征可用，下面应该尝试设计 lunarjev 了”。本文件是新研究方案，不是已经实现、训练或验证的模型；它不批准恢复下载、使用 Test 或新增付费资源。

## 1. 为什么现在开始

来源：[P3.1 进度](../reports/p3_1_progress.md)、[Age](../reports/p3_1_age_baseline.md)、[Geology](../reports/p3_1_geology_baseline.md)、[最终六组审计](../reports/evidence/p31_prefix_runs_20261104/sequence_verification.json)。

- 已完成 5,135 Train / 1,155 Val 上 Age、Geology 各 seeds42/43/44；三种子均超过 Train prior。Age Val Accuracy 58.6436%±2.0070个百分点，Geology 43.6075%±2.2583个百分点。
- 这足以支持“输入存在可学习信号、已有编码器可作为第一版工程起点”。**不等于**冻结特征在任意新头/任务上都足够，也不等于 NASA/MG-MAE 预训练或语义增益已被验证。
- 当前已有预处理、split、checkpoint、逐样本概率和回放工具，不需要再等完整 HDF5、全月 OOD 或 GPU 满载才能设计决策模块。
- 原 LunarJEV 方向是 `observations + task + candidate meanings -> decisions / confidence`。低秩高斯后验不是其必需组件；本版也不是对 RSJEV/Clef 论文实现的复现声明。Choice 是首个实现原语；Noul、Score 预留，不虚造现有数据没有的安全/连续标签。

**P3.1 验收关闭：保留六组结果，不新增 scratch 种子，不事后延长 seed44、不因少数类表现修改旧预算。** 性能定位可做独立小探针，但不是 P3.2 的前置关卡。

## 2. 本轮问题与非目标

本轮使用新的假设编号，避免与 LunarRisk 的 H1/H2 混淆：

| 假设 | 比较 | 能支持的结论 |
|---|---|---|
| J1：公共候选描述有增量 | 描述条件头 vs ID 条件头；描述正确映射 vs 固定打乱映射 | 在现有闭集任务上，语义结构/知识是否帮助分类，而非仅换了参数化 |
| J2：按问题查询空间证据有增量 | 相同描述、编码器与预算下，空间 query head vs 全局池化 semantic head | 是否需要查询局部视觉证据，而不仅是 pooled feature 分类 |
| J3：复用一个观测编码服务多任务可行 | 同一视觉上下文同时回答 Geology 与 Age；对照独立普通头及 Geology→Age 确定性汇总 | 多任务接口、任务区分和实际代价；不是全新任务零样本能力 |

原始动机中的跨观测条件稳健性保留为后续问题，本轮不据此声称已学到不变物理结构。缺失模态/几何变化、Noul 拒判、连续 Score、LLM 干路、NASA 表征替换分别需要自己的证据，不能一次全部扩张。

首轮仍是 **已反复查看的北半球 Train/Val 开发轨**，不是确认性独立测试、全球泛化或未见类别评测。正式 Test 与旧 LROC Gate4 均 HOLD。

## 3. 模型设计：LunarJEV-lite / Semantic Query Decider

### 3.1 观测上下文可缓存，不重新训练感知主干

默认分别复用 P3.1 的三个 **Geology scratch checkpoint**，一个编码器种子对应一组配对头实验。第一轮工程实现固定 seed42，仅因其是预定首个种子，不因为挑中最佳分数；正式比较必须补齐42/43/44，不选最有利编码器。

使用每个原 run 的 `best_state_dict`（而非最后一步 `state_dict`），核验 receipt/hash，冻结参数、BatchNorm running stats，始终 `eval()`。这是 Geology-supervised encoder，**不是基础模型预训练**。Age 使用它属于相关任务监督迁移；所有对照接受相同额外监督。原 P3.1 独立 Age CNN只作历史参考，不直接作为同特征头部比较的对手。

当前 `ScratchClassifier.features` 最后是全局池化+flatten。新增只读导出适配器从这些层之前取 `[B,256,16,16]` 特征图；不修改既有训练实现与原 checkpoint。固定2D平均池化到4×4，得到 **16个256维空间 token**，再导出它们的平均值作为 pooled control。空间相对位置使用固定二维编码；**候选槽位不加入顺序位置编码**。不加月面经纬度、真实地质ID/年代前缀等样本标签作为视觉输入。

仅导出原协议允许的 Train/Val。每个encoder、每张样本只编码一次；多个任务/候选复用cache。6,290×16×256×float32 的纯token约98.3MiB/seed，三个约295MiB，另加元数据；这是形状计算，不是实测磁盘峰值。缓存本地分批写入、不入Git；保留FP32、既有归一化和样本顺序/哈希。

### 3.2 决策接口

```text
Moonstone pixels [B,28,256,256]
       │ frozen audited encoder, shared once
       ▼
visual tokens V [B,16,256] ──────────────────┐
                                          │
task definition q + candidate descriptions │
       │ fixed text/ontology encoder       │
       ▼                                  ▼
candidate queries [T,K,D] ──> cross-attend to V
                                  │ shared scorer
                                  ▼
                       candidate logits / masked softmax
                       Geology Choice (49), Age Choice (5)
```

建议首版隐藏维128、4个注意力头、2个cross-attention+FFN residual块、FFN维256、dropout0。每个候选query由**任务表示与候选描述表示**共同形成；视觉projection和决策模块可训练，原视觉encoder冻结。先用轻量判别式query模块，不训练Qwen/完整语言主干。它不是任意语言指令跟随模型；任务集合首先只有两个。

将全部候选沿batch/query维并行处理，视觉编码不随候选重复。首版不必加候选间self-attention，避免为了显得复杂增加自由度。若各候选logit独立于其余候选，本模型的子集概率仍等价于自身full logits遮罩归一化；这**不是缺陷，也不是语义增益证据**。不得为打破这一恒等式而强加交互。

接口建议：`encode_observation(pixels)->V`；`score_choices(V, task, candidate_table, candidate_mask)->logits`；`predict_choices(...)->probabilities, selected_candidate_id`。候选ID只用于索引公共候选表和输出映射；语义主模型不得拥有绕过描述的独立可学习class-ID embedding或class-specific bias。显式ID模型另列为控制。

### 3.3 真实语义来源，不把查表改名为语言理解

现有 `reports/evidence/p30_20261103/geologic_units.json` 只有49个符号和映射，例如 `Cc`、`EIp`、`pNc`；不能假定已有完整地质定义。建立版本化候选表：`task, class_id, symbol, public_name, description, source, source_hash, status`，对所有样本统一。

1. 优先用**已有本地审计资料**中的公开地质图例/类定义；有来源才展开术语，不靠模型编造岩性、年代、风险或空间位置。
2. 缺完整图例时，可立即用公开符号和已验证Age映射形成 `TAXONOMY_ONLY_LOCAL` 弱描述（例如某符号对应哪个官方Age类别），进行接口开发与有限弱语义实验。它不等于完整地质语义，不能用其负结果否定语言条件化，不能用其正结果声称自由文本理解。
3. 文本表征优先使用**已在本机可用**且许可/版本可追溯的一个冻结文本编码器，实际运行前固定模型/hash/模板。不得为本轮擅自下载权重或调用计费API。
4. 没有可用文本编码器时，允许固定词/字符TF-IDF表示作为离线lexical/ontology control，词表仅来自对所有样本公开的候选表与任务文字，不来自样本标签、图像描述或Val错误。准确命名为 lexical-semantic prototype，**不称其为预训练语言模型**。这不是停止接口开发的理由。
5. 两种描述/文本encoder tier分开记录，不得在看分数后静默替换。需要额外官方图例或文本权重时，一次性列出具体缺口、最小大小和许可；VPN暂停继续有效，缺口不阻塞特征导出和模块测试。

禁止使用逐样本真实unit name/age prefix构造问题，禁止根据答案为每条样本专门写描述。若描述包含地质年代，这属于公开候选知识；必须额外报告 `Geology probabilities -> official age aggregation`，不能把派生Age收益写成独立发现。

## 4. 首轮公平对照：四个模型足够

| 名称 | 同一冻结encoder输入 | 候选信息 | 目的 |
|---|---|---|---|
| B0 ordinary | 相同token平均池化 | 两个普通任务头；子集用masked-softmax | 必须的任务分类参照 |
| B1 candidate_id_control | 同上 | 已知类ID embedding | 参数共享结构参照，不声称语言能力 |
| B2 pooled_semantic | 同上 | 与LunarJEV完全相同的task/description表示 | 隔离语义信息、尽量匹配非空间scorer深度 |
| J lunarjev_query | 同一16-token cache | 与B2相同描述，query读取空间token | 检验空间任务查询 |

四组都重新训练各自头，输入样本/冻结encoder/额外Geology监督/任务损失/种子和更新预算相同。不能将新训练头与原端到端优化过的P3.1 classifier直接当作唯一公平比较。所有模型都训练Geology+Age，按每任务CE均值再等权平均；Age标签由Geology派生，因此它是辅助任务而非独立证据。若新增Geology-only对照，另列协议，不中途删除对某模型不利的辅助监督。

匹配的是预先限定训练预算及共同信息来源，不虚称参数/FLOPs精确一致；逐模型记录参数量、head时延、共享视觉编码时延、端到端两任务时延及文本表示预计算成本。OnePass在本版只表示一次上下文编码、并行判别，不意味着已经优于普通多头或生成式LLM。

### 推荐的首版固定预算（新实验执行前写入run manifest）

- 使用同一原始5,135 Train / 1,155 Val ID，Test不进入cache。
- frozen-encoder head训练：FP32、batch128（含最后不足批）、AdamW lr1e-3、weight_decay0.01、最多50 epochs、patience8，无class weighting/label smoothing，不进行超参搜索。
- 按Val Geology full-49 Macro-F1严格改善选checkpoint，最早平分胜出；Age一律从**同一个**选中checkpoint报告，不独立挑Age最优epoch。
- 初始化/优化seed42/43/44，分别配对应Geology encoder seed；每个seed四模型都使用同cache。先seed42完成可运行性/有限值/资源检查，再43/44，不能因分数差而单独停掉某模型。
- 工程阶段可用合成输入与少量Train batch检查接口，但不再做另一轮大型scratch训练。固定文本来源、编码器、结构和run config后才运行模型比较。

这是一组**新头部实验**，允许自己的小批训练预算；不修改P3.1既有batch16、种子、epoch、源码或已封存结果。

## 5. 必须随首轮报告的控制和指标

### 数据/数学控制

- frozen encoder梯度为None，BN统计不变化；权重选择和缓存来源可重放；同一个样本在所有模型中ID/target一致。
- Geology与Age候选维数不同；padding/mask不产生概率质量、空/重复/越界候选拒绝。
- 候选排列后复原logit顺序应在数值容差内一致；子集恒等式按实际结构如实报告，不能当科研收益。
- 所有四组都有普通masked-softmax控制；不保证每个“现实候选集”必含真值。首轮全候选分类必然含真值，另造真值缺席拒判任务要单独预注册，不能把它偷渡成安全标签。
- 描述表与样本输入解耦；同一query重复/跨样本不会读取目标。固定地理子集不构成未见类别任务。

### 科学评价

主指标：Geology全49类Macro-F1，辅以Accuracy、支持32类Macro-F1、每类支持/recall、NLL和multiclass Brier；Age全5类同口径，从同一个checkpoint给出。保留三个seed及均值/样本std，不能把std写成置信区间。

对J-B2、B2-B1/B0保存逐样本成对预测；尤其看少数类与混淆类别，但少数类划分阈值由Train支持数事先固定，不按Val得分挑类。配对重采样若加入，先固定空间block规则，不将每个候选问句视为独立月面样本。当前Val被用于模型选择，区间/诊断一律标 exploratory，不当确认性P值。

描述对齐控制：固定一个随机描述-类别置换，对训练与推理均保持同一置换、全部候选信息量相同。它可测试正确语义映射是否优于随机编码，但闭集训练可能重新学习置换，不能预设其必须跌至随机。只在推理时打乱的降分不单独证明语言理解。首轮矩阵完成后再跑这一单项预先定义的控制，不追加无穷prompt搜索。

任务控制：对比Age专用输出与通过公开类映射汇总Geology概率；同时输出不同task的响应，不能仅因task_id查表成功就称“任意问题泛化”。未知类别、同义描述、模态缺失、校准/拒判各自是后续协议，不混写进官方分数。

## 6. 立即执行的 Coding Agent 清单

| 项 | 现在做什么 | 交付 |
|---|---|---|
| P32-01 | 只读核验三个本地Geology checkpoint，按原Train/Val ID分批导出16-token cache和pooled features | 新 `scripts/p32_cache_context.py`、测试、`reports/p3_2_context_cache.md` |
| P32-02 | 建公共候选表，标注真实描述/弱taxonomy/缺项；盘点本地文本encoder并固定一种表示 | `protocols/p32_candidate_catalog.json`、文本缓存manifest、来源缺口（如有） |
| P32-03 | 实现B0/B1/B2/J统一接口、并行任务查询、冻结与mask控制；不动旧runner | 新 `src/lunarrisk/lunarjev/`、必要单测、CPU synthetic和真实cache smoke |
| P32-04 | 将上述默认设计与实际asset/hash锁定为run manifest，运行四头×三种子 | 新 `scripts/p32_train_heads.py`、逐样本预测/checkpoint、`reports/p3_2_lunarjev_mvp.md` |
| P32-05 | 归纳J1/J2/J3，提出下一项而非无限扩张 | `reports/p3_2_decision.md`；有增量再考虑LLM/领域预训练替换，无增量如实定位 |

这些脚本名是**待实现接口**，本次文档提交未创建可运行模型。报告使用PASS/FAIL/BLOCKED并给出实际命令、源代码/data/cache/text hash与回放结果。缺完整语义时只把对应科学主张标BLOCKED，继续可完成的context缓存、模块实现和明确命名的弱语义控制；不以权重未取得为理由停掉所有工作。

吞吐探针可复用少量Train输入分开测GPU常驻、CPU常驻、真实HDF5三条路径。不得更改封存的六组实验或为了GPU占用率更高增大研究模型；不以profiler尚未完成阻塞设计。缓存头训练不会每epoch重复解压原始月球像素。

## 7. 资源与保护

- **继续暂停VPN大文件/权重/图例下载**；只用已有本地数据、checkpoint、代码与文本资产。用户当前授权是推进设计与已有资产上的可逆本地实现，不是恢复40.48GB余量传输。
- 单卡RX7900XT与当前ROCm；不替换环境、不买云、不启动四卡。小型新cache先检查磁盘，原始pixel/weights/caches不入Git。
- 旧P2.2负结果、P3.1六组证据及全部冻结配置不改写；旧LROC Gate3 NOT_FROZEN/Gate4 HOLD，公开Test scoring HOLD。
- 先跑现有测试及新增关键模块测试、compileall；579项是旧代码证据，不是新模型验证。每阶段实际完成后按用户习惯push main，提交前核对远端、防止非快进，不force-push。

**阶段目标：先获得一个真正接收任务与候选描述、共享视觉证据、可公平比较的 LunarJEV-lite。是否比普通头更好由新实验决定，而不是用已有scratch成绩替它背书。**
