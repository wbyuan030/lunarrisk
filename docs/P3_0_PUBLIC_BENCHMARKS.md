# P3.0 — LunarJEV：公开 Benchmark 接入与最小决策实验

**日期：2026-11-03。状态：执行计划，尚未下载新数据、接通新模型或产生 benchmark 成绩。**

基线提交：`461316fb50c10ba4570aa55ba56f96a056e6df32`。用户已选择“使用公开 benchmark 验证 LunarJEV”的新方向；本轮先做可逆、低成本的接入审计和普通任务头，不恢复旧版 v5.x 全套大架构。

当前入口：[AGENTS.md](../AGENTS.md)；机器可读范围：[p30_public_benchmark_plan.json](../protocols/p30_public_benchmark_plan.json)。这份计划不是已冻结的实验预注册，精确数据版本、配置、预算和测试安排在 P30-00/01 中确定。

## 1. 研究重心与不变边界

原始 LunarJEV 的问题是：**月球领域表征 + 任务/候选条件，是否支持可复用、可信且高效的决策？** 当前 LunarRisk 只验证了其中一种四属性后验实现，没有检验所有 LunarJEV 主张。

P2.2 的工程结论 `STOP_OR_REFRAME` 保留，H1/H2 稳定增益仍为 `NOT ESTABLISHED`。不要删除负结果、反复重跑旧开发集找正结果，或把这次方向调整称为原假设已被证实。已有 AST、校准、配对统计、checkpoint/replay 和数据审计可复用，但联合高斯不是新路线的必需前置。

**本轮主线：Moonstone 年代分类先跑通，地质分类为核心主任务。** SomBench 只允许一个必要的 LFM 接入探针；不得并行展开冰、成分、分割、检测、主动路由和所有大模型。

旧外部 LROC `heldout_v2_external` 与产品级冻结文件全部保持不变：Gate 3 `NOT_FROZEN`、Gate 4 `HOLD`；公开 benchmark 的测试流程是新的独立协议，不得借此解锁旧 Gate 4。

## 2. 公开来源：已核对的内容与尚待实测的内容

下列是 2026-11-03 读取的作者数据卡/代码入口，不是已经安装或复现的证据。Coding Agent 必须解析并固定仓库 commit、HF revision、文件大小、校验和与许可证。

| 来源 | 支持的事实 | 尚未验证的接入事项 |
|---|---|---|
| [Moonstone 数据卡](https://huggingface.co/datasets/ayushprd/Moonstone/blob/main/README.md) | 28 通道、约 237m/px；任务含 49 类地质、5 类年代；主 benchmark HDF5 标称 16,200 切片；准备好的 benchmark 约74GB，pretraining 部分约207GB | 本机资产是否存在、真实字段/过滤后行数、传输范围、完整下游依赖 |
| [Moonstone 官方仓库](https://github.com/ayushprd/Moonstone) 与 [README](https://github.com/ayushprd/Moonstone/blob/main/README.md) | 发布任务代码、scratch/linear/finetune 入口与 MG-MAE 架构 | README 中的 `checkpoints/latest.pt` 是使用示例，不证明公开预训练权重已可下载 |
| [官方 downstream_dataset.py](https://github.com/ayushprd/Moonstone/blob/main/downstream_dataset.py) | 运行时从全局 mmap 读取像素，HDF5 提供位置与标签；默认读取存储的 split；地理模式按绝对纬度分带 | 数据卡称 benchmark 独立可下载，但现有 loader 依赖 mmap；不能假定只下载 HDF5 就原样可跑 |
| [NASA–IBM LFM 模型卡](https://huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model/blob/main/README.md) 与 [官方代码](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model) | 原生 NAC/WAC 模态与几何上下文接口、公开 backbone/config 路线 | 与 Moonstone 28 通道、尺度和观测上下文的映射；现有 ROCm 兼容性与真实资源消耗 |

代码与数据许可分别记录，不能用代码许可替代所有数据许可。Moonstone 数据卡与源码如有分歧，保留双方原文、revision 和差异；不得自行协调成一个未经验证的“官方事实”。

本计划不引用论文精度数字作为验收承诺，不声称单卡运行时间或显存已经测过。

## 3. 任务定义：只借用现成真值，不重造主 benchmark

### T1 — Age：第一项接入与基线任务

沿用官方五类地质年代映射、顺序与未知标签过滤。它不是连续绝对年龄回归，不输出“多少亿年”的新真值。`dominant_unit_name` 及其年代前缀是监督元数据，**不得作为每张切片的输入或 prompt**。

### T2 — Geology：核心主实验

沿用官方 49 类映射、索引和有效标签规则。标签是切片主导地质单元，不是像素分割掩码。保留每类样本数、混合单元/主导类别的局限，不把类别描述来自地质图等同于模型看到真实答案。

Age 与 Geology 标签相关，不能当作两份完全独立证据。在两项标签映射能严格对齐时，增加“地质预测概率按年代聚合”的简单参照；它是模型输出的映射，不得读取样本真实地质标签作为年龄输入。

### 暂缓任务

FeO/TiO2、冰前景度、IMP、撞击坑不进入首轮主矩阵。后续若开展成分回归，遵循官方目标通道排除；若开展冰任务，区分前景度参考产品与实测冰含量；若把检测/分割改为二分类，必须标为新任务，不再沿用原 benchmark 指标比较。

## 4. 阶段与明确交付

| 阶段 | 立即程度 | 动作 | 验收/产物 |
|---|---|---|---|
| P30-00 | 现在开始 | 公共资产、权重、磁盘、许可证与官方代码审计 | `reports/p3_0_benchmark_audit.md`；版本化 asset manifest；可达性与缺口 |
| P30-01 | 00 后 | 数据加载器、标签、split、通道/mask/归一化等价性 | `reports/p3_0_loader_validation.md`；离线测试和真实小样本 I/O |
| P30-02 | 01 后 | Age 普通任务头与最小真实训练；冻结表征可行性 | `reports/p3_0_age_baseline.md`；checkpoint、逐样本概率、资源日志 |
| P30-03 | 02 后 | Geology 普通头及同表征的候选条件头 | `reports/p3_0_geology_baselines.md`；公平实验配置与结果 |
| P30-04 | 条件式后续 | 候选排列/子集、模态缺失、校准扩展 | 新协议先冻结，不能伪装成官方原指标 |

每阶段在 `reports/p3_0_progress.md` 写 PASS/FAIL/BLOCKED、commit、命令、测试数、真实资产/哈希。计划中的脚本与报告名称均是**待实现产物**，不是已经存在的可运行接口。

## 5. P30-00 — 审计先行，不盲目下载

1. 先读本地已有资产、空闲磁盘和运行环境；保留现有可工作的 ROCm PyTorch。不要运行无版本控制的上游 `pip install -e .` 覆盖 torch；检查依赖后在隔离路径安装必要组件。
2. 获取作者代码、数据、模型仓库的 metadata/listing，记录实际 revision 与文件字节数。权重需验证真实可访问链接、格式、架构配置和 hash；`latest.pt` 示例、论文宣称开源、空占位文件均不算已获得权重。
3. 生成精确 allowlist 下载计划：数据文件、标签映射、normalization、必要权重及临时/解压/缓存空间。**默认 dry-run；禁止无过滤的 snapshot_download 或重建全球数据流水线。**
4. 初次审计/小样本网络传输上限为 **1 GiB（本轮工程护栏，不是用户新增算力预算）**；大权重和全量数据不算“元数据”。先检查 HEAD/Range 能力，小范围请求被忽略时立即中止，不能把整文件流入内存。
5. 不能部分读取 HDF5 或本地无可用像素时，明确 `BLOCKED_PARTIAL_DATA_ACCESS`，提交最小下载/硬盘方案供下一轮审阅；不得自动退回约207GB的预训练下载。约74GB主数据也不在初次自动下载范围内。
6. 若可用预训练权重不存在，标记 `BLOCKED_PRETRAINED_WEIGHTS`；仍可完成 loader/小网络 smoke。禁止从头预训练 MG-MAE 冒充权重复现，禁止为找正结果无界更换底座。

**验收**：资产清单区分 verified/unknown/blocked；磁盘与总下载量可计算；至少给出“从哪些公开文件到一条有效带标签样本”的真实路径。没有实际读取不能标记 PASS。

## 6. P30-01 — 官方协议忠实复现与数据泄漏检查

### 6.1 像素来源与等价性

数据卡的 standalone HDF5 和 mmap-backed loader 是已知接入风险。先检查实际 HDF5 keys、shape、dtype、是否含像素、是否已归一化、NoData/mask、通道顺序、裁片起点与经度回绕。

若实现 HDF5 替代 loader，必须对一组固定 ID 与官方 mmap 路径比较：像素、通道顺序、有效 mask、归一化和标签/split。允许的数值容差需事前写定。无法取得双方像素时，只能称 `HDF5_DERIVED_PILOT`，不能声称官方 loader 等价性已通过。

不要把归一化后的真实0值随意当作缺失，也不要悄悄“修正”官方缺失语义；忠实复现与改进版应分开记录。如官方使用公开全局 normalization，官方复现轨按其约定并注明统计来源；train-only 改动属于另一协议，不能混报。

### 6.2 任务/输入隔离

- 逐 Split 输出有效样本数、类别直方图、未知标签排除、sample_id 与空间 footprint。
- `dominant_unit_id/name`、age、地质真值 raster 等只进 target，不进影像特征、数值标量或逐样本候选构造。
- 候选类别描述可以由公开 class map 统一生成，但对全部样本一致；不能暗中只为正确类提供答案提示。
- 不把多个 prompt、选项排列、相邻同源 crop 当作新独立地点；统计仍按 tile/source/空间组声明。
- 不将 P2.2 的三米坡度标签拼接到237m的 Moonstone任务上；二者独立建数据空间。

### 6.3 Split 两条轨分开

**O：官方复现轨。** 使用存储的70/15/15 split和官方过滤；不能重新随机切分以凑整数。先核对上游训练/选模型准则，按原任务输出原始 Accuracy/Macro-F1。默认随机 split 不称地理 OOD。

**G：官方地理扩展轨（后续）。** 按锁定代码的绝对纬度边界实现：Train ≤60°；Val (60°,70°]；Test (70°,80°]。写出排除区与实际计数。只能称下游地理留出；预训练可能看过全球地图，未审计前不得称预训练未见区域。

官方公开测试可以在模型/配置锁定后进行基准评估，但不能反复用测试分数选学习率、prompt、候选措辞或 early stopping。

**离线测试**：标签映射/未知类、分界纬度、ID重复、split不漂移、目标泄漏、缺失mask、通道顺序、归一化、worker-safe HDF5读取；再做真实像素smoke。Synthetic 测试不能替代数据验证。

## 7. P30-02/03 — 最小可解释的模型矩阵

首先选择一套已验证的可见模态和输入映射，所有同表征对照固定输入、分辨率、sample ID、训练预算与随机种子。首个smoke只用一个seed；正式小矩阵计划用42/43/44，并保留上游官方配置的区别。

| 模型/控制 | 回答的问题 |
|---|---|
| 简单多通道视觉模型/官方scratch基线 | 不用领域预训练的参照 |
| 可获得的月球预训练encoder + 普通线性/MLP分类头 | 领域表征是否有帮助 |
| **同一encoder、同一特征 + 任务/候选条件化小头** | 决策条件是否有独立收益 |
| 普通分类头 + 候选子集mask后重新归一化 | 避免把“能接受候选子集”包装成独有能力 |

### NASA LFM 与 Moonstone MG-MAE 分开

MG-MAE 原生适配 Moonstone 的模态分组，但权重可达性尚待00确认。NASA LFM 是另一套编码器：其 NAC/WAC 通道键、原生尺度和几何上下文不与28通道任意互换。

必须先提交 input mapping：每个可见通道的物理含义、单位、normalization、尺度、对应adapter、是否新增参数及缺失模态处理。不能将未知观测角伪造为已测0值，也不能给候选模型额外图层而不给对照。

若 NASA LFM 仅支持经过验证的模态子集，先在该子集做内部公平对照；全28通道官方成绩另列，不能归因于同一变量。若映射无法成立，保留 BLOCKED，不靠伪RGB或任意填充声称原生多物理场适配。

先做冻结特征探针。冻结未改善不等于基础模型无效；后续可提出**一次预先限定**的adapter/LoRA试验，但先通过I/O、存储和普通头基线，不做无界架构搜索。本轮不启动完整Qwen/语言主干联合微调、Latent TiM生成或主动传感器路由。

特征缓存记录模型权重、输入ID/通道、normalization与dtype的hash；仅在预处理/模型不变时复用。不能在缓存特征上报告端到端微调成绩或忽略特征提取成本。

## 8. 决策扩展与校准：在基础复现之后

新增分析与官方原任务成绩分表：

- **选项顺序**：相同候选内容的置换，输出应按相同语义逆置换对齐；不能把顺序稳健称为新类别泛化。
- **候选子集**：用固定生成器与seed、保留正确类、记录子集与难度。所有方法用同一子集；与masked-softmax普通分类器比较。此协议不代表开放集拒识。
- **模态缺失**：固定缺失集合与mask语义，保持标签和样本一致；不人为规定每个样本缺失后熵必须增大。
- **校准/拒判**：官方原始准确率优先；额外概率实验使用明确的校准轨。若在Val内预先划分 `val_select/val_cal`，须在任何调参前锁定ID并与完整官方Val复现分开标注。不能对官方Test拟合温度/拒判阈值，也不能用同一批Val先反复选模型再宣称它是独立Cal。
- 报告 raw/calibrated multiclass Brier（明确求和还是类别均值）、NLL、Macro-F1、每类召回、可靠性图、实际拒判覆盖率；不得把此处的分类错误风险称为着陆安全风险。
- OnePass 效率需要同输入/硬件/batch/精度、完整视觉编码与预处理时间；区分冷启动、热缓存与多query摊销。一个共享上下文或规则编译可复用的基线不能被迫每张图重新编译以制造速度优势。

## 9. 资源、测试与提交纪律

- 主机为已有 RX7900XT/ROCm，不覆盖PyTorch，不向共享路径写入未经允许的数据；新增依赖隔离并记录。
- 初轮不使用四卡4090。以后即便获准，也保持个人数据≤10GiB，只考虑必要特征/小型权重，不能拷贝主HDF5。
- 既有¥500云预算不等于购买授权；不创建付费实例、不充值、不购买任何资源。
- 原始数据、P2.2 evidence、旧labels/cache、protocols/heldout_* 和CDR变更状态都不改动；新文件使用 `p30_` 命名空间。
- 禁止将原始大数据、缓存、模型权重、访问token、机器私有路径直接推Git；提交代码、脱敏日志、manifest及结果哈希。
- 所有upstream代码先pin commit，再执行。任何冲突先协调，不force-push，不用reset丢弃他人工作。

现有回归命令：

```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts src tests
```

P2.2 历史记录为444项（443通过、1跳过）；这是旧版本证据，不是P3.0新功能已通过。新增loader与模型代码须提供自己的测试；已有模型/数据的检查跳过必须说明原因，不能以缺本地数据伪造成功。

## 10. 本次 Coding Agent 应立即做什么

**第一提交：P30-00 审计脚本/资产清单 + P30-01 loader设计与离线测试。** 若已有可用像素，立刻读取一批真实Age样本并验证映射；没有则报告准确的最小下载阻塞，不重启旧LROC工程。

**第二提交：数据门槛通过后，Age普通头真实smoke与可恢复checkpoint。** 不需要先写语言模型、联合概率或复杂多槽位系统；只要能清楚复现一个公开任务。

**第三提交：固定输入下的Geology普通头与候选条件头，先在Train/Val验证。** 再冻结小矩阵和测试安排，公开测试不参与选择。

审计、离线代码和预算内小样本工作无需重复询问用户；超过初次下载范围、修改共享环境、付费/四卡、完整大模型或旧外部Gate4则必须保持HOLD。

**阶段目标不是“跑出一个好看的月球安全分数”，而是得到可审计的公开任务基线，并检验月球表征与任务条件化各自有没有增量。**
