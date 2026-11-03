# P3.0 下一阶段决策

2026-11-03。状态：**需要讨论研究路线与扩大数据获取授权；不需要现在购买 GPU 或磁盘。** 机器可读提案：[p30_next_acquisition_proposal.json](../protocols/p30_next_acquisition_proposal.json)，明确 PENDING_USER_DECISION，不代表批准。

## 当前已做到哪里

现有资源内完成了真实 HDF5 I/O、官方标签/split 审计、28 通道固定行配对、五类 Age 训练/恢复、49 输出 Geology 同 cache 两头训练/恢复与候选排列控制。最新 482 tests（481 PASS/1 旧数据可选 SKIP）、compileall PASS。没有 Test 像素或评分，没有付费/四卡，没有替换 ROCm。

[Geology 实测](p3_0_geology_baselines.md) 的 20 行、980 个概率在独立进程回放后字节一致；候选排列误差 3.73e-9，普通头完整 mask 控制误差 0。它们说明接口正确，不说明候选条件化有效。现有主 pilot 只有 5 Train/5 Val、Geology 5/49 类；冻结前端还是同批 Train 上 2 步 Age 监督的 scratch CNN。继续增加少量 smoke 更新不会检验“月球预训练表征＋任务条件化”的研究问题。

累计实际响应体 **78,502,175 字节**，保守累计 **83,468,143 字节**，初轮 **1 GiB** 上限仍未修改。剩余预算足以做更多局部探针，但并不授权全量数据或权重；本轮没有用耗尽预算替代研究选择。

## 需要先决定的研究问题

**建议先建立数据充足的 scratch/shared-encoder 普通基线，同时保持预训练轨 BLOCKED。** 这能检验监督表征下候选头相对 ordinary/masked-softmax 的增量，但必须明确它没有验证领域预训练收益。正式模型/训练预算/Val 选择与候选协议应在看新结果前锁定。

另一条合理路线是优先取得可验证的原生 MG-MAE checkpoint 后才做正式矩阵。当前已审计官方来源没有核实权重，不能从头预训练冒充。NASA backbone 虽真实可达，但其 Python 兼容性、模态/物理量/643nm 与观测上下文缺口仍在；单独下载 2.43GB backbone 不能解决输入映射，也不在当前初轮上限内。

当前 candidate_id 头只条件于已知的学习类 ID，没有自然语言地质定义、候选间交互或新类别语义能力。是否把它作为最小科学比较，还是先定义统一的公开地质描述/更强任务条件，是研究设计选择。不能用十张图的低 loss 为任一路线背书。

## 具体可审批的数据方案

固定版本只下载一个文件：

- 仓库 `ayushprd/Moonstone`，revision `53099efd0f2db81eee96b28eae87e1efdd281310`。
- `benchmark/lunar_patches_v4.h5`，**74,449,119,347 字节（69.34 GiB）**。
- 完整 SHA256 需下载后核验：远端声明 `e3e96bccc044d856cd55883c9cda7628f02afbdc1b2d1818b401bbcc590df83f`，目前只有 Range/局部 hash 证据。
- stats/class map 已在本地并已校验，无需重复下载。不给完整 pretraining snapshot、aligned rasters、NASA 权重或其他仓库隐式授权。
- 建议预留 **80 GiB** 新工作空间：新 revision 目录直接写 `.partial`，校验完整大小/hash 后原子改名，避免 snapshot/cache 再复制一份。80 GiB 是预留方案，不是已测下载器峰值。
- 另可审批不超过 **256 MiB** 的指定 28 mmap 通道固定范围，用于代表性有效模态的完整裁片配对；不得因此拉取全部 mmap。读取前锁定 ID、容差与 mask 条件。大量细 Range 的延迟尚未测定，不承诺完成时间。

本轮实测可用磁盘 **514,972,385,280 字节（约 479.6 GiB）**；单份 HDF5 下载后，按该时点尚有约 410.3 GiB 可用。因此需要的是扩大数据传输范围及研究路线确认，**当前无需购买磁盘**。原样保留旧原始数据、cache、P2.2 evidence 与旧 heldout 协议。

这条 HDF5 路线在代表性全裁片配对前仍为 derived 轨，不直接冒充官方 mmap 复现。真实官方存储 Train/Val/Test 继续保持 11,400/2,400/2,400，不重新抽样；Geology 49 输出仍保留，Train 只出现 46 类这一事实不能静默过滤。

## 若优先要求官方 mmap 原路径

HDF5 + 28 个完整 float32 mmap 最少 **193,357,637,747 字节（约 180.08 GiB）**。packed masks 额外 3,715,894,784 字节，总共 **197,073,532,531 字节（183.54 GiB）**；官方 Age/Geology loader 实际不读取 packed masks。该路线在本机磁盘上可容纳，但网络与文件范围显著扩大，必须单独审阅，不是本提案自动执行的推荐项。

没有额外下载就能给出的科学结论到此为止：工程链路可用；正式 Age/Geology 收益、领域预训练增益、候选语义增益仍未建立。旧 LROC Gate 3 NOT_FROZEN、Gate 4 HOLD，P2.2 STOP_OR_REFRAME/NOT ESTABLISHED 均不变。
