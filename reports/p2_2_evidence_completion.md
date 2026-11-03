# P2.2 — 概率、重放与配对统计实测（2026-11-03）

**本轮开发集验证已完成；工程决定保持 `STOP_OR_REFRAME`，H1/H2 的稳定普遍增益为 `NOT ESTABLISHED`。**
修复评价与证据链后，恢复的 coarse-logistic 在两个主事件上的 Brier/NLL 均优于 Joint 的三个 seed。
Joint−Diagonal 的方向依赖 seed，不能据此宣称稳定联合相关性收益，也不能把未确立增益写成整体假设被否决。

本页取代旧报告的完成状态与结论口径。旧数值保留供溯源；以下全部来自本次真实执行，
仅涉及反复查看的 **110 Tile / 25 Group development_test**。外部 Gate 3 `NOT_FROZEN`、Gate 4 `HOLD`。

## 1. 执行与不可变源码

- 实验源码提交：`ac93b100d17dbfa1d5e048b8785d04e33f4a8551`；日志的 `code_sha=ac93b10` 指向该提交。
- 用 `git archive HEAD` 在全新源码目录展开该提交，没有覆盖容器原副本。
- 数据根目录下的运行目录：`outputs/p22_20261103_validation`；日志目录：`outputs/p22_20261103_logs`。
- 公开证据使用 `${SOURCE_ROOT}` / `${PROJECT_DATA}` / `${ROCM_CONTAINER}` 占位符，表示本地源码根、数据根与既有容器。
- 本地 RX 7900 XT，Python 3.10.12，torch 2.5.1+rocm6.2，HIP `6.2.41133-dd7f95766`；numpy 2.2.6、pandas 2.3.3。
- 仅用既有标签与缓存；没有下载、扩容、付费资源、大模型或外部模型预测。

真实启动命令的可分享版本（先把三个变量设为本地路径和容器名；仅替换位置标识）：

```bash
docker exec -w "$SOURCE_ROOT" \
  -e LUNARRISK_GIT_SHA=ac93b10 -e OMP_NUM_THREADS=1 "$ROCM_CONTAINER" \
  python3 scripts/run_p22_evidence.py \
  --run "$PROJECT_DATA/outputs/p22_20261103_validation" \
  --logs "$PROJECT_DATA/outputs/p22_20261103_logs" \
  --labels "$PROJECT_DATA/outputs/labels_hydrated.csv" \
  --cache "$PROJECT_DATA/outputs/cache_v3.npz"
```

复跑须选择新的源码/产物目录；脚本拒绝覆盖旧运行。
源码哈希、输入哈希、全部协议文件哈希、每条实际命令、退出码与日志 SHA 见
[execution.json](evidence/p22_20261103/logs/execution.json)。下表是命令墙钟时间，不是 GPU kernel 时间。

| 阶段 | 实际退出码 | 秒 | 结果 |
|---|---:|---:|---|
| 全仓 unittest | 0 | 11.25 | `Ran 444 tests`，`OK (skipped=1)`：443 通过、1 跳过 |
| compileall scripts/src/tests | 0 | 0.04 | 通过 |
| 三 seed 比较 | 0 | 91.90 | 6 模型 × 3 seed 产物，后验含 3 种同 checkpoint 对照 |
| 重放 + 4 次独立重训 | 0 | 64.45 | 全部通过 |
| 18 条 Train 阈值 AST 诊断 | 0 | 11.26 | 4 模型族 × 3 seed × 18 规则，216 条分数完整保留 |
| 历史 v1 centroid collision | **2** | 0.02 | 预期阻塞：17 对冲突，8 对触及候选选择 |
| v2 centroid collision | 0 | 0.02 | 0 对冲突，仍为 **UNCERTIFIED** |

跳过的是 `test_feature_list_present_in_hydrated_labels`：不可变源码快照不含 `outputs/labels_hydrated.csv`。
后续比较通过绝对路径读取真实 760 行标签并实际使用全部 coarse 特征。
[完整单测日志](evidence/p22_20261103/logs/tests.log)保留测试名、结果和跳过原因。
预检曾因 coarse-logistic 浮点重算的末位差异触发严格逐位断言；已改为 `rtol=1e-14, atol=1e-15` 后重新运行全套。

## 2. 修正范围与共同样本

修复了 CSV 全集合抽样与 JSON 分 split 抽样不一致；两者现在使用相同 split 顺序、ID 顺序、draws 与 sampling seed。
checkpoint 保存全部 Cal 温度、scaler、AST、split ID、group、变换 eps、采样种子、执行源码哈希和输入 provenance。
基线同样有可加载 checkpoint；恢复原 coarse-logistic + Cal-only Platt，direct-event 保留原始专用 sigmoid 分数。
物理空间四目标 Gaussian Ridge 单独标注为失配参照：不保证 `g15≤g10`，不能称其必然比历史事件基线更强。

配对验证现在拒绝重复 ID、不同 group/split/四属性真值、缺失缓存与非法概率；重用产物须通过配置、源码和文件哈希校验。
固定覆盖率按风险及 sample ID 排序，严格接受 `floor(n*c)` 个样本；Cal 阈值部署保留实际覆盖率和平票影响。
零放行/零危险分母返回未定义及原因，而非自动记为零。

| 项目 | 本轮实测 |
|---|---|
| 760 行中过滤后使用 | 749：Train 399 / Val 120 / Cal 120 / Test 110 |
| 组数 | Train 195 / Val 26 / Cal 26 / Test 25 |
| 排除 | NAC valid fraction <0.99：Train 1、Test 10 |
| 18 个逐模型/seed CSV 的并集 / 交集 | 749 / 749 |
| 原标签 SHA256 | `eb2ff9326cd3d3180409355b53b2cf135c02ddd324cec87e583686abccdaafb5` |
| canonical 标签 SHA256 | `418561718334355843bc3609dad64a0964ca1b5565d2e8ea9ac4419a5ae05596` |
| cache SHA256 | `e1bc2134cfdb9290b8b557394c842c59e431102734166700c813f5cdba3e92fa` |
| 排序 ID SHA256 | `5c3d696547972cd7646d68eed0a2099d1751961ff7bed9b51be13bae0b3bc22b` |

参数量 Point/Diagonal/Joint 分别为 **98,464 / 99,008 / 100,096**。
epochs 上限40、batch32、lr0.001、4096 普通 MC draws、2000 次组配对 bootstrap，seeds42/43/44。
coarse-logistic 是同一确定性拟合的三份记录，不能当作三次独立训练；Ridge 拟合也相同，seed 变化来自 MC。
危险率是原规则 AST 对真实标签的计数；例如 g10 主事件 Test **76/110**，g15 主事件 **53/110**。

## 3. 重放与独立重复

[完整审计](evidence/p22_20261103/artifact_audit/audit.json)记录每个 checkpoint 文件和权重 SHA。

- 9 个后验 checkpoint 重放全部 split；Joint 同时重放 full/shared U=0/refit U=0/latent matched。
  **最大事件概率差 0，最大 JSON−CSV Brier 差 0**。
- 9 个基线 checkpoint 在独立新 Python 进程重载 scaler、权重与校准：
  coarse-logistic 最大差 `1.11e-16`，Ridge 最大差 `2.84e-14`，direct-event 最大差 `2.98e-8`，
  均小于事先代码中定义的 `1e-7` 容差；此处范围包含导出均值及概率，不能写成逐位完全相同。
- Point seed42、Joint seed42 各开两个新进程从头训练，共四次；与矩阵原始运行比较，
  **权重 SHA、完整 T、U=0 refit T 均一致，最大概率差 0**。各次命令、日志和报告均保留。
- 单测另含真实小网络原始推理→保存→新进程恢复的全变体/分支测试，及一次真实 CPU 训练后 JSON/CSV/replay 一致性测试。

这些检查只证明本硬件/软件/配置的执行复现；不证明换环境也确定，也不构成独立地理验证。
早期差异归因于 MIOpen kernel/reduction 的说法未被隔离验证，已撤回。

## 4. 五条既有规则的结果

下表为 seed42/43/44 的**描述性算术均值**，不提供把同一批开发组视为独立重复的 SEM。
NLL 是 epsilon-clipped **Bernoulli event NLL**，不是物理连续密度 NLL。
物理密度的正号真值 Jacobian 公式有回归测试，但端点点质量仍未建模，不作为最终论文指标。
全部每 seed 分数、ΔNLL 和区间见 [compare.json](evidence/p22_20261103/compare.json)。

| 规则 / 指标 | coarse_logistic | coarse_ridge | direct_event | point | diagonal | joint |
|---|---|---|---|---|---|---|
| g10_gt_0.05 / Brier | 0.0728 | 0.1020 | 0.0796 | 0.0748 | 0.0897 | 0.0857 |
| g10_gt_0.05 / NLL | 0.2283 | 0.3006 | 0.2447 | 0.2451 | 0.2995 | 0.2905 |
| g15_gt_0.02 / Brier | 0.0751 | 0.1425 | 0.0880 | 0.0924 | 0.1149 | 0.1197 |
| g15_gt_0.02 / NLL | 0.2549 | 0.4120 | 0.2843 | 0.3034 | 0.3756 | 0.3968 |
| g10_gt_0.10_or_g15_gt_0.03 / Brier | — | 0.1176 | — | 0.0779 | 0.0960 | 0.0967 |
| g10_gt_0.10_or_g15_gt_0.03 / NLL | — | 0.3341 | — | 0.2520 | 0.3243 | 0.3288 |
| g10_gt_0.10_and_rough_gt_0.05 / Brier | — | 0.0579 | — | 0.0859 | 0.0564 | 0.0776 |
| g10_gt_0.10_and_rough_gt_0.05 / NLL | — | 0.2002 | — | 0.2679 | 0.1847 | 0.2661 |
| not_both_safe / Brier | — | 0.1243 | — | 0.1325 | 0.1117 | 0.1196 |
| not_both_safe / NLL | — | 0.3388 | — | 0.3883 | 0.3278 | 0.3461 |

负差代表左侧模型更好。下表均为 25 个 Test 组的 pointwise 95% 区间，2000 次成组配对重抽样。

| 比较 | 规则 | seed42 ΔBrier [95% CI] | seed43 | seed44 |
|---|---|---|---|---|
| diagonal−point | g10 | +0.0209 [-0.0017, +0.0414] | -0.0086 [-0.0226, +0.0052] | +0.0325 [+0.0217, +0.0442] |
| diagonal−point | g15 | +0.0262 [+0.0073, +0.0451] | -0.0097 [-0.0236, +0.0026] | +0.0510 [+0.0276, +0.0737] |
| joint−diagonal | g10 | -0.0258 [-0.0415, -0.0056] | +0.0278 [+0.0084, +0.0436] | -0.0139 [-0.0221, -0.0060] |
| joint−diagonal | g15 | -0.0161 [-0.0317, -0.0013] | +0.0484 [+0.0263, +0.0680] | -0.0179 [-0.0337, -0.0022] |
| joint−coarse_logistic | g10 | +0.0036 [-0.0080, +0.0160] | +0.0248 [-0.0033, +0.0439] | +0.0105 [-0.0076, +0.0247] |
| joint−coarse_logistic | g15 | +0.0196 [-0.0014, +0.0389] | +0.0600 [+0.0315, +0.0850] | +0.0542 [+0.0251, +0.0795] |

H1 在主事件没有稳定改善；H2 存在两个 seed 改善、另一个明显变差的情况。
不采用不存在的“预注册 2/3 显著”标准，不把某两次改善包装成独立确认。
Diagonal 在已有 roughness 组合规则上较好，但这不足以证明所有属性/规则上的普遍增益。

## 5. 风险计数与部署阈值

下表完整展示 g10 主事件在名义20%处的三 seed 计数；Test 危险总数均为76。
Oracle 接受数严格为22，Cal20 是 **Cal 概率20%分位数**导出的阈值，实际 Test 覆盖率可以不同。
不能跨实际覆盖率直接认定 UAR 更低即更优。10%/40%、g15 及其余可用规则完整记录在机器表中。

| 模型 | seed | Oracle20 危险/放行 | Cal20 危险/放行 | 实际覆盖率 | UAR | HMR |
|---|---|---|---|---|---|---|
| coarse_logistic | 42 | 3/22 | 2/19 | 17.27% | 0.1053 | 0.0263 |
| coarse_logistic | 43 | 3/22 | 2/19 | 17.27% | 0.1053 | 0.0263 |
| coarse_logistic | 44 | 3/22 | 2/19 | 17.27% | 0.1053 | 0.0263 |
| coarse_ridge | 42 | 3/22 | 3/19 | 17.27% | 0.1579 | 0.0395 |
| coarse_ridge | 43 | 3/22 | 3/18 | 16.36% | 0.1667 | 0.0395 |
| coarse_ridge | 44 | 3/22 | 2/17 | 15.45% | 0.1176 | 0.0263 |
| direct_event | 42 | 1/22 | 0/17 | 15.45% | 0.0000 | 0.0000 |
| direct_event | 43 | 1/22 | 0/17 | 15.45% | 0.0000 | 0.0000 |
| direct_event | 44 | 1/22 | 1/14 | 12.73% | 0.0714 | 0.0132 |
| point | 42 | 2/22 | 2/18 | 16.36% | 0.1111 | 0.0263 |
| point | 43 | 2/22 | 1/18 | 16.36% | 0.0556 | 0.0132 |
| point | 44 | 0/22 | 0/20 | 18.18% | 0.0000 | 0.0000 |
| diagonal | 42 | 3/22 | 3/21 | 19.09% | 0.1429 | 0.0395 |
| diagonal | 43 | 1/22 | 0/11 | 10.00% | 0.0000 | 0.0000 |
| diagonal | 44 | 3/22 | 2/14 | 12.73% | 0.1429 | 0.0263 |
| joint | 42 | 3/22 | 1/15 | 13.64% | 0.0667 | 0.0132 |
| joint | 43 | 2/22 | 0/18 | 16.36% | 0.0000 | 0.0000 |
| joint | 44 | 2/22 | 2/20 | 18.18% | 0.1000 | 0.0263 |

本次新增的是**固定 Cal 阈值**的 Test-group 配对风险区间，bootstrap 内不重拟合阈值。
例如 g10、Cal20 的 Joint−coarse_logistic：

| seed | ΔUAR [95% CI] | ΔHMR [95% CI] | UAR 有效重复数 |
|---|---|---|---|
| 42 | -0.0386 [-0.1158, +0.0000] | -0.0132 [-0.0395, +0.0000] | 2000/2000 |
| 43 | -0.1053 [-0.3529, +0.0000] | -0.0263 [-0.0789, +0.0000] | 2000/2000 |
| 44 | -0.0053 [-0.0642, +0.0318] | +0.0000 [+0.0000, +0.0000] | 2000/2000 |

完整其他模型差、规则、覆盖率、有效/未定义重复数在 `summary.deployable_pairs`。
零危险放行导致的零宽经验区间仅描述该观测样本的重抽样，不能解释为未来零风险保证。
这些区间不包含 Train/Cal 重抽取或模型训练不确定性，未作多规则/多 seed 的联合校正。

一个真实误放行实例：Joint seed42，`M1222396634_r10028_c2624` / `G_M1222396634`，
真实 g10=`0.091892`，预测危险概率约 `0.185303`，低于 Cal20 阈值 `0.242919921875`，
因此是15个放行中的1个危险样本。它仅是可追溯失败案例，未据此修改规则或重训。

## 6. 同 checkpoint 消融与潜变量边缘

| seed | full/shared T | U=0 refit T | g15 ΔBrier shared−full | refit−full | latent matched−full |
|---|---|---|---|---|---|
| 42 | 0.979576 | 2.102237 | -0.020258 | -0.008556 | -0.006388 |
| 43 | 0.979807 | 1.338256 | -0.007852 | -0.002757 | -0.011057 |
| 44 | 0.839345 | 1.525023 | -0.005934 | -0.011018 | -0.008960 |

三个分支不能混为一个消融。U=0 改变潜变量边缘方差；matched 固定的是 `diag(Σ_z)`。
`g15=sigmoid(z1)*sigmoid(z2)`，改变 Cov(z1,z2) 可以改变物理 g15 的整个边缘分布，
所以改善不能归因为“固定物理边缘之后相关性有害”。本次 matched 与 full 的 g15 每 site
抽样均值平均绝对差为 **0.0353 / 0.0378 / 0.0449**。

相反 g10/roughness/relief 各自仅依赖一个潜变量，理论物理边缘仍相同。
g10 公共噪声样本也逐位相同，三个 seed 的该规则 ΔBrier 均为0。
roughness/relief 的有限 MC 样本未逐位相同，抽样矩差不是其理论边缘变化的证据。
尤其 seed42 relief 的两组抽样均值平均绝对差约 **60,870**，提示 `exp(z)` 极端抽样使均值/方差估计很不稳定；
不能把该数字当成真实地形测量或证明另一种物理边缘变化。
原始数值见 [physical_marginal_changes.csv](evidence/p22_20261103/rule_diagnostic/physical_marginal_changes.csv)。
这些现象值得后续方法诊断，本轮没有据 Test 结果修改模型。

## 7. MC 与18条规则诊断

固定 checkpoint/seed 的普通 MC 检查使用256/1024/4096 draws，未声称 QMC。
g10 的最大 site 概率差为1024−256：0.0693、4096−256：0.0842；
各规则4096−256最大差约0.0503–0.0947。4096 draws 对闭式 g10 CDF 的最大绝对误差，
Point/Diagonal/Joint 为 **0.0151 / 0.0150 / 0.0189**。概率尚有有限 MC 误差，不能把微小评分差当确定优势。

18 条 AST 由四个 Train 属性的25/50/75分位数及固定三对 AND/OR 中位数组合构成；
阈值清单在诊断预测之前写入 [rules.json](evidence/p22_20261103/rule_diagnostic/rules.json)。
测试验证改变全部 Val/Cal/Test 值不会改变规则。它是**事后开发诊断**，不是新预注册，未筛选有利规则。
四属性 Ridge 可以转移到这些规则；两种固定事件分类头不能转移。

下表为全部18条规则的描述性三 seed 平均 Brier。
全部216条单次分数、实际危险数、逐样本概率及162个组配对 Brier 区间均已归档。
诊断没有额外生成逐规则 ΔNLL/部署风险区间；这些主分析统计已在原五规则比较中提供，不能声称诊断具有同等确认范围。

| 规则 | coarse_ridge | diagonal | joint | point |
|---|---|---|---|---|
| g10_AND_roughness_9m_train_medians | 0.1453 | 0.1675 | 0.1649 | 0.1808 |
| g10_OR_roughness_9m_train_medians | 0.1087 | 0.1131 | 0.1175 | 0.1270 |
| g10_train_q25 | 0.0959 | 0.0897 | 0.0876 | 0.0870 |
| g10_train_q50 | 0.0871 | 0.0957 | 0.0906 | 0.0796 |
| g10_train_q75 | 0.0426 | 0.0676 | 0.0687 | 0.0556 |
| g15_AND_relief_p95_p5_train_medians | 0.0642 | 0.0571 | 0.0979 | 0.0578 |
| g15_OR_relief_p95_p5_train_medians | 0.0887 | 0.0712 | 0.0804 | 0.0784 |
| g15_train_q25 | 0.1089 | 0.1018 | 0.0988 | 0.1114 |
| g15_train_q50 | 0.1404 | 0.1102 | 0.1214 | 0.1011 |
| g15_train_q75 | 0.0618 | 0.0828 | 0.0917 | 0.0649 |
| relief_p95_p5_train_q25 | 0.0281 | 0.0301 | 0.0555 | 0.0348 |
| relief_p95_p5_train_q50 | 0.0245 | 0.0353 | 0.0879 | 0.0558 |
| relief_p95_p5_train_q75 | 0.0198 | 0.0426 | 0.1709 | 0.0912 |
| roughness_9m_AND_relief_p95_p5_train_medians | 0.1185 | 0.1257 | 0.1570 | 0.1478 |
| roughness_9m_OR_relief_p95_p5_train_medians | 0.0738 | 0.0740 | 0.0811 | 0.0942 |
| roughness_9m_train_q25 | 0.1349 | 0.1171 | 0.1218 | 0.1347 |
| roughness_9m_train_q50 | 0.1939 | 0.1988 | 0.2111 | 0.2339 |
| roughness_9m_train_q75 | 0.2323 | 0.2331 | 0.2757 | 0.3038 |

## 8. 证据位置与后续边界

仓库 [evidence/p22_20261103](evidence/p22_20261103/) 保存日志的路径脱敏副本、比较 JSON、逐模型报告、
所有主分析和诊断逐样本概率（`.csv.gz`）、规则清单、独立重复报告与审计。
只将机器绝对路径、容器名称换成占位符，未修改任何测试结果、数值或科学配置。
未经脱敏的原始日志继续保留在本地运行目录。
[archive_manifest.json](evidence/p22_20261103/archive_manifest.json) 同时记录原始输入文件 SHA256、
脱敏后未压缩内容 SHA256 和归档 SHA256；执行清单内的 `log_sha256` 仍指向本地原始日志，
校验公开日志时须使用归档清单的 `content_sha256`。因此不能把公开脱敏日志称为原始文件的逐字副本。
`matrix_run.json` 另含原始18个 checkpoint、CSV、报告的文件 SHA256。
所有 `.pt` 保留在上述本地新运行的 `ckpt/`，独立重训 checkpoint 在 `artifact_audit/*_repeat*/model.pt`，
未将大型权重或输入缓存写入 Git。`pandas.read_csv(path)` 可直接读取归档 `.csv.gz`。

[H1/H2 决定](p2_2_hypothesis_decision.md)仍为 STOP_OR_REFRAME：停止以这些开发结果支持大模型扩展，
把未确立的稳定增益如实保留为研究结果。后续若做新方法，需要另定可审计问题与确认协议。

本轮 v1 检查的 exit2 是明确阻塞信号，v2 exit0 仅表示没有 centroid 警报，绝非外部独立性认证。
所有协议哈希记录在执行清单，原冻结文件没有改写。
更广泛662景桥接/同源审计、源 provenance、EDR/CDR 路线变更签核仍未完成；
外部协议保持 `PENDING_SIGNOFF`，Gate3 `NOT_FROZEN`，Gate4 `HOLD`。
