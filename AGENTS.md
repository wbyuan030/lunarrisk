# 当前任务：P3.3 — native256真实输入预检PASS，官方执行仍阻塞（更新2026-11-05）

**最新本地交付：** `f74af37`安全拉取后，现有私有runtime执行75项P33及完整701项测试，零跳过PASS，compileall/diff PASS。先查明旧LROC缓存测试缺DELTA产品页，补齐合成夹具并断言无urlopen调用后才运行完整回归；云端旧partial日志仍保留为INCOMPLETE。既有Train2705/target43原始裁片、batch和固定NASA配置SHA再次认证，新provenance显式选择native256，真实input-only CLI exit0/PASS；独立逐元素归一化及prepared hash完全相等。见[本地报告](reports/p3_3_native256_local_validation.md)、[22个源码hash及验证](reports/evidence/p33_20261105/native256_local_01/verification.json)。这只确认单裁片输入准备，`official_model_execution=NOT_RUN`，历史strict失败、原资产和P3.2结果不改写。

**最新云端交接：** 先读[`docs/P3_3_MOONSTONE_NATIVE256.md`](docs/P3_3_MOONSTONE_NATIVE256.md)和[`reports/p3_3_native256_policy.md`](reports/p3_3_native256_policy.md)。在任何迁移分数前新增独立`moonstone_native256_v1`协议：保留Moonstone原256网格、原NASA配置归一化和256/16模型位置几何，不复现NASA数据resize512/crop/flip，不声称60m物理尺度等价；默认strict路径仍拒绝原差异。必须显式选择协议、匹配固定配置SHA和新provenance，贯通preflight/dataset/model/manifestV2/run/replay。独立源码/diff审查通过；本轮75项P33中59PASS/16SKIP，compileall/diff检查PASS；完整套件因旧LROC网络请求被停止，只有部分日志，绝非全套PASS。本地旧690PASS不能延伸到新代码。

**最新runtime方案：** 见[`docs/P3_3_PRIVATE_RUNTIME_PLAN.md`](docs/P3_3_PRIVATE_RUNTIME_PLAN.md)、[`protocols/p33_private_runtime_plan.json`](protocols/p33_private_runtime_plan.json)和[研究报告](reports/p3_3_private_runtime_plan.md)。已只读核实27个固定源码文件及官方发行信息：独立Python3.12 + torch2.12.1/torchvision0.27.1的ROCm7.2轮子是候选，绝非已安装/已兼容环境。正常包初始化会导入训练/数据依赖，不能只补torch/einops。完整依赖解析/大小/hash仍缺；宿主版本及驱动release映射已补齐；候选ROCm7.2不在已安装driver6.2.x的官方支持行中，不能靠venv或容器绕过。

**宿主只读核实（历史记录）：** `7784886`后已确认Ubuntu22.04.5/kernel6.8.0-138/glibc2.35、已加载及磁盘amdgpu模块6.8.5和DKMS包`1:6.8.5.60204-2070768.22.04`；目标卷可用462,071,136,256字节。当时已知模块与包身份，但尚未查清驱动release映射；此缺口现已由下段官方包目录证据解决。不能把模块6.8.5当ROCm版本。见[实际宿主报告](reports/p3_3_host_runtime_audit.md)、[证据](reports/evidence/p33_20261105/host_runtime_01/host_audit.json)。未安装、下载权重或改驱动；75/701与input-only均未重跑。

**最新官方兼容结论：** `69763c5`之后只读核实AMD包目录，已安装版本对应driver6.2.4/family6.2.x；其官方支持行不包含候选ROCm7.2。这是支持矩阵不覆盖，不是实测不能运行。官方ROCm6.4/7.0索引最高torch2.9.1/2.10.0均低于NASA要求；NASA初始官方提交已要求torch≥2.12，回退祖先也无解。见[兼容路径报告](reports/p3_3_driver_runtime_compatibility.md)及[新证据](reports/evidence/p33_20261105/driver_support_01/public_evidence.json)。历史宿主审计保持原样，原“映射未知”现已解决为具体支持边界。

**最新CPU候选准备：** 已核实Python3.12的官方torch2.12.1+cpu/vision0.27.1+cpu元数据，以及28个直接候选约束和72个直接包间约束；28个直接轮子公布大小合计444,858,870字节，不是完整预算。OmegaConf依赖的已查ANTLR4.9.3没有兼容轮子，完整transitive/build闭包仍缺。见[CPU报告](reports/p3_3_cpu_feasibility.md)、[候选计划](protocols/p33_cpu_feasibility_plan.json)。新增独立离线metadata检查入口不导入torch/NASA、不联网，不等于运行兼容性PASS，也不批准新建/安装环境；模型与原守卫不改。新增18项离线测试PASS，独立审查及三文件compile检查通过；不延伸旧701证据。

**本地下一步：** 当前保持系统/驱动不动的范围内，没有核实到受官方支持且满足固定NASA依赖的GPU方案。停止在路径决定处，不重复inventory、75/701或native256预检，不安装、不拉容器、不改驱动、不下载权重、不降依赖或换源码pin。CPU仅可作为另行批准的有限兼容性研究，不冒充GPU方案；若以后考虑驱动路径，先另行核实RX7900XT/OS/kernel目标、旧ROCm6.2保留及回退，再获明确批准。完整依赖锁/大小/hash和预算仍缺。官方runtime、checkpoint、真实双臂forward/backward、完整cohort和预算锁定仍是独立门槛，Test HOLD。

**首先阅读 [`reports/p3_3_local_readiness.md`](reports/p3_3_local_readiness.md)、[`docs/P3_3_NASA_TRANSFER_HANDOFF.md`](docs/P3_3_NASA_TRANSFER_HANDOFF.md) 和 [`protocols/p33_nasa_transfer_plan.json`](protocols/p33_nasa_transfer_plan.json)。** 上游`8eeab47`已安全拉取，并合并并发`ce8b1ef`的固定根Train身份核验；此前pre-runner本地新增28项及完整654项测试零跳过PASS，compileall PASS。身份/标签认证通过后，真实裁片仍在空间策略处被拒绝；[最新合并验证](reports/evidence/p33_20261105/local_02/verification.json)。固定NASA配置已从既有缓存复核大小/SHA256/Git blob；直连SSH取得两个固定revision的10个所需源码文件并审计，未执行上游Python。原始Train2705裁片已核验，但真实输入CLI因官方`pre_resize512/input_size256`被当前空间策略守卫拒绝，exit1及日志保留，没有成功receipt。

**并行云端更新：已实现 [`scripts/p33_train_pair.py`](scripts/p33_train_pair.py) 的 lock/run/replay 配对训练入口，见 [`docs/P3_3_NASA_PAIRED_RUNNER.md`](docs/P3_3_NASA_PAIRED_RUNNER.md)。** 原始地形直接经认证prefix读取，固定Train/Val，显式预算/源码/资产/runtime锁定，双臂选模和独立进程回放。云端48项P33测试通过、16项因torch/h5py缺失跳过；这不延伸上述本地654项证据到新runner。未执行官方训练或冻结实际预算。保留本地所有审计、pre_resize512/input_size256和runtime阻碍，不静默放宽守卫；自动resume尚未实现。

**新增runner本地验证：** `423005e`安全拉取后，在现有私有runtime执行全部64项P33及完整690项测试，零跳过PASS，compileall PASS；[实际报告](reports/p3_3_runner_local_validation.md)、[日志/18个源码hash](reports/evidence/p33_20261105/runner_local_01/verification.json)。云端16项跳过已在本地执行，但torch训练/重建与HDF5仍是小型合成夹具，mock输出的LOCKED不是实际预算。没有运行官方NASA模型、完整真实cohort扫描、GPU训练或正式独立进程回放；下述空间/runtime阻塞保持，科学源码、旧证据/预算均未改。

## P3.3下一项与真实边界

- 现有主机/ROCm容器均为Python3.10.12、torch2.5.1；官方声明Python>=3.11,<3.13、torch>=2.12，且缺少OmegaConf/timm/einops/TerraTorch/Lightning等依赖。未找到现成兼容runtime，未安装包、创建新环境或替换共享ROCm。不能称官方模型已兼容。
- 已显式声明Moonstone-native256配对空间策略且单个真实Train裁片输入准备PASS；下一步明确兼容私有运行时方案。官方dtm/slope的Untok数据transform先resize512、再归一化及crop/flip；native256独立下游协议保留原网格，backbone接收张量。默认strict仍拒绝旧差异，不代表模型不能接受256张量；不得修改YAML或静默跳过策略检查。
- 已核实源码声明的高程米/坡度度、58–62°SLDEM/LOLA混合、NaN处理与HDF5原通道传递。原始栅格header、完整执行谱系、逐样本fallback未独立复现。236.901m是名义南北/赤道间距，东西地面间距随纬度变化；presence不是像素mask，finite zero保留。
- 两个先决门通过后再核实必要固定checkpoint、真实配对forward/backward，然后使用新runner在任何task分数前扫描固定Train/Val cohort并锁定预算；若需改变cohort或空间策略，另立明确协议后再执行。checkpoint未下载/完整核验，官方模型构造/梯度/训练/迁移收益均NOT_RUN/NOT_ESTABLISHED；不把合成wrapper测试当模型执行。
- 保留P3.2负结果与18run，不重跑。完整地质语义来源核实仍是独立后续项，不与NASA预训练迁移混写。只用Train/Val，Test HOLD；必要数据/权重直连授权及下文其他资源边界不变。

## P3.2已完成阶段与独立语义后续项

**阅读 [`reports/p3_2_progress.md`](reports/p3_2_progress.md)、[`reports/p3_2_lunarjev_mvp.md`](reports/p3_2_lunarjev_mvp.md) 和 [`reports/p3_2_decision.md`](reports/p3_2_decision.md)。** 原设计 [`docs/P3_2_LUNARJEV_MVP.md`](docs/P3_2_LUNARJEV_MVP.md)、[`protocols/p32_lunarjev_design.json`](protocols/p32_lunarjev_design.json) 保留为实施前交接；其未实施措辞不代表最新状态。实际运行清单是 [`protocols/p32_head_run_manifest_v2.json`](protocols/p32_head_run_manifest_v2.json)。P3.1入口可在 [edffc0a](https://github.com/wbyuan030/lunarrisk/blob/edffc0a7c274efb4c1cee03e6c0f3203d87ebba7/AGENTS.md) 追溯。

## 本轮已完成，不能反复重跑

- P32-01：三个原Geology最佳checkpoint/receipt核验，冻结参数和BN，Train5,135/Val1,155各导出16×256 FP32 token及同token均值pooled控制。实际每cache110,118,080字节，留在`outputs/p32/context_03/`；[缓存证据](reports/p3_2_context_cache.md)。
- P32-02：统一公共49+5候选表、120维catalog-only TF-IDF，明确`TAXONOMY_ONLY_LOCAL`。只有官方符号和Age映射，含身份性质literal特征，不冒充完整地质语言定义或预训练模型。逐样本真实unit name/age prefix绝不作为输入。
- P32-03/04：B0/B1/B2/J四头×seeds42/43/44及固定描述置换B2/J×三seed，共18run完成，全部独立进程精确回放六份预测CSV。旧Geology encoder冻结、所有头同cache/ID/任务监督/预算，未改P3.1设置。
- P32-05：J1/J2/J3已如实报告。Val full49 Macro-F1均值±样本std：**B0 .255488±.014813，B1 .212279±.009432，B2 .221628±.014017，J .202774±.020098**。J−B2 −.018854±.029749；正确描述相对固定乱序也无增益。共享上下文工程可行，语义/空间收益未建立。
- 验证：**626项测试零跳过PASS**、compileall PASS、真实GPU有限前后向PASS；最终162个本地/180个归档hash、3个cache hash、108份逐行预测ID/target/概率/指标/选模/停止/聚合与精确回放PASS。[最终核验](reports/evidence/p32_20261105/runs/sequence_verification.json)。首次冷GPU统计初始化和旧schema适配失败保留，不是额外超参试验。

原六组P3.1普通scratch结果和所有源码/预算/evidence保留。完整上游文件hash未核实；Val已反复使用，仅为探索性北半球开发轨。没有独立确认性Test、全月、新类或着陆安全结论。

## 下一项限定工作

1. **核实可溯源完整地质类定义及一个冻结文本表示。** 优先现有公开资产；缺项明确，不编造岩性/年代/风险描述。已有本地文本权重不等于已核验可用，核对模型卡/许可/revision/权重hash/runtime。缺口只阻塞对应完整语义科学主张，不回到反复smoke。
2. **另立完整语义tier和实际manifest再实验。** 当前lexical负结果保留；沿用同特征B0/B1/B2/J、普通masked-softmax、同描述pooled与Geology概率→Age、固定描述置换。所有样本共享公共描述，语义头不加绕过描述的可学习class-ID embedding/class-specific bias。
3. 不为“救”当前J延长预算、挑encoder/seed、事后选prompt、扩大模型或重复旧18run。少数类阈值已由Train≤50预先定义。n=3样本std不能写成置信区间；候选排列/子集恒等式不是语义/新类能力证据。
4. 后续实现用`src/lunarrisk/lunarjev/`、`scripts/p32_*`及独立`outputs/p32/`命名空间。尚未实现、核验或运行的项目不写PASS；更新进度、真实命令和源码/data/cache/text hash。

## 最新授权与资源边界

**用户最新明确指示：拉远端继续推进，若需模型权重或更多数据，不走代理即可下载。** 此指示更新旧的全面暂停边界。必要获取固定来源/revision/大小/hash，显式禁用代理，记录实际直连与许可；不反复索取已获同类授权。当前首轮既有数据足够，未新增下载、未恢复完整HDF5。现保留33,964,443,553字节，尚缺40,484,675,794字节；不做无必要40GB传输或全snapshot。

仅本地RX7900XT/现有ROCm；不替换torch或共享环境、不买云、不启动四卡；共享四卡机器个人存储≤10GiB约束不变。数据、cache、checkpoint、权重、凭据和私有路径不入Git。只用Train/Val，公开Test cache/评分HOLD。

MG-MAE权重未在审计来源核实，不从头预训练冒充；NASA LFM物理模态/尺度/几何/Python兼容性另验，不伪造通道。当前不恢复Qwen/完整OnePass联训、TiM、主动路由、全球预训练或低秩后验扩容。冻结文本表示核实与LLM联训是不同范围。

P2.2负结果和P3.0/P3.1全部旧资产不改写，H1/H2增益仍NOT_ESTABLISHED。旧LROC Gate3 NOT_FROZEN/Gate4 HOLD，不批准EDR/CDR变更、不覆盖heldout/provenance或原始NAC/DTM/CDR、旧labels/cache。

## 验证与提交

云端完整套件曾在legacy LROC网络请求处被停止，历史只有部分日志。本地查明cached-only测试缺少DELTA产品页，补齐合成缓存并断言无网络调用，随后完整701项零跳过PASS；这不改写云端历史失败。后续新增P33优先用`python3 -m unittest discover -s tests -p 'test_p33_nasa*.py' -v`验证；新增网络用例仍须先核实不会触发禁止的数据获取。

```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts src tests
```

需要时沿用已验收私有runtime/现有ROCm，不安装替换共享包。626项是本轮实现证据，不自动涵盖未来代码。每阶段完成后保存PASS/FAIL/BLOCKED、实际资源、checkpoint和逐样本回放，按习惯直接push main，不force-push、不覆盖并发提交。超出下载/付费/环境边界时准确报告，不隐式扩大权限。
