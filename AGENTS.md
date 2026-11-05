# 当前任务：P3.3 — NASA地形迁移本地审计完成，官方执行仍阻塞（更新2026-11-05）

**首先阅读 [`reports/p3_3_local_readiness.md`](reports/p3_3_local_readiness.md)、[`docs/P3_3_NASA_TRANSFER_HANDOFF.md`](docs/P3_3_NASA_TRANSFER_HANDOFF.md) 和 [`protocols/p33_nasa_transfer_plan.json`](protocols/p33_nasa_transfer_plan.json)。** 上游`8eeab47`已安全拉取，并合并并发`ce8b1ef`的固定根Train身份核验；此前pre-runner本地新增28项及完整654项测试零跳过PASS，compileall PASS。身份/标签认证通过后，真实裁片仍在空间策略处被拒绝；[最新合并验证](reports/evidence/p33_20261105/local_02/verification.json)。固定NASA配置已从既有缓存复核大小/SHA256/Git blob；直连SSH取得两个固定revision的10个所需源码文件并审计，未执行上游Python。原始Train2705裁片已核验，但真实输入CLI因官方`pre_resize512/input_size256`被当前空间策略守卫拒绝，exit1及日志保留，没有成功receipt。

**并行云端更新：已实现 [`scripts/p33_train_pair.py`](scripts/p33_train_pair.py) 的 lock/run/replay 配对训练入口，见 [`docs/P3_3_NASA_PAIRED_RUNNER.md`](docs/P3_3_NASA_PAIRED_RUNNER.md)。** 原始地形直接经认证prefix读取，固定Train/Val，显式预算/源码/资产/runtime锁定，双臂选模和独立进程回放。云端48项P33测试通过、16项因torch/h5py缺失跳过；这不延伸上述本地654项证据到新runner。未执行官方训练或冻结实际预算。保留本地所有审计、pre_resize512/input_size256和runtime阻碍，不静默放宽守卫；自动resume尚未实现。

**新增runner本地验证：** `423005e`安全拉取后，在现有私有runtime执行全部64项P33及完整690项测试，零跳过PASS，compileall PASS；[实际报告](reports/p3_3_runner_local_validation.md)、[日志/18个源码hash](reports/evidence/p33_20261105/runner_local_01/verification.json)。云端16项跳过已在本地执行，但torch训练/重建与HDF5仍是小型合成夹具，mock输出的LOCKED不是实际预算。没有运行官方NASA模型、完整真实cohort扫描、GPU训练或正式独立进程回放；下述空间/runtime阻塞保持，科学源码、旧证据/预算均未改。

## P3.3下一项与真实边界

- 现有主机/ROCm容器均为Python3.10.12、torch2.5.1；官方声明Python>=3.11,<3.13、torch>=2.12，且缺少OmegaConf/timm/einops/TerraTorch/Lightning等依赖。未找到现成兼容runtime，未安装包、创建新环境或替换共享ROCm。不能称官方模型已兼容。
- 先明确配对空间策略和兼容私有运行时方案。官方数据transform先resize512、再归一化及crop/resize；backbone接收张量。当前守卫拒绝不代表模型不能接受256张量，但不得修改YAML或静默跳过策略检查。
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

```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts src tests
```

需要时沿用已验收私有runtime/现有ROCm，不安装替换共享包。626项是本轮实现证据，不自动涵盖未来代码。每阶段完成后保存PASS/FAIL/BLOCKED、实际资源、checkpoint和逐样本回放，按习惯直接push main，不force-push、不覆盖并发提交。超出下载/付费/环境边界时准确报告，不隐式扩大权限。
