# 当前任务：P3.1 — 基线交付与后续公平对照（更新2026-11-04）

**首先阅读 [`docs/P3_1_BENCHMARK_BASELINES.md`](docs/P3_1_BENCHMARK_BASELINES.md) 的最新状态和 [`reports/p3_1_progress.md`](reports/p3_1_progress.md)。** 原机器可读交接在 [`protocols/p31_execution_plan.json`](protocols/p31_execution_plan.json)，保留为原任务设计，不能把其待实施措辞当作最新执行状态。P3.0 原入口保存在 [25c8ac9 的 AGENTS.md](https://github.com/wbyuan030/lunarrisk/blob/25c8ac93e6ea94a1f402c3daecf547bba1990278/AGENTS.md)，其他历史任务仍在 docs/reports。

## 当前判断

P31-01可扩展runner、真实pilot精确回放/续训、P31-02授权范围内完整裁片配对与离线复核、P31-04候选ID恒等式/实际延迟控制已交付。完整裁片验收覆盖全部七组、27个有效通道；roughness有效数据处理仍未验证。P3.0十张图smoke保留为回归，不再延长。

**独立本地前缀轨六次普通基线已全部完成：Age与Geology各seeds42/43/44，每次均通过独立进程精确预测回放。** 数据为保留前缀中5,135原Train+1,155原Val完整裁片，北半球可用性选择子集；官方映射/输出维数、stored成员归属和审计归一化不变。预先冻结协议见 [`protocols/p31_local_prefix_eval_plan.json`](protocols/p31_local_prefix_eval_plan.json)，最终报告见 [`reports/p3_1_partial_data_baselines.md`](reports/p3_1_partial_data_baselines.md)。Age Val Accuracy为58.6436%（样本标准差2.0070个百分点）、五类Macro-F1为0.403084±0.035138；Geology为43.6075%（样本标准差2.2583个百分点）、完整49类Macro-F1为0.208791±0.014977（n=3）。完整上游文件SHA256未核实，这些不是完整官方benchmark成绩。Test未评分。

最新实现验收为579项测试PASS、零跳过，compileall与真实冻结计划预览PASS；最终证据核验66个归档/48个本地文件hash、24份预测CSV、24项聚合统计PASS，见 [`sequence_verification.json`](reports/evidence/p31_prefix_runs_20261104/sequence_verification.json)。预测回放与重复训练确定性分开；不能由前者推断后者。模型未使用预训练或Age监督迁移；语义/候选增益及新类别能力仍未建立。

## 接续工作边界

1. **保留下载暂停，不自动恢复。** 用户在大文件授权后明确要求“先用这些数据吧，vpn流量快顶不住了”。当前六次离线实验已结束，不重复训练、不添加seed或事后改预算。
2. **完整官方基线仍待完整数据。** 原 [`protocols/p31_baseline_eval_plan.json`](protocols/p31_baseline_eval_plan.json) 及完整大小/SHA256/ID gate不变，不能将不完整文件传给完整数据入口，也不能用本地前缀成绩填充该轨。流量条件变化并获后续恢复指示后，再沿用单文件续传与冻结配置。
3. **候选对照另立公平实验协议。** 当前CandidateIDHead独立打分已知类ID，子集调用等价于其自身全类别softmax遮罩归一化；保留为`candidate_id_control`。已完成恒等式测试与实际延迟测量，不要用排列/子集接口证明语义或新类能力。自然语言类别定义头必须单独锁定来源、预算和公平对照，不能输入逐样本真实单位名/年龄前缀。
4. 更新`reports/p3_1_progress.md`及真实证据；尚未实现或运行的接口/实验不能声称可用或完成。原始Age/Geology映射、presence及统计来源保持已审计口径，任何改变另立实验轨。

## 数据和预训练授权边界

**2026-11-04 用户曾明确授权“允许大文件下载 推进基线”，随后因VPN流量要求暂停并先用本地数据。** 原新增范围见 [`protocols/p31_acquisition_authorization.json`](protocols/p31_acquisition_authorization.json)：固定revision的单一74,449,119,347字节HDF5及已锁定Age/Geology Train/Val基线。当前保留33,964,443,553字节，未完成40,484,675,794字节；本地暂停记录为`outputs/p31/acquisition_pause.json`，不自动恢复。原 [`protocols/p30_next_acquisition_proposal.json`](protocols/p30_next_acquisition_proposal.json) 保留为授权前历史记录；其旧PENDING状态不否定后来的授权，也不抵消更新的暂停要求。

不下载完整mmap、额外权重或snapshot，不购买资源。原初次1GiB采样累计账本不重置（保守已计165,257,071字节）；大文件为独立新增授权范围。只保留单副本`.partial`，未来恢复时严格Range续传，完整大小/SHA256校验后原子改名；不重复保留snapshot/cache。只用Train/Val，Test评分仍HOLD。原始像素、checkpoint和范围缓存留在本地ignored outputs，不入Git。

MG-MAE权重未在已审计来源核实，不从头预训练冒充；NASA LFM模态/尺度/几何/Python兼容性单独验证，不伪造通道和观测。当前不恢复完整Qwen/OnePass联训、TiM、主动路由、全球预训练或低秩后验扩容。

## 科学纪律与既有资产

- 所有候选比较使用相同encoder/模态/样本/预算；ordinary masked-softmax是必要对照。Age监督迁移是额外监督，明确标注。
- 官方stored/geographic、derived loader、新校准/语义协议分轨。当前stored划分并非新类别留出；49输出不等于已评估zero-shot类别。
- P2.2负结果、P3.0所有smoke/evidence/原提案原样保留。P2.2仍STOP_OR_REFRAME、稳定H1/H2增益NOT_ESTABLISHED。
- **旧LROC Gate3 NOT_FROZEN、Gate4 HOLD**；不批准EDR/CDR变更、不覆盖任何heldout、原始NAC/DTM/CDR或旧labels/cache。
- 本地RX7900XT/现有ROCm为主，不替换torch或实验室共享环境。不购买云、不启动四卡；四卡机器个人存储≤10GiB约束仍在。数据/权重/凭据/私有机器路径不入Git。

## 验证与提交

```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts src tests
```

主机缺少torch/h5py时沿用已验收的私有runtime/既有ROCm方式，命令与环境见进度和evidence；不安装或替换共享包。历史482项通过/跳过记录不代表P3.1新代码。保存本轮命令、PASS/FAIL/BLOCKED、源码/数据hash、逐样本预测与回放证据；完成后按用户既定习惯直接push main，不force-push，不覆盖并发工作。超过下载/付费/环境边界时报告准确阻塞，不隐式扩大权限。
