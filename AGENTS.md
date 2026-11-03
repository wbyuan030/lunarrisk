# 当前任务：P3.1 — 正式 Benchmark 基线与候选对照（2026-11-03）

**首先阅读并执行 [`docs/P3_1_BENCHMARK_BASELINES.md`](docs/P3_1_BENCHMARK_BASELINES.md)。**
机器可读任务在 [`protocols/p31_execution_plan.json`](protocols/p31_execution_plan.json)。这是待实施交接，不是已运行实验或全量下载批准。当前文件只保留有效任务；P3.0 原入口保存在 [25c8ac9 的 AGENTS.md](https://github.com/wbyuan030/lunarrisk/blob/25c8ac93e6ea94a1f402c3daecf547bba1990278/AGENTS.md)，其他历史任务仍在 docs/reports。

## 当前判断

P3.0 已完成来源/标签/split审计、部分真实I/O、五类Age与49输出Geology工程smoke、独立进程回放。最新归档482项测试中481通过、1项旧标签可选检查跳过。主pilot仅5 Train+5 Val，前端为Age监督两步的scratch CNN；没有正式benchmark、预训练或候选语义增益。

**结束反复延长十张图smoke，转向可扩展runner与数据充足的普通基线。** Age先验证正式训练链，Geology为核心主任务；预训练权重核实并行，不阻塞普通scratch基线。

## 现在立即实施

1. **P31-01：新正式训练入口。** 不直接解除旧smoke上限。用分块SHA256替代大文件`read_bytes()`；worker-local只读HDF5和mini-batch DataLoader；验证/特征缓存也分批。记录实际RSS/显存/吞吐、数据与源码hash、checkpoint及回放；保留原smoke作回归。
2. **P31-02：设计一次有代表性的完整裁片验收。** 固定Train/Val ID，覆盖每组有效模态、部分缺失、raw zero与radar；网络前锁定范围和预算，沿用既有容差。原北极八图17通道全缺失，不能只重复零值配对或三行检查。只能在已授权传输范围内执行。
3. **P31-03：锁定普通基线协议。** 官方标签/输出类别/split不变；固定Accuracy/Macro-F1缺失类别口径、Val选择和训练预算。数据具备后先正式单seed运行验证，再按锁定配置做多seed。正常Train/Val学习曲线可以做，Test不得用于选模型。
4. **P31-04：候选对照纠偏。** 当前CandidateIDHead独立打分已知类ID，子集调用等价于其自身全类别softmax遮罩归一化。保留为`candidate_id_control`，不要用排列/子集接口证明语义或新类能力；增加恒等式控制测试，测实际算量/延迟。自然语言类别定义头另立公平协议，不能输入逐样本真实单位名/年龄前缀。

对应报告和脚本路径见P3.1计划；尚未实现的接口不能声称可运行。每项实施后更新`reports/p3_1_progress.md`及真实证据。原始Age/Geology映射、presence及统计来源保持已审计口径，任何改变另立实验轨。

## 数据和预训练授权边界

**2026-11-04 用户明确授权“允许大文件下载 推进基线”。** 执行范围见 [`protocols/p31_acquisition_authorization.json`](protocols/p31_acquisition_authorization.json)：下载固定 revision 的单一74,449,119,347字节HDF5，并推进已锁定的Age/Geology Train/Val基线。原 [`protocols/p30_next_acquisition_proposal.json`](protocols/p30_next_acquisition_proposal.json) 保留为授权前历史记录；其旧PENDING状态不否定这次新增明确授权。此次不下载完整mmap、额外权重或snapshot，不购买资源。原初次1GiB采样累计账本仍不重置（保守已计165,257,071字节）；大文件为独立新增授权范围。

可扩展runner、真实pilot回放/续训、代表性完整裁片配对和基线设计已交付，见 `reports/p3_1_progress.md`。现在直接执行固定单文件下载与正式基线，不再重复十张图smoke。仅单副本`.partial`、严格Range续传、完整大小/SHA256校验后原子改名；不重复保留snapshot/cache。数据校验后先Age42、Geology42核验正常训练与资源，再按冻结配置运行43/44。只用Train/Val，Test评分仍HOLD。

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

历史482项通过/跳过记录不代表P3.1新代码。保存本轮命令、PASS/FAIL/BLOCKED、源码/数据hash、逐样本预测与回放证据；完成后按用户既定习惯直接push main，不force-push，不覆盖并发工作。超过下载/付费/环境边界时报告准确阻塞，不隐式扩大权限。
