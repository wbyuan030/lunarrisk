# 当前任务：P3.2 — 开始设计并实现 LunarJEV-lite（2026-11-04）

**首先执行 [`docs/P3_2_LUNARJEV_MVP.md`](docs/P3_2_LUNARJEV_MVP.md)，机器可读设计见 [`protocols/p32_lunarjev_design.json`](protocols/p32_lunarjev_design.json)。** 用户已明确希望从可用特征转向 LunarJEV。P3.1六组普通基线已经结束，不再要求重复证明CNN能学习，也不把完整数据、NASA/MG-MAE权重或GPU性能优化作为本轮启动前置。

本文件只保留当前有效工作；P3.1交接可在 [edffc0a版本](https://github.com/wbyuan030/lunarrisk/blob/edffc0a7c274efb4c1cee03e6c0f3203d87ebba7/AGENTS.md) 与对应docs/reports追溯。

## 已有证据与边界

5,135 Train / 1,155 Val 的北半球可用前缀上，Age/Geology各seeds42/43/44均超过Train prior；Age Val Accuracy 58.6436%±2.0070个百分点，Geology 43.6075%±2.2583个百分点。见 [`reports/p3_1_progress.md`](reports/p3_1_progress.md) 和 [`sequence_verification.json`](reports/evidence/p31_prefix_runs_20261104/sequence_verification.json)。这足以启动新决策模块，**不证明**冻结特征足够、语义或预训练增益已经成立。

旧实现579项测试通过是P3.1证据，不是新模型通过。六组模型/预测/归档不改写，当前无额外scratch训练需要。完整上游文件hash未验证；Val已反复使用，不称独立确认性评测。Test没有评分。

## 现在直接动手，不再延长smoke阶段

1. **P32-01：缓存共享视觉上下文。** 复用三个已完成的Geology checkpoint，逐个验证receipt并加载 `best_state_dict`。冻结参数和BN，在既有Train/Val上分批导出全局池化前空间特征；固定平均池化为4×4，即16个256维token，并从相同token生成pooled control。第一轮工程固定seed42，不按成绩选encoder；正式头部对照补齐42/43/44。不重新训练encoder、不请求Test像素、不新增网络读取。
2. **P32-02：公共候选语义表。** 现有49类文件主要是符号，不冒充完整地质描述。使用已缓存且可溯源的公共图例/定义，缺项标明；不得编造定义。无完整图例时可用公开符号+官方Age映射的 `TAXONOMY_ONLY_LOCAL` 弱语义推进接口与有限控制，但不能声称完整语言语义。文本表征只用已存在的可核验冻结encoder；没有则使用明确标注的确定性lexical/TF-IDF控制，不擅自下载权重。候选表对所有样本完全相同，绝不把样本真实unit name/age prefix喂给模型。
3. **P32-03：实现四头同特征对照。** B0普通任务头、B1候选ID结构控制、B2 pooled semantic、J任务/候选描述查询空间token。首版隐藏维128、4头、2个cross-attention块、FFN256、dropout0，不使用候选位置编码。共享一次视觉编码，并行回答49类Geology与5类Age，先做Choice；Noul/Score预留，不虚造标签。缺文本资产不阻塞缓存、结构和mask/frozen测试。
4. **P32-04：固定实际asset/run manifest后跑小头部实验。** 同encoder/ID/任务监督/描述来源/预算，FP32、batch128、AdamW lr1e-3/wd0.01、最多50epoch/patience8，无超参搜索；按Val Geology全49类Macro-F1选checkpoint，Age用同一个checkpoint。四组都重新训练头，不拿P3.1原端到端分类头直接充当唯一公平参照。先seed42核验执行，再同配置43/44。记录逐样本概率、class support、参数/真实时延、checkpoint与回放。
5. **P32-05：报告J1/J2/J3。** 描述语义、空间查询、共享上下文分别归因，保留ordinary masked-softmax、相同描述pooled head、Geology概率按官方映射汇总Age等强控制。描述置换是独立固定控制，不事后搜索prompt。候选排列/子集接口正确不等于新类/语言泛化。

所有列出的p32脚本/模块路径是待实现接口，本次交接仅为设计。新增代码在 `src/lunarrisk/lunarjev/` 和 `scripts/p32_*`，日志/缓存用独立 `outputs/p32/`；不要改旧训练runner的科学设置。进度写 `reports/p3_2_progress.md`，最终比较写 `reports/p3_2_lunarjev_mvp.md`。

## 研究口径

- 本版是自定义轻量判别式 Semantic Query Decider，不宣称已经复现RSJEV/Clef或建立任意语言理解。
- 当前主干是Geology监督CNN，不是NASA/MG-MAE预训练。Age由Geology标签派生，复用该encoder是相关任务监督迁移，所有对照获得同一信息。
- 学习类ID和公共文本语义是不同控制；语义模型不加绕过描述的可学习class-ID embedding/class-specific bias。没有真实定义的tier不替完整语义回答科学问题。
- 首版候选可独立打分，子集与自身full-softmax mask等价是预期性质，不为破坏恒等式而强加复杂模块。
- 低GPU占用原因尚未通过profiling确认。可用少量Train数据做独立三路径性能探针，但不得作为设计前置、不得改封存的六组预算。

## 资源与既有资产保护

**VPN下载继续暂停，不自动续传。** 已保留33,964,443,553字节、尚缺40,484,675,794字节；原暂停要求覆盖更早的大文件授权。不下载新mmap、权重、数据snapshot或图例；需要额外小型来源时列清单和准确缺口，不让它阻塞全部本地开发。

本轮可直接进行已有资产上的可逆本地实现、缓存与上述限定的小头训练；不购买云、不启动四卡、不替换ROCm/torch或其他用户环境。特征缓存、原始pixel与checkpoint留本地，不入Git。模板/代码/来源元数据与脱敏结果可归档。

P2.2负结果与P3.0/P3.1证据保留；旧LROC Gate3 NOT_FROZEN/Gate4 HOLD；公开Test scoring HOLD。不能借新路线批准EDR/CDR变更、重写heldout、用Test选择模型、把n=3样本std写成置信区间。

## 验证与提交

```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts src tests
```

沿用已有可工作的私有runtime/ROCm路径，不安装共享包。新增关键测试覆盖冻结/BN、空间token shape、同ID特征、双任务padding/mask、候选排列/子集、label隔离、checkpoint恢复与实际cache有限值。实际成功后记录命令、源码/data/cache/text hash、PASS/FAIL/BLOCKED并push main，不force-push、不覆盖并发工作。尚未实现或运行的项目不能写成PASS。
