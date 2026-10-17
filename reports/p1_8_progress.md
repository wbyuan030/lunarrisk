# P1.8 进度与证据（2026-10-17）

代码状态：`a53937e`（本轮提交）；起点 `61d43a2`。
环境：单卡 RX 7900 XT，PyTorch 2.5.1+rocm6.2，HIP 6.2.41133，Python 3.10.12。
测试：`python3 -m unittest discover -s tests -v` → **148/148 PASS**（本轮新增 29 项：hydration 20、freeze 9）。

| 任务 | 状态 | 证据 |
|---|---|---|
| P18-01 resume 确定性 | **PASS** | 修复死分支；exit 0/2 语义；计数器；一致性拒绝 |
| P18-02 原子追加与 schema 演进 | **PASS** | 原子写入 + 列并集 merge + 400→401 真实 dry-run |
| P18-03 统计 fail-fast 与 NLL 口径 | **PASS** | 真值/事件/分组/协议/概率校验 + 显式 NLL(primary/raw) |
| P18-04 固定参数消融 | **未开始**（设计已定，见下） | — |
| P18-05 冻结 heldout_v2 | **BLOCKED（严格）/ 已冻结产品级候选** | 272/272 groups 已被使用 |
| P18-06 扩容到 3k | **未开始**（依赖 04/05） | — |
| P18-07 V2 一次性评估 | **未开始**（依赖 04/05/06） | — |

## P18-01 — Hydration no-op resume

发现（属实）：`if not records: raise SystemExit` 位于正常 no-op 分支之前，全量下载完再 `--resume` 必然非零退出。

修复：

- `run_hydration()` 成为可注入的驱动（`download_fn`/`hydrate_fn`），返回 `(exit_code, stats)`。
- 全部已存在 → **exit 0**，不重写 labels、不伪造下载计数，仍生成 summary。
- `todo` 为空 ≠ 全部下载失败：后者 **exit 2** 且逐项输出 `{sample_id, error}`。
- 部分失败 → 写入成功的记录，**exit 2** 并报告失败列表。
- 计数：`selected_count / existing_count / todo_count / download_attempted / new_records / labels_written`。
- 显式一致性检查（拒绝而非覆盖）：已存在行的 `split / official_split / group_id / NAC_EDR_ID / LTM_CODE / block_5km`
  与 manifest 不符 → `HydrationError`；`sample_id` 重复 → 拒绝；本地源文件缺失 → 拒绝；
  已存在行不在当前 selection → 拒绝（旧 cohort 必须是新 cohort 的子集）。
- 回归测试：`tests/test_hydrate_tiles.py`（20 项，纯离线）。

## P18-02 — 原子化标签合并

- 旧实现 `mode="a"` + 旧表头 → 新列时列数/表头不一致。现改为
  **读既有 → 按 `sample_id` 合并 → 列并集（保留原顺序，新列稳定追加）→ 校验 → 临时文件 + fsync → `os.replace`**。
- 旧行在新列上填 NaN；同 `sample_id` 只允许一次；无 `sample_id` 的记录被拒绝。
- `--enrich` 同样原子写回，且要求 merge 为 `one_to_one`（manifest / labels 两侧 `sample_id` 唯一），
  行集或顺序变化即拒绝。
- **真实数据 dry-run（备份副本 `/tmp/labels_backup.csv`，不触网）**：

| 检查 | 结果 |
|---|---|
| `select_tiles(400)` vs `select_tiles(401)` | 旧选集合为新集合子集；新增 1 个 `M1099059565_r35940_c6582` |
| cal / val / test 是否被重新分配 | **完全一致**（顺序也一致） |
| 现有 760 行 ⊆ 新 selection | 是 |
| 网络禁用（stub 抛错） | exit 2，labels 文件**逐字节未变** |
| stub 成功（1 条合成分录） | exit 0，761 行，原 760 行的**所有列值完全一致**（0 处数值差异） |
| 列顺序 | 原列顺序保留，新列追加在末尾 |

注：dry-run 因 stub 记录缺少部分列，导致这些列 dtype 由 int64/bool 升为 float64（值不变）；
真实的 `hydrate_one` 返回完整列，不会发生该升位。旧行无任何 NaN 注入。

## P18-03 — 统计一致性、NLL 口径与 fail-fast

`scripts/aggregate_seeds.py`：

- `assert_truth_consistency`：跨 seed 比较 `sample_id / group_id / LTM_CODE / y_g10 / y_g15 / event_*`，任一漂移即退出。
- `assert_probability_quality`：缺列、NaN/Inf、越界、以及"声称 calibrated 却无 calibrated metrics"一律拒绝。
- `assert_seed_protocol_consistency`：seed 间 `input_size / augment / epochs / patience / batch / lr / min_nac_valid /
  channels / rules / calibration / ablation` 必须一致；`protocol.seed` 必须等于运行 seed；`provenance`
  （manifest/labels/cache 的 path+bytes+sha256）必须一致；缺失字段记入 `missing` 并标记 `UNKNOWN`，不猜测。
- NLL 拆分为 **NLL(primary)**（由导出的 primary 概率重算）与 **NLL(raw)**（trainer 的 raw 值），
  并在 `field_sources` 中逐字段注明来源；同时用导出概率重算 Brier 与 `metrics.json` 交叉校验（>1e-6 即拒绝）。

`scripts/compare_ablations.py`：

- 跨 config×seed 比较 `sample_id / group_id / LTM_CODE / y_g10 / y_g15 / event_*`。
- `assert_config_protocol_consistency`：只允许 `ablation` 的几何/光度开关不同；其余超参、规则、
  输入尺寸、增强策略必须一致；`provenance` 必须一致；并校验记录的
  `variant_channels / variant_scalars` 与 `ablation_features()` 的结果自洽（防止 protocol 撒谎）。
- NLL 同样拆分为 primary/raw 并交叉校验 Brier。
- 明确写入 caveats：25 组 cluster bootstrap × 3 固定 seed，**不构成**新地区或随机初始化的覆盖保证；
  110 Tile 为 `development_test`，多配置比较属探索性。

**真实数据复检**（P1.7 的 12 次运行，输出到 `outputs/p18_recheck/`，**不覆盖** P1.7 原结果）：

- 4 个 config 汇总与跨配置对比全部通过；`probability_quality.checked = 6600`。
- P1.7 运行缺少 `batch / lr / min_nac_valid / provenance` → 记为 `missing: [...]` / `provenance: UNKNOWN`，
  未被伪造；P18-04 起的新运行会写入这些字段。

## P18-04 — 固定参数量消融（未开始，设计如下）

计划：`--ablation-mode fixed-mask`，所有变体固定 3 通道输入与**完整**标量 head，
缺失模态以训练集标准化后的 0（中性值）掩码，而不是删通道/改 `Linear` 形状；记录每个变体的
可用通道、掩码字段、参数总数，并在比较时断言一致。另设合法增强对照组
（`--no-augment` vs `--no-geometry --augment`）。该改动未经实测前不得替代 P1.7 的旧消融结论。

## P18-05 — heldout_v2

见 `reports/p1_8_heldout_protocol.md`：严格组级独立 **BLOCKED**（272/272 组已被 760 Tile 覆盖），
另冻结产品级候选 64 Tile 并明确标注非组级独立。

## 止损与约束遵守

- 未使用 4×4090、付费云、LFM/Qwen/联合后验。
- 未修改原始 NAC/DTM 数据；未覆盖 `outputs/labels_hydrated.csv`（dry-run 在副本上进行）。
- 未用旧 `development_test` 挑选任何新模型；未读取 V2 的任何模型指标。
