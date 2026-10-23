# P2.2 P22-01/P22-02 — 统一样本口径下的公平比较矩阵（2026-10-23）

**状态：已完成（development_test 探索性）。**
所有模型在**同一批 sample_id、同一 train/val/cal/development_test、同一规则 evaluator**
下重新训练与评估；逐样本概率落盘；统计区间为 **25 个开发组** 的 group-paired bootstrap。

> `development_test` 是已经反复查看的 110 Tile / 25 Group 开发集，**不是外部 OOD**。
> Gate 3 仍为 NOT_FROZEN，Gate 4 仍为 HOLD；本轮未接触外部候选场景。

## 1. 样本口径（P22-01 第 1 项）

`labels_hydrated.csv` 有 760 行。P1.8 的 CNN 基线要求 `nac_valid_fraction >= 0.99`，
本轮把 **所有模型**限制到这个交集，避免“不同模型用不同 Tile”的历史问题。

| 项 | 值 |
|---|---|
| 总行数 | 760 |
| 使用行数 | **749**（train 399 / val 120 / cal 120 / test 110） |
| 排除 | 11 行（train 1、test 10），全部因 NAC 有效比例 < 0.99 |
| test 组数 | 25 |
| sample_id 排序后 SHA256 | `5c3d696547972cd7646d68eed0a2099d1751961ff7bed9b51be13bae0b3bc22b` |
| 过滤后 labels SHA256 | `418561718334355843bc3609dad64a0964ca1b5565d2e8ea9ac4419a5ae05596` |

排除行清单在 `outputs/p22/compare.json` 的 `manifest.inventory.excluded`，
含 `sample_id / split / group_id / nac_valid_fraction`。
5 个模型 × 3 seed 的逐样本 CSV 均为 **749 行、ID 集合完全相同**
（并集 = 交集 = 749）。

> 历史报告里的 “110 Tile” 与本轮的 test 行数一致；此前一些 120 Tile 的数字属于未过滤口径，
> 不能与本表混排。

## 2. 模型与概率口径（P22-01 第 2–3 项）

| 模型 | 定义 | 概率来源 | 参数量 |
|---|---|---|---|
| `coarse_ridge` | 60 m 坡度 7 维特征 + Ridge(α=1) 回归 **4 个物理属性** | Train 残差 std + Cal 闭式各向同性 T，物理空间高斯 | 7×4+4 |
| `direct_event` | P1.8 `TinyCNN`，2 个 pre-registered 事件 logit（BCE） | sigmoid 原样保留，不可迁移的强参考 | 98,464 |
| `point` | 共享 encoder + 4 输出 | Train 残差 std × Cal-T | 98,464 |
| `diagonal` | 共享 encoder + 8 输出（μ, logσ） | Cal-T 高斯 | 99,008 |
| `joint` | 共享 encoder + 16 输出（μ, logσ, rank-2 U） | Cal-T 高斯 | 100,096 |

- `coarse_ridge` 的 P1.8 版本只回归两个 fraction；本轮扩展到 4 属性是为了让含
  roughness/relief 的 AND/OR 规则也能被同一基线覆盖，这是**更强**的参照。
- 所有温度只在 **cal** 上拟合，test 从不参与阈值或温度选择。
- `joint` 额外报告三个同 checkpoint 对照：
  `u0_shared_T`（共享完整 joint 的 Cal-T）、`u0_refit_T`（在 U=0 模型上重新拟合 T）、
  `matched_marginals`（保留 `diag(Σ_joint)`，只去 off-diagonal）。
- 规则统一为 `src/lunarrisk/rules/ast.py`，GT 与概率使用同一个 evaluator。

## 3. Checkpoint（P22-01 第 4 项）

`outputs/p22/ckpt/{variant}_seed{seed}.pt` 保存完整推理状态：`state_dict`、
trained-scaler 的 mean/std 与列名、`point_sigma`、所有 Cal-T、target eps、seed、
`labels_sha256`、`cache_sha256`、`git_sha`、`weights_sha256`。

round-trip 实测：新进程重载 joint seed42，5 条规则概率与导出 CSV 的
`max|diff| = 0.00e+00`。权重 SHA 例：point `a8d21d1c…`、diagonal `dd6c2b4b…`、
joint `4e1f2b6f…`（git `1d104c1e`）。

## 4. 运行与资源

```bash
docker exec -e LUNARRISK_GIT_SHA=1d104c1e56488436ef6e177c10a8d049b15f056b rocm-lab \
  bash -lc 'cd /data/lunarrisk && python3 scripts/compare_p22.py \
    --seeds 42,43,44 --epochs 40 --draws 4096 --bootstrap-draws 2000 --outdir outputs/p22'
```

- 设备：AMD Radeon RX 7900 XT（单卡），torch 2.5.1+rocm6.2；
- 用时：**1 分 35 秒**（含 15 次训练、逐样本采样、2000 次 group bootstrap、
  256/1024/4096 抽样的 MC 诊断）；
- 数据集：既有 760 Tile 与本地缓存，未下载、未扩容。

## 5. 结果：每规则 Brier / NLL（test，3 seed 均值 ± SEM）

| 规则 | coarse_ridge | direct_event | point | diagonal | joint |
|---|---|---|---|---|---|
| `g10>0.05` Brier | 0.1020 ± .0002 | 0.0796 ± .0120 | **0.0747 ± .0053** | 0.0896 ± .0103 | 0.0858 ± .0058 |
| `g10>0.05` NLL | 0.3006 ± .0007 | 0.2447 ± .0264 | **0.2452 ± .0174** | 0.2993 ± .0260 | 0.2910 ± .0162 |
| `g15>0.02` Brier | 0.1425 ± .0005 | **0.0880 ± .0025** | 0.0925 ± .0036 | 0.1149 ± .0174 | 0.1202 ± .0125 |
| `g15>0.02` NLL | 0.4120 ± .0010 | **0.2843 ± .0115** | 0.3036 ± .0082 | 0.3759 ± .0449 | 0.3982 ± .0290 |
| `g10>0.10 OR g15>0.03` Brier | 0.1176 ± .0002 | — | **0.0781 ± .0051** | 0.0963 ± .0088 | 0.0972 ± .0065 |
| `g10>0.10 AND rough>0.05` Brier | 0.0579 ± .0001 | — | 0.0858 ± .0143 | **0.0559 ± .0043** | 0.0774 ± .0170 |
| `NOT(both safe)` Brier | 0.1243 ± .0005 | — | 0.1331 ± .0183 | **0.1123 ± .0084** | 0.1196 ± .0146 |

危险基率（test）：`g10>0.05` 0.691，`g15>0.02` 0.482（25 组，110 Tile）。
`direct_event` 只输出两个 pre-registered 事件，不能覆盖其他规则（— = 不适用）。

**读法**：
- 点模型在 `g10`、`g15`、OR 规则上与最强的 direct-event 头处同一水平，明显优于 Ridge；
- 含 roughness 的 AND 规则与 NOT 规则上，`diagonal` 最好，点模型最差
  （说明点模型对 roughness/relief 的残差高斯标度不合适，而不是均值错误）；
- `joint` 在任何规则上都没有稳定超过 `point`。

## 6. 组级配对 bootstrap（`g10>0.05`，ΔBrier 与 ΔNLL）

负值 = 前一个模型更好。区间为 25 个开发组重采样的 2.5/97.5 百分位，2000 次。

| seed | 比较 | ΔBrier [95% CI] | ΔNLL [95% CI] |
|---|---|---|---|
| 42 | joint − point | −0.0048 [−0.0199, +0.0115] | +0.0024 [−0.0353, +0.0426] |
| 43 | joint − point | **+0.0191 [+0.0051, +0.0318]** | **+0.0699 [+0.0332, +0.1054]** |
| 44 | joint − point | **+0.0189 [+0.0078, +0.0305]** | **+0.0652 [+0.0367, +0.0949]** |
| 42 | joint − diagonal | **−0.0254 [−0.0414, −0.0044]** | **−0.0524 [−0.0901, −0.0035]** |
| 43 | joint − diagonal | **+0.0277 [+0.0084, +0.0435]** | **+0.0762 [+0.0340, +0.1110]** |
| 44 | joint − diagonal | **−0.0137 [−0.0217, −0.0057]** | **−0.0486 [−0.0727, −0.0247]** |
| 42 | point − ridge | −0.0200 [−0.0439, +0.0072] | −0.0300 [−0.0879, +0.0345] |
| 43 | point − ridge | **−0.0247 [−0.0392, −0.0096]** | **−0.0484 [−0.0864, −0.0102]** |
| 44 | point − ridge | **−0.0372 [−0.0534, −0.0208]** | **−0.0880 [−0.1291, −0.0467]** |
| 42 | point − direct_event | +0.0018 [−0.0417, +0.0392] | +0.0300 [−0.0748, +0.1192] |
| 43 | point − direct_event | −0.0223 [−0.0534, +0.0108] | −0.0389 [−0.1184, +0.0443] |
| 44 | point − direct_event | +0.0057 [−0.0172, +0.0262] | +0.0104 [−0.0566, +0.0677] |

跨规则显著性汇总（Brier，CI 不跨 0 记为 SIG）：

| 规则 | joint−point (42/43/44) | joint−diagonal (42/43/44) | point−ridge (42/43/44) |
|---|---|---|---|
| `g10>0.05` | ns / SIG+ / SIG+ | SIG− / SIG+ / SIG− | ns / SIG− / SIG− |
| `g15>0.02` | ns / SIG+ / SIG+ | SIG− / SIG+ / SIG− | SIG− / SIG− / SIG− |
| OR | ns / SIG+ / SIG+ | ns / SIG+ / SIG− | ns / SIG− / SIG− |
| AND(rough) | ns / ns / ns | ns / SIG+ / ns | ns / ns / ns |
| NOT | SIG− / ns / ns | ns / ns / ns | SIG+ / ns / ns |

## 7. 同 checkpoint 消融（`g15>0.02`，ΔBrier vs 完整 joint）

| seed | U=0 shared_T | U=0 refit_T | matched-marginals |
|---|---|---|---|
| 42 | **−0.0205 [−0.0325, −0.0046]** | **−0.0084 [−0.0127, −0.0044]** | −0.0066 [−0.0124, +0.0010] |
| 43 | −0.0087 [−0.0204, +0.0045] | −0.0033 [−0.0114, +0.0057] | **−0.0117 [−0.0177, −0.0046]** |
| 44 | −0.0066 [−0.0169, +0.0055] | **−0.0115 [−0.0158, −0.0063]** | **−0.0096 [−0.0150, −0.0028]** |

- `g10>0.05` 在 matched-marginals 下 ΔBrier **恒等于 0**（结构性结果：Cholesky 第一行只用
  `noise[0]`，保留边缘方差后 g10 的抽样完全相同）。这说明 matched-marginals 只能用来解释
  跨属性事件，不能用来解释单属性 g10 事件。
- matched-marginals **保留全部边缘方差**，仍让 `g15` Brier 改善 0.007–0.012，且 2/3 seed 的
  CI 不跨 0 → **学习到的 off-diagonal 相关性对 g15 事件概率有害** 是一个稳定的信号；
  但 `joint`（含相关性）本来就比 `point` 差约 0.028，去掉相关性也只能缩小差距，
  不能反超点模型。
- `U=0` 的同时缩小边缘方差，其“改善”幅度随校准口径（shared vs refit）明显变化，
  不能作为“相关性有害”的单独证据；这与 P2.1 勘误的要求一致。

## 8. 固定放行覆盖率下的 UAR（`g10>0.05`，3 seed 均值 / 接受行数）

| 覆盖率 | coarse_ridge | direct_event | point | diagonal | joint |
|---|---|---|---|---|---|
| oracle 10% | 0.030 / 11 | 0.000 / 11 | 0.000 / 11 | 0.146 / 11 | 0.030 / 11 |
| oracle 20% | 0.136 / 22 | 0.045 / 22 | 0.076 / 22 | 0.106 / 22 | 0.106 / 22 |
| oracle 40% | 0.250 / 44 | 0.227 / 44 | 0.242 / 44 | 0.258 / 44 | 0.242 / 44 |
| deployable 10% | 0.000 / 10 | 0.000 / 7 | 0.000 / 8 | 0.141 / 6 | 0.000 / 8 |
| deployable 20% | 0.147 / 18 | 0.024 / 16 | 0.053 / 19 | 0.094 / 16 | 0.087 / 19 |
| deployable 40% | 0.202 / 41 | 0.124 / 35 | 0.168 / 37 | 0.153 / 37 | 0.178 / 39 |

- `oracle` 只按模型自己的概率排序，接受数固定；
- `deployable` 的阈值**只在 cal 上**按目标接受率选，绝不看 test 危险标签；
- 由于 cal/test 分布漂移（`P(g10>0.05)`：train 0.568、val 0.350、cal 0.708、test 0.691），
  deployable 的接受数经常低于名义覆盖率（例如 direct-event 40% 只接受 35 行），
  因此 UAR 必须与接受数一起读，不能只比 UAR。
- 40% deployable 下 direct-event UAR 最低（0.124），但它接受的危险样本数同样最少；
  五种模型在 oracle 40% 下的差异很小（0.227–0.258）。

## 9. Monte-Carlo 误差（普通 MC，不是 QMC）

同一 joint seed42 checkpoint、同一 seed：

| 规则 | max abs diff 1024 vs 256 | max abs diff 4096 vs 256 |
|---|---|---|
| `g10>0.05` | 0.0693 | 0.0842 |
| `g15>0.02` | 0.0605 | 0.0640 |
| OR | 0.0625 | 0.0586 |
| AND(rough) | 0.0430 | 0.0503 |
| NOT | 0.0830 | 0.0947 |

256 抽样时单样本概率误差可达 ~0.03，4096 时降到 ~0.008。正式比较因此固定 **4096 draws**。

**闭式校验**：单属性规则 `P(sigmoid(z1)>t)` 有精确高斯 CDF 解。4096 抽样与闭式解的
最大绝对差：point 0.0151、diagonal 0.0150、joint 0.0189 —— 与 4096 抽样的 MC 噪声
（单点 std ≈ 0.007、110 行最大值 ≈ 3–4σ）一致，说明基于抽样的 AST 积分在单属性规则上可信。
复合规则没有闭式对照，只能依赖 MC 误差上表。

## 10. 可复现性发现（重要）

- 只设 `torch.manual_seed` / `np.random.seed` **不足**：同命令两次运行训练损失打印相同但权重
  逐位不同（MIOpen 归约顺序）。
- 仅在后验训练里开 `torch.use_deterministic_algorithms(True)` 也**不足**：同一进程里先训练的
  `direct_event` 基线使用非确定性 kernel，会改变随后 MIOpen 的 kernel 选择，导致矩阵两次运行
  的 joint/diagonal 结果不同（实测 diagonal seed42 Brier 0.0866 vs 0.1154）。
- 修复：在 **任何模型接触 GPU 之前** 启用确定性算法（`compare_p22.py` 已改）。修复后两次完整
  单 seed 矩阵运行的 point/diagonal/joint checkpoint `weights_sha256` 完全相同
  （`a8d21d1c… / dd6c2b4b… / 4e1f2b6f…`）。
- 本报告所有数字来自修复后的最终一次运行（代码 SHA `1d104c1e`）。

## 11. 真实失败与限制

1. `direct_event` 的 seed 间波动最大（Brier 0.0586–0.1001），且 early-stop epoch 只有 7/15/9；
   它不是稳定的强参考，只是“不可迁移的对照”。
2. `coarse_ridge` 在含 roughness 的 AND 规则上反而最好（0.0579），说明 60 m 粗地形本身
   携带了该事件的强信息；点模型的 AND 事件概率（0.0858）明显更差。
3. 所有数字均为同一开发集，规则与模型都是在本轮之前或本轮预注册的，但没有新的
   external heldout；不能声称 OOD 泛化。
4. 样本过滤去掉了 10 个 test Tile（NAC 有效比例低），这不是按模型表现挑选，
   但确实改变了历史 test 集合。
5. 物理密度 NLL 未作为本表指标（边界点质量问题见 P22-00 报告）。

## 12. 产物

| 路径 | 内容 |
|---|---|
| `outputs/p22/compare.json` | manifest（inventory、规则、SHA）与全部聚合统计 |
| `outputs/p22/pred/*.csv` | 每模型每 seed 的逐样本 truth/mean/概率（含 U 消融分支） |
| `outputs/p22/ckpt/*.pt` | 可加载 checkpoint（含 provenance） |
| `outputs/p22/report/*.json` | 每次训练的历史、T、参数量、消融口径 |
| `outputs/p22/labels_p22.csv` | 749 行统一样本清单 |
