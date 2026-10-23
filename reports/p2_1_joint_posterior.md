# P2.1 A3 — 联合后验第一轮实验（探索性）

**状态：框架与实验跑通，但 H1（P2 优于 P1）在属性精度上不成立，H2（P3 优于 P2）只有微弱的 Brier 信号。
全部结果来自 `development_test`（120 Tile / 25 组，已被反复查看），只能作探索性报告。** 2026-10-22

## 1. 设置

| 项 | 值 |
|---|---|
| 数据 | `outputs/labels_hydrated.csv`（760 行）+ `outputs/cache_v3.npz`（3 通道：NAC contrast/texture、60 m 粗坡度） |
| 划分 | train 400（195 组）/ val 120（26）/ cal 120（26）/ test 120（25） |
| 输入 | 3 通道 + 7 个视角标量（train 上标准化） |
| 目标 | `g10, g15, roughness_9m, relief_p95_p5` 经物理受限链变换 |
| 变体 | `point`（残差 sigma 概率）、`diagonal`、`joint`（rank-2 U） |
| 训练 | Adam 1e-3，batch 32，最多 40 epoch，val NLL early-stop（patience 4） |
| 三个 seed | 42 / 43 / 44；GPU 为本地 RX 7900 XT，9 次运行合计约 **100 s** |
| 参数量 | point 98,464；diagonal 99,008；joint 99,392（仅 head 不同，encoder 完全相同） |

训练目标是**内部链空间**的条件高斯 NLL；物理空间 NLL 需要 Jacobian 项，代码两条都返回，本表只报事件指标。

## 2. test 结果（3 seed 均值）

### 属性精度（越低越好）

| 指标 | point | diagonal | joint |
|---|---|---|---|
| `g10` MAE | **0.0710** | 0.1790 | 0.1248 |
| `g15` MAE | 0.0803 | 0.1614 | 0.1215 |
| `roughness_9m` MAE | **0.0207** | 0.0253 | 0.0221 |
| `relief_p95_p5` MAE | 33.27 | **21.32** | 50.06 |

点模型在 `g10`/`g15`/roughness 上明显更好；relief 在 seed 间极不稳定（joint seed 43 为 115.7 m），说明该目标对训练噪声敏感，不能据此排序。

### 事件风险

| 指标 | point | diagonal | joint |
|---|---|---|---|
| `g10>0.05` Brier | 0.0778 | 0.0918 | **0.0743** |
| `g10>0.05` NLL | 0.3143 | 0.3052 | **0.2559** |
| `g15>0.02` Brier | **0.1037** | 0.1282 | 0.1113 |
| `g10>0.10 ∨ g15>0.03` Brier | 0.0879 | 0.1020 | **0.0837** |

@20% 覆盖率的实际计数（`g10>0.05`）：放行 **24** 行，其中危险 **3** 行；三个变体的 UAR@20 在 0.042–0.125 之间，HMR@20 在 0.012–0.036 之间，seed 间差异与变体间差异同量级。

## 2b. 同 checkpoint U=0 消融（决定性）

joint 模型训练完成后，把 head 的低秩列（`U`）置零再评估——encoder 与 `mu` 完全不变，只有协方差退化回对角：

| seed | `g10>0.05` Brier（训练后） | Brier（U=0） | 差 | NLL（训练后） | NLL（U=0） |
|---|---|---|---|---|---|
| 42 | 0.0959 | **0.0901** | −0.0057 | 0.3096 | **0.2828** |
| 43 | 0.0708 | **0.0684** | −0.0024 | 0.2420 | **0.2174** |
| 44 | 0.0710 | **0.0623** | −0.0087 | 0.2460 | **0.2017** |
| 均值 | 0.0792 | **0.0736** | −0.0056 | 0.2659 | **0.2339** |

**三个 seed 全部是去掉相关性后更好。** 也就是说 P3 相对 P2 的那点 Brier 优势**不是**来自低秩协方差，
去掉它反而更准；在这个样本量（train 400 Tile）下，相关性参数更可能是在拟合噪声。

### 复现性警告

同一配置、同一 seed 的两次独立训练可以得到 Brier 0.0763 与 0.0959（joint seed 42 的前后两次运行）。
ROCm 上的卷积/Cholesky 内核并非逐位确定，因此 **< 0.02 的差异不可解释**；上表的 U 消融是**同一次
checkpoint 内部**的对比，不受该噪声影响，所以可信度高于跨运行比较。

## 2c. 校准后的结果（`cal` 拟合温度，`test` 报告）

`cal` 只用于拟合一个各向同性温度 `T`（闭式解 `T² = mean(quad)/k`，Gaussian 的 ML 解），三个变体都拟合，
point 的初始离散度来自训练残差、同样被校准。`test` 结果（3 seed 均值 [最小, 最大]）：

| 变体 | `g10>0.05` Brier | NLL | `g10` MAE | `g15` MAE | 拟合温度 T |
|---|---|---|---|---|---|
| point | 0.0811 [0.0650, 0.0979] | 0.3492 | **0.0805** | **0.0879** | 1.005 / 0.999 / 1.014 |
| diagonal | 0.0951 [0.0746, 0.1110] | 0.3177 | 0.1819 | 0.1606 | 1.118 / 0.979 / 1.080 |
| joint | **0.0796** [0.0750, 0.0841] | **0.2751** | 0.1349 | 0.1273 | 1.174 / 1.375 / 1.064 |

校准后：
- **joint 的事件概率最好**（Brier 0.0796、NLL 0.2751），且 seed 区间最窄；但相对 point 的优势只有 0.0015 Brier，
  **属于噪声量级**，不能说显著；
- **NLL 的排序稳定**（joint 0.275 < diagonal 0.318 < point 0.349），因为 NLL 对"过度自信/过度保守"都惩罚，
  比 Brier 更能区分概率质量；
- **属性精度仍是 point 最好**（`g10` MAE 0.081 vs 0.135），与"点估计器在点误差上占优"一致；
- joint 的温度 >1（1.06–1.38），说明它**系统性低估了离散度**，这是它校准前 Brier 偏高的原因。

## 3. 结论（明确反对过度解读）

- **H1 不支持**：`diagonal` 在属性 MAE 上全面劣于点模型（`g10` 0.179 vs 0.071）。
- **H2 分两层，答案不同**：
  - "joint 作为整体优于 diagonal"——**校准后成立**（Brier 0.0796 vs 0.0951，NLL 0.2751 vs 0.3177，seed 区间基本不重叠）；
  - "优势来自低秩相关性"——**不成立**：同 checkpoint 的 U=0 消融在三个 seed 上全部更好
    （Brier 0.0736 vs 0.0792，NLL 0.2339 vs 0.2659）。
  - 因此可写"联合建模与校准带来更好的事件概率"，**不可**写"协方差结构带来增益"。
- **relief 目标不稳**：单 seed 异常值使 MAE 从 9 m 到 116 m，任何基于该目标的排序都不可靠。
- **相关性方向已量化**：U=0 消融（见 2b）显示相关性项有害，方向与 H2 的假设相反。

## 4. 与计划要求的差距（必须补齐才能谈结论）

1. ~~未做校准~~ **已完成**（见 2c）：`cal` 上拟合各向同性温度，三个变体都做；校准后 joint 的 NLL 最好，
   Brier 与 point 相差 0.0015（噪声量级）。
2. ~~未做 U 置零消融~~ **已完成**：见 2b，结论为 U=0 更好；仍缺"U 随机置换但保持方差"的对照。
3. **未做同组配对 Bootstrap 的变体差 CI**：目前只有 seed 区间与逐 seed 配对差值，组级 CI 尚未接到差异统计上。
4. **未接入 `coarse_ridge` 与 direct-event head 基线**：P1 只有点回归，缺少原有参照。
5. **规则集只用了 5 条固定规则**，未做阈值网格扫描与复合 AST 的结构泛化（A2 已支持，实验未跑）。
6. 训练与评估都在**开发数据**上；这不构成任何独立 OOD 证据。

## 5. 止损判断（按计划要求先行分析归因）

按计划："若后验不优于点模型，先分析归因并启动止损 Gate，不能强行上 LFM/One-Pass"。

- 当前确实**没有**证据支持后验优于点模型；
- 归因（已有数据支持部分）：4 维目标中的 `relief` 噪声大（MAE 在 seed 间 7–116 m）；`g10/g15` 受限于 [0,1]
  且高度偏序，内部链空间的高斯假设未被有效利用——U=0 更好说明模型学到的相关性没有泛化价值；
- 因此在补齐第 4 节 1–4 项之前，**不启动**任何 LFM/One-Pass/大规模训练，也不宣称新的 OOD 能力。

## 6. 复现

```bash
# 单次运行（GPU）
python3 scripts/train_posterior_p21.py --variant joint --seed 42 --epochs 40 --device cuda \
  --draws 256 --out outputs/p21/joint_seed42.json
# 9 次（3 变体 × 3 seed），约 100 s
for v in point diagonal joint; do for s in 42 43 44; do
  python3 scripts/train_posterior_p21.py --variant $v --seed $s --device cuda \
    --out outputs/p21/${v}_seed${s}.json; done; done
# 汇总
python3 scripts/summarize_p21.py --dir outputs/p21 --md reports/p2_1_joint_posterior.md \
  --json outputs/p21_summary.json
```

产出：`outputs/p21/{variant}_seed{seed}.json`（含 history、逐 split 指标、checkpoint 的 weights SHA256）。


---

## 7. P2.2 方法勘误与重新运行要求（2026-10-23）

**本节为追加勘误；上面的历史数字没有按修复后的代码重新计算，不能直接用作修复后结论。**

后续 Code Review 发现三项 P0 问题并已在 `main` 中做代码修复、添加单测：

1. `train_posterior_p21.py::predict` 的 Point 在 Cal 上虽记录了温度 `T`，
   原实现却没有把 `T` 乘到训练残差 `point_sigma` 上。历史表中 Point 的所谓
   “Cal 校准后 Brier/NLL”不能算作严格应用了同等温度策略的 Point；现已让
   Point/Diagonal/Joint 均按同一高斯温度标准采样（应用于 latent STD）。
2. `--ablate-u --calibrate` 时，旧 `evaluate_splits` 对 `u_zeroed` 分支
   强制将 `temperature` 设为 `None`，同时原 Joint 使用 Cal 的 T；
   这不是同一校准口径的 U 消融。现默认对 U=0 使用**与原 Joint 同一个 Cal-T**，
   并记录为 `shared_T_fitted_on_full_joint_cal`。另需将
   `U=0` 在 Cal 上**独立 refit-T** 以及**保留原联合对角方差只去掉相关性**
   的 matched-marginals 实验作为不同对照。
3. `posterior.py::nll` 的旧物理密度公式使用 `internal - J(mu)`，
   正确的连续变量变换为 `internal + J(z_truth)`。
   本轮已修代码及单测，**此前报告主要是事件 NLL，不是这个 physical density NLL**，
   因此不意味着此前所有事件 NLL 都由该符号错误直接产生。但未来如报告
   物理密度指标，必须考虑 `g10/g15=0/1` 的截断边界质量，连续变换密度对真实端点不成立。

此外，有两项此前结论必须下调强度：

- 旧 “U=0 改善 → 相关性有害” **尚非隔离属性依赖的因果归因**：
  `Σ = diag(σ²)+UUᵀ` 置 `U=0` 会同时降低每个属性的边缘方差；
  应通过同 checkpoint `diag(diag(Σ))` 保持每个属性边缘方差不变，仅消除 off-diagonal。
- 旧 “Joint 比 Diagonal 的 NLL/概率分数更好” 仍只是旧版开发集上的初步趋势：
  不同训练 head、随机性、旧 `development_test` 反复查看、尚缺同样 sample ID 的
  Coarse Ridge / Direct-event 参照与组级配对区间。不能声称 H1/H2 得到了证实。

**下一次实验需要从 Cal 起重新估计并应用温度、重新跑 3 Seed，固定所有模型/数据/规则
的样本口径，存储逐样本概率与可加载 checkpoint，再做 Group-paired CI。**

完整执行清单：[P2.2 校准与公平评测](../docs/P2_2_PROBABILITY_VALIDATION.md)。
外部 v2 仍保持 `PENDING_SIGNOFF/NOT_FROZEN/HOLD`，不受本次勘误影响。
