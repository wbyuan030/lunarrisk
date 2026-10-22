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

## 3. 结论（明确反对过度解读）

- **H1 不支持**：`diagonal` 在属性 MAE 上全面劣于点模型（`g10` 0.179 vs 0.071）。
- **H2 只有微弱信号**：`joint` 在 `g10>0.05` 的 Brier/NLL 上优于 `diagonal`（0.0743 vs 0.0918；0.2559 vs 0.3052），
  且优于 point 的 Brier（0.0743 vs 0.0778），但 seed 间波动（diagonal 0.074–0.106）与被比较的差值同量级，
  **不构成显著性主张**。
- **relief 目标不稳**：单 seed 异常值使 MAE 从 9 m 到 116 m，任何基于该目标的排序都不可靠。
- **相关性方向尚未解释**：没有做"同一 joint checkpoint 把 U 置零"的消融，因此无法判断 P3 的优势来自相关性建模
  还是仅仅来自 head 多出的参数与不同的优化轨迹。这是 H2 的必要条件，**尚未完成**。

## 4. 与计划要求的差距（必须补齐才能谈结论）

1. **未做校准**：`cal` 完全未使用。当前 Brier/NLL 是未校准的，对 `diagonal`/`joint` 不公平；下一步应在 `cal`
   上拟合温度/方差缩放，再在 `test` 报告。
2. **未做 U 置零消融**：同一 joint checkpoint 的 `U=0` 版本必须与 diagonal 对比，且 marginal variance 保持。
3. **未做同组配对 Bootstrap 的变体差 CI**：目前只有 seed 区间与逐 seed 配对差值，组级 CI 尚未接到差异统计上。
4. **未接入 `coarse_ridge` 与 direct-event head 基线**：P1 只有点回归，缺少原有参照。
5. **规则集只用了 5 条固定规则**，未做阈值网格扫描与复合 AST 的结构泛化（A2 已支持，实验未跑）。
6. 训练与评估都在**开发数据**上；这不构成任何独立 OOD 证据。

## 5. 止损判断（按计划要求先行分析归因）

按计划："若后验不优于点模型，先分析归因并启动止损 Gate，不能强行上 LFM/One-Pass"。

- 当前确实**没有**证据支持后验优于点模型；
- 归因（待验证的假设）：4 维目标中的 `relief` 噪声大、`g10/g15` 高度相关且受限于 [0,1]，内部链空间的高斯假设
  可能在 `g10` 接近边界时拟合不良（训练 NLL 约 0.086 vs 点模型残差 sigma 下的 0.088，差别很小，说明模型并未
  真正利用协方差结构）；
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
