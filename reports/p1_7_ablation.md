# P1.7 受控消融 — 实际运行结果

日期：2026-10-17（Asia/Shanghai）
代码状态：`e17efdf`（工作区干净时运行；P1.7 加固提交后的第一次真实重跑）
文档依据：`docs/P1_7_NEXT_STEPS.md` A/B/C 节
环境：单卡 `AMD Radeon RX 7900 XT` 19.98 GiB，PyTorch **2.5.1+rocm6.2**，HIP **6.2.41133**，Python 3.10.12
数据：`outputs/labels_hydrated.csv`（760 Tile）+ 缓存 `outputs/cache_v3.npz`（3 通道，本轮未重建）
测试集：110 Tile / 25 个地理组 / 10 个 LTM zone，危险 76/110（`g10>0.05` 先验 0.691）

## 1. 首跑验证（B 节）

| 检查 | 结果 |
|---|---|
| `python3 -m unittest discover -s tests -v` | **96/96 PASS**（含 P1.7 新增的并列分数、消融特征、非法增强、泄漏断言、真实事件头检查） |
| `scripts/train_baseline.py --help` | 显示 `--augment / --no-augment / --no-geometry / --no-nac-photometry` |
| `--augment`（未配 `--no-geometry`）| 按预期 `parser.error`：静态太阳方位标量不会随空间翻转同步旋转 |
| `--no-geometry --augment` | 可运行（1 epoch smoke 通过） |
| `metrics.json.protocol.ablation` | 四个配置各自正确记录 `use_geometry` / `use_nac_photometry` |
| split 泄漏 fail-fast | 通过（272 组、0 组跨 split，0 产品跨 split） |
| 事件头审计 | 12 次运行 × 每 seed 的 `cnn_*_raw_event_prob_*` 均**非完全相同**（真正的二分类头，不是回归 fraction） |

**并列感知 UAR 的效果**：`constant` 在 `g10>0.05` 的 UAR@20% 由 P1.6 的 0.818 变为 **0.691**（= 该规则先验）。
常数模型分数全并列，旧值来自 CSV 行序；新实现按期望危险数计算，因此 0.691 才是它的真实（无判别力）水平。

## 2. 运行命令（C 节，逐字执行）

```bash
for config in p17_full p17_no_geo p17_no_photo p17_no_both; do
  case "$config" in
    p17_full)     flags="" ;;
    p17_no_geo)   flags="--no-geometry" ;;
    p17_no_photo) flags="--no-nac-photometry" ;;
    p17_no_both)  flags="--no-geometry --no-nac-photometry" ;;
  esac
  for seed in 42 43 44; do
    python3 scripts/train_baseline.py --seed "$seed" --no-augment \
      --out "outputs/${config}_seed${seed}" $flags
  done
  python3 scripts/aggregate_seeds.py \
    --pattern "outputs/${config}_seed{seed}" --out "outputs/${config}_summary"
done
python3 scripts/compare_ablations.py      # 跨配置与差值之差（本轮新增）
```

每 seed 耗时 20–29 s，12 次训练 + 4 次聚合合计 **312 s**。产物：`outputs/p17_{full,no_geo,no_photo,no_both}{_seed42,_seed43,_seed44,_summary}`、
跨配置对比 `outputs/p17_ablation_summary/{summary.json,summary.md}`。

## 3. 跨配置结果（每 seed 均值，主规则 `g10>0.05`）

| config | model | MAE g10 | Brier (标定) | Brier (raw) | NLL | AUC | UAR@20% | UAR@40% |
|---|---|---|---|---|---|---|---|---|
| p17_full | coarse_ridge | **0.0504** | 0.0728 | 0.0702 | 0.2240 | 0.9659 | 0.136 | 0.250 |
| p17_full | cnn_coarse_grid | 0.0705 (0.0052) | 0.0804 | 0.0736 | 0.2419 | 0.9645 | 0.136 | 0.250 |
| p17_full | cnn_fusion | 0.0662 (0.0077) | 0.0673 | 0.0655 | 0.2091 | 0.9835 | 0.045 | 0.227 |
| p17_full | cnn_nac | 0.3148 (0.0625) | 0.1750 | 0.3259 | 1.0946 | 0.7868 | 0.333 | 0.462 |
| p17_no_geo | cnn_coarse_grid | 0.0623 (0.0002) | 0.0825 | 0.0720 | 0.2309 | 0.9627 | 0.121 | 0.250 |
| p17_no_geo | cnn_fusion | 0.0638 (0.0037) | 0.0666 | 0.0579 | 0.1926 | 0.9814 | 0.030 | 0.227 |
| p17_no_geo | cnn_nac | 0.3357 (0.0403) | 0.1743 | 0.3264 | 0.9762 | 0.7801 | 0.394 | 0.462 |
| p17_no_photo | cnn_coarse_grid | 0.0608 (0.0059) | 0.0838 | 0.0721 | 0.2270 | 0.9632 | 0.136 | 0.250 |
| p17_no_photo | cnn_fusion | 0.0655 (0.0082) | 0.0718 | 0.0599 | 0.1947 | 0.9763 | 0.076 | 0.242 |
| p17_no_photo | cnn_nac | 0.3292 (0.0280) | 0.1786 | 0.3408 | 1.0715 | 0.7661 | 0.439 | 0.462 |
| p17_no_both | cnn_coarse_grid | 0.0698 (0.0094) | 0.0821 | 0.0715 | 0.2263 | 0.9623 | 0.121 | 0.250 |
| p17_no_both | cnn_fusion | 0.0556 (0.0068) | 0.0695 | 0.0590 | 0.1883 | 0.9792 | 0.061 | 0.235 |
| p17_no_both | cnn_nac | 0.3254 (0.0044) | 0.1797 | 0.2986 | 0.8847 | 0.7543 | 0.470 | 0.470 |

（`constant` 在四个配置完全相同：MAE 0.3554、Brier 0.2284、UAR@20 0.691；`coarse_ridge` 也是确定性模型，不随消融变化。）

## 4. NAC 增量：`cnn_fusion` − `cnn_coarse_grid`（组级配对自助，1000 次，3 seed 平均）

| config | ΔMAE [95% CI] | 排除 0 | ΔBrier [95% CI] | 排除 0 |
|---|---|---|---|---|
| p17_full | −0.0043 [−0.0113, +0.0033] | 否 | −0.0131 [−0.0369, +0.0079] | 否 |
| p17_no_geo | +0.0014 [−0.0056, +0.0093] | 否 | −0.0158 [−0.0358, +0.0013] | 否 |
| p17_no_photo | +0.0047 [−0.0011, +0.0105] | 否 | −0.0120 [−0.0336, +0.0086] | 否 |
| p17_no_both | **−0.0141 [−0.0212, −0.0071]** | 是 | −0.0126 [−0.0335, +0.0084] | 否 |

差值之差（相对 `p17_full`）：

| config | Δ(ΔMAE) [95% CI] | 排除 0 | Δ(ΔBrier) [95% CI] | 排除 0 |
|---|---|---|---|---|
| p17_no_geo | +0.0058 [+0.0002, +0.0106] | 是 | −0.0027 [−0.0124, +0.0060] | 否 |
| p17_no_photo | +0.0090 [+0.0022, +0.0150] | 是 | +0.0011 [−0.0142, +0.0172] | 否 |
| p17_no_both | −0.0098 [−0.0174, −0.0032] | 是 | +0.0005 [−0.0122, +0.0141] | 否 |

## 5. 关键结论

### 5.1 P1.6 报告的"NAC 增量"不成立，它是数据增强的伪影（本轮自我纠正）

同数据、同 seed、同缓存，唯一差别是**是否开启空间增强**：

| 运行 | 增强 | ΔMAE (fusion−coarse_grid) | ΔBrier |
|---|---|---|---|
| P1.6 v3 | 开（flip/rot90，与固定太阳方位几何标量冲突） | −0.0081 [−0.0137, −0.0025] **排除 0** | −0.0223 [−0.0428, −0.0005] **排除 0** |
| P1.7 `p17_full` | 关（P1.7 修正后的合法协议） | −0.0043 [−0.0113, +0.0033] 不排除 0 | −0.0131 [−0.0369, +0.0079] 不排除 0 |

结论：**在修正后的协议下，没有可统计分辨的单目 NAC 增量。** P1.6 §P1.6.4 第 1 条应视为被本轮否证。

### 5.2 消融无单调模式，且与模型容量混杂

- 移除几何、或移除 NAC 光度通道，都会使融合模型相对粗坡度的优势**缩小**（Δ(ΔMAE) 分别 +0.0058、+0.0090，均排除 0）。
- 但两者**同时**移除反而让优势扩大到唯一显著的水平（−0.0141，排除 0）。
- 该非可加性说明：差异更可能来自参数量/过拟合差异（去掉通道会减少卷积输入通道，去掉标量会缩小 head），
  而不是"哪些物理信息更有用"。C 节消融**没有**控制参数数量，因此不足以支撑信息论式解读。

### 5.3 单模态 `cnn_nac` 在所有配置下都不可用

MAE 0.3148–0.3357（常数基线 0.3554）、原始 Brier 0.2986–0.3408（**差于常数的 0.2284**）、NLL 0.88–1.09
（过度自信），标定后才降到 0.174–0.180。四个配置之间没有实质差别。

### 5.4 风险排序方向一致但幅度小（探索性）

`cnn_fusion` 在四个配置的 UAR@20% 均为最低（0.030–0.076，`coarse_grid` 0.121–0.136、`coarse_ridge` 0.136、
`constant` 0.691），但 20% coverage 只放行 22 个 Tile（其中 1–3 个危险），差距不足以支撑部署结论。

## 6. 限制（必须随结论一起引用）

- **旧 110 个 test Tile 已被多轮查看**，本节所有数字只能称探索性；不得据此挑选"最终模型"并声称独立测试性能。
- 每配置仅 3 个 seed、25 个地理组；组级自助 CI 较宽（Brier CI 宽度约 ±0.02–0.04）。
- 消融未控制参数量（见 5.2），也未做光度归一化的空间建模；几何仅为切片级标量。
- 3 m DTM 的 stereo 源 ID 仍缺失，坐标重采样 + 10 km 中心间距不构成严格的源级独立证明（`GEO_JOIN_PENDING`）。
- 危险规则是工程代理（`g10>0.05` 先验 0.691；平衡规则先验 0.573；复合规则 0.645），不是认证着陆安全。

## 7. 下一步（D 节，尚未执行）

1. **冻结 `heldout_v2`**：从 90,786 索引中选此前未参与任何训练/验证/标定/决策的新站点，
   记录 sample_id/PRODUCT_ID/LTM/分组/缓冲分布/manifest SHA256，并对同源 NAC/stereo DTM 做人工 footprint 审核；
   冻结前不看其模型指标。旧 110 Tile 保留为 `development_test`。
2. 在固定 calibration/val/test 不变的前提下把 train 扩到 3–5k Tile（先确认 `--resume`/`--enrich` 的追加语义）。
3. 固定小型候选配置（如 `p17_full` 与 `p17_no_geo`）后在 `heldout_v2` 上一次性评估，不再调整阈值/校准。

本轮未使用 4×4090、付费云、LFM/Qwen 或联合后验。
