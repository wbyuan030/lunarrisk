# P1.8 / P18-04 — 固定参数量（fixed-mask）模态消融

日期：2026-10-17；代码状态 `a53937e`（本轮 P18-04a 的提交见仓库 HEAD）。
环境：单卡 RX 7900 XT，PyTorch 2.5.1+rocm6.2，HIP 6.2.41133，Python 3.10.12。
数据：既有 760 hydrated Tile（`outputs/labels_hydrated.csv`），缓存 `outputs/cache_v3.npz`；
110 Tile 测试集 / 25 个地理组 —— **`development_test`，仅探索性**。
测试：full test suite **163/163 PASS**，其中 fixed-mask 相关 7 项（`TestFixedMaskAblation`）。

## 1. 实现（P18-04a）

`--ablation-mode {legacy,fixed-mask}`（默认 `legacy`，旧 P1.7 结果保持不变）：

- **fixed-mask**：每个变体统一输入 `[nac_contrast, nac_texture, coarse_slope]` 三通道 +
  完整标量 head `NAC_SCALARS(2) + GEOMETRY_SCALARS(5)`；缺失模态以 **train 上 StandardScaler 后的 0**
  （中性值）相乘屏蔽，**不删通道、不改 `Linear` 形状**。
- 变体语义：`cnn_nac` 屏蔽 coarse 通道；`cnn_coarse_grid` 屏蔽两路 NAC 通道与 NAC 标量；
  `cnn_fusion` 全量。`--no-geometry` 屏蔽几何标量；`--no-nac-photometry` 屏蔽对比度通道与 NAC 标量（保留纹理）。
- 掩码在 `train`/`val`/`cal`/`test` 全部路径生效（`train_variant` 与 `predict_variant` 共用同一掩码）。
- `metrics.json.protocol` 记录 `ablation_mode`、`fixed_mask_specs`、架构、以及
  每个变体的 `parameters{total,trainable}`；fixed-mask 下若变体参数量不一致直接退出。
- 实测参数量：**三个变体均为 98,464**（`fixed_mask_parameter_counts`）。
- 聚合侧新增 `assert_fixed_capacity`（`aggregate_seeds.py`）与跨配置参数量校验
  （`compare_ablations.py`）：拒绝混用 `legacy`/`fixed-mask`、拒绝同 seed 内参数量不一致、
  拒绝跨 seed/config 的掩码方案不一致。

离线不变量测试（7 项）：参数量在全部变体×消融组合下相同；改动被屏蔽通道/标量对输出**无影响**；
改动可见通道/几何标量会改变输出；掩码在序列化（state_dict 保存/加载）后仍生效；
`--no-nac-photometry` 保留纹理；`--no-geometry` 屏蔽全部几何；1 epoch CPU synthetic smoke 通过。

## 2. 运行（P18-04b）

```bash
for config in p18_full p18_no_geo p18_no_photo p18_no_both; do
  case "$config" in
    p18_full) flags="" ;; p18_no_geo) flags="--no-geometry" ;;
    p18_no_photo) flags="--no-nac-photometry" ;;
    p18_no_both) flags="--no-geometry --no-nac-photometry" ;;
  esac
  for seed in 42 43 44; do
    python3 scripts/train_baseline.py --ablation-mode fixed-mask --seed "$seed" --no-augment \
      --out "outputs/${config}_seed${seed}" $flags
  done
  python3 scripts/aggregate_seeds.py --pattern "outputs/${config}_seed{seed}" --out "outputs/${config}_summary"
done
# 合法增强对照（仅在无几何条件下）
for seed in 42 43 44; do
  python3 scripts/train_baseline.py --ablation-mode fixed-mask --seed "$seed" --augment --no-geometry \
    --out "outputs/p18_no_geo_aug_seed${seed}"
done
python3 scripts/aggregate_seeds.py --pattern "outputs/p18_no_geo_aug_seed{seed}" --out "outputs/p18_no_geo_aug_summary"
python3 scripts/compare_ablations.py --configs p18_full,p18_no_geo,p18_no_photo,p18_no_both \
  --reference p18_full --pattern "outputs/{config}_seed{seed}" --out outputs/p18_fixed_mask_summary
```

耗时：15 次训练，每次 24–32 s（wall，单卡），合计约 **431 s**。结果按规则 `g10>0.05`（主规则）报告；
产物 `outputs/p18_{full,no_geo,no_photo,no_both}{_seed42,_seed43,_seed44,_summary}`、
`outputs/p18_no_geo_aug_{seed*,summary}`、`outputs/p18_fixed_mask_summary/`。

## 3. 结果（3 seed 均值，`development_test`）

| config | model | MAE | Brier(primary) | Brier(raw) | NLL(primary) | NLL(raw) | AUC | UAR@20% | UAR@40% |
|---|---|---|---|---|---|---|---|---|---|
| p18_full | coarse_ridge | **0.0504** | 0.0728 | 0.0702 | 0.2283 | 0.2240 | 0.9659 | 0.136 | 0.250 |
| p18_full | cnn_nac | 0.3321 (0.0171) | 0.1843 | 0.2795 | 0.5431 | 0.8362 | 0.7539 | 0.470 | 0.492 |
| p18_full | cnn_coarse_grid | 0.0621 (0.0101) | 0.0835 | 0.0741 | 0.2553 | 0.2346 | 0.9639 | 0.091 | 0.250 |
| p18_full | cnn_fusion | 0.0664 (0.0076) | 0.0674 | 0.0659 | 0.2068 | 0.2099 | 0.9832 | 0.045 | 0.227 |
| p18_no_geo | cnn_nac | 0.3089 (0.0315) | 0.1760 | 0.2363 | 0.5220 | 0.6899 | 0.7726 | 0.424 | 0.447 |
| p18_no_geo | cnn_coarse_grid | 0.0676 (0.0005) | 0.0826 | 0.0724 | 0.2509 | 0.2274 | 0.9625 | 0.121 | 0.250 |
| p18_no_geo | cnn_fusion | 0.0665 (0.0092) | 0.0679 | 0.0611 | 0.2072 | 0.1946 | 0.9814 | 0.045 | 0.227 |
| p18_no_photo | cnn_nac | 0.3484 (0.0322) | 0.1820 | 0.3481 | 0.5409 | 1.1117 | 0.7532 | 0.439 | 0.492 |
| p18_no_photo | cnn_coarse_grid | 0.0703 (0.0090) | 0.0798 | 0.0704 | 0.2436 | 0.2236 | 0.9659 | 0.121 | 0.250 |
| p18_no_photo | cnn_fusion | 0.0561 (0.0054) | 0.0721 | 0.0655 | 0.2154 | 0.2049 | 0.9785 | 0.076 | 0.227 |
| p18_no_both | cnn_nac | 0.3471 (0.0301) | 0.1828 | 0.3097 | 0.5419 | 0.9844 | 0.7414 | 0.439 | 0.515 |
| p18_no_both | cnn_coarse_grid | 0.0713 (0.0093) | 0.0844 | 0.0740 | 0.2549 | 0.2338 | 0.9627 | 0.121 | 0.250 |
| p18_no_both | cnn_fusion | 0.0566 (0.0037) | 0.0810 | 0.0756 | 0.2405 | 0.2291 | 0.9737 | 0.061 | 0.227 |

（`constant`：MAE 0.3554、Brier 0.2284、UAR@20 0.691；`coarse_ridge` 为确定性模型，四个配置完全相同。）

### 3.1 NAC 增量：`cnn_fusion` − `cnn_coarse_grid`（组级配对自助，1000 次，3 seed 平均）

| config | ΔMAE [95% CI] | 排除 0 | ΔBrier [95% CI] | 排除 0 |
|---|---|---|---|---|
| p18_full | +0.0043 [−0.0018, +0.0116] | 否 | −0.0161 [−0.0374, +0.0044] | 否 |
| p18_no_geo | −0.0011 [−0.0131, +0.0151] | 否 | −0.0147 [−0.0355, +0.0040] | 否 |
| p18_no_photo | **−0.0142 [−0.0217, −0.0062]** | 是 | −0.0078 [−0.0248, +0.0126] | 否 |
| p18_no_both | **−0.0147 [−0.0225, −0.0059]** | 是 | −0.0034 [−0.0249, +0.0237] | 否 |

差值之差（相对 `p18_full`）：`no_geo` ΔMAE −0.0054（不排除 0）；`no_photo` **−0.0184 [−0.0267, −0.0117]**；
`no_both` **−0.0190 [−0.0279, −0.0112]**、ΔBrier **+0.0127 [+0.0019, +0.0288]**。

### 3.2 运行间噪声（同配置重复，重要）

同配置、同 seed 重复一次（`p18_full` vs `p18_full_repeat`，仅 MIOpen 运行间不确定性）：

| model | MAE run1 | MAE run2 | 差值 |
|---|---|---|---|
| cnn_fusion | 0.0664 | 0.0664 | 0.0000 |
| cnn_coarse_grid | 0.0621 | 0.0671 | 0.0050 |
| cnn_nac | 0.3321 | 0.2973 | 0.0348 |

**重复性地板约 0.005（MAE，融合/粗坡度模型）**；`cnn_nac` 的波动更大（~0.035）。
因此：`no_photo`/`no_both` 的 −0.014 ~ −0.019 增量约为噪声地板的 3–4 倍，可视为真实但幅度有限；
`p18_full` 的 +0.0043 完全落在噪声内。Brier 的组级 CI 已覆盖采样不确定性，**未覆盖**运行间噪声，
两者需一起引用。

### 3.3 合法增强对照（`--no-geometry`，fixed-mask）

| 设置 | cnn_fusion MAE | cnn_coarse_grid MAE | cnn_nac MAE | fusion Brier |
|---|---|---|---|---|
| aug-off | 0.0665 | 0.0676 | 0.3089 | 0.0679 |
| aug-on | **0.0561** | 0.0643 | 0.3124 | **0.0621** |

几何条件下的空间增强是合法的（无方位标量），且融合模型 MAE 改善 0.0104（> 0.005 地板）。
这与 P1.8 文档的判断一致：**合法增强可能有真实正则化收益；P1.6 的较好结果不能仅凭"增强开关"完全否定**，
但也不能把 P1.6 与 P1.7/1.8 的数字直接横比。

## 4. 结论

1. fixed-mask 下三个变体**参数量严格相同**，此前的"删除通道"消融不再用于容量相关结论。
2. **在开启 NAC 光度通道（默认 full）时，没有可分辨的 NAC 增量**（ΔMAE +0.0043，落在噪声内）。
3. 屏蔽 NAC 绝对对比度通道与 NAC 光度标量后，融合模型相对粗坡度仅有 **0.014–0.019 MAE** 的优势
   （CI 排除 0，且约为重复性地板 3–4 倍）——即增益来自 **纹理 + 粗坡度**，而不是绝对对比度/亮度统计。
4. 事件概率（Brier）在 equal-capacity 下**没有**任何配置出现排除 0 的 NAC 增量。
5. `cnn_nac` 单模态在所有配置下仍不可用（MAE 0.31–0.35，raw Brier 差于常数基线）。

## 5. 限制

- `development_test` 已多轮查看，全部结论为**探索性**；不得据此声称独立测试性能。
- 3 seed、25 个地理组；CI 不含运行间不确定性（见 3.2）。
- 无几何/光度标量的空间建模；几何仍是切片级标量。
- 3 m DTM 的 stereo 源 ID 在本语料中仍缺失（严格 OOD 的新来源审计见
  `reports/p1_8_new_heldout_sources.md`）。
- 未使用 4×4090、付费云、LFM/联合后验。
