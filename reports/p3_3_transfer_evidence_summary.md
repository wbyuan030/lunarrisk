# P3.3 transfer evidence: what the frozen-feature experiments now support

2026-11-13. Consolidated reading of the 2026-11-10/12/13 phases. Every number below is a frozen-feature result on the fixed 5,135 Train / 1,155 Val cohort under `moonstone_native256_v1`; Test has never been read or scored.

## The chain of evidence

| Phase | Question | Result |
|---|---|---|
| Terrain product identity | Is the cohort the published product? | 6,290 rows bitwise identical to the published mmap after the published z-score; 0 non-finite pixels; masks declare no NaN anywhere |
| NASA GPU preflight (single crop) | Does the official encoder run and load strictly? | PASS, 116 encoder keys, 37,681 trainable head params, no training |
| Frozen probe, 3 seeds | Does pretraining beat a same-seed random encoder? | **Yes**: 0.188510 ± 0.008293 vs 0.072138 ± 0.001763 (paired +0.116372 ± 0.009017) |
| Feature ablation | Is that gain bigger than in-domain features? | **No**: Moonstone pooled tokens reach 0.235176 (linear) / 0.248381 (balanced) versus NASA 0.188510 / 0.196045 |
| Task matrix | Does head capacity or class balance matter? | Balanced CE helps everywhere (+0.0075 to +0.019); wider MLP heads do not |
| Fusion | Are the two encoders complementary? | **Yes for classification**: 0.268314 (Geology linear) / 0.286517 (balanced), 0.429163 / 0.449843 (Age); fusion beats Moonstone in all four cells with 3/3 positive seeds |
| Composition regression | Does fusion help on continuous targets? | **No**: FeO 0.00685 vs 0.00675 RMSE, TiO2 0.00494 vs 0.00511 (within seed noise) |
| Latitude stratification | Is the fusion gain one-region? | No: +0.0152 equatorial, +0.0140 mid, +0.0179 polar (mid 3/3 seeds, others 2/3) |
| Learned block gate | Does learned weighting beat concatenation? | No (0.2695 vs 0.2683); the gate settles at Moonstone 1.62 vs NASA 0.75 |

## What is established

- **Pretraining transfers**: the verified NASA encoder is worth roughly +0.116 macro-F1 over an identically initialised random encoder on 49-class Geology, consistent in sign across all three seeds.
- **Transfer and in-domain features are complementary**: concatenation adds +0.022 to +0.038 over Moonstone alone on Geology and Age, and the gain appears in all three latitude bands on average.
- **The cross-domain block is a minority partner**: a learned gate keeps both blocks but weights NASA at about 45% of Moonstone's block weight.
- **Task dependence is real**: the fusion win does not extend to FeO/TiO2 regression, where Moonstone-only is already at the leakage ceiling.

## What is not established

- **No clean in-domain control exists locally.** The P3.2 Moonstone encoder was Geology-supervised on this same Train split, and its 28 input channels include the LPGRS FeO/TiO2 rasters. Its side of every comparison is therefore inflated; the NASA arm is the honest self-supervised measurement. A leakage-free in-domain baseline would require training a fresh unsupervised encoder, which has not been done.
- **No fine-tuning or adapters.** Everything here is frozen features plus a head. `input_affine` and partial unfreezing remain unrun; they require gradient through the encoder (roughly 0.1 s/sample, hours per seed) and separate authorisation.
- **No Test, no new classes, no operational claims.** Val selects the epoch everywhere; three seeds give a sample standard deviation, not a confidence interval.

## Practical use

The reusable asset is the feature bank: one ~5-minute GPU pass per arm produces cached vectors that make any head experiment seconds-long. Concrete uses today are (a) NASA-only features where label leakage must be avoided, (b) concatenated features where the best frozen classification accuracy is wanted, and (c) the fused representation as the input layer for new task heads without re-encoding the cohort.

Recommended next experiments, in ascending cost:
1. A leakage-free in-domain control: retrain an unsupervised Moonstone-style encoder on the same cohort (expensive, needs approval) - this is the only way to settle the "which encoder is better" question properly.
2. `input_affine` adapter on the frozen NASA encoder (hours per seed) to test whether the normalisation/domain shift costs real accuracy.
3. Partial unfreezing of the last encoder blocks (still more expensive; needs a new protocol and a frozen budget).

Evidence index: [product identity](p3_3_mmap_terrain_linkage.md), [preflight](p3_3_real_nasa_preflight.md), [probe](p3_3_frozen_probe.md), [ablation](p3_3_feature_ablation.md), [fusion](p3_3_feature_fusion.md), [composition](p3_3_composition_regression.md), [stratification](p3_3_fusion_stratified.md).
