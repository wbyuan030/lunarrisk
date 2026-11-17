# P3.3 transfer evidence: what the frozen-feature experiments now support

2026-11-13. Consolidated reading of the 2026-11-10/12/13 phases. The table mixes terrain-product identity, a model preflight and frozen-feature results on the fixed 5,135 Train / 1,155 Val cohort under `moonstone_native256_v1`. No Test tile was selected, compared, scored or used for training/selection; the metadata ID table contains all 16,200 IDs, including 2,400 Test IDs, and metadata/whole-file byte access is not Test scoring.

## The chain of evidence

| Phase | Question | Result |
|---|---|---|
| Terrain product identity | Is the cohort the published product? | 6,290 rows bitwise identical to the published mmap after the published z-score; 0 non-finite pixels; masks declare no NaN anywhere |
| NASA GPU preflight (single crop) | Does the official encoder run and load strictly? | PASS, 116 encoder keys, 37,681 trainable head params, no training |
| Frozen probe, 3 seeds | Does pretraining beat a same-seed random encoder? | **Yes, in this probe**: 0.188510 ± 0.008293 vs 0.072138 ± 0.001763 (paired +0.116372 ± 0.009017) |
| Feature ablation | Is that gain bigger than in-domain features? | **No, descriptively**: Moonstone pooled tokens reach 0.235176 (linear) / 0.248381 (balanced) versus NASA 0.188510 / 0.196045; unequal supervision/input histories prevent a clean domain comparison |
| Task matrix | Does head capacity or class balance matter? | Balanced CE helps everywhere (+0.007534 to +0.020680); the tested wider MLP shows no reliable benefit (slightly higher on NASA pretrained, lower on Moonstone/random) |
| Fusion | Do NASA features add predictive information to the Moonstone block? | **Yes, descriptively for classification**: 0.268314 (Geology linear) / 0.286517 (balanced), 0.429163 / 0.449843 (Age); fusion beats Moonstone in all four cells with 3/3 positive seeds |
| Composition regression | Does fusion help on continuous targets? | **No reliable benefit**: FeO 0.00685 vs 0.00675 RMSE (slightly worse), TiO2 0.00494 vs 0.00511 (slightly better, within seed noise) |
| Latitude stratification | Is the fusion gain positive across broad bands? | Yes on average: +0.0152 equatorial, +0.0140 mid, +0.0179 polar (mid 3/3 seeds, others 2/3); a coarse robustness check, not regional uniformity |
| Learned block gate | Is a meaningful improvement over concatenation established? | No: observed difference +0.001224 (0.269538 vs 0.268314). The gate scalars (Moonstone ~1.61, NASA ~0.75) are non-identifiable parameter values, not importance weights |

## What is established

- **Pretraining transfers in this probe**: on this fixed cohort, with possible pretraining/Val geography overlap and Val-selected epochs, the verified NASA encoder is worth roughly +0.116 macro-F1 over an identically initialised random encoder on 49-class Geology, consistent in sign across all three head seeds. There is one encoder realization per arm; this is not a generalization result.
- **NASA features add predictive information to the Moonstone block**: concatenation adds +0.022 to +0.038 macro-F1 over Moonstone alone on Geology and Age (descriptive, reused-Val), and the gain is positive on average in all three broad latitude bands. No capacity-matched or clean domain comparison establishes why.
- **The learned gate does not establish a meaningful gain over concatenation**: the observed difference is +0.001224 (0.269538 vs 0.268314). The scalars settle at Moonstone ~1.61 vs NASA ~0.75, but because each scalar multiplies a block before an unconstrained linear layer, they are not identifiable as relative feature importance or contribution percentages; there is no "minority partner" claim.
- **Task dependence is observed but limited**: the fusion win does not extend to a reliable FeO/TiO2 regression benefit (FeO slightly worse, TiO2 slightly better within seed noise). Two classification tasks and one leakage-affected regression do not establish a general task law or a "leakage ceiling".

## What is not established

- **No clean in-domain control exists locally.** The P3.2 Moonstone encoder was Geology-supervised on this same Train split, and its 28 input channels include the LPGRS FeO/TiO2 rasters. That confounds a clean in-domain-versus-cross-domain comparison. Train-label supervision is not validation-target leakage by itself, so the internal additive observation stands; the NASA arm is cleaner on target-source leakage but is still subject to pretraining/geography overlap and Val selection. A leakage-free in-domain baseline would require training a fresh unsupervised encoder, which has not been done.
- **No fine-tuning or adapters.** Everything here is frozen features plus a head. `input_affine` and partial unfreezing remain unrun; they require gradient through the encoder and separate authorisation.
- **No Test, no new classes, no operational claims.** Val selects the epoch everywhere; three seeds give a sample standard deviation, not a confidence interval. No Test tile was selected, compared, scored or used for training/selection; metadata and whole-file byte access are not Test scoring.
- **Auditability and protocol deviations.** Fusion and stratification retained only summary aggregates; per-seed predictions and head states were not saved, so those scores cannot be independently reconstructed from predictions. The ablation used inverse-square-root class weights although the protocol says inverse-frequency; the stratification plan declared balanced heads but only unweighted cells were run (9 retained runs vs 12 planned). The four dedicated protocols are author-declared pre-score plans; their chronology is not independently provable from the repository.

## Practical use

The reusable asset is the feature bank: one ~5-minute pass for both arms (287 s, 12,580 forwards) produces cached vectors that make any head experiment seconds-long. Concrete uses today are (a) NASA-only features where label leakage must be avoided, (b) concatenated features where the best frozen classification macro-F1 among tested cells is wanted (not necessarily accuracy; Age linear accuracy is slightly lower for fusion), and (c) the fused representation as the input layer for new task heads without re-encoding the cohort.

Recommended next experiments, in ascending cost:
1. A leakage-free in-domain control: retrain an unsupervised Moonstone-style encoder on the same cohort (expensive, needs approval) - this is the only way to settle the "which encoder is better" question properly.
2. `input_affine` adapter on the frozen NASA encoder (hours per seed) to test whether the normalisation/domain shift costs real accuracy.
3. Partial unfreezing of the last encoder blocks (still more expensive; needs a new protocol and a frozen budget).

Evidence index: [product identity](p3_3_mmap_terrain_linkage.md), [preflight](p3_3_real_nasa_preflight.md), [probe](p3_3_frozen_probe.md), [ablation](p3_3_feature_ablation.md), [fusion](p3_3_feature_fusion.md), [composition](p3_3_composition_regression.md), [stratification](p3_3_fusion_stratified.md).
