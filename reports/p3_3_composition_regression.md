# P3.3 composition regression: fusion adds nothing, and Moonstone's edge is partly leakage

2026-11-11. **`COMPOSITION_REGRESSION_COMPLETE`.** Predicting patch-mean FeO and TiO2 from frozen features, the Moonstone features dominate (FeO R² 0.985, TiO2 R² 0.916), the NASA features are clearly weaker but still well above the mean baseline (0.879 / 0.722 versus -0.002 / -0.010), and **fusion does not improve FeO and barely moves TiO2** (FeO 0.00685 vs 0.00675 RMSE, TiO2 0.00494 vs 0.00511). Unlike Geology and Age, concatenation is not a win here.

| Feature set | FeO RMSE | FeO R² | TiO2 RMSE | TiO2 R² |
|---|---:|---:|---:|---:|
| Train-mean baseline | 0.05603 | -0.0024 | 0.01786 | -0.0104 |
| Moonstone 256d | **0.00675 ± 0.00037** | **0.9854 ± 0.0016** | 0.00511 ± 0.00080 | 0.9161 ± 0.0256 |
| NASA 768d | 0.01942 ± 0.00030 | 0.8795 ± 0.0037 | 0.00937 ± 0.00003 | 0.7221 ± 0.0018 |
| Fusion 1024d | 0.00685 ± 0.00033 | 0.9850 ± 0.0015 | **0.00494 ± 0.00068** | **0.9217 ± 0.0204** |

## Why this result is contaminated in a specific way

The Moonstone 28-channel product includes **LPGRS FeO and TiO2 as input channels**, and the P3.2 feature encoder was trained for Geology on the same cohort. Label-leaky inputs and label-supervised features are exactly the situation here, so the Moonstone numbers should be read as an upper bound under leakage, not as evidence that its representation is intrinsically better at geochemistry. The NASA encoder never saw these rasters or these labels, so its 0.879 / 0.722 is the cleaner measurement of genuine transfer, and it is still strong against a mean baseline of R² ≈ 0.

This also reframes the earlier task matrix: Moonstone's Geology advantage is likely inflated by label supervision on the same Train split, while the NASA arm is self-supervised. The fusion gain on Geology and Age survives that asymmetry - which makes it more interesting, not less - but no cell here should be read as a clean representation-quality comparison.

## Setup

Labels are the mean of finite LPGRS pixels inside each fixed 256x256 Train/Val tile; every one of the 6,290 selected tiles was fully finite, so no label is a partial-coverage average. Rasters were downloaded from the fixed revision (FeO `1a984808…`, TiO2 `ebaee6c5…`, 86,012,479 and 108,829,989 bytes). Targets were z-scored with Train statistics and metrics are reported on the original scale. Model: `Linear(d,2)` with MSE, 20 epochs, patience 5, AdamW 1e-3/1e-2, batch 8, seeds 42/43/44, earliest-tie Val selection on mean normalised RMSE. Labels and feature rows were required to cover the cohort manifest exactly before training.

## Caveats

- Val selects the epoch in every cell; the numbers are mildly optimistic and are not Test results.
- Three seeds give a sample standard deviation, not a confidence interval.
- The TiO2 fusion improvement (3%) is within seed spread and should not be called a gain.
- LPGRS products are coarse geochemical maps, not in-situ measurements; nothing here supports operational geochemistry or landing-safety interpretation.

Evidence: [labels summary](evidence/p33_20261111/composition_labels_01/summary.json), [labels CSV](evidence/p33_20261111/composition_labels_01/labels.csv), [regression summary](evidence/p33_20261111/composition_regression_01/summary.json), protocol [pre-registered](../protocols/p33_composition_regression_plan.json).
