# P3.3 fusion by latitude and learned block gating

2026-11-12. **`FUSION_STRATIFIED_COMPLETE`.** The fusion gain is positive on average in all three Val latitude bands (**+0.0152 equatorial, +0.0140 mid, +0.0179 polar**), though only the mid band is positive for all three seeds. Giving the two blocks learnable scalar weights instead of fixed 1.0 shows no established meaningful improvement over plain concatenation (overall 0.269538 vs 0.268314; observed difference **+0.001224**), and the optimizer settles at roughly **1.62 for the Moonstone block and 0.75 for the NASA block**. Those scalars are parameter values only and are not identifiable as relative feature importance.

| Band (Val rows) | Moonstone | Fusion | Gated fusion | Fusion - Moonstone | Per-seed deltas |
|---|---:|---:|---:|---:|---|
| Equatorial 9-30 (355) | 0.190933 | 0.206086 | 0.205830 | +0.015152 | -0.0017 / +0.0135 / +0.0336 (2/3 positive) |
| Mid 30-60 (450) | 0.129616 | 0.143631 | 0.144312 | +0.014015 | +0.0138 / +0.0171 / +0.0111 (3/3 positive) |
| Polar 60-90 (350) | 0.114087 | 0.131967 | 0.131462 | +0.017879 | +0.0389 / +0.0220 / -0.0073 (2/3 positive) |
| Overall (1,155) | 0.235176 | 0.268314 | 0.269538 | +0.033138 | +0.0437 / +0.0368 / +0.0190 (3/3 positive) |

Cell values are means over seeds 42/43/44; band macro-F1 uses the full 49-class denominator, and each band has a different class support, so band values and deltas are descriptive and not directly comparable effect sizes across rows.

## What it says

- **The additive fusion gain is positive on average in all three broad bands**, with the mid-latitude band the most consistent. This is a coarse robustness check, not proof that the effect is uniform or absent from any single region; the equatorial and polar bands each have one seed that goes slightly negative, so the per-band claim is directional, not established.
- **Learning the block weights does not establish a meaningful improvement.** The gated model is +0.001224 above plain concatenation overall (0.269538 vs 0.268314), a difference of the same order as seed noise, not a demonstrated gain. The learned scalars (~1.62 vs ~0.75, consistent across seeds) are parameter values only: each scalar multiplies a block before an unconstrained linear layer, so scalar magnitudes are not identifiable as relative feature importance or contribution percentages. The model remains a linear classifier over concatenated features.
- **Absolute band differences are small relative to the overall gain**, which is expected because each band has 350-450 Val rows, many classes are absent, and the 49-class macro-F1 over a subset is non-additive.

## Setup

Latitude comes only from the recorded metadata export keyed by patch id; Val is northern hemisphere only (9° to 89°), so there is no southern-hemisphere band. Training reuses the frozen budget (20 epochs, patience 5, AdamW 1e-3/1e-2, batch 8, seeds 42/43/44, earliest-tie Val selection) on the same caches as the fusion matrix; the retained overall means and best epochs reproduce the earlier matrix, though per-seed overall scores and prediction hashes were not retained. The gated model multiplies the Moonstone and NASA blocks by two learnable scalars initialised at 1.0 and otherwise shares the same head and budget. No Test tile was selected, compared, scored or used for training/selection; the metadata export contains all 16,200 IDs, including 2,400 Test IDs, and was read as a full ID table.

## Caveats

- Bands hold 350-450 Val rows and three seeds; many classes are absent or very sparse (21/24/30 of 49 absent in the equatorial/mid/polar bands), so the per-band numbers are exploratory and not a significance test.
- No regional, geological or latitudinal causal interpretation is supported; the bands are only a coarse robustness check, and full-class macro-F1 on subsets is non-additive.
- The Moonstone encoder was Geology-supervised on this same Train split and consumes 28 channels including the LPGRS FeO/TiO2 rasters. This confounds a clean in-domain-versus-cross-domain comparison. Train-label supervision is not by itself validation-target leakage, so the internal additive observation stands, but the cross-encoder interpretation does not.
- **Auditability limitation:** only aggregated metrics and scalar weights were retained; per-seed predictions, confusion matrices and head states were not saved, so the scores cannot be independently reconstructed from predictions.
- **Protocol deviation:** the plan declared `(linear, linear_balanced)` heads; only the unweighted cells were run. The retained matrix is 9 runs, while the planned stratified portion alone specifies 12.

Evidence: [summary](evidence/p33_20261112/fusion_stratified_01/summary.json) with every per-seed band value and the learned weights; protocol [author-declared pre-score plan; chronology independently unverified](../protocols/p33_fusion_stratified_plan.json).
