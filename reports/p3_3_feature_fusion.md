# P3.3 feature fusion: NASA features add predictive information even though weaker alone

2026-11-11. **`FEATURE_FUSION_COMPLETE`.** Concatenating the 256-d Moonstone pooled tokens with the 768-d NASA pretrained tokens gives the best macro-F1 among the tested frozen-feature cells on both tasks (descriptive, reused-Val; see the caveats), and the gain over Moonstone alone is positive for **every** seed: Geology linear **0.268314 ± 0.008927** versus **0.235176 ± 0.004210** (+0.0331, seeds +0.0437/+0.0368/+0.0190); Geology balanced **0.286517 ± 0.011506** versus **0.248381 ± 0.009623** (+0.0381); Age linear **0.429163 ± 0.003382** versus **0.407454 ± 0.012380** (+0.0217); Age balanced **0.449843 ± 0.004326** versus **0.423393 ± 0.008451** (+0.0265).

| Feature set (width) | Geology linear | Geology balanced | Age linear | Age balanced |
|---|---:|---:|---:|---:|
| Moonstone 256d | 0.235176 ± 0.004210 | 0.248381 ± 0.009623 | 0.407454 ± 0.012380 | 0.423393 ± 0.008451 |
| NASA 768d | 0.188510 ± 0.008293 | 0.196045 ± 0.000176 | 0.411073 ± 0.035003 | 0.430482 ± 0.017190 |
| **Fusion 1024d** | **0.268314 ± 0.008927** | **0.286517 ± 0.011506** | **0.429163 ± 0.003382** | **0.449843 ± 0.004326** |
| Fusion - Moonstone | +0.033138 (3/3 seeds) | +0.038136 (3/3) | +0.021709 (3/3) | +0.026450 (3/3) |

## What this changes

- **Under this head procedure, NASA features add predictive information to the Moonstone block.** The previous ablation showed NASA features lose to Moonstone alone on Geology (-0.047). Adding them to the Moonstone block nevertheless improves the fused result: concatenation beats Moonstone by +0.033 to +0.038, with no seed contradicting it. This is an observed additive gain on reused Val, not a clean comparison of pretraining domains: no capacity-matched nuisance-feature control was run, and the Moonstone block has a different supervision and input history (see caveats).
- **Age behaves differently from Geology.** NASA alone matches or slightly beats Moonstone on the 5-class Age task (0.4111 vs 0.4075 linear, 0.4305 vs 0.4234 balanced), while losing badly on 49-class Geology. The reason for this task difference is not established by these runs; the "broad terrain context rather than fine lithology" explanation was not tested.
- **Class-balanced loss stays the cheapest win**: +0.007534 to +0.020680 across the tested feature sets and tasks.

## Design and gates

36 runs: 3 feature sets x 2 tasks x 2 heads x 3 seeds, all under the frozen budget (20 epochs, patience 5, AdamW 1e-3/1e-2, batch 8, earliest-tie Val selection). Fusion is a plain column concatenation in a verified row order; no learned gating, no per-feature normalization beyond what each cache already carries. Ordered Train/Val IDs were required to match across the NASA manifest and every Moonstone cache before any cell ran; Age targets came only from the recorded P3.2 cache; macro-F1 uses the full class count including absent classes. No Test tile was selected, compared, scored or used for training/selection; the metadata ID table includes Test IDs, but no Test pixels or targets entered the evaluation.

Age support: Train `[811, 1697, 2244, 283, 100]`, Val `[232, 366, 471, 62, 24]` over classes 0-4; the rarest Val class has 24 examples, so the Age tail is noisy.

## Caveats

- Val selects the epoch in every cell, so numbers are mildly optimistic and not Test results.
- Three seeds give a sample standard deviation, not a confidence interval; the fusion deltas are consistent in sign but the per-seed spread reaches 0.025 on Geology.
- Pretraining-data overlap cannot be excluded for either encoder, and the Age labels are the P3.2 derived targets, not an independent label source.
- The Moonstone encoder was Geology-supervised on this same Train split and consumes 28 channels including the LPGRS FeO/TiO2 rasters. That confounds a clean in-domain-versus-cross-domain comparison. Train-label supervision is not validation-target leakage by itself, so the internal additive observation stands; the cross-encoder interpretation does not.
- Concatenation widens the head input to 1024 dimensions (4x Moonstone's 256; 4/3x NASA's 768), but the head stays tiny (1024x49 = 50,225 parameters), so this is still a frozen-feature result.
- **Auditability limitation:** this matrix retained only the summary; per-seed predictions and selected head states were not saved, so the scores cannot be independently reconstructed from predictions (unlike the frozen probe). Feature-cache content hashes were recorded after evaluation, not checked against pinned expected hashes before it.

## Cost

The whole 36-run matrix took 130 s on CPU heads; the underlying feature caches are the fixed assets from the two earlier phases (NASA extraction 287 s, covering both arms = 12,580 forwards; Moonstone P3.2 export). Evidence: [summary](evidence/p33_20261111/feature_fusion_01/summary.json), protocol [author-declared pre-score plan; chronology independently unverified](../protocols/p33_feature_fusion_plan.json).
