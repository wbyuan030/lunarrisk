# P3.3 feature fusion: NASA features are complementary even though weaker alone

2026-11-11. **`FEATURE_FUSION_COMPLETE`.** Concatenating the 256-d Moonstone pooled tokens with the 768-d NASA pretrained tokens gives the best frozen-feature result on both tasks, and the gain over Moonstone alone is positive for **every** seed: Geology linear **0.268314 ± 0.008927** versus **0.235176 ± 0.004210** (+0.0331, seeds +0.0437/+0.0368/+0.0190); Geology balanced **0.286517 ± 0.011506** versus **0.248381 ± 0.009623** (+0.0381); Age linear **0.429163 ± 0.003382** versus **0.407454 ± 0.012380** (+0.0217); Age balanced **0.449843 ± 0.004326** versus **0.423393 ± 0.008451** (+0.0265).

| Feature set (width) | Geology linear | Geology balanced | Age linear | Age balanced |
|---|---:|---:|---:|---:|
| Moonstone 256d | 0.235176 ± 0.004210 | 0.248381 ± 0.009623 | 0.407454 ± 0.012380 | 0.423393 ± 0.008451 |
| NASA 768d | 0.188510 ± 0.008293 | 0.196045 ± 0.000176 | 0.411073 ± 0.035003 | 0.430482 ± 0.017190 |
| **Fusion 1024d** | **0.268314 ± 0.008927** | **0.286517 ± 0.011506** | **0.429163 ± 0.003382** | **0.449843 ± 0.004326** |
| Fusion - Moonstone | +0.033138 (3/3 seeds) | +0.038136 (3/3) | +0.021709 (3/3) | +0.026450 (3/3) |

## What this changes

- **The two encoders are complementary, not redundant.** The previous ablation showed NASA features lose to Moonstone alone on Geology (-0.047). Fusion flips that: concatenation beats Moonstone by +0.033 to +0.038, with no seed contradicting it. Cross-domain representations carry information the in-domain encoder does not, even when their standalone accuracy is worse.
- **Age behaves differently from Geology.** NASA alone matches or slightly beats Moonstone on the 5-class Age task (0.4111 vs 0.4075 linear, 0.4305 vs 0.4234 balanced), while losing badly on 49-class Geology. Coarser targets benefit more from the transfer, which is consistent with cross-domain features encoding broad terrain context rather than fine lithology.
- **Class-balanced loss stays the cheapest win**: +0.008 to +0.019 in every feature set and task.

## Design and gates

36 runs: 3 feature sets x 2 tasks x 2 heads x 3 seeds, all under the frozen budget (20 epochs, patience 5, AdamW 1e-3/1e-2, batch 8, earliest-tie Val selection). Fusion is a plain column concatenation in a verified row order; no learned gating, no per-feature normalization beyond what each cache already carries. Ordered Train/Val IDs were required to match across the NASA manifest and every Moonstone cache before any cell ran; Age targets came only from the authenticated P3.2 cache; macro-F1 uses the full class count including absent classes; Test was never read.

Age support: Train `[811, 1697, 2244, 283, 100]`, Val `[232, 366, 471, 62, 24]` over classes 0-4; the rarest Val class has 24 examples, so the Age tail is noisy.

## Caveats

- Val selects the epoch in every cell, so numbers are mildly optimistic and not Test results.
- Three seeds give a sample standard deviation, not a confidence interval; the fusion deltas are consistent in sign but the per-seed spread reaches 0.025 on Geology.
- Pretraining-data overlap cannot be excluded for either encoder, and the Age labels are the P3.2 derived targets, not an independent label source.
- Concatenation doubles the head input width but the head stays tiny (1024x49 = 50,225 parameters), so this is still a frozen-feature result.

## Cost

The whole 36-run matrix took 130 s on CPU heads; the underlying feature caches are the fixed assets from the two earlier phases (NASA extraction 287 s, Moonstone P3.2 export). Evidence: [summary](evidence/p33_20261111/feature_fusion_01/summary.json), protocol [pre-registered](../protocols/p33_feature_fusion_plan.json).
