# P3.3 feature-source and head ablation: in-domain beats cross-domain

2026-11-11. **`FEATURE_ABLATION_COMPLETE`.** Under one identical budget (20 epochs, patience 5, AdamW 1e-3/1e-2, batch 8, seeds 42/43/44, Val-selected), the authenticated Moonstone pooled tokens still beat the NASA pretrained features on Geology: **0.235176 ± 0.004210** versus **0.188510 ± 0.008293** for linear heads, and **0.248381 ± 0.009623** versus **0.196045 ± 0.000176** with class-balanced loss. The earlier frozen-probe gain (+0.116) is therefore a gain **over random initialization**, not over the in-domain encoder.

| Feature set | Head | Val macro-F1 (mean ± sample std) | Val accuracy | Head params | Best epochs (42/43/44) |
|---|---|---:|---:|---:|---|
| NASA pretrained (768-d) | linear | 0.188510 ± 0.008293 | 0.4433 | 37,681 | 10/9/20 |
| NASA pretrained (768-d) | MLP 256 | 0.190067 ± 0.005625 | 0.4496 | 209,457 | 16/17/8 |
| NASA pretrained (768-d) | linear + balanced CE | 0.196045 ± 0.000176 | 0.4165 | 37,681 | 20/17/14 |
| NASA random (768-d) | linear | 0.072138 ± 0.001763 | 0.3388 | 37,681 | 10/9/8 |
| NASA random (768-d) | MLP 256 | 0.059356 ± 0.000766 | 0.3429 | 209,457 | 6/5/4 |
| NASA random (768-d) | linear + balanced CE | 0.072525 ± 0.000775 | 0.2886 | 37,681 | 12/11/10 |
| Moonstone pooled (256-d) | linear | **0.235176 ± 0.004210** | 0.4739 | 12,593 | 4/9/10 |
| Moonstone pooled (256-d) | MLP 256 | 0.225787 ± 0.005490 | 0.4615 | 78,385 | 6/8/9 |
| Moonstone pooled (256-d) | linear + balanced CE | **0.248381 ± 0.009623** | 0.4560 | 12,593 | 14/13/9 |

## What the matrix says

- **Feature source dominates head design**: the 256-d in-domain features beat the 768-d cross-domain features by about +0.047 macro-F1 in both the linear and balanced cells, despite being three times narrower.
- **Class-balanced cross-entropy is the cheapest win**: +0.0075 on NASA pretrained and +0.0132 on Moonstone, with the tightest NASA seed spread (0.000176) but a slightly lower accuracy, so the trade is F1 for calibration-style balance.
- **Wider heads do not help**: the 5.6x-larger MLP is flat on NASA pretrained (0.1901) and worse on Moonstone (0.2258) - the frozen features, not head capacity, are the limit.
- **Random-init encoders are not a strawman for regularisation**: random NASA features already reach 34% accuracy; the pretrained arm's advantage is a real representation gain, but the in-domain encoder's advantage is larger still.

## Model sizes and runtime

| Item | Measured |
|---|---|
| NASA base encoder (decoder pruned) | 85,348,608 parameters ≈ 341 MB fp32 |
| NASA adapters / linear head / MLP head | 4 / 37,681 / 209,457 parameters |
| Upstream NASA checkpoint file | 2,425,755,440 bytes (2.43 GB, includes the generative decoder that the preflight removes) |
| Selected inference tensors actually loaded | 85,761,024 elements (116 keys) |
| Moonstone frozen encoder | checkpoint not present locally; P3.2 recorded 16x256 tokens per sample (pooled 256-d) and head sizes 305,921-337,462 for its own B0/B1/B2/J designs |
| NASA feature extraction | 12,580 forwards (6,290 samples x 2 arms) in 287 s = 43.8 forwards/s including HDF5 decode |
| Head training | 3.4-8.6 s per 20-epoch run at batch 8 (~1,000-1,500 samples/s on CPU); the whole 27-run matrix took 123 s |

The extraction number is the practical cost of the approach: one ~5-minute pass over the cohort (per arm) buys a feature bank that makes every head experiment in this table essentially free.

## Caveats

- Val is used for early stopping and selection, equally in every cell; the numbers are mildly optimistic and not a Test result.
- Three seeds give a sample standard deviation, not a confidence interval or a significance test.
- Pretraining-data overlap with Val geography cannot be excluded for either encoder.
- The Moonstone cells pair cache seed with head seed, matching the P3.2 convention; the NASA features are one fixed encoder with varying head seeds.
- P3.2's own B0 number (0.2555 ± 0.0148) used a 50-epoch/patience-8/batch-128 budget and a 337k-parameter head, so it is not directly comparable to the 0.2352 linear cell here.

Evidence: [summary](evidence/p33_20261111/feature_ablation_01/summary.json) with every per-epoch history, cache hashes and per-run timings; protocol [pre-registered](../protocols/p33_feature_ablation_plan.json).
