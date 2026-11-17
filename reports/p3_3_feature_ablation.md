# P3.3 feature-source and head ablation: in-domain features lead this frozen-feature matrix

2026-11-11. **`FEATURE_ABLATION_COMPLETE`.** Under one identical budget (20 epochs, patience 5, AdamW 1e-3/1e-2, batch 8, seeds 42/43/44, Val-selected), the Moonstone pooled tokens from the recorded P3.2 caches beat the NASA pretrained features on Geology: **0.235176 ± 0.004210** versus **0.188510 ± 0.008293** for linear heads, and **0.248381 ± 0.009623** versus **0.196045 ± 0.000176** with class-balanced loss. The earlier frozen-probe gain (+0.116) is therefore a gain **over random initialization**, not over the in-domain encoder.

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

- **Feature source and supervision history dominate this comparison**: the 256-d in-domain features beat the 768-d cross-domain features by about +0.047 macro-F1 in both the linear and balanced cells, despite being three times narrower. The Moonstone features also come from a Geology-supervised, Val-selected encoder, so this is not a clean source-only comparison.
- **Class-balanced cross-entropy is the cheapest win**: +0.0075 on NASA pretrained and +0.0132 on Moonstone, with the tightest NASA seed spread (0.000176) but a slightly lower accuracy. No calibration analysis was run, so "calibration-style balance" is not supported.
- **The tested wider MLP does not show a reliable benefit**: it is slightly higher on NASA pretrained (0.190067 vs 0.188510), lower on Moonstone (0.225787 vs 0.235176) and lower on random NASA (0.059356 vs 0.072138). One MLP design does not establish that head capacity is the limiting factor.
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
| Head training | 2.053-8.6 s per 20-epoch run at batch 8 (~1,000-1,500 samples/s on CPU); the whole 27-run matrix took 123 s |

The extraction number is the practical cost of the approach: one ~5-minute pass over the cohort for both arms (287 s, 12,580 forwards) buys a feature bank that makes every head experiment in this table essentially free.

## Caveats

- Val is used for early stopping and selection, equally in every cell; the numbers are mildly optimistic and not a Test result.
- Three seeds give a sample standard deviation, not a confidence interval or a significance test.
- Pretraining-data overlap with Val geography cannot be excluded for either encoder.
- The Moonstone cells pair cache seed with head seed, matching the P3.2 convention; the NASA features are one fixed encoder with varying head seeds.
- P3.2's own B0 number (0.2555 ± 0.0148) used a 50-epoch/patience-8/batch-128 budget and a 337k-parameter head, so it is not directly comparable to the 0.2352 linear cell here.
- **Protocol implementation deviation:** the plan specifies inverse-frequency class weights; the implementation uses inverse-square-root frequency. The reported numbers are from the implementation as run.
- **Protocol chronology:** this is an author-declared pre-score plan; the repository cannot independently prove when the working copy was written relative to execution.
- No Test tile was selected, compared, scored or used for training/selection; the metadata ID table contains Test IDs, but no Test pixels or targets entered this matrix.

Evidence: [summary](evidence/p33_20261111/feature_ablation_01/summary.json) with every per-epoch history, cache hashes and per-run timings; protocol [author-declared pre-score plan; chronology independently unverified](../protocols/p33_feature_ablation_plan.json).
