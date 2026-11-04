# P3.1 Age baseline

Updated: 2026-11-04. **Local-prefix Age seed42 PASS, including independent exact prediction replay.** Complete-data official baseline remains pending; acquisition stays paused for VPN quota. These are separate experiment tracks.

The partial-data result uses [`P31_LOCAL_PREFIX_STORED_DERIVED_V1`](../protocols/p31_local_prefix_eval_plan.json): original stored Train/Val membership, 5,135 Train and 1,155 Val, official five-class targets and audited HF normalization. It is availability-selected from a retained northern HDF5 prefix, not the full official benchmark. Full upstream integrity remains unverified. Test was not scored. [Subset scope and reproduction](p3_1_partial_data_baselines.md).

Seed42 completed 28 epochs under the unchanged maximum50/patience5 budget and selected **epoch23** by strict Val full-five-class Macro-F1. Accuracy and Macro-F1 below come from that same selected checkpoint. Train loss fell from 1.117469 at epoch1 to 0.687579 at epoch28. [Geology seed42](p3_1_geology_baseline.md) subsequently completed and passed independently, with no Age-supervised encoder transfer; Age seed43 is now running under the frozen settings.

| Baseline | Train Accuracy | Train Macro-F1, all 5 | Val Accuracy | Val Macro-F1, all 5 |
|---|---:|---:|---:|---:|
| Train empirical prior | 43.7001% | 0.121642 | 40.7792% | 0.115867 |
| Independent scratch CNN, seed42 | 72.6582% | 0.659738 | **57.3160%** | **0.431332** |

All five classes have support in both splits, so supported-class and full-five-class macro agree here. The prior is fitted only to Train labels; its argmax predicts Imbrian. The CNN's selected Val Accuracy exceeds the prior by 16.54 percentage points and Macro-F1 by 0.315464. This is a single-seed ordinary-baseline result; seeds43/44 remain pending and no multi-seed standard deviation is claimed. No candidate, semantic, pretrained or unseen-class gain is established.

| Age class | Train support | Val support | Selected Val F1 | Selected Val recall |
|---|---:|---:|---:|---:|
| Pre-Nectarian | 811 | 232 | 0.391509 | 0.357759 |
| Nectarian | 1,697 | 366 | 0.611850 | 0.691257 |
| Imbrian | 2,244 | 471 | 0.682119 | 0.656051 |
| Eratosthenian | 283 | 62 | 0.213115 | 0.209677 |
| Copernican | 100 | 24 | 0.258065 | 0.166667 |

Val NLL is 1.077131 and multiclass Brier sum is 0.574599 (Train prior: 1.296575 and 0.692515). The small Eratosthenian/Copernican supports and weaker class results are visible; no per-class reweighting or retuning was introduced.

Executed training source: `2f90c5b3903da9bbd5ce31171b8446595f3df9f7`. The local prefix SHA256 is `77f9e3dbb31601182557308717c705c6efbf8be29eeed3c2920069d158d5adf8`; selected weight fingerprint is `d8b85fdcc1d8af3f0557a64fae88b7bdb82899399e2a5c7e6a7beaabd53fa103`. [Completed report](evidence/p31_prefix_runs_20261104/age_seed42/report.json), [completion receipt](evidence/p31_prefix_runs_20261104/age_seed42/completion.json), [epoch history](evidence/p31_prefix_runs_20261104/age_seed42/history.json), [independent replay](evidence/p31_prefix_runs_20261104/age_seed42/replay_report.json) and [archive hashes](evidence/p31_prefix_runs_20261104/age_seed42/evidence_manifest.json) are committed. All four Train/Val model/prior prediction CSVs are adjacent and retain exact original bytes. Checkpoint remains local at `outputs/p31/local_prefix_baselines_01/age_seed42/checkpoint.pt`; its SHA256 is recorded in the receipt.

Completed-run verification recomputed checkpoint/history/config/source/input/ID consistency, selected epoch, class counts and confusion matrices from per-sample probabilities. The fresh process reproduced all four prediction CSVs exactly. The receipt was rechecked before archiving. This establishes replay for these weights/data, not repeat-training determinism on the larger subset.

Measured training-window throughput was **28.7506 examples/s**, over 5,000.946 seconds and 143,780 examples. Training plus final evaluation/export took 6,336.519 seconds, excluding the 43.734-second input-hash pass and the separate replay process. Parent peak RSS was 2.5523 GiB with workers0. Torch GPU peak allocated/reserved were 337.65/422.00 MiB; these are allocator measurements, not whole-card/process VRAM. Batch16 and FP32 ran on the existing RX7900XT/ROCm without installing packages or changing the shared environment.

The full-data design remains [p31_baseline_eval_plan.json](../protocols/p31_baseline_eval_plan.json), with 11,400 Train / 2,400 Val and its original full file/statistics/map/split checks. Age is five-class classification with unknown prefixes filtered, not continuous age regression. It retains independent scratch CNN and Train prior, maximum50 epochs, patience5, batch16, FP32, seeds42/43/44, and Val full-five-class Macro-F1 selection. Retaining the final batch, FP32 and HDF5-derived normalization are explicit upstream differences. The old five-per-split runner regression remains an engineering check only. Complete-data execution can occur only after complete size/SHA256 verification; this partial result does not satisfy that gate. Test scoring remains HOLD.
