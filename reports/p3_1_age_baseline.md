# P3.1 Age baseline

Updated: 2026-11-04. **Local-prefix Age seeds42/43/44 COMPLETE and PASS, including independent exact prediction replay for every run.** All six planned Age/Geology runs are complete. Complete-data official baseline remains pending; acquisition stays paused for VPN quota.

This result uses [`P31_LOCAL_PREFIX_STORED_DERIVED_V1`](../protocols/p31_local_prefix_eval_plan.json): 5,135 original stored Train and 1,155 original stored Val rows, official five-class targets, fixed 28 channels and audited HF normalization. This availability-selected northern prefix is a separate derived track, not the full official benchmark. Full upstream integrity remains unverified and Test was not scored. [Subset scope/reproduction](p3_1_partial_data_baselines.md).

All three seeds use independent scratch CNNs and the predeclared AdamW/cosine budget: maximum50 epochs, patience5, batch16, FP32, workers0, no label smoothing or class weighting. Selection is strictly higher Val full-five Macro-F1, earliest epoch on ties. Both metrics for each seed come from that same selected checkpoint. Geology models are initialized independently without Age-supervised transfer.

| Baseline / seed | Completed epoch | Selected epoch | Train Accuracy | Train Macro-F1 | Val Accuracy | Val Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Train empirical prior | — | — | 43.7001% | 0.121642 | 40.7792% | 0.115867 |
| Scratch CNN, 42 | 28 | 23 | 72.6582% | 0.659738 | 57.3160% | 0.431332 |
| Scratch CNN, 43 | 20 | 15 | 69.1723% | 0.531061 | 60.9524% | 0.414184 |
| Scratch CNN, 44 | 9 | 4 | 63.2717% | 0.444409 | 57.6623% | 0.363737 |
| Scratch CNN, mean ± sample std | — | — | 68.3674 ± 4.7448% | 0.545069 ± 0.108346 | **58.6436 ± 2.0070%** | **0.403084 ± 0.035138** |

The aggregate uses all three predeclared seeds (n=3, sample standard deviation with ddof=1); Accuracy standard deviations are percentage points. Full-five and supported-class Macro-F1 agree because all five classes have support. The [four-run snapshot](evidence/p31_prefix_runs_20261104/age_seed44/aggregate_at_completion.json) already contains final Age results; the [final six-run aggregate](evidence/p31_prefix_runs_20261104/aggregate_final.json) now also includes all Geology seeds. The [final sequence audit](evidence/p31_prefix_runs_20261104/sequence_verification.json) verifies all six runs and their predictions. An [independent aggregate/archive check](evidence/p31_prefix_runs_20261104/age_three_seed_verification.json) recomputed Train/Val means and sample standard deviations and verified 33 per-seed archived artifact hashes, shared provenance and replay receipts. Prior probabilities and metrics are deterministic from the same Train labels; repetitions across seeds are not independent prior fits.

Each seed exceeds the Train prior on this fixed Val set. Mean Val Accuracy exceeds the prior by 17.8644 percentage points and Macro-F1 by 0.287217. Seed44's earlier stopping and lower Macro-F1 are included in the aggregate. The 0.363737–0.431332 Macro-F1 range shows seed/selection variability; three seeds on this northern subset do not establish global benchmark performance or semantic/candidate/pretrained/unseen-class gain. Val supports are 232/366/471/62/24 in official Pre-Nectarian/Nectarian/Imbrian/Eratosthenian/Copernican order, so minority-class evidence is limited. Each per-seed report retains complete class F1/recall/counts/confusion.

| Seed | Val NLL | Val multiclass Brier sum | Training examples/s | Parent peak RSS, GiB | Torch GPU allocated / reserved peak, MiB |
|---|---:|---:|---:|---:|---:|
| 42 | 1.077131 | 0.574599 | 28.7506 | 2.5523 | 337.65 / 422.00 |
| 43 | 1.036316 | 0.550267 | 28.4116 | 2.2917 | 337.65 / 422.00 |
| 44 | 1.048789 | 0.556662 | 28.4239 | 2.2896 | 337.65 / 422.00 |

Train-prior Val NLL/Brier are 1.296575/0.692515. Throughput measures training windows only. Their cumulative training times were 5,000.946 / 3,614.721 / 1,625.923 seconds; training plus final evaluation/export wall times were 6,336.519 / 4,641.947 / 2,208.084 seconds, excluding input-hash passes (43.734 / 45.753 / 45.625 seconds) and the separate replay processes. GPU fields measure the torch allocator, not whole-card/process VRAM. These runs use the existing RX7900XT/ROCm runtime without package installation or shared-environment changes.

All runs use training source `2f90c5b3903da9bbd5ce31171b8446595f3df9f7`, local-prefix SHA256 `77f9e3dbb31601182557308717c705c6efbf8be29eeed3c2920069d158d5adf8`, and the same source, input and split hashes; configurations differ only in the predeclared seed. Checkpoints remain local under `outputs/p31/local_prefix_baselines_01/age_seed{42,43,44}/checkpoint.pt`; receipts record checkpoint SHA256 and reports record selected-weight fingerprints.

| Seed | Report | History | Completion/replay receipt | Archive hashes |
|---|---|---|---|---|
| 42 | [Report](evidence/p31_prefix_runs_20261104/age_seed42/report.json) | [History](evidence/p31_prefix_runs_20261104/age_seed42/history.json) | [PASS](evidence/p31_prefix_runs_20261104/age_seed42/completion.json) | [Manifest](evidence/p31_prefix_runs_20261104/age_seed42/evidence_manifest.json) |
| 43 | [Report](evidence/p31_prefix_runs_20261104/age_seed43/report.json) | [History](evidence/p31_prefix_runs_20261104/age_seed43/history.json) | [PASS](evidence/p31_prefix_runs_20261104/age_seed43/completion.json) | [Manifest](evidence/p31_prefix_runs_20261104/age_seed43/evidence_manifest.json) |
| 44 | [Report](evidence/p31_prefix_runs_20261104/age_seed44/report.json) | [History](evidence/p31_prefix_runs_20261104/age_seed44/history.json) | [PASS](evidence/p31_prefix_runs_20261104/age_seed44/completion.json) | [Manifest](evidence/p31_prefix_runs_20261104/age_seed44/evidence_manifest.json) |

Each evidence directory contains exact original bytes for four Train/Val model/prior prediction CSVs and a separate fresh-process replay report. Completed-run verification checked checkpoint/history/config/source/input/ID consistency, selection and stopping, class counts/confusion from per-sample probabilities, and finite resource measurements. Every fresh process reproduced all four CSVs exactly; receipts were rechecked before archiving. This establishes prediction replay for each fixed checkpoint/data, not repeat-training determinism.

The full-data design remains [p31_baseline_eval_plan.json](../protocols/p31_baseline_eval_plan.json), with stored 11,400 Train / 2,400 Val and its original full file/statistics/map/ordered-ID gates. Age remains official five-class classification, with unknown prefixes filtered, not continuous age regression. FP32, retained final batch and HDF5-derived pixels are disclosed upstream differences. Old five-per-split pilot regression remains engineering evidence. Complete-data results require a complete verified asset; this subset result does not satisfy that gate. Test scoring remains HOLD.
