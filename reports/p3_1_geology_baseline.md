# P3.1 Geology baseline

Updated: 2026-11-04. **Local-prefix Geology seeds42/43/44 COMPLETE and PASS, with independent exact prediction replay for every run. All six planned Age/Geology runs are complete.** Complete-data official baseline remains pending; acquisition stays paused for VPN quota.

The core task remains official 49-output classification with unit_id minus one and audited HF global normalization. This result uses the separately locked [`P31_LOCAL_PREFIX_STORED_DERIVED_V1`](../protocols/p31_local_prefix_eval_plan.json): 5,135 original Train and 1,155 original Val rows whose complete pixel chunks are present in the retained prefix. Train supports 36 classes and Val supports 32; all Val-supported classes occur in Train. Original stored membership/order is retained. This northern availability-selected subset is not the full official benchmark or an unseen-class protocol. Upstream full-file integrity remains unverified; Test was not scored. [Subset scope/reproduction](p3_1_partial_data_baselines.md).

Each scratch CNN was initialized independently for Geology, without Age supervision. All three predeclared seeds use the same maximum50-epoch/patience5 budget, batch16, workers0, FP32, AdamW and cosine schedule. Selection is strictly greater Val full-49 Macro-F1, earliest epoch on ties. Per-seed metrics come from that same selected checkpoint.

| Seed | Completed epoch | Selected epoch | Val Accuracy | Val Macro-F1, all 49 | Val Macro-F1, supported 32 |
|---|---:|---:|---:|---:|---:|
| 42 | 18 | 13 | 41.0390% | 0.223322 | 0.341961 |
| 43 | 23 | 18 | 45.2814% | 0.209648 | 0.321023 |
| 44 | 20 | 15 | 44.5022% | 0.193405 | 0.296151 |
| **Mean ± sample std** | — | — | **43.6075 ± 2.2583%** | **0.208791 ± 0.014977** | **0.319712 ± 0.022933** |

All three fixed seeds are included (n=3, sample standard deviation with ddof=1); Accuracy standard deviations are percentage points. Full-49 Macro-F1 includes zero F1 for absent classes, matching the pinned upstream definition. Supported-class macro is secondary. The final [six-run aggregate](evidence/p31_prefix_runs_20261104/aggregate_final.json) preserves per-seed, Train/Val, full/supported and prior results, with no missing seeds.

| Baseline | Train Accuracy | Train Macro-F1, all 49 | Val Accuracy | Val Macro-F1, all 49 | Val Macro-F1, supported 32 |
|---|---:|---:|---:|---:|---:|
| Train empirical prior | 16.2415% | 0.005703 | 16.2771% | 0.005714 | 0.008749 |
| Scratch CNN, three-seed mean ± sample std | 61.5125 ± 3.5059% | 0.343186 ± 0.028979 | 43.6075 ± 2.2583% | 0.208791 ± 0.014977 | 0.319712 ± 0.022933 |

Every seed exceeds the Train prior on this fixed Val set. Mean Val Accuracy exceeds it by 27.3304 percentage points and full-49 Macro-F1 by 0.203078. The prior is fitted only to Train labels and is identical across seeds; repeated prior values are not independent fits. These results establish an ordinary scratch reference on this subset. They do not establish semantic/candidate/pretrained gains, unseen-class generalization or full official benchmark performance. Complete per-class support/F1/recall/predicted counts/confusion are in each report.

| Seed | Val NLL | Val multiclass Brier sum | Training examples/s | Parent peak RSS, GiB | Torch GPU allocated / reserved peak, MiB |
|---|---:|---:|---:|---:|---:|
| 42 | 1.884506 | 0.751598 | 28.3307 | 2.2907 | 337.83 / 422.00 |
| 43 | 1.797948 | 0.708132 | 28.4231 | 2.2908 | 337.83 / 422.00 |
| 44 | 1.874705 | 0.719928 | 28.5694 | 2.2901 | 337.83 / 422.00 |

Train-prior Val NLL/Brier are 2.908877/0.924160. Per-seed training-window times were 3,262.534 / 4,155.244 / 3,594.753 seconds; training plus final evaluation/export wall times were 4,209.140 / 5,304.933 / 4,624.206 seconds, excluding input hashing (55.450 / 45.460 / 50.357 seconds) and the separate replay processes. GPU fields measure the torch allocator, not total card/process VRAM. The existing RX7900XT/ROCm runtime was retained without package installation or shared-environment changes.

All runs use training source `2f90c5b3903da9bbd5ce31171b8446595f3df9f7`, local-prefix SHA256 `77f9e3dbb31601182557308717c705c6efbf8be29eeed3c2920069d158d5adf8`, and the same source/input/split hashes; configurations differ only in the predeclared seed. Checkpoints remain local under `outputs/p31/local_prefix_baselines_01/geology_seed{42,43,44}/checkpoint.pt`. Receipts record checkpoint SHA256 and reports record selected-weight fingerprints.

| Seed | Report | History | Completion/replay receipt | Archive hashes |
|---|---|---|---|---|
| 42 | [Report](evidence/p31_prefix_runs_20261104/geology_seed42/report.json) | [History](evidence/p31_prefix_runs_20261104/geology_seed42/history.json) | [PASS](evidence/p31_prefix_runs_20261104/geology_seed42/completion.json) | [Manifest](evidence/p31_prefix_runs_20261104/geology_seed42/evidence_manifest.json) |
| 43 | [Report](evidence/p31_prefix_runs_20261104/geology_seed43/report.json) | [History](evidence/p31_prefix_runs_20261104/geology_seed43/history.json) | [PASS](evidence/p31_prefix_runs_20261104/geology_seed43/completion.json) | [Manifest](evidence/p31_prefix_runs_20261104/geology_seed43/evidence_manifest.json) |
| 44 | [Report](evidence/p31_prefix_runs_20261104/geology_seed44/report.json) | [History](evidence/p31_prefix_runs_20261104/geology_seed44/history.json) | [PASS](evidence/p31_prefix_runs_20261104/geology_seed44/completion.json) | [Manifest](evidence/p31_prefix_runs_20261104/geology_seed44/evidence_manifest.json) |

Four Train/Val model/prior prediction CSVs per run retain exact original bytes; each was reproduced exactly by an independent process. Checkpoint/history/config/source/input/ID consistency, selection and stopping, probabilities/counts/confusion and resources passed verification before the next run. The [final independent audit](evidence/p31_prefix_runs_20261104/sequence_verification.json) checked all six Age/Geology runs: 66 archived and 48 live artifact hashes, 24 prediction CSVs, 24 independently recomputed aggregate cells, shared frozen provenance, no Train/Val ID overlap and no missing seeds. This proves prediction replay for each fixed checkpoint/data, not repeat-training determinism.

The complete-data plan remains [p31_baseline_eval_plan.json](../protocols/p31_baseline_eval_plan.json), with stored 11,400 Train / 2,400 Val, complete size/SHA256 and ordered-ID checks, and fixed seeds42/43/44. Full Train has 46 represented classes; its three absent classes are also absent from full Val/Test. Complete-data scores remain pending until the complete asset is available and verified. Old P3.0 Age-supervised five-example features remain engineering evidence. Candidate comparisons still require a common encoder/modality/examples/budget and ordinary masked-softmax; semantic descriptions or new class holdouts need a separate design. Test scoring remains HOLD. No additional training is scheduled under this completed six-run protocol.
