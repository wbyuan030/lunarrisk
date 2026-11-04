# P3.1 local-prefix Train/Val baselines

Updated: 2026-11-04. **All six predeclared runs COMPLETE and PASS: Age and Geology seeds42/43/44, each with independent exact prediction replay.** The fixed sequence used training source `2f90c5b`. Acquisition remains paused under the user's later direction: “先用这些数据吧，vpn流量快顶不住了”. This is a separately frozen partial-data track; complete-data official scores remain pending.

The retained single HDF5 partial contains **33,964,443,553 bytes** of the pinned 74,449,119,347-byte asset. Local-prefix SHA256 is `77f9e3dbb31601182557308717c705c6efbf8be29eeed3c2920069d158d5adf8`; the upstream full-file hash has **not** been verified. Metadata-only scanning found 7,400 physically complete chunks, ten descriptors outside the prefix and 8,790 rows with unavailable chunk-index metadata. Missing intervals raise errors; no file holes/pixels are filled. Previously acquired and individually hashed range caches supply metadata beyond physical EOF, without expanding eligibility to outside-prefix pixel chunks. The partial is read in place without modification or a second data copy.

| Original stored split | Selected complete rows | Age support | Geology support | Patch-center latitude |
|---|---:|---:|---:|---|
| Train | 5,135 | 5/5 | 36/49 | 7°N–89°N |
| Val | 1,155 | 5/5 | 32/49 | 9°N–89°N |
| Test | 0 selected; 1,110 local complete chunks excluded | Not scored | Not scored | Not used |

All 2,400 original Test rows are excluded from model access, including those with local chunks. The scan queried chunk metadata for all original rows and read zero pixel chunks. Original stored membership, indices and order are retained. Latitude ranges describe centers, not crop footprints. Availability selection makes this a geographically biased northern subset, not a representative substitute for the full official benchmark or an unseen-class evaluation.

The pre-score [`P31_LOCAL_PREFIX_STORED_DERIVED_V1` plan](../protocols/p31_local_prefix_eval_plan.json) seals exact IDs/indices/chunk offsets/sizes and prefix/metadata/cache-report/statistics/map/source hashes. It retains audited channel order, HF global normalization, raw-zero/nonfinite semantics, presence definitions and official mappings. Only normalized pixels enter the models. Geology receives no Age-supervised encoder transfer.

| Task / baseline | Val Accuracy | Val Macro-F1, all outputs | Val supported-class Macro-F1 |
|---|---:|---:|---:|
| Age Train prior | 40.7792% | 0.115867 | 0.115867, five classes |
| Age scratch CNN, three-seed mean ± sample std | **58.6436 ± 2.0070%** | **0.403084 ± 0.035138**, K=5 | Same as full-five |
| Geology Train prior | 16.2771% | 0.005714 | 0.008749, 32 classes |
| Geology scratch CNN, three-seed mean ± sample std | **43.6075 ± 2.2583%** | **0.208791 ± 0.014977**, K=49 | **0.319712 ± 0.022933**, 32 classes |

Sample standard deviations use ddof=1 and n=3; Accuracy deviations are percentage points. Every fixed seed is included. Deterministic Train priors repeated per seed are not independent fits. Full-K macro includes absent classes at F1=0; supported macro is separately reported and never substituted for the primary metric. See [Age results](p3_1_age_baseline.md), [Geology results](p3_1_geology_baseline.md) and [final machine-readable aggregate](evidence/p31_prefix_runs_20261104/aggregate_final.json).

| Run | Completed epoch | Selected epoch |
|---|---:|---:|
| Age42 | 28 | 23 |
| Geology42 | 18 | 13 |
| Age43 | 20 | 15 |
| Age44 | 9 | 4 |
| Geology43 | 23 | 18 |
| Geology44 | 20 | 15 |

Each uses independent task scratch CNN plus Train empirical prior, AdamW/cosine, learning rate .001, batch16, FP32, workers0, maximum50 epochs and patience5. Selection is strictly higher Val full-K Macro-F1, earliest epoch on ties. There is one configuration per task; early stops and lower seed scores are retained. The six-run launcher ended with status COMPLETE/exit0. No additional runs are scheduled under this completed protocol and Test remains HOLD.

Every run passed checkpoint/history/selection/probability/confusion/resource/source/input/ID verification and a fresh-process replay with exact Train/Val model/prior CSV bytes before the next run. The [final independent audit](evidence/p31_prefix_runs_20261104/sequence_verification.json) verified **66 archived and 48 live artifact hashes, all 24 prediction CSVs, and all 24 aggregate metric cells**; all planned seeds and shared provenance match, and Train/Val IDs do not overlap. Per-run directories under `reports/evidence/p31_prefix_runs_20261104/` preserve report, asset manifest, receipt, history, replay report, sanitized stdout, four exact prediction CSVs and artifact hashes. Checkpoints/raw pixels/caches remain local under ignored outputs; no weights or private paths are committed.

Across all six runs, training windows consumed 21,254.121 seconds for 605,930 examples; total training/final-evaluation wall time was 27,324.829 seconds, excluding input hashing and separate replay processes. Individual throughput was 28.3307–28.7506 examples/s. Maximum parent peak RSS was 2.5523GiB with workers0; maximum torch GPU allocated/reserved peaks were 337.83/422.00MiB, not whole-card/process VRAM. Existing RX7900XT/ROCm and private host copies of already installed packages were used; no installation, shared-environment change, container write or new data transfer was performed.

The implementation passed **579 repository tests, zero skips**, 13.347 seconds; compileall and the real sealed-plan preview PASS. [Implementation evidence](evidence/p31_prefix_20261104/verification.json) preserves source/input hashes and sanitized test output. The actual six-run results and final audit above are additional evidence; earlier test counts alone do not establish model performance.

Reproduction uses existing local metadata/cache files and the trusted runtime. The planner refuses an existing output; use a fresh ignored plan path when checking the frozen manifest:

```bash
python3 -I "$JOB_TMP/p31_python.py" scripts/p31_plan_prefix.py \
  --data outputs/p31/assets/moonstone_53099efd/lunar_patches_v4.h5.partial \
  --metadata "$LOCAL_AUDITED_METADATA" \
  --range-report outputs/p31/loader_pair_01/report.json \
  --stats reports/evidence/p30_20261103/hf_channel_stats.json \
  --class-map reports/evidence/p30_20261103/geologic_units.json \
  --outplan outputs/p31/recomputed_prefix_plan.json

python3 scripts/p31_run_prefix_baselines.py \
  --data outputs/p31/assets/moonstone_53099efd/lunar_patches_v4.h5.partial \
  --metadata "$LOCAL_AUDITED_METADATA" \
  --range-report outputs/p31/loader_pair_01/report.json \
  --plan protocols/p31_local_prefix_eval_plan.json \
  --runtime-directory "$JOB_TMP/p31_runtime" \
  --outdir outputs/p31/local_prefix_baselines_01
```

The launcher is a preview by default. `--apply` verifies/reuses already completed receipted runs; a new output namespace would start new training and is not needed for this completed experiment. Existing ROCm library environment settings are retained. The complete-data runner and its original size/hash/split guards remain independent and unchanged.

The scratch models exceed Train prior on this subset across all three seeds, providing an ordinary reference for future fair candidate comparisons. Full official benchmark performance, pretrained/semantic/candidate gains, unseen-class capability and stable LunarJEV H1/H2 gains remain unestablished. P2.2/P3.0 evidence, original plans and legacy LROC gates are unchanged. Remote synchronization status is tracked separately from scientific completion in the progress report and sync evidence.
