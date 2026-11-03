# P3.1 formal baseline execution

Updated: 2026-11-04. **Launcher implemented and validated; full asset acquisition RUNNING; no formal score yet.** [Explicit authorization](../protocols/p31_acquisition_authorization.json) now covers the fixed HDF5 and [frozen baseline design](../protocols/p31_baseline_eval_plan.json). This report describes prepared execution controls, not completed training.

`scripts/p31_run_baselines.py` serializes the exact sequence **Age42 → Geology42 → Age43 → Age44 → Geology43 → Geology44**. Each child is a fresh isolated Python process, uses the same existing ROCm package directory, and retains the locked 50-epoch maximum, batch16, workers0, FP32, AdamW/cosine and patience5. No Test scores, tuning or additional supervision. The optional invocation cap limits only how many new runs start; it never changes a model's epoch/LR horizon.

After each completed run, the launcher verifies the full data/stat/map hashes, source/plan identity, official Train/Val IDs/counts, actual checkpoint finite weights, finite training history/NLL, early stopping and earliest-best Val selection, batch/resource records, model/prior CSV probabilities and recomputed confusion, and artifact hashes. Only then does the next run begin. A completion receipt records the evidence. Complete matching runs can be reverified and reused; incomplete/tampered/changed outputs stop without overwriting checkpoints or silently retraining. A process lock prevents concurrent use of the same sequence directory. Source and scientific plan remain fixed once real runs start.

The aggregate snapshots contain only verified completed runs, separated by task, Train/Val and scratch/prior. They report per-seed accuracy, full-K Macro-F1 and supported-class Macro-F1, mean, and **sample standard deviation (ddof=1)**; missing results and standard deviation with fewer than two completed seeds remain null. Repeated deterministic Train priors are explicitly not independent prior fits. No favorable seed or macro definition is selected.

Prepared first-stage command, to execute only after the final HDF5 size/SHA256 passes:

```bash
LUNARRISK_GIT_SHA="$P31_SOURCE_COMMIT" \
LD_LIBRARY_PATH="$P31_EXISTING_ROCM_LIBS" \
python3 -I scripts/p31_run_baselines.py \
  --data outputs/p31/assets/moonstone_53099efd/lunar_patches_v4.h5 \
  --outdir outputs/p31/baselines_v1 \
  --runtime-directory "$P31_EXISTING_PRIVATE_RUNTIME" \
  --apply --max-new-runs 1
```

The launcher is stdlib-only until checkpoint verification; its children explicitly load the trusted package directory while retaining `-I`. No package installation or environment replacement. Subsequent phase invocations use the same command/output root, reverify completed receipts, and start the next one of the six runs. This gives separate completion events for the first Age and first Geology gates. Omitting the cap executes all remaining fixed runs serially. Each apply invocation and each trainer perform bounded full-file hashing; these integrity checks cost additional disk reads, never additional downloads or dataset-sized RAM.

Validation actually executed: **531 repository tests, all PASS, zero skips**, 14.044 seconds; compileall PASS. The 21 new tests comprise 7 tiny fake-HTTP acquisition tests and 14 launcher tests, including mocked serial first-seed/failure/restart paths, tampered evidence, capped invocations, aggregation and actual torch checkpoint finite-state checking. None is a full-data training result. CLI dry-run also PASS without reading the incomplete asset or creating model outputs.

Public sanitized test/preview/source evidence: `reports/evidence/p31_baseline_launch_20261104/`. Runtime model artifacts, checkpoints, receipts and aggregate snapshots will live under ignored `outputs/p31/baselines_v1/`. Formal learning curves/resources/metrics will be archived only after actual completion. Storage of the full HDF5 and hashing its compressed bytes do not authorize model access to Test samples.
