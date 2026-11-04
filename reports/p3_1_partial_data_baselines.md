# P3.1 local-prefix Train/Val baselines

Updated: 2026-11-04. **Offline subset protocol locked and implementation validated; no model scores reported yet.** The user directed: “先用这些数据吧，vpn流量快顶不住了”. Acquisition stays paused. This is a separate availability-selected, HDF5-derived experiment; the complete-data baseline protocol and gates remain unchanged.

The retained single partial contains **33,964,443,553 bytes** of the pinned 74,449,119,347-byte HDF5. The upstream full-file SHA256 has **not** been verified. An offline metadata scan found 7,400 physically complete pixel chunks, ten descriptors outside the retained prefix and 8,790 rows whose chunk-index metadata is unavailable. Missing intervals raise errors; no missing pixels or file holes are filled. Existing individually hashed range-cache bodies provide previously acquired metadata beyond the prefix's physical EOF. They do not expand eligibility to pixel chunks outside the physical prefix. The original partial is read in place without a second data copy or modification.

| Original stored split | Complete eligible rows | Age support | Geology support | Patch-center latitude |
|---|---:|---:|---:|---|
| Train | 5,135 | 5/5 | 36/49 | 7°N–89°N |
| Val | 1,155 | 5/5 | 32/49 | 9°N–89°N |
| Test | 1,110 complete chunks, excluded | Not scored | Not scored | Not used |

The scan queried chunk metadata for all original rows and read zero pixel chunks. All 2,400 original Test rows are excluded from model access, including the 1,110 whose chunks happen to be local. Original stored memberships and source indices are retained. Latitude ranges describe centers, not full crop footprints. This northern subset is geographically biased and is not a representative substitute for the full benchmark. Forty-nine outputs do not imply unseen-class evaluation.

The new [`P31_LOCAL_PREFIX_STORED_DERIVED_V1` protocol](../protocols/p31_local_prefix_eval_plan.json) seals the actual prefix SHA256 `77f9e3dbb31601182557308717c705c6efbf8be29eeed3c2920069d158d5adf8`, exact ordered Train/Val IDs/indices, complete chunk offsets/sizes, source metadata/cache-report/statistics/map hashes and class counts before scores. Audited channel order, HF global normalization, raw-zero/nonfinite semantics and official Age/Geology mappings are retained. Only normalized pixels enter the model.

Training retains the ordinary baseline design: independent scratch CNN per task plus Train empirical prior; five Age or 49 Geology outputs; full-K Macro-F1 with absent F1=0, supported-class macro separately; AdamW, learning rate .001, batch16, FP32, workers0, maximum50 epochs and patience5; earliest epoch on equal Val full-K Macro-F1. Age42 precedes Geology42, followed by fixed seeds43/44 after run verification. Geology receives no Age-supervised encoder. No hyperparameter search or Test scoring is authorized by this track.

`p31_run_prefix_baselines.py` invokes the shared mini-batch trainer under the separate `subset_baseline` purpose and `partial_stored_prefix` track. The ordinary complete-data purpose retains its original size/hash/ID checks. Each completed subset run must pass checkpoint/history/selection/probability/confusion/resource verification and a fresh-process replay with exact Train/Val model and prior prediction CSVs before a completion receipt or next run. Incomplete directories are retained, never overwritten; a source/plan mismatch stops the sequence. Checkpoints, input data and raw caches remain local under ignored outputs.

Reproduction commands (the metadata argument names an existing audited local JSON; `JOB_TMP` is the trusted task-local runtime directory):

```bash
python3 -I "$JOB_TMP/p31_python.py" scripts/p31_plan_prefix.py \
  --data outputs/p31/assets/moonstone_53099efd/lunar_patches_v4.h5.partial \
  --metadata "$LOCAL_AUDITED_METADATA" \
  --range-report outputs/p31/loader_pair_01/report.json \
  --stats reports/evidence/p30_20261103/hf_channel_stats.json \
  --class-map reports/evidence/p30_20261103/geologic_units.json \
  --outplan protocols/p31_local_prefix_eval_plan.json

python3 scripts/p31_run_prefix_baselines.py \
  --data outputs/p31/assets/moonstone_53099efd/lunar_patches_v4.h5.partial \
  --metadata "$LOCAL_AUDITED_METADATA" \
  --range-report outputs/p31/loader_pair_01/report.json \
  --plan protocols/p31_local_prefix_eval_plan.json \
  --runtime-directory "$JOB_TMP/p31_runtime" \
  --outdir outputs/p31/local_prefix_baselines_01
```

The planner refuses an existing plan path. The launcher command above is a preview; explicit `--apply` executes the already authorized fixed local experiments. Both use zero network. `--max-new-runs` limits the number of newly launched complete runs without changing their epoch budgets. Required existing ROCm library environment variables are retained; no package installation or shared-environment changes are involved.

The integrated repository suite passed **579 tests, zero skips**, 13.347 seconds; compileall and the real sealed-plan launcher preview PASS. The 45 added cases cover prefix/cache integrity and holes, immutable Train/Val selection and Test exclusion, worker shutdown/pickling/final batches, planner provenance/exclusive publication, separate trainer configuration, serial completion verification and exact independent replay. [Implementation evidence](evidence/p31_prefix_20261104/verification.json) records actual source and input hashes; sanitized test output is adjacent. These tests do not constitute trained-model results. Historical 534 passing tests are not used as validation of this new code. Current discovery inventory lives locally at `outputs/p31/prefix_inventory_01.json`; it is not a model result or complete upstream-integrity certificate. Full-data benchmark scores remain pending and Test remains HOLD.
