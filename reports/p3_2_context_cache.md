# P3.2 frozen visual context cache

Updated 2026-11-05. **PASS: all three predeclared Geology encoders exported and verified.** Cache files remain local at `outputs/p32/context_03/`, not in Git. The [archived manifest](evidence/p32_20261105/context/manifest.json) records source/input/checkpoint/receipt hashes; [P3.2 progress](p3_2_progress.md) tracks the head experiments separately.

The exporter uses each original P3.1 Geology seed42/43/44 `best_state_dict`, authenticated against unchanged original completion receipts, reports and checkpoint hashes. Classifier weights are loaded strictly; the original classifier is not reused as a P3.2 comparison head. The feature extractor stops before adaptive global pooling/flatten, obtains `[B,256,16,16]`, average-pools to4×4 and exports `[B,16,256]` FP32 tokens. Pooled controls are the exact NumPy mean of these same tokens, not independently extracted features.

| Encoder seed | Original selected epoch | Actual cache bytes | Export/verification wall seconds | Cache SHA256 |
|---|---:|---:|---:|---|
| 42 | 13 | 110,118,080 | 218.051 | `09a5a34061ac711bd4498769bd36a3603ea3c8d0758fa0c0e2395d5a77107d43` |
| 43 | 18 | 110,118,080 | 224.161 | `84cecee9b7e17c9628451f8be4200edf749196521d0bc03ef4b341c180de6138` |
| 44 | 15 | 110,118,080 | 222.694 | `38a06f49546d0f524a56fe5ee01e8a843a18feb3ed47b3584c64025075bb05f0` |

Total actual files:330,354,240 bytes, including tokens, pooled vectors, both task targets, IDs and HDF5 overhead. The standalone pure-token shape estimate is103,055,360 bytes per seed. Existing disk had432GiB free before export. Batch16/workers0/final partial batch retained; original normalization and raw-zero/nonfinite handling are reused unchanged. Prefix SHA256 was streamed once for the three-cache invocation, taking49.669 seconds, with no full-upstream integrity claim.

Each cache contains **5,135 original Train / 1,155 original Val rows**. Exact ordered ID hashes, finite FP32 values, both task support counts and row-by-row target arrays are checked after writing. Train/Val IDs are disjoint. Every parameter has gradients disabled; all layers remain eval and full original weight/BN fingerprints before/after export match. `FrozenContext.train(True)` still keeps eval mode. The first batch of each split per encoder was encoded twice with exact equality; this is a finite repeat probe, not independent re-export of every cached pixel.

Only the original frozen subset indices reach pixel reads. All original Test rows remain outside the cache and no Test scores were computed. Full source prefix is authenticated locally as `77f9e3dbb31601182557308717c705c6efbf8be29eeed3c2920069d158d5adf8`; upstream74,449,119,347-byte integrity remains unverified. Data retain the northern availability-selection limitation and previously reused Val. Geology-supervised features give Age correlated-task supervision shared by all new comparison heads; this is not foundation-model pretraining.

Eight focused context tests cover shape, pooling, frozen parameters/BN, repeatability, invalid inputs, finite values, wrong IDs/support and Test rejection. They are included in the final **626-test** P3.2 repository suite. The first two export attempts failed before pixel export on provenance-schema mismatches and are documented in [implementation evidence](evidence/p32_20261105/implementation/verification.json). The corrected third invocation completed; no old run/data file was overwritten.

Actual invocation, using the existing trusted runtime and local audited metadata:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
TMPDIR="$JOB_TMP" python3 -I "$JOB_TMP/p31_python.py" \
  scripts/p32_cache_context.py \
  --data outputs/p31/assets/moonstone_53099efd/lunar_patches_v4.h5.partial \
  --metadata "$LOCAL_AUDITED_METADATA" \
  --range-report outputs/p31/loader_pair_01/report.json \
  --runs-root outputs/p31/local_prefix_baselines_01 \
  --outdir outputs/p32/context_03
```

The output directory is exclusive; do not rerun into it. Original P3.1 source/configuration/evidence is unchanged. Raw features, weights and private paths stay local; only source and sanitized provenance/results are committed. No new data/weights, package installation, shared-environment replacement, paid resources or four-GPU work occurred.
