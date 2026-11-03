# P3.1 loader pair — offline lock and bounded runner

Date: 2026-11-03. **Offline implementation and selection complete; real full-crop I/O NOT EXECUTED.** No network requests, downloads, environment installations, commits or changes to classifier/candidate files were made for this task. Existing `http_range.py` is unchanged. This report does not certify real loader equivalence.

## Locked selection

The input is the existing real HDF5 inspection containing all 16,200 metadata records. Its SHA256 is `3b3ed8cade200f79f720beeba6a666365d66383dbb623337cd96ebec31a0a98d`. File paths on private machines are excluded. The plan also hashes the archived listing, HF statistics, 49-class map and all five runtime source files. Its canonical content lock is `672ae6d64401390fe79c3131b6a25b7b41f0aed322e476fb9dbbdacca828184c`.

Selection uses coverage metadata, not model scores, errors, target class or accuracy. Require one stored Train and one stored Val crop; each must contain every channel available anywhere in Train/Val, have absolute latitude ≤60°, and have strictly partial M3 coverage. Choose the lowest numeric patch ID satisfying those conditions separately in each split. Two crops are the minimum possible while exercising both splits. Fail offline if this two-crop solution is unavailable; do not silently expand acquisition.

| Patch ID / original index | Stored split | Tile row, col | Latitude | Geology ID / target | Name | Age target | M3 valid fraction |
|---|---|---|---|---|---|---|---|
| 2705 | Train (0) | 15, 5 | 59° | 44 / 43 | Nt | 1 | 0.7902679443359375 |
| 2700 | Val (1) | 15, 0 | 59° | 41 / 40 | Nc | 1 | 0.9296417236328125 |

Both have 27/28 metadata channels present, covering all seven groups, including both radar channels. **Roughness has zero present samples in all 11,400 Train and all 2,400 Val records.** Its real valid-data normalization cannot be certified by any Train/Val selection under this metadata; the plan retains it in the complete 28-channel comparison and explicitly records this coverage limit. No Test pixel is selected. The metadata `valid_fraction` field is zero for both chosen records despite 27 present channels; it is preserved as a source diagnostic and is not treated as a proxy for complete absence.

Finite raw-zero coverage cannot be inferred from `has_channel` or M3 fraction. It remains **UNKNOWN before pixels**. The runner counts finite raw zeros explicitly and requires at least one in the two selected crops to pass coverage. If none exists, the run fails coverage and stops; it does not adaptively fetch more patches. Likewise, metadata presence is a selection prior, not proof that every selected channel has actual valid normalized pixels; each split must demonstrate all 27 available channels and all seven groups after I/O.

## Prelocked source, comparison and transfer

Data revision: `ayushprd/Moonstone@53099efd0f2db81eee96b28eae87e1efdd281310`; audited code commit: `cbd4720653eb05afd975d2a36280858c5c56a275`. The plan allowlists exactly the benchmark HDF5 and 28 named mmap files with pinned URLs, sizes and upstream LFS SHA256 claims. Whole remote files are **not** downloaded or claimed to have been independently hashed.

Compare two complete `28×256×256` crops: **3,670,016 pixel positions**, including all missing positions. Fix `rtol=1e-5`, `atol=1e-6`; compare exact full finite/nonzero normalized masks separately from numeric tolerance. Compare channel ANY and group ANY presence, and verify original index/ID, stored split, tile coordinates, geology ID/name and metadata presence against the remote HDF5 before extraction. Labels remain outside pixels and are never selection scores or features. The fixed HF statistics are the tested route; Git statistics are not substituted after observing errors. Finite raw zeros retain their existing normalization semantics; raw nonfinite values return to zero; radar is not logged a second time.

For each sample and channel, read all rows `y=0..255` with:

```text
start = ((tile_row * 256 + y) * 46080 + tile_col * 256) * 4
length = 1024 bytes
row stride = 184320 bytes
```

The JSON plan stores 56 explicit first-row runs with row stride/count/length and a SHA256 over all 14,336 expanded request descriptors. It never merges across the unselected columns between these two crops, including their different splits. Files are headerless little-endian float32 of shape `(23040,46080)`, exactly 4,246,732,800 bytes each. Bounds and sizes are checked before requests.

| Budget item | Locked amount |
|---|---:|
| Existing actual body total (unchanged by this task) | 78,502,175 bytes |
| Existing conservative cumulative total | 83,468,143 bytes |
| Original cumulative initial cap, NOT reset | 1,073,741,824 bytes |
| Exact mmap requests | 14,336 |
| Exact mmap response bodies | 14,680,064 bytes (14 MiB) |
| HDF5 extraction body ceiling | 67,108,864 bytes (64 MiB) |
| HDF5 range-attempt ceiling | 2,048 |
| Total range-attempt upper bound | 16,384 |
| New body upper bound | 81,788,928 bytes (78 MiB) |
| Cumulative conservative upper bound from present ledger | 165,257,071 bytes |
| Remaining cap at that upper bound | 908,484,753 bytes |
| Cache file count upper bound, including raw crops and report | 16,387 |
| Cached binary payload upper bound, excluding NPY headers/JSON/filesystem overhead | 96,468,992 bytes (92 MiB) |

**Exact HDF5 compressed bytes, read offsets, request/cache counts and elapsed time are UNKNOWN before apply.** HDF5 metadata traversal discovers its chunk offsets. This is an explicit bounded dynamic part of the plan, not a claim that exact HDF5 ranges have already been discovered. If strict advance enumeration of every HDF5 physical range is required, that remains blocked on a separately executed, bounded discovery step; the present implementation locks the selected chunk coordinates, file/revision, exact-read policy and hard ceilings instead. The mmap ranges and body count are fully enumerated offline.

HDF5 uses an unbuffered `ExactRangeFile` with the existing strict `range_get`, instead of rounding reads into cache blocks. h5py reads only the two selected single-patch compressed chunks plus necessary file/metadata bytes. The adapter requests exactly the length h5py asks for, with no deliberate read-ahead or gap overfetch. A real h5py synthetic chunk-layout test verified that requests for selected chunks do not overlap a designated unselected Test pixel chunk. Actual HDF5 ranges, including repeated metadata reads, are cached and hashed at execution for audit. No claim is made that metadata for other IDs is inaccessible: metadata was already available for all 16,200 records; the restriction is on unselected pixel chunks and model use of targets.

Four mmap workers each receive their entire nonoverlapping lane's byte allocation **before any worker starts**: 3,670,016 bytes and 3,584 range requests per worker. HDF5 has its separate 64 MiB reservation, so no mutable budget is shared across threads. Each worker uses a persistent requests Session with retries disabled, streaming bodies and manually bounded redirects. Every response still passes the existing HTTP 206, exact Content-Range, Content-Length and encoding checks before its body is read. Redirect bodies are closed without reading; at most eight redirects are followed per range, so HTTP/header round trips can substantially exceed the range-attempt count. There are no automatic retries, gap coalescing or resume/replay requests.

Returned body counts and elapsed time are reported. For conservative cumulative accounting the runner charges **the full 81,788,928-byte reservation even on failure**, because a transport may consume some bytes before raising without returning them to the helper. Unused allocation must not be reclaimed without a separate audit. If other authorized downloads happen first, pass the updated higher conservative ledger via `--prior-transfer-bytes`; lower values than 83,468,143 are rejected. The runner rechecks the cumulative cap.

No throughput measurement exists for this request pattern, and no completion time is promised. One range body per cache file creates substantial filesystem overhead: mmap payload alone is 14 MiB but 14,336 files can occupy about 56 MiB with 4 KiB allocation units. Reserve at least **256 MiB free disk/inodes** for this bounded run and its JSON report; this is a conservative planning allowance, not a measured peak. The runner refuses an existing output directory and never overwrites older evidence.

## Implementation and commands

Owned files:

- `scripts/p31_loader_pair.py`: default offline dry-run, separate offline lock creation and explicit `--apply`.
- `src/lunarrisk/benchmarks/p31_pair.py`: selection, exact rows, full comparisons, isolated worker reservations and session adapter.
- `tests/test_p31_loader_pair.py`: synthetic coverage, mask/layout/budget and real h5py in-memory adapter checks.
- `protocols/p31_loader_pair_plan.json`: locked records, assets, row runs, hashes and ceilings.
- This report.

From a current source checkout, using variables for private paths:

```bash
python3 -I scripts/p31_loader_pair.py \
  --metadata "$P31_METADATA_JSON"

# Explicit bounded execution, for the authorized executor; NOT run for this delivery.
python3 -I scripts/p31_loader_pair.py \
  --metadata "$P31_METADATA_JSON" \
  --prior-transfer-bytes 83468143 \
  --outdir "$P31_FRESH_OUTDIR" \
  --apply
```

The default listing/stat/map/plan paths resolve relative to the script's checkout. Runtime requires existing h5py, NumPy and requests. The host currently lacks h5py; use the existing ROCm Python with **the current source and locked plan/evidence staged consistently**, rather than invoking an old container copy. No package installation or shared environment replacement is needed. Source/input hash mismatch rejects apply before network access. To create a genuinely new lock, `--write-plan --plan <new-path>` is an offline-only invocation; it cannot be combined with apply and refuses to overwrite an existing plan.

Execution creates `report.json`, exact HDF5/mmap range caches, and two raw `.npy` crops in the fresh output directory. It records each successful range's offset, length, cache name and SHA256; raw/source/input hashes; per-channel valid/missing/raw-zero/nonfinite counts and max error; full mask disagreement counts; channel/group presence; labels/split/IDs; attempts, body accounting and timing. A transport or ceiling failure produces BLOCKED evidence with an error type only, excluding signed URLs/proxy details/private paths. Only complete numeric/mask **and** coverage acceptance yields `VALIDATED_WITHIN_SELECTED_SAMPLE_COVERAGE`; this is not a worldwide or every-channel-valid-data certificate.

## Executed validation

- `python3 -m unittest discover -s tests -p 'test_p31_loader_pair.py' -v`: **19 tests discovered; 18 PASS, 1 host h5py dependency skip**, 0.156 s in the recorded run.
- The skipped integration test was run using the **existing container's h5py and only in-memory buffers**, loading the current module/test source through stdin: **1 PASS**, 0.130 s. It reads selected synthetic compressed chunks and verifies no range overlaps the designated Test chunk; no files or network are used. An earlier attempt to run all tests inside the container was rejected by automatic approval review because the digest test creates temporary files; that action did not execute, and the narrowed in-memory check succeeded.
- Default dry-run rebuilt the plan from real local metadata, compared all locked inputs/source hashes and exited successfully with **zero network transfer**. `--write-plan` was also exercised offline for initial creation; the final lock includes runtime source hashes.
- `python3 -m compileall -q scripts src tests`: **PASS**.
- Full host `python3 -m unittest discover -s tests -v` at the time of the run: **264 tests, 20 errors, 9 skips**; errors arise from missing host dependencies including torch, tifffile, shapefile, xarray and pyproj. This is not a clean full-suite result and does not replace the targeted results. No environment was changed to address it.

Real selected HDF5 extraction, full mmap I/O, physical source-range audit and raw-zero coverage confirmation remain **NOT EXECUTED**, deliberately left to the executor. Existing P3.0 partial-row matches and all-missing roughness agreement cannot be relabeled as this acceptance result. No scientific accuracy, pretrained benefit or loader PASS is reported here.
