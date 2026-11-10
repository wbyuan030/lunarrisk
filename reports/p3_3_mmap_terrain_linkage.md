# P3.3 terrain product identity: fixed Train/Val cohort equals the published mmap

2026-11-10. **`TERRAIN_MMAP_LINKAGE_PASS`:** every fixed Train/Val elevation and slope pixel equals the published `pretraining/mmap` arrays of revision `53099efd` bit for bit after the published z-score. This establishes product identity for the cohort, not an independent re-execution of the raw LOLA/SLDEM ingestion, and it does not train, rank, score or read Test.

## What was actually compared

The pinned builders define the relation being tested: [`step04_tile.py`](https://github.com/ayushprd/Moonstone/blob/cbd4720653eb05afd975d2a36280858c5c56a275/data_preparation/step04_tile.py) writes HDF5 channel 1 from `aligned/elevation_blended.tif` and channel 2 from `derived/slope.tif` by rasterio window in physical units, and [`step15_build_mmap.py`](https://github.com/ayushprd/Moonstone/blob/cbd4720653eb05afd975d2a36280858c5c56a275/data_preparation/step15_build_mmap.py) converts the same GeoTIFF grid to headerless float32 with the published z-score, NaN-to-zero and `packbits(~isfinite)`. The new [`p33_audit_mmap_linkage.py`](../scripts/p33_audit_mmap_linkage.py) re-derives that z-score from the retained HDF5 physical values and compares the full 256×256 window with the published array at the authenticated `row_idx`/`col_idx`.

| Split | Rows | Pixels per channel | Bitwise exact | Beyond tolerance | Max abs difference | Declared-NaN bits |
|---|---:|---:|---:|---:|---:|---:|
| Train | 5,135 | 336,527,360 | 336,527,360 | 0 | 0.0 | 0 |
| Val | 1,155 | 75,694,080 | 75,694,080 | 0 | 0.0 | 0 |

Both channels are counted separately; 824,442,880 terrain pixels were compared in total. The pre-registered tolerance `rtol=1e-5, atol=1e-6` was never needed: the maximum absolute difference is exactly zero. The packed mask files contain no set bit anywhere on the grid, so the publisher declares no non-finite elevation or slope pixel at all; `slope_mask.npy` is byte-identical to `elevation_mask.npy` at this revision.

## Inputs, mapping and identity checks

- Published assets were downloaded with the proxy explicitly disabled from `hf-mirror.com`, three transfers with zero retries, and each SHA256 matched the declared LFS digest: `elevation.npy` `2013f78a…ffbef` (4,246,732,800 bytes), `slope.npy` `38c84594…d4b8` (4,246,732,800 bytes), shared mask `1c09aaba…a7511` (132,710,528 bytes); 8,626,176,128 bytes total. The aligned `elevation_blended.tif` (4.39 GB) and the unpublished `derived/slope.tif` were not acquired.
- The cohort authority is unchanged: the pinned P31 plan, its ordered-ID hashes, the pinned metadata archive and the retained 33,964,443,553-byte prefix all re-authenticated before the scan.
- The historical metadata export at SHA256 `3b3ed8ca…a98d` is **not present locally** any more. The audit therefore read the HDF5 `metadata` group through the guarded prefix adapter and required exact agreement with the pinned archive on 16,200 rows - patch IDs, stored split, dominant unit ID and name, `row_idx`, `col_idx`, and centre latitude/longitude within half a pixel - before using that table. The rebuilt export hash is recorded, not passed off as the lost file.
- Each of the 6,290 selected tiles is unique, Train/Val IDs do not overlap, and the ordered-ID digests recompute to the frozen plan values. Tiles of Train/Val are disjoint from Test tiles, so no held-out pixel was read.

## What this does and does not establish

Established: the fixed Train/Val terrain channels are the published product at the fixed revision, with the publisher's own NaN declaration, and no additional filtering, imputation or sentinel substitution is needed to reach that product. The earlier `finite_only_no_finite_sentinels`-style uncertainty is now bounded by product identity: any remaining anomaly is one the publisher shipped, not one introduced downstream.

Not established, and not claimed: an independent re-execution of the raw LOLA/SLDEM ingestion, the actual GDAL/rasterio versions, per-stage intermediate headers, intermediate reuse, or a byte comparison against the aligned GeoTIFFs themselves. `derived/slope.tif` is not published, so the published mmap channel is the only available form of that derived product. Finite values may still be bilinear-interpolated or produced by [`step02_align.py`](https://github.com/ayushprd/Moonstone/blob/cbd4720653eb05afd975d2a36280858c5c56a275/data_preparation/step02_align.py)'s SLDEM/LOLA blend (SLDEM inside 58°, LOLA outside 62°, linear transition, LOLA where SLDEM is non-finite). Git-repository channel statistics remain a diagnostic path only.

## Execution and cost

Private Python 3.12.9 / NumPy 2.4.4, `-I -B`, inside `unshare --user --map-root-user --net`; torch was not imported. Elapsed 226.8 s, process max RSS 4,224,768 KiB, no model, no optimizer step, no budget lock, no score, no Test pixel. Three earlier aborted launches in this phase were engineering iterations before the first successful comparison and produced no evidence artifacts; only the successful run is archived.

Verification: 91 focused P33 tests PASS (13 new), zero skips, 5.8 s; `compileall` and `git diff --check` PASS. The full suite ran 721 tests in 15.9 s with 714 PASS and 7 legacy failures that this change does not touch: `test_posterior_rules_integration` and `test_build_manifest` break on NumPy 2.4 float conversion and pandas NaN semantics (the same manifest test passes under the host NumPy 1.26 environment), and the LROC DTM/EDR modules fail on import for missing optional dependencies in the private interpreter. Those are environment artifacts of the isolated runtime, not results of this audit, and they were not fixed.

Evidence: [summary](evidence/p33_20261110/mmap_linkage_01/summary.json), [Train rows](evidence/p33_20261110/mmap_linkage_01/train_rows.csv), [Val rows](evidence/p33_20261110/mmap_linkage_01/val_rows.csv), [acquisition receipt](evidence/p33_20261110/mmap_linkage_01/acquisition.json), [metadata export](evidence/p33_20261110/mmap_linkage_01/metadata_export.json) and [verification](evidence/p33_20261110/mmap_linkage_01/verification.json) with an independent standard-library recount of every CSV total.

Next: this removes product identity from the provenance gate. What remains is a scientific decision, not more pixel work - whether to run the paired pretrained-versus-random experiment under a protocol that explicitly records the residual upstream-ingestion limit, and with which frozen actual budget. That needs explicit review before any score; training, the actual budget and Test all remain unstarted.
