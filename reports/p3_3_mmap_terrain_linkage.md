# P3.3 terrain product identity: fixed Train/Val cohort equals the published mmap

2026-11-10. **`TERRAIN_MMAP_LINKAGE_PASS`:** every fixed Train/Val elevation and slope pixel has an identical float32 bit pattern to the published `pretraining/mmap` window after the published z-score, and no terrain pixel in those windows is non-finite in either source. This establishes product identity for the cohort, not an independent re-execution of the raw LOLA/SLDEM ingestion, and it does not train, rank or score anything.

## What was actually compared

The pinned builders define the relation being tested: [`step04_tile.py`](https://github.com/ayushprd/Moonstone/blob/cbd4720653eb05afd975d2a36280858c5c56a275/data_preparation/step04_tile.py) writes HDF5 channel 1 from `aligned/elevation_blended.tif` and channel 2 from `derived/slope.tif` by rasterio window in physical units, and [`step15_build_mmap.py`](https://github.com/ayushprd/Moonstone/blob/cbd4720653eb05afd975d2a36280858c5c56a275/data_preparation/step15_build_mmap.py) converts the same GeoTIFF grid to headerless float32 with the published z-score, NaN-to-zero and `packbits(~isfinite)`. The new [`p33_audit_mmap_linkage.py`](../scripts/p33_audit_mmap_linkage.py) re-derives that z-score from the retained HDF5 physical values and compares the full 256×256 window with the published array at the authenticated `row_idx`/`col_idx`.

| Split | Rows | Pixels per channel | Bitwise identical | Numerically equal | Beyond tolerance | Max abs difference | Raw non-finite | Declared-NaN bits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 5,135 | 336,527,360 | 336,527,360 | 336,527,360 | 0 | 0.0 | 0 | 0 |
| Val | 1,155 | 75,694,080 | 75,694,080 | 75,694,080 | 0 | 0.0 | 0 | 0 |

Both channels are counted separately; 824,442,880 terrain pixels were compared in total, 151,388,160 of them in Val. The pre-registered tolerance `rtol=1e-5, atol=1e-6` was never needed: the maximum absolute difference is exactly zero, and every finite float32 payload matched bit for bit (`bitwise_exact` equals the pixel count in both channels). The packed masks contain no set bit anywhere on the grid - 0 nonzero bytes and 0 set bits for both channels - so the publisher declares no non-finite elevation or slope pixel at all. The two published masks are byte-identical at this revision, and each was downloaded and SHA256-verified separately so that the agreement check is not one local file compared with itself.

## Inputs, mapping and identity checks

- Published assets were downloaded with the proxy explicitly disabled from `hf-mirror.com`, four transfers with zero retries, and each SHA256 matched its declared digest: `elevation.npy` `2013f78a…ffbef` and `slope.npy` `38c84594…d4b8` (4,246,732,800 bytes each), `elevation_mask.npy` and `slope_mask.npy` `1c09aaba…a7511` (132,710,528 bytes each); 8,758,886,656 bytes total. `channel_stats.json` and `channel_index.json` were separately digest-checked, and the index declares the fixed grid shape with `normalized: true`. The aligned `elevation_blended.tif` (4.39 GB) and the unpublished `derived/slope.tif` were not acquired.
- The cohort authority is unchanged: the pinned P31 plan, its ordered-ID hashes, the pinned metadata archive and the retained 33,964,443,553-byte prefix all re-authenticated before the scan, and every published asset was re-hashed again after it.
- The historical metadata export at SHA256 `3b3ed8ca…a98d` is **not present locally** any more. The audit therefore read the HDF5 `metadata` group through the guarded prefix adapter and required exact agreement with the pinned archive on 16,200 rows - patch IDs, stored split, dominant unit ID and name, `row_idx`, `col_idx`, and centre latitude/longitude within half a pixel - before using that table. The rebuilt export hash is recorded, not passed off as the lost file.
- Tile policy is now mechanically enforced: the metadata declares 16,200 unique tiles (13,800 Train/Val, 2,400 Test), the 6,290 selected tiles are unique, inside the Train/Val set, and their intersection with the Test tile set is empty.

## What this does and does not establish

Established: the fixed Train/Val terrain channels are the published product at the fixed revision, bit for bit, with the publisher's own NaN declaration and no downstream filtering, imputation or sentinel substitution. Any remaining anomaly is one the publisher shipped, not one introduced by this pipeline.

Not established, and not claimed: an independent re-execution of the raw LOLA/SLDEM ingestion, the actual GDAL/rasterio versions, per-stage intermediate headers or intermediate reuse, and a byte comparison against the GeoTIFFs themselves. `derived/slope.tif` is not published, so the published mmap channel is the only available form of that derived product. Finite values may still be bilinear-interpolated or produced by [`step02_align.py`](https://github.com/ayushprd/Moonstone/blob/cbd4720653eb05afd975d2a36280858c5c56a275/data_preparation/step02_align.py)'s SLDEM/LOLA blend (SLDEM inside 58°, LOLA outside 62°, linear transition, LOLA where SLDEM is non-finite). Git-repository channel statistics remain a diagnostic path only.

Test scope is stated precisely: no Test tile was selected, compared or scored, and `test_tiles_compared` is 0. Whole-file SHA256 of the 4.2 GB arrays and the global packed-mask count necessarily stream bytes that cover every latitude, Test geography included; those bytes are hashed or counted, never decoded into per-tile values for Test. Do not describe this phase as "zero Test bytes received".

## Execution and cost

Private Python 3.12.9 / NumPy 2.4.4, `-I -B`, inside `unshare --user --map-root-user --net`; torch was not imported. Elapsed 236.3 s, process max RSS 4,236,596 KiB, no model, no optimizer step, no budget lock, no score. Earlier aborted launches in this phase were engineering iterations before the accepted run and produced no evidence artifacts; only the final run is archived, and the weaker first receipt is explicitly superseded in the verification file.

An independent review (see [the review report](p3_3_mmap_linkage_review.md)) of the first receipt found four real weaknesses - the "zero Test bytes" wording, value equality standing in for bit identity, a mask-agreement check that compared one file with itself, and a byte count labelled as a bit count. All four were fixed before this re-execution: the `bitwise_exact` comparison, the separate mask download, the byte/bit split, the raw non-finite count, the tile-disjointness checks and the corrected Test wording are visible in the archived evidence. The review retained the first receipt unchanged and added synthetic tests; this receipt is the re-execution it asked for.

Verification: 23 audit tests PASS and the focused P33 suite runs 101 PASS with zero skips; the full suite runs 727 tests with 720 PASS and the same 7 legacy NumPy-2.4/pandas and missing-`shapefile` failures recorded on the previous receipt. `compileall` PASS and `git diff --check` PASS for source, Markdown and JSON; the CRLF evidence CSVs are flagged as trailing whitespace exactly like the already committed 2026-11-07 `rows.csv`.

Evidence: [summary](evidence/p33_20261110/mmap_linkage_01/summary.json), [Train rows](evidence/p33_20261110/mmap_linkage_01/train_rows.csv), [Val rows](evidence/p33_20261110/mmap_linkage_01/val_rows.csv), [acquisition receipt](evidence/p33_20261110/mmap_linkage_01/acquisition.json), [metadata export](evidence/p33_20261110/mmap_linkage_01/metadata_export.json) and [verification](evidence/p33_20261110/mmap_linkage_01/verification.json) with an independent standard-library recount of every CSV total.

Next: product identity is no longer part of the provenance gate. What remains is a scientific decision - whether to run the paired pretrained-versus-random experiment under a protocol that explicitly records the residual upstream-ingestion limit, and with which frozen actual budget. That needs explicit review before any score; training, the actual budget and Test all remain unstarted.
