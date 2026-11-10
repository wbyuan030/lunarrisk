# Terrain source-validity follow-up

Read-only research on 2026-11-07, following [the completed Train/Val numerical diagnostic](p3_3_cohort_terrain_diagnostic.md) at ee3945effab20182cf69cffdfe2895c0adf8faa2. No upstream source code execution, raster acquisition, user-computer access, Test reads, installation, or training.

**Result:** the diagnostic records are consistent, but accepted full-cohort source validity remains unestablished. Do not repeat the completed scan, assert `finite_only_no_finite_sentinels`, lock an actual budget or start training on the strength of this follow-up. [Structured public evidence](evidence/p33_20261107/source_validity_01/public_evidence.json).

**Later boundary update (2026-11-07, 9c292b0):** [Subsequent requests intended to retrieve headers also received raster strips](p3_3_aligned_header_observations.md). Further raster requests are stopped. The earlier window-comparison suggestions in this report and its linked public-evidence receipt are historical, not current next steps or permission. Do not issue raster HEAD/range/window requests or decode received strips to determine geographic coverage. Continue only with existing text records, source code and standalone text metadata; changing this boundary or contacting the producer requires separate user authorization. No source-validity guard, budget or Test-scoring restriction is relaxed.

## Independent committed-record check

A standard-library review verified nine recorded file/source SHA256 values, all 6,290 CSV rows against frozen-plan ordered IDs and source indices, target class totals, and every per-channel count and extremum. Six source-recheck entries match the earlier pinned-source audit. [Review receipt](evidence/p33_20261107/source_validity_01/record_consistency_review.json). This checks the published records, not a second pixel scan. Per-row targets were not independently compared against the metadata archive in this review; the local agent recorded that separate check. The 78-test result remains the local agent's execution record, not a rerun here.

## Newly checked primary evidence

The exact upstream label [SLDEM2015_128_60S_60N_000_360_FLOAT.LBL](https://imbrium.mit.edu/DATA/SLDEM2015/GLOBAL/FLOAT_IMG/SLDEM2015_128_60S_60N_000_360_FLOAT.LBL) was read using a 16,384-byte response cap. Actual content: 4,588 bytes; SHA256 `948d15d1a52db229521a6dfad64494c0f3dc437a55f048440f65f5d19b5a7493`. The digest was calculated during the single bounded read; the original bytes were not retained for an independent digest check. No IMG was requested. This is the current publicly served label, not proof of the bytes used in Moonstone's historical build.

It identifies product V2.0; 15,360 lines x 46,080 samples; 32-bit PC_REAL; height in kilometres relative to the 1737.4 km reference sphere. Its HEIGHT formula uses DN times SCALING_FACTOR=1; OFFSET=1737.4 is for planetary radius, not an extra height correction. No MISSING_CONSTANT, INVALID_CONSTANT or NULL declaration appears. Lack of such a declaration does not establish that every historical pixel is valid.

Its structured MAP_RESOLUTION=128 pix/deg and MAP_SCALE=0.236901 km/pix disagree with a prose description of approximately 60 metres. The structured grid supports approximately 236.9 m equatorial spacing, not a 60 m equivalence claim. This reinforces the existing native256 scale caveat.

## Specific pipeline checks that remain unresolved

Pinned source: [Moonstone cbd4720653eb05afd975d2a36280858c5c56a275](https://github.com/ayushprd/Moonstone/tree/cbd4720653eb05afd975d2a36280858c5c56a275).

1. [step02_align.py lines 144-161](https://github.com/ayushprd/Moonstone/blob/cbd4720653eb05afd975d2a36280858c5c56a275/data_preparation/step02_align.py#L144-L161) copies the source profile, maps equality with src.nodata to NaN, and writes float32 pixels. It does not read a separate validity mask or explicitly change the copied nodata metadata to NaN. If a source uses an external/internal validity mask or incomplete NoData metadata, this function alone cannot establish its handling. This is a conditional code-path concern, not evidence of an observed data defect.
2. [step02_align.py lines 188-228](https://github.com/ayushprd/Moonstone/blob/cbd4720653eb05afd975d2a36280858c5c56a275/data_preparation/step02_align.py#L188-L228) uses the scaled file's src.nodata in reprojection, while destination NoData is NaN. Actual source/scaled metadata and the GDAL/rasterio version therefore matter. The scaling step reads band 1 without `masked=True`; the source contains no explicit `read_masks`, `write_mask`, mask-flag or alpha handling. Reprojection passes `rasterio.band(src,1)` to GDAL, so this static review does not establish whether GDAL internally honors a particular input mask or alpha band. Do not claim masks were definitely lost. The scaling writer does not explicitly assign destination `scales` or `offsets`; actual metadata must be observed rather than inferred from a copied profile.
3. [step02_align.py lines 294-338](https://github.com/ayushprd/Moonstone/blob/cbd4720653eb05afd975d2a36280858c5c56a275/data_preparation/step02_align.py#L294-L338) reuses nonempty scaled intermediates. Such existence is not content identity or evidence of the generating code/version.
4. [step03_derive.py lines 246-250](https://github.com/ayushprd/Moonstone/blob/cbd4720653eb05afd975d2a36280858c5c56a275/data_preparation/step03_derive.py#L246-L250) chooses the aligned LOLA-only DEM if the blended DEM file is absent. Record the actual slope input, rather than assuming every slope came from a blend. This branch was not independently observed in the historical build.
5. [step03_derive.py lines 69-100](https://github.com/ayushprd/Moonstone/blob/cbd4720653eb05afd975d2a36280858c5c56a275/data_preparation/step03_derive.py#L69-L100) computes slopes from gradients and arctangent. Finite contaminated elevations can still produce slopes in [0,90]. Thus the new range scan is valuable numerical eligibility evidence but does not independently prove source validity.

Scaling, interpolation and blending can change a finite missing-value code into many other finite values. Searching four exact terminal values cannot rule out those cases. Do not mask observed zero values, add more candidate equality tests as proof, or assert `finite_only_no_finite_sentinels` from this research.

## Smallest useful next local-agent receipt

First inventory existing provenance records only, without rescanning pixels or acquiring raw products. Report present/absent for historical labels, source receipts, converted/scaled/aligned DEM headers, slope header, and build/reuse logs. Hash only small labels, exported header records and existing receipts in this inventory. For large original/intermediate rasters, reuse an existing authenticated hash receipt; do not read the whole raster merely to create a new hash. Record product/version identity, dimensions/CRS/transform, dtype, nodata, scale/offset, mask flags and mask provenance, library versions, and whether a cached intermediate was reused. Distinguish observed headers from reconstructed expectations.

Tie source and intermediate identities to the actual HDF5 lineage: raw/converted -> scaled -> aligned -> blended (or documented LOLA-only fallback) -> slope -> V1 -> V2 -> V4. A current public label or source checkout is not that linkage. If source/headers/logs do not exist locally, report that precise gap; do not replace it with a blind all-valid assertion or download multi-GB rasters.

If authentic records exist, evaluate them before requesting further pixel work. A subsequent bounded comparison could use already-held fixed Train/Val windows and their needed gradient/reprojection neighbourhoods, but must be separately scoped and must not accidentally read Test pixels through spatial halos. A small spot check can detect a mismatch, not certify all 6,290 samples. Full scientific acceptance needs evidence covering the cohort, or an explicitly different downstream protocol with clearly limited claims; the current guard must remain unchanged.

## Access and verification limits

GitHub pinned source and the exact MIT label were read successfully. Web rendering rejected the label's octet-stream content type; bounded direct text retrieval succeeded. The USGS product page open returned HTTP 403 and was not retried or bypassed. A search snippet alone was not used to establish its NoData value. No original raster header, historical GDAL execution, or per-pixel lineage was independently verified here.
