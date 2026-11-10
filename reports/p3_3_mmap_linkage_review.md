# P3.3 mmap linkage: source/text review and numerical-claim correction

2026-11-10. Reviewed main at `b709e9b21acd00b9492de1afef6fa04240983833`, which integrates local audit `e231448a` and prior handoff `d83ddbfe`. This review used repository source and small archived text evidence in cloud scratch. It did not access the user's computer, obtain mmap/HDF5/raster assets, inspect real pixels, rerun the linkage or train a model.

## Result retained, claim narrowed

The existing Train/Val CSVs independently recount to 5,135 / 1,155 rows, respectively. For each of elevation and slope, Train has 336,527,360 numerical exact matches and Val has 75,694,080. All 824,442,880 channel pixels compare numerically equal, zero are beyond tolerance, and maximum absolute difference is 0.0. Ordered-ID digests, unique row IDs/tiles, CSV bytes/hashes and summary hash match the archived receipt. Train and Val IDs do not overlap.

The archived script's `exact` used floating-point `==`, so the original “bit for bit” wording was unsupported: +0.0 and -0.0 compare equal despite different bits. The result establishes numerical equality after the published normalization. Bitwise equality on the actual windows was not measured, and no new pixel comparison is needed merely to strengthen that wording.

The new and 2026-11-07 diagnostic summaries contain the same raw-terrain ordered hashes for both splits. That binds this comparison to the already diagnosed all-finite terrain; reuse that evidence rather than rescanning it. Product identity remains distinct from independently reproducing the historical LOLA/SLDEM ingestion, intermediate reuse, or proving absence of every undocumented finite sentinel.

## Small implementation correction

- `mask_nonzero` previously counted nonzero packed bytes, although its API and row field called them bits. One byte with eight bits set incorrectly returned 1. It now unpacks only the selected tile's mask bytes and counts individual set bits. The archived masks were all zero, so their zero finding is unchanged. The separate global field is correctly named `declared_nan_bytes_global` and still counts bytes.
- `compare` keeps `exact` as numerical equality and adds a distinct `bitwise_exact` finite-float32 counter, including row/aggregate output. Signed zeros remain numerical matches but only identical bit patterns count as bitwise matches. Equal infinities, like NaNs, cannot count as valid terrain matches.
- Six new synthetic tests cover multi-bit bytes, adjacent-tile exclusion, signed zeros, nonfinite pairs, noncontiguous windows and representably different values within tolerance. This is verification of future code behavior, not a new result for real pixels.

The historical `mmap_linkage_01` evidence and pre-comparison linkage plan are unchanged. Their source hashes still identify the original script at the reviewed commit. Current handoff/report/transfer-plan wording points here; archived claims do not override this correction. The original [upstream step15](https://github.com/ayushprd/Moonstone/blob/cbd4720653eb05afd975d2a36280858c5c56a275/data_preparation/step15_build_mmap.py) computes `nan_mask` before replacing missing values, then stores that retained mask after normalization. Saving it later does not recompute the mask from zero-filled data.

## Acquisition and held-out scope

The historical acquisition receipt records two complete global value arrays plus a global packed mask, totaling 8,626,176,128 bytes. The original script also counted nonzero mask bytes over the whole grid. Consequently, “no Test pixel was requested/read” cannot describe the entire acquisition/inspection history: the global products cover Test geography. The code restricts numerical value-window comparisons to Train/Val, and no Test task score was produced. This distinction does not authorize inspecting more held-out data or repeating downloads. Test scoring stays HOLD.

## Verification and next gate

On Python 3.12.14 / NumPy 2.3.5, all **19 focused synthetic tests passed, zero skips**. Compilation passed for the materialized script/test and seven package source files; diff whitespace checks passed. An independent source review found no blocking defect. The full repository suite and old P33/preflight tests were not rerun. The local audit report's earlier 714-pass/7-failure full-suite run remains a mixed result, not a clean full-suite pass.

Evidence: [new verification receipt](evidence/p33_20261110/mmap_linkage_review_01/verification.json) and [synthetic test log](evidence/p33_20261110/mmap_linkage_review_01/synthetic_tests.log). No actual budget was frozen. The safe next step remains scientific review of a separately declared provenance contract that explicitly records upstream limits, then an explicit paired budget before separately authorized training. Existing source-validity guards are unchanged; do not silently claim `finite_only_no_finite_sentinels`, repeat the completed scans/preflight, resume large acquisition, or unlock Test.

## Follow-up (same day, after this review)

This review deliberately did not re-execute. A re-execution under the strengthened predicate was nevertheless run locally on the already-held arrays and is archived at [mmap_linkage_01](evidence/p33_20261110/mmap_linkage_01/verification.json): `bitwise_exact` equals the pixel count for both channels in both splits (412,221,440 pixels per channel), no finite pixel differs in its float32 bit pattern, raw non-finite pixels are zero, and both published masks were downloaded and verified separately. It used no new source raster and read no Test tile, so the conclusions above stand and the bitwise wording is now backed by measurement rather than inference. The review's synthetic edge cases remain in the test file.
