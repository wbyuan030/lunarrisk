# P3.3 fixed Train/Val terrain diagnostic

2026-11-07. **Authenticated full-cohort numerical scan COMPLETE:** all original5,135 Train and1,155 Val rows have finite raw elevation and slope pixels, and all slope values lie within[0,90]degrees. **This is not proof of absent undocumented finite NoData, accepted training provenance, a budget lock, training, or a transfer score.** Test remains HOLD.

The new [diagnostic entry point](../scripts/p33_audit_terrain_cohort.py) hashes the retained33,964,443,553-byte prefix and historical metadata/range report/statistics/class map against the fixed P31 plan, authenticates ordered Train/Val IDs and targets against the pinned plan/metadata archive, and uses the existing read-only prefix adapter and physical chunk checks. It reads raw contiguous elevation/slope slices, never the normalized/imputing dataset item method. Both complete original cohorts remain unchanged; no rows were dropped or repaired. HDF5 can decompress complete28-channel chunks internally, but only terrain slices are materialized by the diagnostic. No additional pixels were downloaded or exported.

| Split/channel | Finite pixels | Nonfinite | Observed zeros | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| Train elevation(m) |336,527,360|0|5|−6547.27246|9537.69922|
| Train slope(degrees) |336,527,360|0|193|0|87.22196|
| Val elevation(m) |75,694,080|0|32|−6850.18164|8332.93066|
| Val slope(degrees) |75,694,080|0|53|0|86.42428|

Exploratory equality counts at−9999,−32768,32767 and65535 are zero in both channels. These four values are a bounded diagnostic list, not claimed official sentinels or an exhaustive search. Finite zero remains observed. HDF5 declares fillvalue0; that declaration does not establish individual pixel provenance. Every selected row has a complete allocated physical chunk authenticated against the historical plan, but original raster execution history and within-chunk source validity were not independently reproduced.

Six pinned Moonstone source files were rehashed against the prior source audit. The source declares masking of **declared** raster NoData to NaN and nonfinite propagation; missing raster NoData declarations or reused scaled intermediates remain lineage limitations. This scan cannot distinguish an ordinary finite measurement from an undocumented finite sentinel. It therefore explicitly records `finite_sentinel_absence=NOT_ESTABLISHED`; it does not create the runner's `finite_only_no_finite_sentinels` assertion or call `lock`.

Execution used existing private Python3.12.9/NumPy2.4.4 with `-I -B`, in `unshare --user --map-root-user --net`. Torch was not imported; NASA source/checkpoint was not loaded. Elapsed215.554s including prefix authentication; process maxRSS127,968KiB. No model output, optimizer step, task metric or Test pixel was requested. The completed Train2705 real GPU preflight was not repeated.

[Summary and hashes](evidence/p33_20261107/cohort_diagnostic_01/summary.json), [per-row diagnostics](evidence/p33_20261107/cohort_diagnostic_01/rows.csv), [source recheck](evidence/p33_20261107/cohort_diagnostic_01/source_recheck.json), [commands](evidence/p33_20261107/cohort_diagnostic_01/commands.json), and [verification](evidence/p33_20261107/cohort_diagnostic_01/verification.json) preserve this run. A separate CSV verification checked all6,290 ordered IDs/targets/ordinals against the authority, recomputed every summary count and extremum from the CSV, and matched the CSV hash. This verifies aggregation, not an independent reread of pixels. Private asset paths are excluded from the archive.

Verification:78 focused P33 tests PASS, zero skips,6.166s, including three new tests for nonfinite/zero separation, candidate counts without mutation, fully missing ranges, and schema rejection. Syntax parsing and `git diff --check` PASS. The focused suite includes existing synthetic model tests; no official model execution occurred in this phase. The full regression suite was not repeated here.

Next remains a justified source-validity decision and a separately chosen actual paired budget before scores. Existing default strict spatial policy, native256 protocol, completed preflight, P3.2 negative evidence and Test HOLD remain unchanged. No accepted full-cohort provenance or training manifest was frozen by this diagnostic.
