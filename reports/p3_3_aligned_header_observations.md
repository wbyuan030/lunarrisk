# P3.3 published aligned terrain headers

2026-11-07. The fixed-revision published blended, LOLA and SLDEM aligned GeoTIFF directories all declare **float32, NaN NoData and a 46,080 × 23,040 grid**. Their pixel scale and tiepoint match each other. The mirror's revision, linked sizes and hashes match the earlier official listing. These are partial-file observations; neither the complete files nor their role in the historical HDF5 build has been verified.

Direct no-proxy requests to the previously used AI FastHub mirror timed out. A second public mirror returned a redirect and each requested 1 MiB range as HTTP 206, 3,145,728 response-body bytes total. Existing `tifffile` parsed the directories without decoding raster pixels. All three files have LZW compression, one row per strip, and their first image strip starts at byte 277,269.

**Scope deviation:** the 1 MiB ranges therefore included compressed image strips, not only metadata. Their geographic coverage was not checked and may include held-out Test areas. The bytes were not decoded or used for analysis, and no model predictions or scores were produced. Further raster requests were stopped. This is not a claim that no Test-related bytes were received.

[Observation receipt](evidence/p33_20261107/aligned_headers_01/observations.json) records sizes, partial hashes, identity limits and the deviation. NaN metadata does not rule out finite invalid values, identify the exact source or intermediates used in construction, or establish that every cohort pixel is valid. The existing guard and Test scoring HOLD remain; no training or actual budget was locked.
