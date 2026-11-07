# P3.3 local historical terrain lineage inventory

2026-11-07. **Inventory complete within the three inspected project/history roots; original raster execution lineage remains NOT_ESTABLISHED.** No original LOLA/SLDEM raster or configured terrain intermediate was found there. This is a scoped local inventory, not proof of absence elsewhere on the host, a remote builder or unpublished upstream storage.

The inspected roots are identified only as `WORKSPACE` (current project), `HISTORICAL_REPO` (prior project copy supplying the authenticated HDF5 inspection), and `HISTORICAL_JOB_TMP` (prior task temporary workspace). Absolute private paths are excluded. Filename/stat enumeration covered 31,248 files: 21,128 / 9,934 / 186 respectively. There were no walk errors. The [inventory](evidence/p33_20261107/local_lineage_inventory_01/inventory.json) explicitly lists 106 excluded dependency, Git, bytecode and symbolic-link directories; links were not followed. These exclusions and three-root scope limit the absence conclusion.

| Expected historical artifact | Found in inspected scope |
|---|---|
| `Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif` | No |
| `SLDEM2015_128_60S_60N_000_360_FLOAT.IMG` / `.LBL` | No original file |
| `lola_dem_scaled.tif` | No |
| `sldem2015_128ppd_converted.tif` / `sldem2015_128ppd_meters.tif` | No |
| `elevation_lola.tif` / `elevation_sldem.tif` / `elevation_blended.tif` | No |
| `slope.tif` | No |

The recently fetched official SLDEM label exists as a compressed evidence artifact; its decompressed hash matches the existing public-source observation. It is current public metadata, not the historical label used by the dataset builder. The current public LOLA header observation is also present in the source evidence. Neither links historical raster bytes to this HDF5 revision. Unrelated historical NAC/DTM/mosaic raster filenames were found and recorded only by name/size; their headers and pixels were not opened.

The inventory read 344 existing `.log`, `.out` and `.prt` text files, each at most 10 MB; none exceeded that bound. Nine literal runtime markers from the pinned Moonstone source—LOLA/SLDEM scaling, scaled-file reuse, alignment, blending, slope computation and V2 building—had no matches. The exact markers and per-log result are archived. Log traversal excluded symbolic links and directory basenames drawn from the initial prune list. Zero literal matches does not prove that differently named, differently worded or unretained build logs never existed.

Existing receipts establish narrower facts: fixed-source acquisition explicitly did not execute upstream code; the prefix download manifest records downstream HDF5 acquisition attempts; the selected pair report records downstream crops/cached ranges; the source observation records current public headers. None is a historical raw-raster ingestion/build/reuse receipt. Only the first 4,096 text bytes of the existing HDF5 inspection were read here to inspect root attributes: channel names, `n_channels=28`, `patch_size=256`, `ppd=128`. Those root attributes do not contain original raster checksums, actual build commands, GDAL/rasterio versions or intermediate-reuse identities. No HDF5 dataset was opened or sampled during this inventory.

The unresolved evidence is specific: historical source LOLA TIFF and SLDEM label/image identities; converted/scaled raster headers and masks; actual GDAL/rasterio versions and conversion settings; whether each intermediate was rebuilt or reused; aligned/blended/slope product identities; and a manifest connecting those artifacts through V1 to the fixed V4 HDF5. The pinned source explains intended handling, including existing-file reuse, but cannot establish those executed facts.

The next useful evidence would be the original producer's retained build manifest/logs and small raster metadata exports, or bounded metadata of fixed published aligned products. Original source/intermediate identities and per-stage validity handling must remain explicit gaps until obtained. No source-validity assertion or actual training budget was frozen by this inventory.

[Verification](evidence/p33_20261107/local_lineage_inventory_01/verification.json) records the scope and inventory hash. This phase made zero network requests, repeated zero large-file hashes or cohort scans, read no raw raster headers/pixels or Test pixels, and ran no model, training or score. Tests were not rerun for this read-only inventory/report phase. Test remains HOLD; the previously completed full-cohort numerical scan and single-crop GPU preflight were not repeated.
