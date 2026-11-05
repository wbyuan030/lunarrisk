# P3.3 native256 local validation and real input preparation

2026-11-05, based on `f74af37973658788dd307c3928ad4a987ebf6122`. **75 P33 tests and 701 full regression tests PASS with zero skips; authentic Train2705 input-only preflight PASS under `moonstone_native256_v1`. Official NASA execution and transfer benefit remain NOT RUN / NOT ESTABLISHED.**

[Verification and 22 source hashes](evidence/p33_20261105/native256_local_01/verification.json) · [Evidence hashes](evidence/p33_20261105/native256_local_01/evidence_manifest.json) · [Actual commands](evidence/p33_20261105/native256_local_01/commands.json).

## Actual results

| Check | Result |
|---|---|
| Focused P33 suite in existing private runtime | **75 PASS, zero skips**, 3.502 s |
| Complete local suite after offline fixture correction | **701 PASS, zero skips**, 14.508 s |
| compileall / diff checks | PASS |
| Existing raw crop, batch, old provenance and original NASA config | SHA256 verified; original assets unchanged |
| Raw crop → batch channel values and validity masks | Exact equality; Train ID2705 / Geology target43 authenticated |
| Real input-only CLI with original config and explicit native256 policy | **Exit 0, input_contract PASS** |
| Independent prepared-array check | Both FP32 `[1,1,256,256]` arrays exactly equal manual elementwise normalization; output fingerprint matches CLI |
| Official model / checkpoint / paired forward-backward | **NOT RUN** |
| Complete real-cohort scan / actual budget lock / training / official fresh-process replay | **NOT RUN / NOT FROZEN / NOT RUN / NOT RUN** |

[Focused log](evidence/p33_20261105/native256_local_01/focused_tests.log) and [full log](evidence/p33_20261105/native256_local_01/full_tests.log). All sixteen checks skipped in the cloud executed locally. These remain synthetic CPU torch backbones, temporary HDF5 and mocked orchestration. `LOCKED` stdout is a mock fixture result; no real experiment manifest was frozen.

## Legacy offline-test repair

The cloud partial log stopped in `TestMainWithCache.test_summary_counts_from_cached_pages`. Inspection found its `--scope all` invocation includes DELTA, but the fixture supplied only DELTA's scene page, omitting its product page. The product-page cache miss reaches `urlopen` despite the test's cached-only intent. Added the missing synthetic no-links product page and mocked `urlopen` for this test with a final `assert_not_called()`. Both scope invocations now use only fixture pages. Production LROC code and old data/gates are unchanged. Only after inspecting and fixing this cause was the local complete regression run started. The earlier cloud partial log remains INCOMPLETE; it is not relabeled PASS.

## Authentic input preparation

The existing raw crop SHA256 is `da95889c23cd001a43b3ab9dc388fe4271f86d2a7be5ab626bcf3978507c9008`. The unchanged input NPZ is `572cb90d9ed06bc73ad5ea372e1190d0ff28399009d9ce9fafc39aa50e73a9cd`. Its two modalities exactly match original channels1/2; finite masks exactly match raw `isfinite`. The pinned plan and metadata authenticate ID2705/target43 against the original 5,135 Train rows; no Val or Test crop was substituted.

Created new ignored `outputs/p33/native256_input_01/provenance.json` and output files. The new provenance retains the original source and raster-lineage limitations and adds explicit native256 policy plus anisotropic equirectangular ground sampling. It makes no whole-cohort finite-sentinel assertion. Nominal north–south spacing236.90117518866782m and latitude-dependent east–west spacing are not NASA60m equivalence.

The unchanged original config bytes match SHA256 `fd94d20eb2b1cd4379516da3d2c1912a56a8684e1952ee9331abd63e7c59e619`. The CLI preserves the whole256 grid and uses that config's std normalization. Independent NumPy calculations checked every output value against `(raw - float32(mean)) / float32(std)` for both modalities. No resize, crop, interpolation, reprojection or filtering was applied. Prepared fingerprint: `7942299d25842593b32765c28d6866e6e6efa8dfcc1abbc966195f890d77029b`.

[Actual CLI receipt](evidence/p33_20261105/native256_local_01/input_preflight.json), [new provenance](evidence/p33_20261105/native256_local_01/input_provenance.json), [independent checks and asset hashes](evidence/p33_20261105/native256_local_01/input_independent_input_verification.json), [stdout](evidence/p33_20261105/native256_local_01/input_preflight.log). No pixels or weights are committed. Earlier strict-policy exit1 receipts in local_01/local_02 and old provenance remain preserved; the strict default's rejection is also covered by the new tests.

## Remaining execution gates

Input preparation is now verified for this one existing Train crop. It does not authenticate original raster headers, executed source fallback, every sample's missingness or the complete retained HDF5. The earlier runtime audit remains applicable: existing Python3.10.12/torch2.5.1+rocm6.2 does not meet the pinned NASA declarations, and required packages are absent. No official constructor, weight load, GPU model execution or paired gradient check occurred in this phase.

Next resolve a compatible private NASA runtime plan within the existing no-install/no-replacement boundary, retaining raw-source and finite-sentinel uncertainties. Then the fixed checkpoint size/hash and complete inference keys/shapes, actual paired forward/backward, full unchanged Train/Val eligibility scan and actual explicit paired budget lock are still required before scores. No weights/data downloaded, packages installed, shared environment replaced, previous experiments repeated or Test pixels/scores accessed in this phase. P3.2's18-run negative results and all old budgets remain preserved. Session-only upstream monitoring continues.
