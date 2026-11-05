# P3.3 local configuration, terrain and runtime audit

2026-11-05. Pulled `8eeab47` fast-forward from `origin/main` after checking a clean local main and no active experiments, then merged concurrent `ce8b1ef` Train-membership authentication before publication. **Latest integrated local checks: 28 new tests and 654 full tests PASS, zero skips; official NASA model execution and transfer training remain BLOCKED / NOT RUN.** This report supplements the original [cloud handoff](p3_3_nasa_transfer.md), preserving its actual test results and all P3.2 evidence.

## Actual local results

| Check | Result |
|---|---|
| New NASA tests in existing private ROCm runtime | **28 PASS, zero skips**, including all nine torch wrapper tests skipped in the cloud; 2.297 s |
| Complete local regression suite | **654 PASS, zero skips**, 19.721 s |
| `python3 -m compileall -q scripts src tests` | PASS |
| Cached fixed-revision NASA config | Actual byte count, SHA256 and Git blob PASS |
| Fixed source acquisition | Direct SSH, no proxy; ten selected public source files hash/blob verified; no upstream Python executed |
| Existing raw Train crop 2705 | Actual original receipt SHA256/size, FP32 shape, frozen Train membership and target verified |
| Unchanged input CLI with that raw crop and authentic NASA config | **Exit 1 at spatial guard**, no success receipt; expected rejection recorded |
| Official model construction/checkpoint loading/real forward-backward | **NOT RUN**; incompatible existing runtime and unresolved spatial policy |
| Transfer task scores / actual training runner and manifest | NOT RUN / NOT IMPLEMENTED / NOT FROZEN |

[Latest integrated verification and source hashes](evidence/p33_20261105/local_02/verification.json), [readiness blockers](evidence/p33_20261105/local_01/readiness.json), [latest full log](evidence/p33_20261105/local_02/full_tests.log). The initial 26/652-test checks at `8eeab47` remain preserved in [local_01](evidence/p33_20261105/local_01/verification.json), followed by a justified revalidation after upstream code changed; those historical receipts were not rewritten. The fixed-root cohort validator authenticates the real ID2705/target43 against the 5,135 original Train rows; the unchanged batch then still fails at the spatial guard. [Integrated real-input receipt](evidence/p33_20261105/local_02/real_input_preflight.json) and [stderr](evidence/p33_20261105/local_02/real_input_preflight.log) record this sequence. Raw-pixel identity is bound separately to the original crop SHA256, not inferred from membership alone. [All local evidence hashes](evidence/p33_20261105/local_02/evidence_manifest.json). The nine torch tests use tiny synthetic backbones, not the official model. Their passing status does not establish NASA/ROCm compatibility or transfer benefit.

## Authentic NASA config recovered locally

The file was already cached under ignored `outputs/p30_asset_probe/nasa_lfm/backbone/config.yaml`. It is **12,102 bytes**, SHA256 `fd94d20eb2b1cd4379516da3d2c1912a56a8684e1952ee9331abd63e7c59e619`, Git blob `17b6b828dd625584720aa5ea5172d35dffaec3b6`. Both identities match the earlier fixed-model-revision `b657bdfe6fc4c0445d918f77a2d7b919805f3bce` [asset receipt](evidence/p30_20261103/asset_probe.json) and [listing](evidence/p30_20261103/nasa_lfm_listing.json). Thus the cloud's missing-config blocker is resolved locally without acquiring new config bytes. [Actual check](evidence/p33_20261105/local_01/config_audit.json).

| Domain | Channels / input / patch | pre_resize | scaler | Mean / std | Declared source gsd_m |
|---|---|---:|---|---|---:|
| dtm | 1 / 256 / 16 | 512 | std | −637.2084107556901 / 1920.7233898703932 | 60 |
| slope | 1 / 256 / 16 | 512 | std | 7.980291966975289 / 6.310896777386625 | 60 |

The current `validate_domain` deliberately rejects `pre_resize != input_size`. The authentic file therefore fails with `pre_resize differs from input_size; audit spatial preprocessing first`, even though the raw patch and model's declared input are both 256×256. This is a rejection by our conservative preprocessing contract, **not proof that the official backbone cannot accept 256×256 tensors**. The pinned `LunarTransform.preprocess` resizes to `pre_resize`, then normalizes; its augmentation subsequently crops/resizes. The backbone forward consumes tensors directly. A paired inference spatial policy must explicitly decide which data transforms apply and document the physical-scale shift; do not change the YAML or silently bypass the guard. No such new policy was frozen here.

## Raw terrain source and one authenticated Train crop

[Source audit](evidence/p33_20261105/local_01/source_audit.json) records all ten files, exact revisions, Git blobs, SHA256 and pinned public URLs. Source text remains local and was not executed.

At Moonstone Git revision `cbd4720653eb05afd975d2a36280858c5c56a275`:

- `config.py` declares elevation index 1 in metres and slope index 2 in degrees, radius 1,737,400 m, 128 pixels/degree. Nominal north–south/equatorial spacing is **236.90117518866782 m**, while east–west ground spacing varies with latitude. This is not isotropic 237 m/px lunar terrain.
- `step02_align.py:282–338` converts LOLA half-metres with ×0.5 and SLDEM kilometres with ×1000. Lines 341–413 use finite SLDEM within |latitude|<58°, a 58–62° transition when both sources are finite, and LOLA beyond 62°; absent/invalid SLDEM can retain LOLA. It is incorrect to label every Moonstone elevation patch as SLDEM.
- Scaling maps declared source NoData to NaN; reprojection writes NaN NoData. `step03_derive.py:34–108` computes slope in degrees with latitude-corrected gradient spacing and a cosine floor of .01; nonfinite terrain propagates. `step04_tile.py`, `step12_m3_integrate.py` and `step14_build_v4.py` retain raw channels through the HDF5 stages. These establish the declared source-code pipeline; original raster headers, executed build history and every sample's actual fallback mixture were not independently reproduced.

The existing `raw_2705.npy` is **7,340,160 bytes**, SHA256 `da95889c23cd001a43b3ab9dc388fe4271f86d2a7be5ab626bcf3978507c9008`, matching its original [full-crop pair receipt](evidence/p31_pair_20261104/manifest.json). It has raw FP32 shape 28×256×256; ID2705 belongs to the original frozen Train cohort, Geology target43, latitude58–60°N and longitude170–168°W. It lies in the declared blending transition. Exact source mix was not recomputed.

Both terrain channels have 65,536 finite pixels; elevation range −487.6218…2316.4143, slope .0167079…31.521456. Neither has zero pixels in this particular crop. This does not change the established rule that finite zeros are observed. [Numerical audit](evidence/p33_20261105/local_01/raw_terrain_audit.json) preserves its earlier, pre-source-review status; the subsequent source review is separate immutable evidence. No whole-HDF5 rehash or complete-cohort eligibility filtering was performed.

A new local B=1 NPZ uses raw values, explicit finite masks consistent with the audited code pipeline, Unicode ID and target only. Its provenance states that original raster headers/build execution were not independently audited; no source-validity claim is inferred merely from finite values. It is not an accepted training dataset. Batch/provenance SHA256 are recorded in [actual CLI result](evidence/p33_20261105/local_01/real_input_preflight.json). The CLI fails at the authentic config spatial guard before preparing NASA inputs; [stderr](evidence/p33_20261105/local_01/real_input_preflight.log) is preserved. No input-success receipt was produced, no model ran, and no Test pixels were read.

## Existing runtime and acquisition outcomes

Host and running `rocm-lab` container both use Python **3.10.12**. No existing Python3.11/3.12 interpreter was found in the checked PATH, private uv/venv/conda locations or container. The existing private runtime has torch **2.5.1+rocm6.2**, NumPy2.2.6, h5py3.16.0, PyYAML5.4.1; GPU is available. OmegaConf, timm, einops, TerraTorch and Lightning are absent. [Runtime inventory](evidence/p33_20261105/local_01/runtime.json).

Pinned NASA `pyproject.toml` declares Python `>=3.11,<3.13`; `requirements.txt` declares torch `>=2.12,<3`, NumPy `>=2.4.4,<3` and additional dependencies. The existing environment does not meet these declarations. The unchanged constructor's Python guard was actually exercised and rejected it before any upstream import or construction. This does not prove every declared package is an inference minimum, or justify ignoring version checks. No compatible reduced dependency path was validated. No packages were installed, no torch/ROCm/shared runtime was replaced, and no alternative runtime was created.

Direct HTTPS produced three timeouts and one verified 33,902-byte backbone-source file; remaining requests were stopped after locating the cached config/card. [Attempt receipt](evidence/p33_20261105/local_01/direct_https_attempt.json) preserves the completed outcomes. Subsequent direct SSH with proxy environment removed successfully fetched shallow filtered objects at the two exact source commits, then selected ten text files totaling143,980 extracted bytes. This sum is **not measured network transfer**, which includes Git objects/protocol overhead and was not byte-metered. No proxy fallback, credentials change, model weights, tokenizers, full snapshot, or additional raw data acquisition occurred. The 2,425,755,440-byte checkpoint was not downloaded while earlier execution gates remained blocked.

## Commands and next gate

Actual local checks used the existing private copies of the ROCm runtime:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
PYTHONPATH="$TRUSTED_RUNTIME" \
python3 -I "$JOB_TMP/p31_python.py" -m unittest discover -s tests -p 'test_p33_nasa*.py' -v

LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
PYTHONPATH="$TRUSTED_RUNTIME" \
python3 -I "$JOB_TMP/p31_python.py" -m unittest discover -s tests -v

python3 -m compileall -q scripts src tests
```

The exact real-input CLI and hashes are in its receipt. Its output filename is intentionally absent because the audit failed; reusing an already written success output would remain prohibited.

Next work is an explicit paired spatial policy plus a compatible private NASA runtime plan preserving the working ROCm environment. Original raw-raster provenance limits and final complete-valid Train/Val IDs must remain explicit. Only after those gates pass should the fixed checkpoint be acquired/verified, an official paired forward/backward run, and a separate actual training runner/budget/cohort be frozen before task scores. No transfer benefit is established. P3.2's 18 runs, all older evidence and budgets remain unchanged; Test and legacy gates remain HOLD. Periodic upstream-main monitoring continues.
