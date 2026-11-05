# P3.3 explicit Moonstone-native256 spatial protocol

2026-11-05. **Protocol and code declared before any P3.3 transfer scores; official execution remains NOT RUN.** This is a separately named downstream transfer policy, not a silent relaxation of the old interface. [Machine-readable protocol](../protocols/p33_native256_spatial_policy.json), [implementation/test report](../reports/p3_3_native256_policy.md).

## Why this is a separate policy

The pinned NASA config has `input_size=256`, `patch_size=16`, `pre_resize=512`. Those fields serve different purposes. The official factory passes input/patch sizes to image embeddings, while `pre_resize` is consumed by the dataset transform. The two terrain modalities use **UntokLunarTransform**: inherited preprocessing resizes to512 and normalizes, followed by crop and flip. The earlier local audit's generic crop/resize description of LunarTransform did not distinguish this untokenized subclass.

Moonstone tiles already contain256×256 physical samples. The named `moonstone_native256_v1` policy keeps the entire raw tile unchanged and applies the unchanged pinned NASA normalization only. It deliberately does **not** apply the NASA pretraining resize/crop/flip pipeline, interpolate, reproject, filter IDs or synthesize missing pixels. Both random and pretrained arms receive identical prepared inputs. This tests transfer on the Moonstone-native grid, not reproduction of NASA's pretraining distribution.

### Tensor and positional geometry

At the pinned source revision, ImageEncoderEmbedding creates a256/16=16×16 positional grid and uses its existing positional tensor directly when runtime geometry matches. The matching patch size also avoids patch-weight resizing. The constructor is unchanged; no input_size/patch_size/config override is permitted. The wrapper continues to reject any runtime tensor other than its canonical size.

This is source-level architectural rationale, not proof that the actual checkpoint matches. NASA's state loader can interpolate mismatched positional state; our **pre-load** complete selected-key/shape audit remains mandatory and rejects mismatch before that loader can adapt it. Actual fixed-checkpoint verification and real paired forward/backward remain execution gates.

### Physical scale remains shifted

Moonstone's declared grid is equirectangular,128 pixels/degree on radius1,737,400m: nominal north–south spacing236.90117518866782m and east–west spacing that depends on latitude. A256 tile spans2°×2°, not a fixed isotropic physical square. The NASA stats declare60m source sampling; that is not an exact effective post-transform sampling comparison. This policy does not claim equivalent physical scale, source sensor, crop distribution or terrain provenance. Original raster headers and executed per-region source lineage are still not independently reproduced.

## Explicit use and receipts

The default remains `strict_config_v1` and still rejects `pre_resize != input_size`. Native policy requires all of:

- Explicit `--spatial-policy moonstone_native256_v1` on preflight, lock and run. Fresh-process replay propagates this flag.
- Actual config bytes matching audited SHA256 `fd94d20eb2b1cd4379516da3d2c1912a56a8684e1952ee9331abd63e7c59e619`. A self-hash of another YAML is not enough. Do not edit the YAML.
- Unchanged input256/patch16/pre_resize512, exactly the two one-channel terrain inputs and the old normalization/missingness/unit guards.
- A new audited provenance JSON retaining earlier limitations and declaring the physical grid below. The existing `spatial_transform` remains `none`; do not relabel an already transformed input.

Add these fields to the actual input/runner provenance, without erasing source evidence or uncertainties:

```json
{
  "spatial_policy": "moonstone_native256_v1",
  "meters_per_pixel": 236.90117518866782,
  "ground_sampling": {
    "grid": "equirectangular",
    "pixels_per_degree": 128,
    "lunar_radius_m": 1737400,
    "east_west_spacing": "latitude_dependent",
    "nasa_native_60m_equivalence": false
  }
}
```

The value is nominal north–south spacing, not an independently measured pixel resolution. Existing batch and provenance hashes must reflect actual new files. Use a new output namespace; preserve the historical failed strict-policy receipt.

First, an input-only preflight may be run with the actual audited config and existing authenticated raw Train batch, adding the policy flag to the [original command](P3_3_NASA_TRANSFER_HANDOFF.md). This does not import torch, acquire weights or establish official model compatibility. It must still say `official_model_execution: NOT_RUN`.

Only after private runtime, provenance and asset checks pass should a new **actual paired forward/backward** receipt be produced under the same policy/device/adaptation. The runner now requires the policy receipt and locks it in `P33_NASA_TERRAIN_PAIRED_V2`; missing/old/different policy receipts fail. The config, policy source, other implementation sources, assets and runtime remain locked. Run/replay cannot switch policy, and complete fixed-cohort scanning plus an actual budget lock are still mandatory before scores. No existing real budget or manifest has been frozen.

## Local input validation completed

The local follow-up at `f74af37` passes75 focused and701 full tests with zero skips, plus authentic Train2705 input-only CLI and independent exact elementwise normalization/hash checks. The original config and raw batch are unchanged; new provenance and receipts live under ignored `outputs/p33/native256_input_01/`. [Local report](../reports/p3_3_native256_local_validation.md) and [verification](../reports/evidence/p33_20261105/native256_local_01/verification.json). Historical strict failures remain preserved. This is one-crop input preparation; official model execution remains NOT_RUN and no actual training manifest is locked. Do not repeat this completed input check without a new implementation or unresolved concern.

## Still blocked or unverified

- No compatible private NASA runtime is available in the existing local audit. Do not replace the shared Python/ROCm/torch environment, install packages or treat passing toy tests as compatibility.
- No fixed checkpoint has been locally verified, no official model has run, no full real-cohort eligibility scan has passed, and no transfer benefit is established.
- Raw source lineage and finite-sentinel provenance must remain explicit. No cohort filtering, fake masks or imputation is introduced.
- Test remains HOLD; previous P3.2 results/budgets are preserved.

## Pinned source evidence

All source links refer to `d54c67aad513cb9daca444afa425cfb278e4fbf8`:

- [ImageEncoderEmbedding, geometry and load handling](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/blob/d54c67aad513cb9daca444afa425cfb278e4fbf8/ni_lfm/models/encoder_embeddings.py#L328-L613)
- [Embedding factory](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/blob/d54c67aad513cb9daca444afa425cfb278e4fbf8/ni_lfm/models/lfm_utils.py#L582-L650)
- [Tensor-only encoder path](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/blob/d54c67aad513cb9daca444afa425cfb278e4fbf8/terratorch_integration/lunar_backbone.py#L332-L383)
- [dtm/slope untokenized transform mapping](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/blob/d54c67aad513cb9daca444afa425cfb278e4fbf8/ni_lfm/data/modality_info.py#L168-L204)
- [Preprocessing and UntokLunarTransform](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/blob/d54c67aad513cb9daca444afa425cfb278e4fbf8/ni_lfm/data/modality_transforms.py#L316-L349)
- [Existing authenticated config audit](../reports/evidence/p33_20261105/local_01/config_audit.json) and [Moonstone source audit](../reports/evidence/p33_20261105/local_01/source_audit.json)
