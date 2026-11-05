# P3.3: NASA terrain-transfer handoff

Updated 2026-11-05. **Local follow-up:** [actual local readiness audit](../reports/p3_3_local_readiness.md) now verifies the cached pinned config, ten pinned source files, one authenticated raw Train crop, all28 new tests and the full654-test suite with zero skips after integrating upstream Train-membership authentication. The unchanged real-input CLI fails closed on authentic `pre_resize512/input_size256`; existing Python3.10/torch2.5.1 also does not meet declared NASA runtime requirements. Official model execution/transfer remain NOT RUN. No packages or weights were installed/downloaded; direct SSH source acquisition succeeded without proxy. The cloud-only missing-config/dependency observations below are historical, not the latest local asset status.

**Input/weight guardrails implemented; official NASA execution and transfer experiments NOT RUN.** This is the next Git collaboration handoff, not a replacement for the completed P3.2 evidence. The P3.2 18-run negative result, original budgets, Test HOLD, and all raw assets remain unchanged.

## What is ready

- `src/lunarrisk/nasa_transfer/contracts.py`: raw elevation→`dtm` and slope→`slope`; audited units, provenance and resolution; explicit validity masks; normalization from the supplied NASA config. Valid zeros remain observations. Missing/nonfinite pixels fail closed. No optical-band padding, invented geometry or per-sample geology text.
- `torch_adapter.py`: official pinned NASA base constructor; strict selected encoder checkpoint coverage, including positional buffers/register tokens; no permissive missing-key success. Unused generative decoder state is removed before loading/transfer. A frozen encoder plus ordinary 49-way head supports a **small input-affine adapter**, with gradients retained through the frozen encoder. This is not LoRA or full fine-tuning.
- Paired random/pretrained construction uses the same architecture, canonical geometry, head initialization and adapter initialization. Task inputs, IDs, ordering, labels, optimizer and selection budgets must additionally be paired by the future training runner.
- `cohort.py`: authoritative Train membership from the already frozen P31 plan and metadata archive, both verified against fixed repository-pinned SHA256 values. Declaring `split=train` alone is insufficient; unknown, Val/Test and wrong-target rows fail before model execution.
- `scripts/p33_nasa_preflight.py`: executable input audit and optional one-batch paired forward/backward. No training loop, optimizer step, weight acquisition or Test scoring. Outputs are new files, never overwrite existing evidence.

## Verified source and unresolved assets

Official code: `NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model`, commit `d54c67aad513cb9daca444afa425cfb278e4fbf8` (Apache-2.0). Model repository revision: `b657bdfe6fc4c0445d918f77a2d7b919805f3bce`.

The previous P3.0 remote listing declares `backbone/checkpoint.pt` at 2,425,755,440 bytes, SHA256 `5843824dfa56f89552b37b5295aa865ed28777c5f2b33286db2ef81b4cf08e5b`. **This is not a newly verified local checkpoint.** The loader checks both before `torch.load(weights_only=True, mmap=True)` and requires complete selected inference state.

`backbone/config.yaml` was not available here. A bounded explicit no-proxy request failed at DNS resolution. No config values were invented. The local execution owner must obtain the fixed-revision config, record its actual SHA256/source, and pass that checksum. The CLI accepts a user-audited hash; it does not prove the config's origin merely by hashing it.

The cloud runtime is Python 3.12.14 with NumPy 2.3.5/PyYAML 6.0.3, but **no torch/h5py or NASA weights/data**. The old local Python 3.10.12 is outside NASA's `>=3.11,<3.13`; current upstream declares torch `>=2.12,<3` and many additional packages. Do not run unbounded `pip install -e .` against the working ROCm environment. Inspect an existing private compatible runtime; if none is usable, report that exact blocker before changing environment scope. No packages were installed for this handoff.

## Local execution order

1. `git pull --ff-only origin main`, inspect concurrent work and this handoff. Do not rerun P3.2's completed 18 runs or overwrite its artifacts.
2. Audit the two **raw physical** HDF5 channels, not `MoonstoneHDF5Pilot.__getitem__`'s already normalized output. Confirm elevation metres, degree slope, NoData sentinels/masks, grid and per-region elevation provenance. Moonstone elevation blends SLDEM2015/LOLA with latitude/fallback behavior; degree slope and roughly 237 m/px do not make these NASA-native terrain.
3. Check actual pinned YAML `data.domains.dtm`/`slope`: channels, input size, patch size, scaler and stats. The first interface performs **no resize/crop/reprojection** and rejects shape mismatch at both preprocessing and model boundaries. If NASA geometry differs from raw Moonstone 256×256, implement and predeclare an explicit, paired spatial policy before continuing. Do not bypass this check by relabeling the provenance.
4. Freeze eligible IDs before scores. Start from the existing 5,135 Train/1,155 Val IDs, but do not claim exact cohort equality if complete-valid terrain filtering changes it. Publish exclusions and reasons; both arms use the same retained IDs. This initial implementation rejects any partially missing pixels rather than silently imputing unknown terrain.
5. Acquire only needed fixed-revision backbone/config if absent, using the user's authorized direct/no-proxy route. No tokenizer/full snapshot is needed for the two untokenized image modalities. Preserve data/weights/cache under ignored local output directories, not Git.
6. Run the new wrapper tests in the private torch runtime, then one real Train-batch preflight. Record PASS/FAIL/BLOCKED, exact artifact hashes, actual device/memory and source checks. A failed or skipped model check is not a transfer result.
7. Only after compatibility passes, write a **separate actual training manifest/runner**: same-input random versus pretrained ordinary 49-class head, fixed Train/Val selection and paired seeds/budgets. The handoff includes no training runner or frozen training budget. Use old 28-channel B0 as context only, not as the modality-matched baseline. A conditional, separately declared input-affine or LoRA comparison can test adaptation; a negative frozen probe does not refute transfer learning.

## Batch and provenance contract

The CLI automatically authenticates `protocols/p31_local_prefix_eval_plan.json` and `reports/evidence/p30_20261103/metadata_rows.csv.gz`. Optional `--train-plan`/`--train-metadata` paths may relocate identical files but cannot replace the pinned hashes. A changed training cohort needs a separately reviewed protocol update, not a user-supplied override hash. The receipt records the authority hashes and 5,135-row ordered-ID digest. This validates membership and labels, **not raw-pixel identity**; the local data extractor must still bind pixels to the authenticated HDF5 rows.

Local NPZ fields, exactly:

- `elevation`, `slope`: FP32 arrays `(B,1,H,W)`, raw metres/degrees, B=1 or 2.
- `valid_elevation`, `valid_slope`: explicit boolean masks with identical shapes. True means genuinely observed, including zero-valued pixels. Do not mark NoData as observed to pass.
- `sample_ids`: unique string Train IDs, not unit-name strings.
- `targets`: integer Geology labels 0..48. These are used only for loss, never as encoder input.

Provenance JSON must contain:

```json
{
  "input_space": "raw_physical",
  "units": {"elevation": "m", "slope": "degree"},
  "spatial_transform": "none",
  "domain_shift": "MOONSTONE_TERRAIN_NOT_NASA_NATIVE",
  "meters_per_pixel": 237.0,
  "elevation_source": "REPLACE with actual per-region source audit",
  "slope_source": "REPLACE with actual source/derivation audit",
  "units_evidence": "REPLACE with audited source revision/evidence",
  "nodata_evidence": "REPLACE with actual mask/sentinel audit",
  "split": "train",
  "batch_sha256": "REPLACE with actual NPZ SHA256"
}
```

237.0 above illustrates the field, **not a newly measured grid value**. Populate actual audited resolution. The placeholders are documentation, not evidence.

Input-only audit (no torch import):

```bash
python3 scripts/p33_nasa_preflight.py \
  --batch "$TRAIN_BATCH" --provenance "$PROVENANCE_JSON" \
  --config "$NASA_CONFIG" --config-sha256 "$AUDITED_CONFIG_SHA256" \
  --output outputs/p33/input_preflight_01.json
```

After actual runtime and asset verification, add `--forward-backward --upstream "$PINNED_NASA_CHECKOUT" --checkpoint "$NASA_CHECKPOINT" --device cuda`. Set a fresh output filename. `--adapt-inputs` additionally checks nonzero finite gradients through the frozen encoder to the affine adapters. Do not add it retrospectively to re-label a frozen-probe score. Without `--forward-backward`, the receipt explicitly says `official_model_execution: NOT_RUN`.

## Verification boundary

The cloud passed **19 runnable new tests**, including CLI subprocess checks. **9 torch wrapper tests are skipped** here and need the compatible private runtime. They use a tiny synthetic stand-in even when run, and do not establish official NASA compatibility. `compileall` passes. The full cloud suite is dependency-blocked, with the same 29 missing-dependency errors as the baseline-only suite; this does not supersede P3.2's earlier 626-test local pass.

Actual NASA config semantics, artifact loading, real terrain I/O, ROCm finite gradients, runtime memory and task scores remain **NOT VERIFIED / NOT RUN**. See `reports/p3_3_nasa_transfer.md` and the machine-readable plan.

## Sources

- [Pinned NASA wrapper](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/blob/d54c67aad513cb9daca444afa425cfb278e4fbf8/terratorch_integration/lunar_backbone.py)
- [Pinned NASA normalization](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/blob/d54c67aad513cb9daca444afa425cfb278e4fbf8/ni_lfm/data/modality_transforms.py)
- [Official terrain-subset recipe](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/blob/d54c67aad513cb9daca444afa425cfb278e4fbf8/terratorch_integration/configs/ice_prosp/ablation/ni_lfm_ps8_m2.yaml)
- Moonstone source audit starting points: [elevation](https://github.com/ayushprd/Moonstone/blob/main/data_preparation/step02_align.py), [slope](https://github.com/ayushprd/Moonstone/blob/main/data_preparation/step03_derive.py). Pin these to the actual dataset revision before using as provenance.
