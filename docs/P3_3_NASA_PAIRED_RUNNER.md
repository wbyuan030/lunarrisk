# P3.3 paired NASA training runner

**New explicit spatial protocol (2026-11-05):** [Moonstone-native256](P3_3_MOONSTONE_NATIVE256.md) now provides a separately declared downstream path for the unchanged audited256/16/pre_resize512 config. It requires an explicit policy flag and physical-scale provenance; strict default remains unchanged. Policy is bound through preflight and manifestV2/run/replay. Actual input/model execution under it is NOT RUN; runtime/checkpoint/full-cohort/budget gates remain. Historical strict-policy failures and local690-test evidence below are preserved, not relabeled as results for this new code.

2026-11-05. **Latest local validation:** [64 P33 tests / full690 tests PASS, zero skips](../reports/p3_3_runner_local_validation.md), including all16 cloud-skipped tests. These use synthetic CPU torch/HDF5 fixtures and mocked locks; no official NASA runtime/model, real cohort scan or actual training/replay was executed. Earlier cloud verification below remains historical.

**Implementation available; no real NASA training, GPU validation or transfer score has run in this cloud workspace.** The earlier preflight-only handoff is extended by `scripts/p33_train_pair.py` and `nasa_transfer/{dataset,experiment,training}.py`.

**Concurrent local findings integrated:** [local readiness report](../reports/p3_3_local_readiness.md) verifies the actual cached config SHA256 `fd94d20eb2b1cd4379516da3d2c1912a56a8684e1952ee9331abd63e7c59e619` and an authenticated Train crop. The current guard rejects the real `pre_resize=512` / `input_size=256` config. The new runner retains that default guard. The explicit Moonstone-native256 policy is now declared but real input execution remains unverified; a compatible private runtime is still required before it can lock or train. The prior local654-test pass does not cover this newly added runner.

## Scope and current blockers

Two arms share the same official NASA base architecture, two physical input channels, frozen historical Train/Val cohort, ordinary 49-way head initialization, epoch permutations and explicit budget. Only the encoder initialization differs: random versus verified pretrained. `frozen` trains the head; `input_affine` additionally trains four scalar affine input-adapter parameters through the frozen encoder. Neither is LoRA or full-backbone fine-tuning.

All scoring is exploratory on the reused northern development split. No Test-scoring option exists. Do not repeat P3.2's old 18-run semantic matrix or compare this two-channel effect against 28-channel B0 as if inputs matched.

The cloud lacks torch, h5py, raw Moonstone assets, NASA config and weights. Its runnable tests cover pure contracts, mocked I/O/orchestration, metrics and subprocess failures. Torch training/replay tests and the real-HDF5 fixture are **skipped**, not passed. Before actual execution, the local agent must supply the original assets and a compatible isolated Python/ROCm runtime without replacing working shared torch. Upstream declares Python >=3.11,<3.13 and torch >=2.12,<3; the previously recorded local Python3.10/torch2.5 runtime is not asserted compatible.

## Data and preprocessing

`open_datasets` authenticates the retained prefix, metadata JSON, cached-range report, channel statistics and class map against the fixed historical plan. The repository-pinned metadata archive independently authenticates all 5,135 Train / 1,155 Val IDs and Geology labels. Existing physical chunk checks and file-mutation guards remain active. Pixels are selected from the raw HDF5 data, **not** the normalized P31 `__getitem__` output.

Only raw `elevation` and `slope` reach `prepare_terrain`. Reading two channel slices can still cause HDF5 to decompress the stored 28-channel chunk internally. Nothing is exported or downloaded. Both official dataset reader handles are read-only, and Test indices cannot be selected.

The common physical provenance JSON must contain the existing audited fields plus:

```json
{"nodata_policy": "finite_only_no_finite_sentinels"}
```

Use that assertion only after actually verifying the source has no finite NoData sentinels. The current tier rejects any missing pixel, unexpected unit/scaler or noncanonical spatial shape. It never filters IDs, imputes unknown terrain or resizes implicitly. If the complete historical cohort fails, this run is **BLOCKED**. A narrower cohort, missing-data handling or spatial adapter requires a separate explicit protocol/code update before scores, not a bypass flag.

`lock` reads every Train/Val sample before any model score. Prefix SHA256 is streamed once per process, not once per epoch; subsequent reads retain mutation checks. Each fresh replay process independently authenticates the prefix again, so this strict approach has extra I/O cost.

## Budget and lock

The example `protocols/p33_nasa_budget_example.json` proposes seed42, maximum20 epochs, patience5, batch2, FP32, AdamW lr.001/wd.01, constant schedule and gradient clipping1. **It is an example, not an executed or already frozen manifest.** Choose the budget before scores, after compatibility/memory checks. Allowed seeds are the complete cohort `[42]` or `[42,43,44]`; no favorable seed/arm selection. Every budget field is required, and unsupported/extra fields fail. AdamW betas are fixed(.9,.999), epsilon1e-8, workers0 and no augmentation.

Both arms have the same maximum budget and early-stop rule, not necessarily the same realized epoch count. Val full49 Macro-F1 selects the checkpoint, strictly greater only, earliest epoch wins ties. Report realized training visits and time. A single-seed difference has no standard deviation; three seeds report sample standard deviation, never a confidence interval.

A successful **actual paired forward/backward** receipt from the updated `p33_nasa_preflight.py`, for the same config, adaptation and device, is required. Input-only receipts are rejected. The preflight records device identity; old receipts lacking that field must not be relabeled. No receipt can substitute for actual model/runtime compatibility.

In a compatible local runtime, after creating a matching real paired preflight and explicit native-grid provenance as described in the [spatial guide](P3_3_MOONSTONE_NATIVE256.md), assemble paths to the audited assets. The example below explicitly selects the new Moonstone-native256 protocol:

```bash
COMMON=(
  --data "$LOCAL_PREFIX" --metadata "$LOCAL_METADATA_JSON"
  --range-report "$RANGE_REPORT" --stats "$HF_CHANNEL_STATS"
  --class-map "$GEOLOGY_CLASS_MAP" --nasa-config "$NASA_CONFIG"
  --nasa-config-sha256 "$AUDITED_CONFIG_SHA256"
  --spatial-policy moonstone_native256_v1
  --nasa-checkpoint "$NASA_CHECKPOINT" --upstream "$PINNED_NASA_CHECKOUT"
  --provenance "$TERRAIN_PROVENANCE_JSON" --preflight "$REAL_PREFLIGHT_RECEIPT"
  --manifest outputs/p33/pair_01_manifest.json --device cuda
)
python3 scripts/p33_train_pair.py --mode lock "${COMMON[@]}" \
  --budget protocols/p33_nasa_budget_example.json
```

`lock` records exact Python/NumPy/torch versions and hashes inputs/checkpoint/source, validates the preflight, completes the input scan, then writes a new immutable manifest. It does not train or score a model. A failure does not create a successful lock. Save the printed actual digest:

```bash
python3 scripts/p33_train_pair.py --mode run "${COMMON[@]}" \
  --manifest-sha256 "$ACTUAL_LOCKED_MANIFEST_SHA256" \
  --outdir outputs/p33/pair_01
```

`run` refuses source/asset/budget/device/runtime-version changes, existing output directories and a missing/mismatched lock digest. It executes every declared seed and both arms. No CLI budget override is accepted. Never change a budget after examining scores and reuse the old experiment identity.

## Outputs, replay and failure handling

Each arm writes:
- Per-epoch task-head/adapter checkpoints, history, training row counts/order hashes and loss.
- Selected head/adapter checkpoint plus reconstructed frozen-backbone identity hash.
- Ordered Train/Val prediction CSVs, full49 F1/accuracy, confusion/support, NLL and sum-form multiclass Brier.
- Resource/time measurements, parameter counts, hashes and `report.json` with replay-pending status.

The launcher deletes the training model, collects/relinquishes unused GPU memory, and starts **a new process** to reconstruct the exact frozen encoder/selected head and replay both prediction CSVs. A mismatch or process failure stops the sequence. The arm gets `completion.json` only after replay passes; the matrix gets `summary.json: COMPLETE` only after all arms replay and paired initialization/order checks pass. Model state and data are not placed in Git.

Deterministic algorithms are required. For NVIDIA, a valid `CUBLAS_WORKSPACE_CONFIG` is configured/checked before CUDA initialization; ROCm does not use that setting. Unsupported deterministic operations fail explicitly, rather than silently relaxing reproducibility.

Automatic resume is **not implemented**. Interrupted outputs and completed epoch checkpoints are preserved, never overwritten or silently retrained. Record the failure and decide how to continue under the unchanged scientific protocol before a new run namespace. This implementation limitation does not authorize changing the budget or concealing partially observed scores.

## Before publishing any experiment claim

Run all tests in the real compatible runtime; recheck input provenance and hashes; obtain finite actual device forward/backward; inspect complete training/replay receipts, fixed IDs and losses. Report actual PASS/FAIL/BLOCKED and any skipped checks. A successful mock or toy-wrapper test does not establish official NASA execution, and a negative frozen probe alone does not refute transfer learning.
