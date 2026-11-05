# P3.3 NASA terrain transfer: implemented runner, no experiment scores

Updated 2026-11-05. Latest implementation integrates concurrent local main `39323a42b665fa367d39b1ae8d4f5311d0276a34` without replacing its audit/evidence. **Training runner implemented; official model/runtime execution, real training and transfer benefit remain NOT RUN / NOT ESTABLISHED.**

**Local follow-up 2026-11-05:** [local readiness audit](p3_3_local_readiness.md) resolves the missing-config/source-text gaps and passes28 new/654 full tests with zero skips after integrating the Train-membership correction. One authenticated raw Train batch was exercised against the actual config; the CLI exits1 at the unchanged spatial guard. Existing runtime compatibility and explicit paired preprocessing remain BLOCKED; official model and transfer scores NOT RUN. This section preserves the original cloud verification below rather than replacing its results. These654 local passes cover the pre-runner revision, not the new training engine.

**Latest runner-local validation:** [64 focused / 690 full tests PASS, zero skips](p3_3_runner_local_validation.md), covering the new runner and all16 tests skipped in the cloud. Torch/HDF5 fixtures are synthetic and the successful lock fixture is mocked; official NASA model execution, full real cohort scan, budget lock, training and fresh-process replay remain NOT RUN. Prior local654 and cloud logs below are preserved for their own revisions.

[Execution guide](../docs/P3_3_NASA_PAIRED_RUNNER.md) · [Input/source handoff](../docs/P3_3_NASA_TRANSFER_HANDOFF.md) · [Plan](../protocols/p33_nasa_transfer_plan.json).

## Implemented

- Raw read-only elevation/slope streaming directly from the authenticated retained HDF5 prefix; exact original 5,135 Train/1,155 Val IDs, targets and chunk records. No normalized-loader substitution, Test path, missing-pixel imputation, hidden filtering or implicit spatial resizing.
- Fixed historical Train/Val authority hashes. A forged `split=train`, invented ID, real Val/Test ID, wrong target or changed manifest cannot pass the preflight membership gate.
- Official pinned NASA base wrapper, complete inference-weight coverage, unchanged canonical geometry, frozen encoder state and optional input-affine adapter. Same-head initialization and minibatch order for random/pretrained arms. Input-affine is not LoRA.
- A required actual preflight receipt, complete pixel eligibility scan and explicit no-default budget become an immutable source/asset/runtime manifest before scores. Run/replay require its exact hash and forbid budget overrides.
- Val full49 Macro-F1 selection with earliest ties, per-epoch checkpoints, selected trainable state, Train/Val prediction CSVs, metrics/resources, and fresh-process reconstruction/replay. Completion requires exact prediction hashes and all declared arms; interrupted outputs are preserved. Automatic resume is not implemented.

## Actual verification

| Check | Actual outcome |
|---|---|
| P33 focused suite | **64 tests: 48 PASS, 16 SKIP**, no errors/failures |
| Completed full cloud suite before documentation/evidence merge | 448 discovered: 350 PASS, 69 SKIP, **29 missing-dependency ERROR** |
| Error comparison | The same 29 error test IDs as the original baseline-only suite |
| compileall / diff checks | PASS |
| Actual raw HDF5 / NASA load / GPU training / official fresh-process replay | **NOT RUN** |

[Latest verification/log hashes](evidence/p33_20261105/paired_runner/verification.json). [Original preflight evidence](evidence/p33_20261105/verification.json) and [Train-membership correction evidence](evidence/p33_20261105/train_membership_fix/verification.json) are preserved.

The runnable tests cover NumPy contracts/metrics, authenticated metadata, mocked raw-reader and orchestration behavior, and CLI subprocess checks. The 16 skips are torch-dependent wrapper/training tests and one real-HDF5 fixture. Even a passing tiny synthetic torch fixture would not prove official NASA compatibility. No packages installed, weights downloaded, paid resources used or actual model trained.

Independent static review identified and corrected model-boundary geometry checks, cached upstream-module origin checks, byte-string receipt serialization, authoritative Train membership, replay command path handling and release of reserved training GPU memory before launching a child replay. The final targeted review found no additional fatal issue; dynamic model behavior remains unverified.

## Local next action and limits

Use the newly audited local config/source evidence, preserve its remaining original-raster/build-lineage limits, and explicitly decide the paired pre_resize512/input_size256 policy. The runner deliberately retains the existing guard until that policy is implemented and audited. Resolve the compatible private Python/ROCm runtime without replacing shared torch. Verify checkpoint bytes/hash, run the actual device preflight, then use the new runner to scan and lock an explicit budget before training. The example budget is **not** an executed or frozen manifest. Shape/missingness failures stop the full-cohort tier; they do not authorize changing IDs or inventing channels.

The cloud has no torch/h5py, actual config/checkpoint or raw pixels. Upstream Python/torch constraints remain a real compatibility gate. The retained prefix has authenticated local bytes, not verified full upstream HDF5 integrity. All scores remain exploratory reused northern Val, and a frozen-probe negative result cannot refute transfer learning. Existing 28-channel B0 is not a same-input baseline. P3.2 negative evidence, budgets, Test HOLD and all old assets remain unchanged.

After integrating the local audit, the focused suite again completed with48 passes/16 skips. The repeated full-suite invocation was blocked by tool review because legacy tests may initiate external LROC HTTPS access. It was not retried or routed around. There are no Python-source changes relative to the last completed full-suite run; its original log is preserved. Only documentation and concurrent local audit evidence changed in that merge.
