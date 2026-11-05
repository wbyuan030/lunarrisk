# P3.3 NASA terrain integration: cloud handoff

2026-11-05, based on main `aea52f28444598df7692f76ac67bbd0466b65ff7`.

**Local follow-up 2026-11-05:** [local readiness audit](p3_3_local_readiness.md) resolves the missing-config/source-text gaps and passes26 new/652 full tests with zero skips. One authenticated raw Train batch was exercised against the actual config; the CLI exits1 at the unchanged spatial guard. Existing runtime compatibility and explicit paired preprocessing remain BLOCKED; official model and transfer scores NOT RUN. This section preserves the original cloud verification below rather than replacing its results.

**Implemented a fail-closed local NASA terrain preflight, not a completed transfer experiment.** See the [execution handoff](../docs/P3_3_NASA_TRANSFER_HANDOFF.md) and [plan](../protocols/p33_nasa_transfer_plan.json).

- Raw elevation/slope only; explicit units/provenance/grid and validity masks. No Hapke→five-band WAC substitution, invented geometry, label leakage, inferred-zero missingness or guessed normalization. Missing pixels and implicit geometry changes are blocked.
- Official pinned NASA base encoder constructor with unchanged canonical geometry. Checkpoint size/SHA256 and complete selected inference tensors, including buffers/register tokens, are required. Generative decoder/unused modality exclusions are reported, not counted as loaded encoder parameters.
- Ordinary head, paired random/pretrained initialization, frozen encoder eval state, optional input-affine adapter with autograd through the encoder. Input-affine is not LoRA. The runnable CLI does not train or score Test.
- An independent static review found missing model-boundary geometry checks and incomplete cached-module origin checks. Both were corrected and regression tests added. No official-model dynamic validation was possible.

## Actual checks

| Check | Outcome |
|---|---|
| `python3 -m unittest discover -s tests -p 'test_p33_nasa*.py' -v` | 28 discovered: **19 PASS, 9 SKIP**, no failures/errors |
| `python3 -m unittest discover -s tests -v` | 412 discovered: 321 PASS, 62 SKIP, **29 ERROR** due missing dependencies |
| Baseline-only suite, excluding new P33 tests | 384 discovered: 302 PASS, 53 SKIP, **same 29 ERROR test IDs** |
| `python3 -m compileall -q scripts src tests` | PASS |
| `git diff --check` | PASS |
| Actual NASA checkpoint load / raw terrain / ROCm forward-backward / task metrics | **NOT RUN / BLOCKED by missing assets and runtime** |

Torch, h5py and older optional geo dependencies are absent in this cloud executor. No package installation, torch replacement, paid compute, GPU training or weight download occurred. Full-suite dependency failure is not a pass; the previous P3.2 local 626-test pass remains historical evidence for its own code/runtime. New torch tests use a tiny synthetic wrapper fixture and will still not prove official NASA compatibility when they pass.

[Latest verification and log hashes](evidence/p33_20261105/train_membership_fix/verification.json). [Original pre-fix verification](evidence/p33_20261105/verification.json) is preserved. Logs contain only tests with paths sanitized; no raw data, checkpoints, cache or credentials are committed.

## Remaining work

Local Git collaborator: audit the actual pinned config and raw terrain; resolve private compatible runtime without replacing shared ROCm; verify the declared checkpoint bytes/hash; run one real paired preflight. If input geometry differs, add an explicit audited paired spatial policy rather than bypassing validation. Then freeze eligible Train/Val IDs, actual training manifest and runner before comparing pretrained versus random. Neither the training runner nor the budget is claimed implemented/frozen here.

The frozen probe alone cannot refute NASA transfer. A separately predeclared adaptation comparison is appropriate if required. Existing 28-channel B0 is not a same-input baseline. P3.2 negative evidence, Test HOLD and all earlier experiment assets remain unchanged.

## Train-membership correction

Independent review identified that the original preflight trusted the batch's `split=train` declaration. The correction authenticates the historical P31 plan and metadata archive with fixed SHA256 roots, reconstructs the 5,135-row Train ID→Geology mapping, verifies stored split/chunk membership/ordered IDs/class counts, and rejects unknown, Val/Test or wrong-target batch rows before any model execution. It records authority hashes in the receipt. New subprocess regressions cover those cases and mutated authority files. This does not prove raw pixels came from the named HDF5 row; the local extractor still owes that provenance check. No real model or data experiment was run. The actual training runner remains **NOT IMPLEMENTED** and its budget **NOT FROZEN**.
