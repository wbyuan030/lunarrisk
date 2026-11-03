# P3.1 progress — 2026-11-03

This report separates implementation, runner validation, acquisition and formal results. P3.0 evidence and P2.2 negative results remain unchanged.

| Item | Current state | Evidence / next action |
|---|---|---|
| P31-01 scalable runner | Implemented; validation in progress | `scripts/p31_train_classifier.py`, `src/lunarrisk/benchmarks/classifier.py`, `tests/test_p31_classifier.py`; streaming hash, mini-batch Train/Val, final partial batch, stateful checkpoint, batched feature export |
| P31-02 representative complete crops | Offline lock/dry-run complete; real I/O pending | `protocols/p31_loader_pair_plan.json`, `reports/p3_1_loader_pair_validation.md`; IDs 2705/2700, all seven groups, 27 available channels; roughness metadata absent throughout Train/Val |
| P31-03 ordinary baselines | Design locked; formal runs BLOCKED | `protocols/p31_baseline_eval_plan.json`; full fixed HDF5 not locally acquired/authorized |
| P31-04 candidate ID control | Identity test and latency runner implemented; measurements pending | `tests/test_p31_candidate_control.py`, `scripts/p31_candidate_control.py`; no semantic head or unseen-class claim |

Formal baseline settings are fixed before scores: independent task scratch CNN and Train empirical prior; official stored split; full-K Macro-F1 matching upstream absent-class zeros, supported macro separately; strict Val full-K best selection with earliest tie; AdamW/cosine, maximum 50 epochs, patience 5, batch 16, FP32, workers 0; seeds 42/43/44 after first-seed validation. FP32, retained final batch and HDF5-derived pixels are disclosed differences from upstream. Test scoring remains HOLD.

The initial transfer ledger remains 78,502,175 returned body bytes / 83,468,143 conservative bytes, under the original 1,073,741,824-byte cumulative cap. The full-crop lock reserves at most 81,788,928 new body bytes; no P3.1 network execution has occurred at this implementation checkpoint. The original 74,449,119,347-byte asset proposal is still `PENDING_USER_DECISION_NOT_AUTHORIZED`.

The host has no installed torch/h5py. Automatic approval rejected a delegated test command that would write temporary files in the container under that delegation's read-only permission. It did not execute. A safer host-only execution path uses private scratch copies of existing container packages and the host's existing ROCm libraries; no package installation, shared environment change or container filesystem write. This runtime is being verified separately; historical 482 tests are not a P3.1 result.
