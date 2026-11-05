# P3.3 paired runner: local dependency-complete test validation

2026-11-05. Fast-forward pulled `423005e` after checking clean main and no running work. **All64 P3.3 tests and the full690-test local suite PASS, zero skips; compileall PASS.** These results cover the new paired runner on the existing private runtime. Official NASA construction/loading, GPU validation, full real-terrain scan, actual training and official fresh-process replay remain **NOT RUN**. [Verification/source hashes](evidence/p33_20261105/runner_local_01/verification.json).

| Check | Local result | Boundary |
|---|---|---|
| `test_p33_nasa*.py` | 64 PASS / 0 SKIP, 3.413 s | Executes all16 tests skipped in the cloud |
| Complete suite | 690 PASS / 0 SKIP, 20.699 s | New runner plus existing regression tests |
| compileall | PASS | No package installation |
| Synthetic HDF5 terrain slicing | PASS | Temporary generated fixture; no full real-prefix scan |
| Tiny frozen/affine training, paired initialization/order, selected-state reconstruction | PASS | CPU synthetic backbones; not official NASA |
| Lock/preflight/source drift/budget/output guards | PASS | Mocked successful assets and orchestration; no actual NASA lock |
| Real model compatibility/paired device preflight/training scores | NOT RUN | Prior runtime and spatial-policy blockers remain |

[Focused log](evidence/p33_20261105/runner_local_01/focused_tests.log), [full log](evidence/p33_20261105/runner_local_01/full_tests.log). The `LOCKED` line printed by a mocked lock unit test refers only to that temporary synthetic fixture. It is not an actual training manifest, an authorized executed budget, or a successful official-model receipt. Training unit tests reconstruct selected state and compare predictions; they do not establish the launcher's official NASA fresh-process execution.

The existing private runtime is Python3.10.12, torch2.5.1+rocm6.2, NumPy2.2.6 and h5py3.16.0. Torch/h5py availability lets these fixtures run without meeting official NASA runtime declarations. NASA's Python>=3.11,<3.13 and declared torch>=2.12 compatibility gates were not relaxed. No shared environment was replaced, no packages installed, no new source/weights/data downloaded and no GPU model experiment launched.

Actual commands, using the same existing private runtime as earlier local work:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
PYTHONPATH="$TRUSTED_RUNTIME" \
python3 -I "$JOB_TMP/p31_python.py" -m unittest discover -s tests -p 'test_p33_nasa*.py' -v

LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
PYTHONPATH="$TRUSTED_RUNTIME" \
python3 -I "$JOB_TMP/p31_python.py" -m unittest discover -s tests -v

python3 -m compileall -q scripts src tests
```

The previous [654-test/local terrain audit](p3_3_local_readiness.md) and the [cloud runner evidence](evidence/p33_20261105/paired_runner/verification.json) remain historical, unchanged. The current local invocation completed normally; it does not reinterpret or bypass the separate cloud tool-review rejection recorded in the upstream report. No Python scientific source changed in this local follow-up.

Next execution gate remains an explicit paired policy for authentic `pre_resize512/input_size256`, a compatible private NASA runtime preserving existing shared torch, verified physical provenance/finite-sentinel semantics, and a real paired preflight with fixed-checkpoint verification. Only then can the implemented runner scan the fixed cohort and lock a chosen budget before scores. The example budget is still **not frozen**. No policy bypass, cohort filtering, imputation, synthetic successful preflight, additional P3.2 runs or Test access was used. [Runner guide](../docs/P3_3_NASA_PAIRED_RUNNER.md), [current plan](../protocols/p33_nasa_transfer_plan.json). Upstream monitoring continues.
