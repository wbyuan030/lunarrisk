# P3.3 runtime planning research

2026-11-05, based on `c9245665fab0213b598b23031c50e9c68c997cf0`. Safe fast-forward integrated local 75 P33/701 full zero-skip passes, the offline LROC fixture repair and actual Train2705 native256 input-only success. Those completed checks were read, not rerun.

**New result:** a concrete, uninstalled private Python 3.12 / PyTorch 2.12.1 / torchvision 0.27.1 / ROCm 7.2 candidate, complete pinned NASA requirement ranges, and the next read-only host-driver feasibility check. [Actionable guide](../docs/P3_3_PRIVATE_RUNTIME_PLAN.md), [machine-readable plan](../protocols/p33_private_runtime_plan.json).

## Verification performed

- Read 27 public source files at exact NASA revision d54c67aad513cb9daca444afa425cfb278e4fbf8. Recomputed every Git blob: 27/27 match. Parsed 25 Python files as AST without importing/executing them. [Source manifest, SHA256 and conclusions](evidence/p33_20261105/runtime_plan_01/source_audit.json).
- Inspected official PyTorch release instructions and wheel indexes: matching cp311/cp312 manylinux_2_28_x86_64 ROCm 7.2 pair is listed. Inspected AMD Radeon and ROCm 7.2 matrices. These are public-document observations, not downloaded wheel, resolved dependency or hardware execution evidence. [URLs and findings](evidence/p33_20261105/runtime_plan_01/public_runtime_sources.json).
- Checked plan JSON, local links, requirement-range reproduction, source hashes and diff whitespace. No scientific Python, model code, budgets or old evidence changed. No full/focused regression rerun is needed for these documentation-only changes; no new test-pass count is claimed.

The normal package import eagerly loads training/data components. Even the backbone's core import graph reaches scipy, hdf5plugin, torchvision, h5py and pandas. A torch/einops-only workaround is not equivalent. Full external dependency closure remains unverified. The official constructor allocates decoder blocks transiently before pruning; memory fit has not been measured.

## Local host follow-up

The [read-only local audit](p3_3_host_runtime_audit.md) at `7784886` completed the missing OS/module/package/capacity facts: Ubuntu22.04.5/kernel6.8.0-138, amdgpu module6.8.5 and DKMS `1:6.8.5.60204-2070768.22.04`, with462,071,136,256 free bytes on the intended volume. The inspected official matrices do not map that exact installed package to their supported driver release row. Exact driver/runtime support remains NOT_ESTABLISHED; the local phase stops before provisioning. [Evidence](evidence/p33_20261105/host_runtime_01/verification.json). No completed tests/preflight or Python/package inventory were repeated.

## Original remaining blocker (host identity now recorded)

The existing host's amdgpu module/package identity and exact OS release were not captured by the prior Python/HIP inventory. The candidate's full dependency resolution and size/hash budget are also unfinished. New interpreter/environment installation is outside this phase. The local Git collaborator can next collect only the missing driver/OS/capacity facts; no repeated native256 checks, installations, host modifications or asset downloads are instructed.

Official runtime compatibility, model construction/loading/GPU gradients, whole-cohort eligibility, actual budget lock and transfer benefit remain unestablished. No user desktop was accessed by this cloud task, no packages/models/data were downloaded or installed, no upstream Python ran, and no Test pixels/scores or old P3.2 runs were touched.
