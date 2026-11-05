# P3.3 local host and driver feasibility audit

2026-11-05, based on `7784886`. **Read-only host identity and storage collection completed; support for the exact installed driver with the proposed ROCm7.2 runtime remains NOT ESTABLISHED.** No environment was created, resolver started or driver changed. The completed75/701 tests and native256 input preflight were not repeated.

[Actual host facts and commands](evidence/p33_20261105/host_runtime_01/host_audit.json) · [Official matrix comparison](evidence/p33_20261105/host_runtime_01/support_comparison.json) · [Verification](evidence/p33_20261105/host_runtime_01/verification.json).

## Actual observations

| Item | Read-only result |
|---|---|
| OS | Ubuntu22.04.5 LTS, VERSION_ID22.04, Jammy |
| Kernel / architecture | 6.8.0-138-generic / x86_64 |
| libc | glibc2.35 |
| Loaded amdgpu module | version6.8.5, srcversion48F9726EF994ECFE76EC464 |
| On-disk module | modinfo version/srcversion match loaded sysfs values; vermagic matches running kernel |
| Installed package | amdgpu-dkms `1:6.8.5.60204-2070768.22.04`, install ok installed |
| Workspace/private-runtime volume | Same filesystem; free462,071,136,256 bytes, approximately430.3GiB at observation |

The loaded module is present; driver identity is no longer UNKNOWN. Its **support mapping** remains unknown. The module version6.8.5 must not be interpreted as a ROCm release, and the package suffix is not used to infer a release mapping. Private paths, hostnames and device identifiers are omitted from the published evidence. Capacity is a point-in-time observation, not reserved space or proof that the unresolved candidate footprint fits. No cache was cleared.

## Comparison and stopping point

The official [Radeon ROCm7.2 page](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html) lists RX7900XT and Ubuntu22.04.5 Desktop/HWE kernel6.8. The observed OS release/kernel family agrees with those fields; this is not validation of all host details or the exact proposed runtime. Its production PyTorch entry is2.9.1, while the NASA-compatible candidate is2.12.1.

The official [ROCm7.2 compatibility matrix](https://rocm.docs.amd.com/en/docs-7.2.0/compatibility/compatibility-matrix.html) lists AMD GPU Driver releases30.30.0,30.20.1,30.20.0,30.10.2,30.10.1,30.10 and6.4.x. The inspected pages do not map the actual module6.8.5 / exact DKMS package to that driver row. Comparing these different version namespaces numerically would be invalid. Consequently exact host-driver support is **not established**, rather than a demonstrated incompatibility.

Per the [runtime handoff](../docs/P3_3_PRIVATE_RUNTIME_PLAN.md), this phase stops here. A container or venv would still use the host kernel driver. No driver upgrade, private interpreter/environment creation, package installation, Docker pull, pip dry-run, wheel/sdist acquisition, NASA import/constructor, weight download, model preflight or training was performed. A later provisioning decision still needs explicit driver-support evidence, a complete platform-specific dependency lock and download/storage budget; none is supplied by the free-space observation.

## Verification and preserved boundaries

Validated new JSON, local document links, evidence hashes and documentation diffs. Production scripts, scientific Python, tests, data, budgets and historical evidence were unchanged. No new regression-pass count is claimed. The original runtime source audit and public-source observations remain separate historical evidence.

The existing Python3.10/torch2.5.1+rocm6.2 remains intact. Official runtime/model compatibility, checkpoint loading, actual paired gradients, full fixed-cohort eligibility, real budget lock, training and transfer benefit remain unestablished. Test HOLD and previous P3.2 results remain unchanged. Upstream monitoring continues; no repeated host inventory is needed unless the host or support evidence changes.
