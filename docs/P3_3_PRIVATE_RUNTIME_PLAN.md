# P3.3 private NASA runtime plan: candidate only

2026-11-05, after `c924566`. The local native256 input preparation and 75/701 test passes are complete; do not repeat them. This handoff adds **read-only source/package-platform research**, not an installation, accepted runtime, dependency lock or model result. [Plan](../protocols/p33_private_runtime_plan.json), [static source evidence](../reports/evidence/p33_20261105/runtime_plan_01/source_audit.json), [public runtime sources](../reports/evidence/p33_20261105/runtime_plan_01/public_runtime_sources.json).

## Concrete candidate, not permission to execute

A new private **Python 3.12** environment with the matched **torch 2.12.1+rocm7.2 / torchvision 0.27.1+rocm7.2** pair is a candidate. PyTorch publishes the pair and its official indexes list cp312 and cp311 Linux x86_64 manylinux_2_28 wheels. No wheel was downloaded or hashed, no dependency resolution or install was run, and the Python patch release/source is not chosen. [PyTorch releases](https://pytorch.org/get-started/previous-versions/), [torch index](https://download.pytorch.org/whl/rocm7.2/torch/), [torchvision index](https://download.pytorch.org/whl/rocm7.2/torchvision/).

This pair respects the pinned NASA Python >=3.11,<3.13, torch >=2.12,<3 and torchvision >=0.27,<0.28 declarations. Installing the latest torch/torchvision pair blindly is unsuitable: newer torchvision 0.28/0.29 falls outside that bound. Python 3.10, the existing torch 2.5.1, and the AMD recipe's torch 2.9.1 do not meet this pinned declaration. Do not lower bounds or remove the Python guard to force a run.

All 28 official requirements remain in scope, including NumPy >=2.4.4,<3. The [plan](../protocols/p33_private_runtime_plan.json) preserves their exact ranges from the [pinned requirements](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/blob/d54c67aad513cb9daca444afa425cfb278e4fbf8/requirements.txt). This is **not** a resolved requirements lock: transitive compatibility, available wheels, hashes, install footprint and download budget are still unknown. The optional wandb extra is not needed or selected.

## Why a minimal torch/einops fix is insufficient

A normal import of `terratorch_integration.lunar_backbone` first executes the package initializer. It eagerly imports registration, data adapter, task families, necks/decoders and determinism. This brings in TerraTorch, Lightning, TorchGeo, TorchMetrics and data/plotting dependencies even for our two-channel encoder path. Separately, modality_info→modality_transforms imports scipy, hdf5plugin and torchvision; io_utils imports h5py and pandas. hdf5plugin is unconditional in this path, even though the data adapter tolerates its absence elsewhere.

The selected dtm/slope constructor does not construct VQ/text tokenizers or diffusion models. That does not prove their declared packages, or others not directly reached by these NASA source files, can safely be omitted: external packages' import closure was not executed or exhaustively audited. Bypassing package initialization, stubbing modules, loading selected files outside normal imports or modifying upstream requirements would be a different compatibility approach, not a faithful installation. It is not selected here.

The base constructor initially allocates encoder **and decoder** blocks before our wrapper prunes unused decoder state. Plan for transient CPU allocation; no measured memory estimate or fit claim exists yet. With checkpoint_path=None the audited constructor does not request pretrained weights, but official import/construction has still NOT RUN.

## Host compatibility is the next missing fact

AMD's ROCm 7.2 Radeon matrix lists RX 7900 XT and Ubuntu 22.04.5 HWE/kernel 6.8. Its production-tested PyTorch entry is 2.9.1, not the candidate 2.12.1. Thus published PyTorch wheels plus listed GPU support are **not** AMD certification of this exact NASA environment. [Radeon matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html).

The existing report records kernel 6.8/glibc 2.35 and torch's HIP 6.2 build. That does not identify the host amdgpu driver or exact OS release. AMD's pinned ROCm 7.2 driver column lists 30.30.0,30.20.1/30.20.0,30.10.2/30.10.1/30.10 and 6.4.x; it does not establish that an unknown host or a 6.2.x driver supports 7.2. A venv or container does not replace the kernel driver. **Do not upgrade it or infer compatibility from old torch.cuda.is_available().** [ROCm 7.2 matrix](https://rocm.docs.amd.com/en/docs-7.2.0/compatibility/compatibility-matrix.html).

## Local host follow-up completed

At `7784886`, the read-only local check recorded Ubuntu22.04.5/kernel6.8.0-138/glibc2.35, matching loaded/on-disk amdgpu module6.8.5 and installed DKMS package `1:6.8.5.60204-2070768.22.04`. The intended private volume had462,071,136,256 bytes free. [Actual host audit](../reports/p3_3_host_runtime_audit.md), [facts](../reports/evidence/p33_20261105/host_runtime_01/host_audit.json), [support comparison](../reports/evidence/p33_20261105/host_runtime_01/support_comparison.json).

Host identity collection is complete. Neither official matrix provides a verified mapping from this exact module/package to its ROCm7.2 driver-release row; that support remains NOT_ESTABLISHED, not demonstrated incompatibility. Module6.8.5 must not be read as a ROCm version. The phase stops before provisioning or host changes. Do not repeat the host inventory, input check or tests without changed host/source evidence. Capacity does not resolve dependency sizes or authorize installation.

## Original local Git handoff: read-only collection completed

Do not repeat the Python search, package inventory, native256 preflight or701 tests already completed. Collect the facts missing from those reports:

1. Exact OS VERSION_ID, architecture and existing amdgpu module/package identity. Read-only sources can include `/etc/os-release`, `uname -r`, `modinfo -F version amdgpu`, or `dpkg-query -W amdgpu-dkms` where available. A missing command, blank module version or distro-provided module is **UNKNOWN**, not a reason to install tooling or guess a driver release.
2. Record available capacity on the intended private-environment volume. Keep absolute private paths, hostnames and hardware identifiers out of Git. Do not create the environment or clear caches to make room. The shared four-GPU machine and paid compute remain excluded.
3. Compare the exact supported combination against the linked official matrices, preserving the difference between source/package version compliance and hardware/runtime validation. If driver support remains unknown or would require host changes, stop and report that precise blocker.

No installation or Docker pull is authorized in this phase. Do not use `pip --dry-run` as a supposedly zero-download probe: dependency resolution may fetch large wheels/sdists. No big assets, NASA import/constructor, new model preflight, Test access or old experiment reruns are requested.

## Later gates require a separate decision

After host feasibility is established, obtain a separate decision for the resolution/provisioning phase. Bounded public metadata research can continue read-only, but any resolver that might fetch wheels or sdists must have an explicitly allowed scope and download limit before it runs. Prepare a complete platform-specific dependency lock with version/source/hash/size information; obtain explicit approval before private interpreter/environment creation, installation or large acquisition. Stop on conflicts or missing compatible wheels. Do not substitute a CPU build as an answer to RX 7900 XT compatibility, widen package bounds or upgrade host ROCm/driver implicitly.

Any approved environment must keep the working Python 3.10/torch 2.5.1+rocm6.2 untouched, avoid system-site-packages and inherited old ROCm library paths, and preserve the fixed NASA checkout clean. Do not install upstream editable into the audited checkout: generated metadata can violate the wrapper's clean-tree requirement. The existing wrapper imports the source directly; dependency installation and any project packaging must preserve that invariant.

Only then validate distribution closure, normal official imports, actual device/library identity, constructor memory, and real finite paired forward/backward in fresh processes. The wrapper's current Python/source/config guards alone do not check every declared distribution bound and are not a compatibility certificate. Checkpoint acquisition/strict loading, unchanged full-cohort scan and real budget lock remain separate later steps. Transfer benefit remains NOT_ESTABLISHED.
