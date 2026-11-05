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

## Current host decision: driver mapped, candidate unsupported by matrix

The completed local audit at `69763c5` records Ubuntu22.04.5/kernel6.8.0-138/glibc2.35 and amdgpu package `1:6.8.5.60204-2070768.22.04`. New official package evidence maps that version to driver release **6.2.4 / family 6.2.x**. The published driver row does not include the candidate ROCm7.2. This supersedes the earlier unresolved mapping, not the historical host measurements. It is a support-matrix exclusion, not an empirical execution failure. See the [decision and official sources](../reports/p3_3_driver_runtime_compatibility.md) and [public evidence](../reports/evidence/p33_20261105/driver_support_01/public_evidence.json).

The checked official ROCm6.4/7.0 torch indexes offer at most2.9.1/2.10.0, below NASA's declared floor. The initial official ancestor already has the same dependency floors, so a source rollback does not solve them. A venv/container does not change the host kernel driver. No verified official GPU combination has been found within the unchanged-host boundary.

The exact Python3.12/torch2.12.1/vision0.27.1 candidate is still uninstalled and unvalidated. AMD's [Radeon7.2 matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html) lists RX7900XT/Ubuntu22.04.5 HWE, but its production-tested torch2.9.1 entry does not certify this NASA combination.

## Local Git handoff: stop before provisioning

Read [the completed host audit](../reports/p3_3_host_runtime_audit.md) and current decision; do not repeat OS/package/storage inventory, native256 input preparation,75/701 tests or old experiments. Do not install, pull containers, change the driver, lower requirements or silently switch NASA/model pins. `pip --dry-run` can fetch wheels/sdists and is not a zero-download probe.

Potential later choices are a bounded CPU-only import/construction feasibility study (not GPU compatibility), or a separately reviewed host-driver path. Driver families6.4.x and30.10.0/.1/.2 have matrix rows containing both oldROCm6.2 and new7.2, but no exact RX7900XT/OS/kernel upgrade or rollback has been selected or validated. Neither option is permission to execute. With the current no-system-change boundary, the GPU run remains blocked pending a scoped decision, not another inventory cycle.

## Later gates require a separate decision

No provisioning path is currently selected. Obtain a separate decision for any resolution/provisioning phase after its platform feasibility is established. Bounded public metadata research can continue read-only, but any resolver that might fetch wheels or sdists must have an explicitly allowed scope and download limit before it runs. Prepare a complete platform-specific dependency lock with version/source/hash/size information; obtain explicit approval before private interpreter/environment creation, installation or large acquisition. Stop on conflicts or missing compatible wheels. Do not substitute a CPU build as an answer to RX 7900 XT compatibility, widen package bounds or upgrade host ROCm/driver implicitly.

Any approved environment must keep the working Python 3.10/torch 2.5.1+rocm6.2 untouched, avoid system-site-packages and inherited old ROCm library paths, and preserve the fixed NASA checkout clean. Do not install upstream editable into the audited checkout: generated metadata can violate the wrapper's clean-tree requirement. The existing wrapper imports the source directly; dependency installation and any project packaging must preserve that invariant.

Only then validate distribution closure, normal official imports, actual device/library identity, constructor memory, and real finite paired forward/backward in fresh processes. The wrapper's current Python/source/config guards alone do not check every declared distribution bound and are not a compatibility certificate. Checkpoint acquisition/strict loading, unchanged full-cohort scan and real budget lock remain separate later steps. Transfer benefit remains NOT_ESTABLISHED.
