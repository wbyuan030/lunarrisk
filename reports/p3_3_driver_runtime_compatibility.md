# P3.3 driver/runtime compatibility decision

2026-11-05, based on `69763c5b8514a2f8a6239311b2555869c0873f4e`. **The driver-release mapping is now established by official package metadata. The installed release family is not listed for ROCm 7.2. No verified official GPU runtime has been found that both leaves this host unchanged and meets the pinned NASA requirements.** This is a support-evidence result, not an observed execution failure.

[Public evidence](evidence/p33_20261105/driver_support_01/public_evidence.json) · [Current runtime plan](../protocols/p33_private_runtime_plan.json) · [Validation](evidence/p33_20261105/driver_support_01/verification.json).

## What the new evidence resolves

The [AMD 6.2.4 package directory](https://repo.radeon.com/amdgpu/6.2.4/ubuntu/pool/main/a/amdgpu-dkms/) lists `amdgpu-dkms_6.8.5.60204-2070768.22.04_all.deb`, exactly matching the installed version after its Debian epoch `1:`. This maps the recorded package version to driver release 6.2.4 / family 6.2.x. No `.deb` was downloaded, and installed binary identity was not compared. The earlier host audit already matched loaded/on-disk module versions; module 6.8.5 is a different version namespace, not ROCm 6.8.5. AMD's [versioning explanation](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/rocm-offline-installer.html) distinguishes the earlier matched driver/ROCm release names and the independent 30.10 naming introduced with ROCm 7.0.

The [current official driver matrix](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/compatibility/compatibility-matrix.html) lists driver 6.2.x with ROCm 6.0/6.1/6.2/6.3/6.4/7.0, not 7.2. The [versioned 7.2 matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/user-kernel-space-compat-matrix.html) is consistent. Installing a venv or container alone does not change this host kernel-driver gate. The correct classification is **not in the published support matrix**, not proof that execution can never work.

## Paths checked and their limits

- **Keep driver, use an older official ROCm wheel:** the checked [ROCm 6.4 torch index](https://download.pytorch.org/whl/rocm6.4/torch/) tops out at 2.9.1 and [ROCm 7.0 index](https://download.pytorch.org/whl/rocm7.0/torch/) at 2.10.0 for the relevant modern ROCm builds. Neither meets pinned NASA torch >=2.12. No wheel payload was fetched. This is a bounded official-index check, not an exhaustive rejection of all future, third-party, nightly or custom source builds.
- **Roll back NASA to an older official ancestor:** both dependency files have only their initial change, at root commit [4eb8d61](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/commit/4eb8d610bc5cf0e10e8f4ddf2b7705fb00afd9a0). Its requirements already declare torch >=2.12 and torchvision >=0.27,<0.28, and pyproject requires Python >=3.11,<3.13. The four commits to our pin do not lower these floors. The [comparison](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/compare/4eb8d610bc5cf0e10e8f4ddf2b7705fb00afd9a0...d54c67aad513cb9daca444afa425cfb278e4fbf8) changes inference normalization/metadata helpers despite unchanged architecture files. Rollback offers no dependency escape and is not selected. Unrelated branches/forks were not audited.
- **Keep current NASA pin and use torch 2.12.1/vision 0.27.1 + ROCm 7.2:** package declarations and published wheel availability are known; this host's driver support is not met. The matrix has potential driver families 6.4.x or 30.10.0/.1/.2 containing both old ROCm 6.2 and new 7.2. They are investigation candidates, not approved upgrades: exact RX7900XT/Ubuntu/kernel support, rollback and effects on the existing stack are unvalidated. Newer is not automatically safer because other newer rows drop ROCm 6.2.
- **Private CPU runtime:** [PyTorch publishes a CPU recipe for the same 2.12.1/0.27.1 pair](https://pytorch.org/get-started/previous-versions/), so a separately approved, bounded CPU import/construction feasibility study could avoid the GPU driver gate. Full dependency closure, memory and performance remain unknown. It would not establish RX7900XT compatibility or satisfy the GPU training objective. No CPU environment or execution is selected here.

Lowering dependency bounds, bypassing imports/guards, modifying upstream code, or building against an unlisted stack would be a separately disclosed and unvalidated compatibility experiment. None was selected or performed.

## Git handoff and stopping condition

Read the completed host audit; **do not repeat inventory, native256 input preparation, the 75/701 tests, or old P3.2 experiments**. The missing fact is no longer driver identity: the concrete issue is lack of published support for this candidate on the installed family. With the current no-system-change boundary, stop before provisioning. Do not resolve by changing packages, pulling a container, adjusting the driver, downloading assets, or silently moving the NASA/model pins.

A later decision may authorize a precisely scoped CPU-only feasibility study, or a separately reviewed host-driver path. A driver path first needs an exact supported GPU/OS/kernel target and rollback plan, with explicit approval before any system change; a matrix intersection alone is insufficient. Any provisioning path still needs the full platform-specific dependency lock, artifact sizes/hashes and an approved download/storage budget. No install command is provided or authorized by this report.

## Validation and unchanged science

Official public metadata and fixed-revision GitHub history were reviewed independently. New JSON, local links, plan consistency, changed-file scope and whitespace were checked; see the validation record. No scientific code, source/model pin, native256 policy, physical-scale claim, budget or prior evidence changed. Prior host evidence is preserved as a historical observation, not rewritten.

No desktop access, install, environment creation, package/model/data download, upstream import or constructor, model forward/backward, training, full-cohort scan or Test access occurred. The completed local tests and single Train input-only PASS are not new runtime evidence. Official model execution remains NOT_RUN and transfer benefit NOT_ESTABLISHED.
