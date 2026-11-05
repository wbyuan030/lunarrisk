# P3.3 RX7900XT driver upgrade readiness

2026-11-05, based on `6bb0108`. The user stopped the five-minute Git loop, then requested driver-upgrade verification and a conditional upgrade if no problems were found. **A concrete driver candidate and bounded package transaction have been prepared, but unattended installation/reboot is not ready.** A stale installed maintenance record, incomplete independent recovery backups and unconfirmed console/reboot availability require resolution. No driver, firmware, system package, boot configuration or runtime was changed.

[Verification](evidence/p33_20261105/driver_upgrade_01/verification.json) · [Host readiness](evidence/p33_20261105/driver_upgrade_01/host_readiness.json) · [Package hashes/control scripts](evidence/p33_20261105/driver_upgrade_01/package_inspection.json) · [APT simulations](evidence/p33_20261105/driver_upgrade_01/apt_simulations.json) · [Recovery review](evidence/p33_20261105/driver_upgrade_01/recovery_review.json) · [Plan](../protocols/p33_driver_upgrade_candidate.json).

## Supported candidate and limits

AMD's [Radeon ROCm7.2 matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html) includes RX7900XT and Ubuntu22.04.5 Desktop/HWE kernel6.8. The host matches that OS/kernel family. The [current driver matrix](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/compatibility/compatibility-matrix.html) explicitly lists driver30.10.2 with both ROCm6.2.x and7.2.x. The [versioned ROCm7.2 matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/user-kernel-space-compat-matrix.html) also lists the30.10.x family. The [ROCm7.0.2 matrix](https://rocm.docs.amd.com/en/docs-7.0.2/compatibility/compatibility-matrix.html) lists driver30.10.2 and Ubuntu22.04.5/HWE6.8. These intersecting support statements establish a candidate; they do not certify this exact `6.8.0-138` build, old desktop graphics libraries, Python packages or NASA execution.

The exact two-package target from AMD's [30.10.2 jammy repository](https://repo.radeon.com/amdgpu/30.10.2/ubuntu/dists/jammy/main/binary-amd64/Packages) is:

| Package | Candidate version | Bytes | SHA256 |
| --- | --- | ---: | --- |
| amdgpu-dkms | 1:6.14.14.30100200-2226257.22.04 | 11,891,236 | ded060d9fdfaee21c8fe7c2cf717562ae44cee7b40f65463e3427e2ed906c20f |
| amdgpu-dkms-firmware | 30.10.2.0.30100200-2226257.22.04 | 16,434,424 | 586e4434f772be909093966de80c933de2c4549434a04ee844503fd42c6839aa |

The old6.2.4 DKMS and firmware packages remain available from AMD's [6.2.4 repository](https://repo.radeon.com/amdgpu/6.2.4/ubuntu/dists/jammy/main/binary-amd64/Packages). All four packages were acquired with proxies disabled, retained only under ignored `outputs/p33/driver_upgrade_01/packages/`, and checked against signed repository metadata. Both Release signatures validate against the already installed AMD public key; Release→Packages→payload SHA256 chains pass. Total acquired package payload is54,553,752 bytes. No package code was executed: only `dpkg-deb --ctrl-tarfile` and in-memory static control-file reads were used. Public metadata snapshots and exact local commands are retained under the same ignored output directory; Git contains sanitized records and hashes, never package payloads.

Do not use the broad Radeon `graphics,rocm` installer for this plan: that changes substantially more than the two-package scope and the Radeon guide recommends uninstall/reinstall rather than in-place upgrades. The [driver-only package-manager guide](https://instinct.docs.amd.com/projects/amdgpu-docs/en/docs-30.10.2/install/detailed-install/package-manager/package-manager-ubuntu.html) separates the kernel driver from ROCm userspace. No removal, repository edit, `apt update`, global upgrade, `autoremove`, library replacement or container pull occurred here.

## Local transaction checks

Secure Boot is disabled. Headers and extra modules for running138 and fallback124 are installed, together with DKMS and required build helpers. `dpkg --audit` is empty and there are no APT holds. The current APT cache points at6.2.3, whereas installed driver packages are6.2.4; a default `apt install` is therefore not an exact upgrade or rollback plan.

Explicit local-package APT simulation with `--no-download --no-install-recommends --no-remove --allow-downgrades` changes exactly DKMS plus firmware and removes zero packages. Other graphics and ROCm package versions remain untouched at dependency-resolution level. Firmware is numerically a **downgrade** because old firmware has Debian epoch `1:` and new firmware has no epoch. Both target files must be explicitly selected; silently retaining the old firmware is not the reviewed target.

A second simulation restores the exact old package pair using a copied hypothetical status file with only target versions/Depends changed. It also changes only those two packages. This is a bounded dependency simulation, not an actual post-upgrade system or demonstrated rollback. No maintainer script, kernel build, initrd generation, bootloader update or reboot ran. `--no-remove` does not stop maintainer scripts from removing DKMS modules.

## Concrete recovery issues

1. `/var/tmp/amdgpu-dkms-kernels` still lists only `6.8.0-48-generic`, whose kernel package and initrd are absent, although residual module-directory/config files remain. The installed old `postrm` handles `upgrade` and runs `update-initramfs -u -k` for this stale entry before new `postinst` can refresh it. Local `update-initramfs` does not simply skip a missing image. This may build an unintended48 image or fail depending on remaining files/hooks; no failure was reproduced. Back up the record and reconcile it deliberately against actual138/124 state before changing packages.
2. Both postinst scripts write an amdgpu blacklist before invoking DKMS under `set -e`. A helper failure can exit before cleanup. Check the prior blacklist state, actual module results and residual blacklist before any reboot.
3. The new package's `prerm` removes its DKMS version with `--all` on upgrade, including rollback to old packages. Rollback is another build/configuration operation that can fail. It is not an atomic restoration.
4. Both138 and124 have vmlinuz/initrd files and generated GRUB normal/recovery entries. However, persistent initrd backups are disabled, GRUB is hidden/default0, and automatic DKMS coverage of124 is unproven. Preserve matching old modules/firmware/initrd outside package-managed locations and verify fallback coverage explicitly. Two kernel names alone do not preserve two complete independent driver environments.
5. Both the discrete GPU and boot-VGA integrated GPU use amdgpu. A failed module can affect desktop recovery as well as compute. The host already had a reboot-required flag for libc6/gnome-shell before this work. Local or out-of-band recovery access and a maintenance window are still unconfirmed; do not initiate an unattended reboot.

These are preparation gates, not proof that the candidate cannot work. Existing free-space observations are ample for the four package payloads, but are not a measured DKMS build/backup peak or a guarantee of boot recovery.

## Proposed attended sequence

1. Confirm someone can use the local console or equivalent recovery access during a maintenance window. The user's conditional upgrade permission is already recorded; do not repeatedly ask whether to research the same candidate.
2. In the approved maintenance scope, back up current package inventory, public AMD source/pin files, relevant driver/firmware/module trees, both known-good initrds and boot policy to ignored local storage. Verify backup hashes and readability; keep old package pair already acquired. Do not collect credentials or unrelated configuration.
3. Reconcile the stale kernel-target record using actual installed138/124 targets. Do not delete leftover kernels or modify other boot policy as an incidental cleanup. Re-run the exact two-package simulation immediately before execution; stop if package scope expands or dependency/package state is unhealthy.
4. Install only the two authenticated local30.10.2 packages with explicit versions/files and the same no-download/no-recommends/no-remove constraints. No broad installer, global upgrade or userspace ROCm change. Check each exit status; never chain an unconditional reboot.
5. Before reboot, verify completed package configuration, DKMS build logs and explicit module/vermagic/dependency/initrd coverage for138 and124; check blacklist/hook state and the recovery backup. If anything fails, stop before reboot and repair or restore old pair with explicit both-kernel rebuild/verification. Generated GRUB entries exist but physical menu accessibility remains a recovery prerequisite.
6. Reboot only in the agreed window. Verify display/login and loaded driver, then perform a bounded check of the existing ROCm6.2/PyTorch environment. A driver matrix cannot replace that functional check. Do not rerun scientific experiments or historical75/701 suites for this purpose.
7. A new private NASA GPU runtime still needs its own exact interpreter, complete GPU dependency/build lock and budget. AMD's [Radeon PyTorch7.2 guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/native_linux/install-pytorch.html) validates torch2.9.1, below NASA's frozen torch≥2.12 floor; generic PyTorch2.12.1+ROCm7.2 wheel availability is separate from AMD's validated Radeon combination. No NASA runtime, checkpoint or training is authorized merely by successful driver maintenance. Test remains HOLD.

Actual driver installation, system-file repair, backup restoration, DKMS build, reboot, old-runtime execution and NASA execution are **NOT_RUN**. The stopped Git loop remains stopped. Historical scientific code, pins, scores and evidence are unchanged.
