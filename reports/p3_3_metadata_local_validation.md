# P3.3 offline runtime metadata tool: local validation

2026-11-05, based on `2f59986`. **18 newly added offline metadata tests PASS, zero skips,0.425s; the three new files compile.** This verifies the standalone auditor and recorded candidate declarations. No NASA runtime was provisioned or executed, and compatibility remains NOT_ESTABLISHED.

[Local verification, actual commands and hashes](evidence/p33_20261105/metadata_local_01/verification.json) · [New focused log](evidence/p33_20261105/metadata_local_01/focused_tests.log) · [CPU candidate research](p3_3_cpu_feasibility.md).

## Actual checks

- Ran only `test_p33_runtime_metadata.py` in the existing private runtime. Tests use injected metadata and CLI fixtures that enforce no torch/NASA imports, socket/network, subprocess helpers or hardware access in the audited process. Exclusive output creation is checked. The CLI fixtures read distribution metadata as part of new-tool testing; no separate inventory or old-environment compatibility audit was repeated.
- Compiled only `scripts/p33_runtime_metadata.py`, `src/lunarrisk/nasa_transfer/runtime_metadata.py` and `tests/test_p33_runtime_metadata.py`. Diff whitespace check PASS. No full-suite pass is claimed for this revision, and the prior75/701 tests and real native256 preflight were not rerun.
- Authenticated all11 upstream CPU-candidate delivery file hashes before updating handoff documents. All22 source hashes in the previous native256 snapshot still match. Existing model wrappers, fixed NASA/model pins, policies and scientific implementation remain unchanged.
- Independently recomputed28 candidate pins against immutable NASA requirements and all72 recorded applicable direct-to-direct constraints from the archived requirements. Rechecked Python3.12/version/wheel tags and non-yanked status for26 non-torch wheels, plus both CPU torch/vision tags and Python declarations. These are checks against saved metadata, not a public metadata refetch or binary audit.
- Recomputed advertised sizes:194,106,305 bytes for the CPU pair plus250,752,565 for26 other direct wheels =444,858,870 bytes. This remains an advertised direct subtotal, not a complete download/storage budget or verified payload hash.

## Scope and remaining gaps

The audit tool reads direct installed-distribution metadata; even DIRECT_METADATA_MATCH reports compatibility NOT_ESTABLISHED and model/upstream execution NOT_RUN. It does not validate transitive closure, native libraries, source cleanliness, CPU model behavior or GPU support. No tool implementation or original guard was changed by this local follow-up.

The existing candidate plan remains unapproved for provisioning. Full transitive dependencies and extras are unresolved; the examined ANTLR4.9.3 metadata lacks a compatible wheel, while the other permitted4.9 releases and source-build path are not exhaustively reviewed. Interpreter patch/source, complete artifact hashes, payload verification, full download/storage budget, normal official imports and constructor memory remain unknown. No resolver/build backend, environment creation, installation, package/data/weight download, driver change, GPU inspection, NASA import/construction or Test access occurred in this phase.

The GPU support conclusion is preserved: the installed driver6.2.x row does not cover candidateROCm7.2. CPU candidate preparation does not solve the GPU training objective or establish transfer benefit. Keep the working runtime, historical evidence and P3.2 results; await a separately scoped path/provisioning decision. Do not repeat these completed local checks without implementation or evidence changes.
