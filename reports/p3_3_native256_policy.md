# P3.3 Moonstone-native256 policy implementation

2026-11-05, based on `e9d0f61`. Safe fast-forward integrated the local runner validation:64 P33 and full690 tests passed locally with zero skips at the previous commit. That historical result does not validate the code added here.

**New result:** explicitly declared Moonstone-native256 downstream transfer policy, fixed config identity, unchanged tensor/patch geometry and normalization, explicit anisotropic physical-scale provenance, and policy binding through preflight/data/model/lock/run/replay. Default strict behavior still rejects the authentic pre_resize512/input256 mismatch. See [policy explanation and local handoff](../docs/P3_3_MOONSTONE_NATIVE256.md).

## Actual verification

- Focused P33 suite: **75 discovered,59 PASS,16 SKIP**, no failures/errors,3.201s. The sixteen skipped torch/HDF5 checks require dependencies absent here. [Log](evidence/p33_20261105/native256_policy/focused_tests.log).
- `python3 -m compileall -q scripts src tests`: PASS. `git diff --check`: PASS.
- Independent source/diff review found no blocking implementation defect; independently ran16 spatial/runner tests, all PASS. Review prompted correct factory function attribution and an additional negative CLI config test, both implemented.
- Full regression suite: **INCOMPLETE, NOT PASS**. A legacy LROC test reached a network request and execution was stopped by the no-large-download boundary. The retained log is partial, with no suite summary, and includes an earlier import ERROR. No retry or alternative route was attempted. No test process remained after the stop. [Partial log](evidence/p33_20261105/native256_policy/full_tests_partial.log.gz).
- The final focused checks cover strict default rejection, explicit native geometry, no resampling/mutation, missingness and degree guards, pinned-config requirement, physical-scale declaration, dataset propagation, matching preflight/lock policy and refusal to switch at run, plus replay flag propagation. The new CLI negative test proves a correctly self-hashed synthetic YAML cannot enable the native policy.

Direct-function tests use synthetic arrays/domains and explicitly supplied test digests. They do not prove authentic config-file loading. Production CLIs and the official constructor verify actual file bytes. Mock `LOCKED` output in the log is not a real experiment lock. No actual NASA config/weights/raw pixels were available here; no official model/runtime, GPU execution, complete real-cohort scan, training or independent official replay occurred. [Machine-readable verification and hashes](evidence/p33_20261105/native256_policy/verification.json).

## Source rationale and limits

At pinned256/16 geometry, the official image embedding has16×16 positions and directly uses them at matching runtime shape, without positional interpolation or patch-weight resizing. The strict pre-load selected-state shape audit is retained because upstream loading otherwise supports adaptations. dtm/slope use UntokLunarTransform: inherited resize512→normalization then crop+flip, not generic LunarTransform's crop-resize path. The new protocol intentionally does not reproduce that dataset geometry. Moonstone's236.901m nominal north–south and latitude-varying east–west spacing are not NASA-native60m physical sampling.

This establishes a declared, guarded engineering input path only. It does not establish checkpoint/runtime compatibility, original raster lineage, actual prepared-input success or transfer benefit. No packages or shared runtime were changed, no model/data downloads were initiated by the new P33 path, no user computer was accessed, and no Test pixels/scores were used. P3.2's18 runs, old budgets and all prior evidence remain untouched.

## Next local action

Pull main safely and inspect concurrent work. Validate the new focused tests in the existing private runtime; all tests still use synthetic backbones, not official NASA. Then, if the existing authenticated raw Train batch and source provenance are suitable, create a new provenance file explicitly declaring this spatial policy and run **input-only** preflight with the original audited config and policy flag. Preserve the old strict-policy failure. This may establish actual input preparation, not official model success. Runtime, checkpoint, actual paired preflight, complete fixed-cohort scan and budget lock remain required before training. Do not install/replace environments, acquire weights while runtime remains unresolved, resume large data transfers or access Test.
