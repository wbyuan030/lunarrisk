# P3.2 LunarJEV-lite progress

Updated 2026-11-05. **Implementation, three frozen-context caches, all18 paired head/control runs, exact independent-process prediction replay and final audit COMPLETE/PASS.** Current J does not improve on B0 or B2. [Final comparison](p3_2_lunarjev_mvp.md), [J1/J2/J3 decision](p3_2_decision.md). Remote design `3e2599a` was pulled fast-forward before implementation. Original P3.1 six runs/settings/evidence are unchanged.

| Item | Actual state | Evidence |
|---|---|---|
| P32-01 frozen context | PASS: all three encoders exported and verified | [Cache report](p3_2_context_cache.md), [manifest](evidence/p32_20261105/context/manifest.json); authenticated best weights/receipts, frozen parameters/BN, Train/Val only,16×256 FP32 tokens and exact pooled mean; each110,118,080 bytes |
| P32-02 candidate semantics | Local taxonomy tier complete; full geological definitions/text model not evaluated | [Public catalog](../protocols/p32_candidate_catalog.json);49 symbols plus audited Age map,54 fixed candidate rows/120D catalog-only TF-IDF, no invented definitions or sample-specific text |
| P32-03 four heads | PASS: CPU/GPU controls and actual-cache training/replay | B0 ordinary, B1 ID, B2 pooled taxonomy, J spatial query; frozen text, no learned ID shortcut in B2/J, masks/permutations/subsets and shared-context two-task interface |
| P32-04 paired runs | COMPLETE/PASS:12 primary +6 fixed descriptor controls | [Actual pre-score manifest](../protocols/p32_head_run_manifest_v2.json), [aggregate](evidence/p32_20261105/runs/aggregate_final.json), [audit](evidence/p32_20261105/runs/sequence_verification.json) |
| P32-05 J1/J2/J3 | Reported; semantic/spatial benefit not established | [Decision](p3_2_decision.md), [paired diagnostics](evidence/p32_20261105/runs/paired_comparisons.json), [shared-context timing](evidence/p32_20261105/shared_context_timing.json) |

| Method | Val Geology Accuracy | Val full49 Macro-F1 |
|---|---:|---:|
| B0 ordinary | **46.8110±.8183%** | **.255488±.014813** |
| B1 ID | 44.9928±1.1787% | .212279±.009432 |
| B2 pooled taxonomy | 46.1472±2.0138% | .221628±.014017 |
| J spatial taxonomy | 44.9062±1.0894% | .202774±.020098 |

All fixed seeds42/43/44 included; sample std/ddof=1, Accuracy deviations in percentage points. J−B2 full49 F1 is−.018854±.029749 and J−B0−.052714±.012323. Correct-versus-fixed-permuted descriptors give B2−.009904±.008298 and J−.002867±.018259. Full language semantics remain untested; these lexical prototypes include identity-like literal-symbol features. All development results use the previously reused northern Val, with no confirmation/global/unseen-class claim. Age supervision/aggregation and confidence limitations are reported in full.

Actual trainable head parameters: B0 337,462, B1 305,921, B2/J329,729. The frozen Geology encoder is shared per seed and does not train. All methods receive identical extra Geology supervision for derived Age, the same cached observations/targets and equal Geology+Age CE. Budget is fixed before scores: FP32/batch128/AdamW.001/wd.01/max50/patience8; strict Val Geology full49 F1 selects one checkpoint for both tasks. No hyperparameter or prompt search, added scratch seeds, or favorable-seed selection.

Independent final audit verifies **162 live artifact hashes,180 archived artifact hashes,3 cache hashes and108 prediction CSVs**. Every CSV row's ID and target matches its frozen cache, all probabilities/metrics/selection/stopping/checkpoints/aggregates pass, and each run's six CSVs reproduce exactly in a fresh process. This is prediction replay, not repeated-training determinism. Checkpoints/cache/raw pixels remain local under ignored `outputs/p32/`; reports/receipts/history/sanitized logs/exact predictions/hashes are public evidence.

Resident Train pixel timing on RX7900XT, batch16, FP32,5 warmups/20 synchronized repeats: shared visual context2.014ms; encoder+both task heads B0 **2.330ms**, B1 3.033ms, B2 3.080ms, J **3.128ms**, averaging per-seed medians. Pixel I/O and transfer are excluded. All methods already share one visual encoding; J has no speed advantage. Text catalog load/verification/TF-IDF construction takes3.016ms once, warmed CPU median1.813ms and is excluded from inference. [Text timing](evidence/p32_20261105/text_precompute_timing.json).

All18 training windows total243.156 seconds/2,557,230 observation-visits; train/final export wall totals270.154 seconds excluding startup/asset verification/replay/probes. Cache export took218.051/224.161/222.694 seconds plus49.669 seconds for the single prefix authentication pass. Full upstream integrity remains unverified; original northern subset selection and all Test exclusions remain unchanged.

## Verification and engineering failures

The final suite passed **626 tests, zero skips**; [verification](evidence/p32_20261105/implementation_v2/verification.json), compileall PASS. Four real-GPU synthetic heads passed finite forward/backward and exact repeated inference. Existing runtime used supported fallback BLAS/attention paths; no experimental flags were enabled. No shared installation or torch/ROCm replacement occurred.

The first full-suite invocation omitted inherited trusted runtime for an existing subprocess test; the next exposed a same-size file-mutation fixture that could write twice within one filesystem clock tick. Explicitly advancing its temporary mtime made this observable-metadata fixture deterministic; the planner and P3.1 training sources remain unchanged. The then-current624 tests passed before the final two GPU initialization regressions were added. [Preserved earlier verification/logs](evidence/p32_20261105/implementation/verification.json).

The first two cache attempts failed closed before pixel export on adapter assumptions about old provenance schema names: reports omit `plan_sha256`, while plan `sha256`/`bytes` correspond to checkpoint `data_sha256`/`data_bytes`. Corrected `context_03` completed; failed empty namespaces are preserved locally. First head launch `heads_01` failed before any epoch at cold GPU peak-counter reset (`RuntimeError: Invalid device argument`). Explicit device initialization, two unit regressions and a real cold-process preflight pass. The first run manifest is preserved; the corrected source and **identical scientific configuration** were sealed in v2 before any scores. Active completed runs are `outputs/p32/heads_02`.

## Authorization and next bounded item

The latest user instruction permits needed models/data **directly without proxy**, superseding older blanket no-download wording. No new data/weight requests were needed or issued in this tier. The retained33,964,443,553-byte HDF5 partial was not resumed merely to show progress. No paid resources, four-GPU work, package installation or shared-environment changes.

A local Qwen3-Embedding snapshot was inventoried but not used; complete model-card/license/integrity/runtime verification is absent. The next bounded item is source-verified geological class definitions and one audited frozen text representation, with a separate full-semantic manifest. Do not repeat this completed matrix or change budgets to rescue negative results. The full-data route remains separate, Test cache/scoring HOLD, old LROC Gate3 NOT_FROZEN/Gate4 HOLD, and P2.2 negative conclusions unchanged.

This stage's code/evidence is ready for normal main synchronization; publication outcome is recorded in the final delivery rather than inferred from local success.
