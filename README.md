# LunarRisk / LunarJEV

**Active sprint — P3.2, 2026-11-04: build a task- and candidate-semantic LunarJEV prototype using existing lunar features.**

Start with [AGENTS.md](AGENTS.md), the [LunarJEV MVP design](docs/P3_2_LUNARJEV_MVP.md), and the [machine-readable scope](protocols/p32_lunarjev_design.json).

The six P3.1 scratch baselines are complete. They provide enough evidence of learnable input signal to begin a decision architecture; they do **not** already establish frozen-feature sufficiency, pretrained gains, language understanding or semantic gains. P3.2 is a **new design and implementation handoff, not completed model results**. More scratch runs, a full dataset download, pretrained checkpoints and performance profiling are not prerequisites.

## Completed reference results

The separate local-prefix track uses 5,135 original Train and 1,155 original Val samples, selected by complete chunks available in the retained northern HDF5 prefix. All fixed seeds42/43/44 are included:

| Task | Val Accuracy, mean ± sample std | Val full-K Macro-F1, mean ± sample std |
|---|---:|---:|
| Age, K=5 | 58.6436% ± 2.0070 percentage points | 0.403084 ± 0.035138 |
| Geology, K=49 | 43.6075% ± 2.2583 percentage points | 0.208791 ± 0.014977 |

These are development-set results on an availability-selected northern subset, **not the full Moonstone benchmark, a global holdout or an unseen-class test**. Each task has an independently initialized scratch encoder; Geology did not receive Age supervision. Archived prediction replays are exact. The final implementation suite records 579 passing tests; that is P3.1 evidence, not validation of the new model.

Sources: [progress](reports/p3_1_progress.md), [Age](reports/p3_1_age_baseline.md), [Geology](reports/p3_1_geology_baseline.md), [final aggregate](reports/evidence/p31_prefix_runs_20261104/aggregate_final.json), [sequence audit](reports/evidence/p31_prefix_runs_20261104/sequence_verification.json).

## What LunarJEV-lite will test

```text
Existing lunar observations -> frozen encoder -> cached spatial evidence
                                                   ^
Task definition + public candidate descriptions -> semantic queries
                                                   |
                                 candidate logits / Choice probabilities
                                  Geology (49) and Age (5)
```

The first prototype is a lightweight discriminative **Semantic Query Decider**, not a claimed RSJEV/Clef reproduction. It reuses one observation context for parallel candidate/task queries. This operational meaning of one-pass does not imply a speed advantage over ordinary multi-head classification or arbitrary-language reasoning.

Use each existing Geology checkpoint's verified `best_state_dict`, frozen and in eval mode. Export the pre-global-pooling feature map, fixed-pool it to 4×4, and cache 16×256 float32 tokens per sample. All head comparisons receive the same encoder, sample IDs, task supervision and cache; a pooled control is derived from these same tokens. The pure token payload is about98.3MiB per encoder for Train+Val, calculated from shape, not measured storage peak.

| Method | Evidence | Candidate information | Role |
|---|---|---|---|
| B0 ordinary | Pooled same-context features | Ordinary task heads + masked-softmax | Required baseline |
| B1 candidate_id_control | Same pooled features | Learned known-class IDs | Structural/parameter-sharing control |
| B2 pooled_semantic | Same pooled features | Fixed public task/candidate descriptions | Separate semantic input from spatial querying |
| J lunarjev_query | Same spatial-token cache | Same descriptions as B2 query spatial evidence | Proposed LunarJEV-lite |

All four heads are fitted anew. Do not use the original end-to-end P3.1 classifier as the sole supposedly identical-feature control. The Geology-supervised encoder is extra related supervision for Age, applied equally to all models. Age labels derive from Geology, so include a Geology-probability-to-Age aggregation control rather than presenting Age as independent evidence.

The first scope is **Choice**. Noul/refusal, continuous Score, LLM backbone training, latent modality synthesis, active routing and NASA/MG-MAE replacement are later, separately evaluated extensions. Do not invent hazard labels or unseen-class claims from the existing classification set.

## Semantic provenance, not another ID head under a new name

The existing49-class JSON primarily contains geological symbols, not complete verified descriptions. Build a source-traceable public catalog shared by all samples. Prefer already local audited definitions and a verified local frozen text encoder. Never inject a sample's true unit name or age prefix into its prompt.

If richer local definitions or text weights are missing, implementation can proceed using an explicitly marked `TAXONOMY_ONLY_LOCAL` catalog and deterministic lexical representation. Such results are weak-semantic/ontology controls, **not evidence of pretrained-language or full geological-semantic reasoning**. Missing assets limit the relevant scientific claim, not the entire cache/module workflow. No new data or weight download is authorized here.

The current ID head and a candidate-independent semantic scorer satisfy their own full-softmax/subset-mask identity. Candidate ordering and subset support are interface checks, not evidence of semantic or new-class generalization. Record actual parameter count and compute; similar parameter counts do not establish equal runtime.

## Execution

1. Export audited frozen-context caches from existing Train/Val and verify IDs, weights and source hashes.
2. Assemble the global candidate catalog, document its semantic tier, and fix one local text representation before new scores.
3. Implement the four heads, dual-task masking and essential leakage/frozen-state/replay tests in new `p32_` modules.
4. Freeze the actual run manifest and train small heads: default FP32/batch128/AdamW1e-3, max50 epochs/patience8, paired seeds42/43/44. Select by Val Geology full-49 Macro-F1; report Age from the same checkpoint. No hyperparameter search or scratch-backbone reruns.
5. Report J1 semantic contribution, J2 spatial-query contribution and J3 shared-context capability, with ordinary masked-softmax and a predeclared description-alignment control. All development analyses remain exploratory.

The planned scripts/modules in the design are **not yet implemented by this handoff**. Future results go to `reports/p3_2_progress.md` and `reports/p3_2_lunarjev_mvp.md`; local binary artifacts use `outputs/p32/`. Preserve all prior code and results for replay.

## Acquisition, resources and protected experiments

**Acquisition remains PAUSED for VPN quota.** The retained file is33,964,443,553 bytes;40,484,675,794 bytes remain. Its local-prefix hash is recorded, but upstream full-file integrity is unverified. The newer pause instruction supersedes the earlier permission to acquire the full asset. Do not resume the transfer or download mmap files, pretrained weights or snapshots automatically.

Use existing RX7900XT/ROCm and local assets. No paid resources, four-GPU runs or shared-environment replacement. Keep raw data, checkpoints, feature/text caches and credentials out of Git. The shared four-4090 machine's10GiB personal-storage limit remains unchanged.

Profiling may independently compare GPU-resident, CPU-resident and real-HDF5 paths using a small local sample. Low allocator memory alone does not prove low compute utilization. Profiling is not a gate for designing LunarJEV, and must not change archived baseline budgets.

P2.2's [negative evidence](reports/p2_2_evidence_completion.md) and [corrected conclusions](reports/p2_2_hypothesis_decision.md) remain intact. The old low-rank route is STOP_OR_REFRAME, not a general rejection of probabilistic decision models. Old LROC Gate3 remains NOT_FROZEN and Gate4 HOLD; public Test model scoring also remains HOLD. New head experiments do not approve EDR/CDR policy changes or rewriting previous heldout assets.

## Verification and history

```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts src tests
```

Use the existing validated private runtime where required. Report real commands, source/data/cache/text hashes, PASS/FAIL/BLOCKED and checkpoint/prediction replay evidence, then push completed changes to main without force-pushing or losing concurrent work.

Historical P3.0/P3.1 plans and P0–P2.2 evidence remain under docs/reports. The previous handoff is preserved at [edffc0a](https://github.com/wbyuan030/lunarrisk/tree/edffc0a7c274efb4c1cee03e6c0f3203d87ebba7). Current instructions are P3.2, not the unexecuted wording in older plans.
