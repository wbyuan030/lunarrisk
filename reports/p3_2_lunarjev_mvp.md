# P3.2 LunarJEV-lite first paired comparison

Updated 2026-11-05. **Implementation and all18 predeclared head runs COMPLETE/PASS. The current taxonomy-query model does not improve on the ordinary pooled head.** Shared-context/two-task execution works; semantic and spatial-query gains are not established. This is an exploratory result on reused northern Train/Val, not an independent confirmation or full official benchmark.

## Assets, protocol and fair controls

The [pre-score actual manifest](../protocols/p32_head_run_manifest_v2.json), SHA256 `c7e853c12d2175954a3f1843f6a4536753ccd6d04c443a34a1649e031f735d5f`, freezes source, caches, original checkpoints, catalog, exact text representation and budgets. It implements the [P3.2 design](../docs/P3_2_LUNARJEV_MVP.md) using `TAXONOMY_ONLY_LOCAL` descriptors. The first manifest/launch are preserved as a failed engineering attempt before any epoch, not another model/configuration trial. [Progress and failure history](p3_2_progress.md).

Three independently trained P3.1 Geology encoders supply their authenticated `best_state_dict` features, frozen including BN. Every method of a given seed receives the same16×256 FP32 spatial-token cache, original5,135 Train/1,155 Val IDs/targets, normalization and related Geology supervision for Age. B0/B1/B2 average the same tokens; J queries the16 positioned tokens. No raw coordinates, sample unit names/age prefixes or targets enter model inputs. [Cache report](p3_2_context_cache.md).

| Method | Input / conditioning | Trainable head parameters |
|---|---|---:|
| B0 ordinary | Pooled same-context features, two ordinary task MLPs; subset masked-softmax | 337,462 |
| B1 candidate_id_control | Same pooled input, learned known-class/task IDs, shared scorer | 305,921 |
| B2 pooled taxonomy | Same pooled input, fixed public task/descriptor features, shared scorer | 329,729 |
| J lunarjev_query | Same task/descriptors as B2; query16 spatial tokens with fixed2D positions | 329,729 |

B1/B2/J use hidden128, four attention heads, two attention/FFN256 residual blocks, dropout0. B2/J have identical trainable architecture/parameter counts and no learned class-ID embeddings or class-specific bias. B1/B2 singleton-context attention has weight1; descriptors and evidence interact through residual FFNs. J-B2 evaluates the whole spatial-context/query/position path, not a separately identified causal effect of language-conditioned attention.

All four heads are newly trained for Geology49+Age5: equal mean task CEs, FP32, batch128 including final partial batch, AdamW constant lr.001/wd.01, maximum50 epochs/patience8, no hyperparameter search or gradient clipping. Select strictly higher Val **full49 Geology Macro-F1**, earliest tie; Age comes from that same checkpoint. Seeds42/43/44 are paired with their corresponding encoder. Original P3.1 end-to-end heads and their six runs are unchanged and are not the sole fair comparator.

## Primary results

| Method | Val Geology Accuracy | Val full49 Macro-F1 | Val supported32 Macro-F1 | Val Age Accuracy | Val Age Macro-F1 |
|---|---:|---:|---:|---:|---:|
| **B0 ordinary** | **46.8110±0.8183%** | **0.255488±0.014813** | **0.391217±0.022682** | 60.3463±0.6061% | 0.458335±0.006906 |
| B1 ID | 44.9928±1.1787% | 0.212279±0.009432 | 0.325052±0.014442 | 60.6926±0.4821% | 0.411307±0.024238 |
| B2 pooled taxonomy | 46.1472±2.0138% | 0.221628±0.014017 | 0.339367±0.021464 | 60.8081±1.4076% | 0.397883±0.026600 |
| J spatial taxonomy query | 44.9062±1.0894% | 0.202774±0.020098 | 0.310498±0.030775 | 58.5859±1.0256% | 0.423375±0.022096 |

All values are mean±sample standard deviation, n=3/ddof=1; Accuracy deviations are percentage points, not confidence intervals. Full49 macro assigns absent classes F1=0; supported32 is secondary. [Full Train/Val aggregate](evidence/p32_20261105/runs/aggregate_final.json) includes NLL/Brier, all seeds, all three output routes and descriptor controls.

| Seed | B0 full49 F1 (selected/completed epoch) | B1 | B2 | J |
|---|---|---|---|---|
| 42 | 0.249918 (26/34) | 0.208920 (16/24) | 0.205779 (7/15) | 0.210188 (16/24) |
| 43 | 0.244268 (26/34) | 0.204986 (12/20) | 0.232396 (15/23) | 0.180022 (5/13) |
| 44 | 0.272279 (45/50) | 0.222930 (28/36) | 0.226708 (20/28) | 0.218112 (25/33) |

Paired J−B2 full49 F1 is **−0.018854±0.029749**, with seed differences +0.004410/−0.052374/−0.008596; one of three is positive. J−B0 is **−0.052714±0.012323**, negative for all three. B2−B1 is +0.009349±0.016020, but B2−B0 is −0.033861±0.019056. These comparisons do not establish a beneficial query or taxonomy-semantic effect. [Paired predictions/diagnostics](evidence/p32_20261105/runs/paired_comparisons.json).

The pre-score minority rule is `0 < Train class support <= 50`:18 classes with89 total Val examples. Their mean F1 across seeds is B0 .288557, B1 .197023, B2 .224189, J .179564. Class IDs and all per-class support/F1/recall/confusion remain archived; this small minority subset is exploratory.

B0 wins the primary F1 criterion but has mean Geology NLL/Brier **2.685249/.805970**, versus B1 **1.976351/.733512**, B2 **1.985266/.733252**, J **2.610682/.796412**. Thus F1 preference does not establish calibrated or universally better confidence. No calibration intervention was selected or fitted on Test.

## Fixed descriptor alignment control and task consistency

The catalog contains only official symbols and their audited public Age mapping; geological public names remain symbols. Its120-dimensional word/literal TF-IDF fits54 candidate descriptions plus two fixed task definitions. Case-sensitive literal tokens preserve class identity. This is a fixed lexical/taxonomy representation, not a pretrained language model or complete geological definitions.

One PCG64 seed20261104 permutation moves entire descriptor rows within each task for **both training and inference**, keeping task vectors and official target/output mapping unchanged. It was fixed before head scores and run only after the primary matrix, for B2/J at all three seeds.

| Method | Aligned Val full49 F1 | Fixed-permuted Val full49 F1 | Paired aligned−permuted |
|---|---:|---:|---:|
| B2 | 0.221628±0.014017 | 0.231532±0.011774 | **−0.009904±0.008298** |
| J | 0.202774±0.020098 | 0.205641±0.002944 | **−0.002867±0.018259** |

Correct alignment does not beat arbitrary descriptor coding here; for B2 it is lower in all three seeds. Closed-set training can relearn a fixed permutation, so the control need not collapse to chance. These results neither establish language understanding nor refute stronger, source-grounded language semantics.

| Method | Age dedicated Accuracy / Macro-F1 mean | Geology-probability→official-Age Accuracy / Macro-F1 mean |
|---|---:|---:|
| B0 | 60.3463% / .458335 | 60.8947% / .462095 |
| B1 | 60.6926% / .411307 | 60.0577% / .414259 |
| B2 | 60.8081% / .397883 | 60.8947% / .397795 |
| J | 58.5859% / .423375 | 60.0289% / .438223 |

Geology probabilities aggregate using the original public class→Age map; the dedicated Age head adds no clear general advantage to this strong deterministic control. Age is derived from Geology and the encoder was Geology-supervised, so this is task consistency and interface evidence, not an independent new-task or zero-shot discovery.

## Measured cost and verification

[Timing evidence](evidence/p32_20261105/shared_context_timing.json) uses the first16 authorized Train observations resident on the RX7900XT, FP32, five warmups and20 synchronized repeats per seed/method. The table averages per-seed medians; regions exclude HDF5 I/O and host/device transfer. Shared visual encoding alone is **2.014ms per16-observation batch**. This probe is not production I/O throughput or a global hardware profile.

| Method | Both-task head latency, batch16 | Encoder + both tasks, batch16 |
|---|---:|---:|
| B0 | .415ms | **2.330ms** |
| B1 | 1.670ms | 3.033ms |
| B2 | 1.746ms | 3.080ms |
| J | 1.740ms | 3.128ms |

All four methods already reuse one visual encoding. Resident-pixel versus reused-context predictions match exactly; encoder/BN fingerprints remain unchanged. J provides no speed advantage over B0. Per-run batch1/batch128 head timings are also in each report. [Text preprocessing timing](evidence/p32_20261105/text_precompute_timing.json): catalog load/source verification plus TF-IDF construction took3.016ms once, with1.813ms median over20 warmed CPU repeats. Python imports are excluded. These fixed features are precomputed outside inference timing; no pretrained text encoder is run at inference.

All18 training windows total243.156 seconds for2,557,230 example-visits; training/final evaluation/export wall totals270.154 seconds, excluding child startup, asset verification, separate replay and latency probes. Counts are repeated observation visits, not independent candidate-level samples. Parent RSS peak1.919GiB; torch allocated/reserved peaks100.05/126.00MiB measure the training/export process allocator before separate timing, not total card memory. Cache export time and its larger memory are separately reported.

[Final independent verification](evidence/p32_20261105/runs/sequence_verification.json) PASS: **18 runs,162 live artifact hashes,180 archived artifact hashes,3 cache hashes and108 prediction CSVs**, all per-row IDs/targets matching the frozen caches. Every run reproduces six Train/Val Geology/Age/aggregated-Age CSVs exactly in a new process; checkpoint/source/selection/stopping/probabilities/NLL/Brier and final aggregates are revalidated. This establishes prediction replay, not repeated-training determinism. Checkpoints remain local at `outputs/p32/heads_02/<run>/checkpoint.pt`; manifests record SHA256. Public evidence preserves reports/history, receipts, sanitized stdout, exact CSVs and hashes. [626-test implementation verification](evidence/p32_20261105/implementation_v2/verification.json), zero skips; compileall PASS.

## Execution and replay

The completed invocation used the existing private runtime with no installation:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 TMPDIR="$JOB_TMP" \
python3 -I "$JOB_TMP/p31_python.py" scripts/p32_run_heads.py \
  --cache-root outputs/p32/context_03 \
  --manifest protocols/p32_head_run_manifest_v2.json \
  --outdir outputs/p32/heads_02 \
  --runtime-directory "$TRUSTED_RUNTIME" --apply
```

The default without `--apply` is preview. Reusing the completed namespace revalidates receipts/replays/immutable aggregates; it never silently resumes incomplete output. A new namespace would train again and is unnecessary for this completed tier. Individual `p32_train_heads.py --mode replay --checkpoint ...` restores a head using the same frozen manifest/cache/catalog and writes to a fresh output directory. The original failed namespace/manifest are retained. Timing is independently reproducible with `scripts/p32_benchmark.py` using the same inputs and completed checkpoint roots; every binary artifact remains local.

## Decision

- **J1: not established in the taxonomy tier; full geological-language semantics untested.** Proper alignment fails the fixed coding control and complete geological definitions/text-model provenance are absent.
- **J2: no positive aggregate support for this fixed configuration.** J is below B2 on average and below B0 for all three seeds. Keep the negative evidence; do not rescue it by extending epochs, selecting favorable encoders or searching prompts on this Val.
- **J3: shared-context/two-task execution demonstrated with actual cost, no speed or arbitrary-task-generalization claim.** B0 also shares context, is faster and wins primary F1; Age aggregation remains a necessary control.

The next bounded research item is source-verified public geological definitions plus an audited frozen text representation, with a separate full-semantic manifest and the same B0/B1/B2/J controls. The latest user instruction permits needed downloads **directly without proxy**; no such download was needed for this completed tier. Do not resume40.48GB of unnecessary data, repeat scratch baselines or scale the model to reinterpret this negative result. Full data/Test/new-class/safety claims remain separate; old P2.2/P3.0/P3.1 evidence and legacy gates are unchanged.
