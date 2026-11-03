# LunarRisk / LunarJEV

**Active sprint — P3.0, 2026-11-03: public benchmark tasks for LunarJEV.**

Start with [AGENTS.md](AGENTS.md) and [P3.0 execution plan](docs/P3_0_PUBLIC_BENCHMARKS.md).
The asset audit, HDF5-derived loader, a bounded 28-channel mmap row probe, and a five-class Age CNN engineering smoke are complete. Fresh-process checkpoint replays produce byte-identical probabilities. **No official benchmark performance result is claimed.**
The machine-readable scope is [protocols/p30_public_benchmark_plan.json](protocols/p30_public_benchmark_plan.json).

## Current research direction

Test whether lunar pretrained representations and task/candidate-conditioned decisions provide useful, reproducible gains on existing public labels.

- **First integration task:** Moonstone age classification, using the official five-class mapping and unknown-label filtering.
- **Main task:** Moonstone geology classification, using the official 49-class task.
- **Required comparison:** ordinary task head versus a task/candidate-conditioned head on the **same encoder, visible modalities, sample IDs and budget**. Include an ordinary classifier with candidate-subset masking and renormalization.
- **Not in the initial sprint:** full language-backbone joint training, Latent TiM generation, active sensor routing, new low-rank posterior scaling, or reconstructing a global lunar data corpus.

Moonstone source pointers: [official code](https://github.com/ayushprd/Moonstone), [dataset card](https://huggingface.co/datasets/ayushprd/Moonstone/blob/main/README.md), [downstream loader](https://github.com/ayushprd/Moonstone/blob/main/downstream_dataset.py).
NASA LFM is a separately audited encoder option: [official model card](https://huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model/blob/main/README.md).
Pinned versions, actual file sizes, licenses and access evidence are recorded in the [asset audit](reports/p3_0_benchmark_audit.md). MG-MAE pretrained weights remain unverified in the audited official sources. NASA LFM has a reachable checkpoint, while its input mapping, Python compatibility and full-weight acquisition remain blocked.

## Begin with an access audit, not a bulk download

The Moonstone card describes a standalone benchmark HDF5, while its current downstream loader reads pixels from global mmap files. Check actual HDF5 fields, normalization, channel order and missing-data masks before choosing a loading path. A replacement loader needs equivalence evidence; otherwise report it as a derived pilot rather than an exact official reproduction.

A README example such as `checkpoints/latest.pt` is not proof of a downloadable pretrained checkpoint. Audit real URLs, revisions, hashes, licenses and architecture compatibility. NASA LFM's native NAC/WAC interfaces must not be treated as interchangeable with arbitrary Moonstone channels.

| Stage | First deliverable |
|---|---|
| P30-00 | Asset/weight/disk/license audit, pinned versions and exact download allowlist |
| P30-01 | Loader and label/split tests; real small-sample I/O when available |
| P30-02 | Ordinary age-classification head smoke and baseline |
| P30-03 | Geology ordinary-head and candidate-conditioned comparisons |
| P30-04 | Separately specified candidate-order/subset, modality-missing and calibration extensions |

Implemented audit, inspection, extraction, metadata-validation, mmap-probe and Age-smoke commands use the `p30_` namespace. See the [loader validation](reports/p3_0_loader_validation.md), [mmap row probe](reports/p3_0_mmap_pair.md), [five-class Age smoke](reports/p3_0_age_fiveclass_smoke.md), and [stage progress](reports/p3_0_progress.md). Actual HDF5 metadata contains 16,200 unique Tiles with stored splits 11,400/2,400/2,400. The original eight-patch single-class smoke is preserved; a second fixed pilot covers all five Age classes with one Train and one Val patch each. Neither is a performance benchmark. Across the original fixed IDs, 172,032 mmap probe pixels and masks match the existing HF normalization exactly. Full-crop equivalence remains unverified, so the protocol stays **HDF5_DERIVED_PILOT**.

Current code checks: **478 tests, 477 passed, one legacy optional-data check skipped; compileall passed**. Actual cumulative first-round response bodies total 78,502,175 bytes (about 74.9 MiB). No Test pixels were requested. Official random-split results, geographic-split results and new decision protocols must remain distinct. Do not tune on the public test split or claim pretraining-unseen geography without a provenance audit.

## Existing LunarRisk evidence remains valid as recorded

The current low-rank terrain-posterior line is **STOP_OR_REFRAME**; stable H1/H2 gains are **NOT ESTABLISHED**, not a general rejection of probabilistic decision methods.

- [P2.2 evidence completion](reports/p2_2_evidence_completion.md): archived six-model/three-seed comparison, replay and independent retraining evidence.
- [Corrected hypothesis decision](reports/p2_2_hypothesis_decision.md): limits of the results and withdrawn overclaims.
- [Evidence archive](reports/evidence/p22_20261103/): public logs, probabilities and hashes with machine locations redacted.
- [External heldout v2 protocol](protocols/heldout_v2_external_protocol_v2.json): still **PENDING_SIGNOFF / NOT_FROZEN / HOLD**.

The last archived P2.2 test run records 444 tests: **443 passed, one skipped**. That is historical evidence, not validation of future P3.0 code. Existing probability/rule tools can support the new work, but success of the four-attribute Gaussian model is no longer a prerequisite.

**No existing raw data, labels/cache, archived evidence or frozen/pending heldout artifacts are changed by this handoff.** Moving to a public benchmark does not authorize old external Gate 4 scoring or approve the EDR-to-CDR policy amendment.

## Resources and reproducibility

Use the existing RX 7900 XT and working ROCm environment; do not replace its PyTorch installation. The initial metadata/sample transfer guardrail is **1 GiB**, allowlist-first and dry-run by default. Full benchmark or pretraining downloads require an explicit disk-safe acquisition plan; do not automatically download entire repositories.

No paid cloud resources or four-GPU runs in this sprint. The shared four-4090 server's 10 GiB personal-storage limit remains in force. Keep large data, weights, credentials and private machine paths out of Git.

Before and after code changes:

```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts src tests
```

Coding Agent reports actual PASS/FAIL/BLOCKED status with commands, commit/data versions and evidence, then pushes to `main` without force-pushing or discarding concurrent work. Missing public assets are a documented blocker, not permission to invent data or claim a reproduction.

## Historical project instructions

Previous P0–P2.2 execution plans and reports remain under `docs/` and `reports/`.
The prior README with the full historical scaffold is preserved at
[commit 461316f](https://github.com/wbyuan030/lunarrisk/blob/461316fb50c10ba4570aa55ba56f96a056e6df32/README.md).
Those dated instructions are historical; the current sprint is defined by `AGENTS.md` and the P3.0 plan above.
