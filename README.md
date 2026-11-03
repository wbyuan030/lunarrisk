# LunarRisk / LunarJEV

**Active sprint — P3.1, 2026-11-03: move from engineering smoke to data-sufficient public-benchmark baselines.**

Start with [AGENTS.md](AGENTS.md), the [P3.1 execution plan](docs/P3_1_BENCHMARK_BASELINES.md), and the [machine-readable task scope](protocols/p31_execution_plan.json).
This update is a handoff, not completed P3.1 training or approval for a bulk download.

## Current research direction

Test whether lunar representations and task/candidate-conditioned decisions improve reproducible performance on public labels. First establish ordinary baselines with adequate data; verify pretrained and semantic contributions separately rather than treating an interface smoke test as evidence.

- **First training integration:** Moonstone five-class age classification.
- **Core task:** Moonstone 49-class geology classification.
- **Controls:** ordinary classifier, ordinary masked-softmax, and candidate-ID structural control on the same encoder/features. A semantic class-description model requires a separately specified experiment.
- **Not presently included:** full language-backbone joint training, Latent TiM generation, active routing, global pretraining, or further low-rank terrain-posterior expansion.

Official source pointers: [Moonstone code](https://github.com/ayushprd/Moonstone), [dataset card](https://huggingface.co/datasets/ayushprd/Moonstone/blob/main/README.md), [downstream loader](https://github.com/ayushprd/Moonstone/blob/main/downstream_dataset.py), and [NASA LFM model card](https://huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model/blob/main/README.md).
Pinned revisions, license distinctions and observed access limits are in the [P3.0 asset audit](reports/p3_0_benchmark_audit.md).

## Completed P3.0 evidence — engineering only

The asset/metadata audit, HDF5-derived partial I/O, five-class Age smoke and same-feature 49-output Geology smoke are complete. The main pilot is **5 Train + 5 Val**. Geology uses the two-update Age-supervised scratch CNN, not MG-MAE or NASA pretrained weights. Independent process replay is documented, but **no official benchmark performance, pretrained gain or semantic gain is claimed**.

The latest recorded P3.0 checks are **482 tests: 481 passed, one optional legacy-data-path check skipped; compileall passed**. These are historical checks, not validation of future P3.1 changes.

The 172,032-pixel mmap row probe matched the pinned HF processing, but the original polar samples have 17/28 channels entirely missing. Full-crop comparison with valid observations across modality groups remains necessary; the existing interface stays **HDF5_DERIVED_PILOT**.

Evidence: [progress](reports/p3_0_progress.md), [loader validation](reports/p3_0_loader_validation.md), [mmap pairing](reports/p3_0_mmap_pair.md), [five-class Age smoke](reports/p3_0_age_fiveclass_smoke.md), [Geology engineering comparison](reports/p3_0_geology_baselines.md), and [next acquisition/research decision](reports/p3_0_next_decision.md).

## P3.1 execution order

| Task | Work | Gate |
|---|---|---|
| P31-01 | New scalable runner: streaming SHA256, batchwise HDF5 reads, minibatch training/inference, checkpoints and memory logs | Implement now; reuse existing pilot for regression |
| P31-02 | Representative full-crop pairing across valid/missing modalities, finite zeros and radar | Lock IDs, tolerances and access budget before reads |
| P31-03 | Data-sufficient ordinary Age and Geology baselines with fixed metrics and Val selection | Requires suitable authorized data and a locked evaluation plan |
| P31-04 | Candidate-ID equivalence/control tests and a separately designed semantic comparison | No semantic/new-class claims from ID lookup or ordering tests |

Do not merely remove limits from the smoke scripts: they hash files using `read_bytes()` and stack the whole small dataset before sending it to the GPU. Those choices are valid for the existing smoke but unsuitable for the full benchmark. Keep the historical runners as regression tools and build a separate production-size training path.

The current candidate-ID scorer evaluates each known candidate independently. Its subset probabilities equal its own full-softmax probabilities masked and renormalized. This does **not** say it learns the same classifier as the ordinary MLP; it says subset support/permutation consistency alone does not establish semantic reasoning. Similar parameter counts also do not imply equal computation. Report head-only and full-pipeline costs separately.

The stored dataset has 11,400/2,400/2,400 Train/Val/Test rows. It has 46 supported training geology classes; the other three are also absent from stored Val/Test, so this is not an unseen-class test. Preserve 49 outputs and document absent-class Macro-F1 handling. Age is derived from geology labels; any Age-supervised encoder is additional supervised training, not independent unsupervised pretraining evidence.

## Data acquisition and pretrained routes

The review recommends the bounded single-HDF5 data-first route, but [the acquisition proposal](protocols/p30_next_acquisition_proposal.json) remains **PENDING_USER_DECISION_NOT_AUTHORIZED**. This handoff does not start or approve the 74,449,119,347-byte download. The initial cumulative **1 GiB** guardrail is unchanged; no full pretraining snapshot, mmap corpus or large weights are implicitly authorized.

Without new download approval, implement the scalable runner, use existing assets for regression, and lock the baseline/pairing protocols. Do not keep retraining ten images as a substitute for research. If the fixed-file acquisition is explicitly approved, use a single new partial file, streaming size/hash checks, atomic finalization, and a fresh disk check; do not duplicate the asset in a second snapshot/cache.

MG-MAE weights remain unverified in the audited sources. NASA LFM has a reachable checkpoint, while physical input mapping, missing geometry, Python compatibility and weight acquisition remain separate blockers. Do not invent channels/context or train from scratch and label it pretrained. Missing pretrained weights need not block an honest scratch baseline.

Full-file storage may contain Test pixels; training, model selection and test scoring must still remain isolated. Public Test scoring waits until model/configuration/analysis is fixed. Official stored, geographic, derived-loader, calibration and new semantic protocols are distinct tracks.

## Existing LunarRisk evidence and resource boundaries

- [P2.2 evidence](reports/p2_2_evidence_completion.md) and [corrected decision](reports/p2_2_hypothesis_decision.md) remain unchanged: **STOP_OR_REFRAME**, stable H1/H2 gains **NOT ESTABLISHED**.
- [External heldout v2](protocols/heldout_v2_external_protocol_v2.json) remains **PENDING_SIGNOFF / NOT_FROZEN / HOLD**. This sprint does not approve EDR-to-CDR changes or old Gate 4 scoring.
- Preserve all previous raw data, labels/cache, frozen/pending protocols and evidence. New outputs use a P3.1 namespace.
- Use the existing RX 7900 XT and working ROCm environment. Do not replace torch, buy cloud resources, start four-GPU runs or write large data/weights/credentials into Git. The shared four-4090 server's 10 GiB personal storage constraint remains.

Validation commands:

```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts src tests
```

Record real PASS/FAIL/BLOCKED status, source/data hashes, commands, predictions and replay evidence in `reports/p3_1_progress.md`. Push completed work directly to main without force-pushing or discarding concurrent changes.

Previous plans/reports remain under docs/reports. The P3.0 README and agent handoff are preserved at [25c8ac9](https://github.com/wbyuan030/lunarrisk/tree/25c8ac93e6ea94a1f402c3daecf547bba1990278); older instructions are historical, not concurrent active sprints.
