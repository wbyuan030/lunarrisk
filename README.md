# LunarRisk: RX 7900 XT First Experiments

> **ACTIVE SPRINT (2026-10-23): [P2.2 probability scoring correctness and H1/H2 fair comparison](docs/P2_2_PROBABILITY_VALIDATION.md).**
> First P2.1 Point/Diagonal/Joint posterior+AST runs and 41-group external pool are implemented,
> but three **evaluation correctness fixes** have been committed: Point calibration now scales
> its residual Gaussian, same-checkpoint U=0 uses the same cal-fitted T, and physical-space
> NLL uses the target-side positive log-Jacobian. See the
> [P2.1 pre-fix result erratum](reports/p2_1_joint_posterior.md).
> **Old Brier/NLL figures are historical until rerun.** The rerun is done: 431 tests pass in the
> ROCm container, all models now share the same 749-tile sample set, checkpoints reload exactly,
> and the group-paired comparison is in
> [reports/p2_2_baseline_comparison.md](reports/p2_2_baseline_comparison.md).
> **Decision: `STOP_OR_REFRAME` — H1 rejected and H2 not established**
> ([reports/p2_2_hypothesis_decision.md](reports/p2_2_hypothesis_decision.md)).
> External Gate 3 remains **NOT_FROZEN**; Gate 4 **HOLD**.
> Coding Agent: [AGENTS.md](AGENTS.md).
>

> **ACTIVE SPRINT (2026-10-22): [P2.1 joint posterior + compositional risk experiments](docs/P2_1_JOINT_RISK_RESEARCH.md).**
> The main research question is whether a *conditional joint distribution* over slope exceedance,
> roughness and relief produces better calibrated, rule-compositional hazard probabilities
> and fewer unsafe acceptances than point or independent marginal estimates.
> Work in parallel on full-pair polygon grouping: current P2.0 `SPARIM1/SPARIM2`
> were split despite identical centroids and recorded zero nearest distance.
> Run `scripts/check_group_pool_collision.py` for a **warning, not a certificate**;
> the current 15-group/18-scene heldout v1 remains **PENDING_SIGNOFF / NOT_FROZEN**, Gate 4 **HOLD**.
> Do not rerun large CDR/ISIS exploratory work or hold science hostage to an unapproved freeze.
> **Coding Agent:** read [AGENTS.md](AGENTS.md) first.
>

> **Active coding sprint (2026-10-20): [P2.0 multi-camera NAC CDR coverage and registration QA](docs/P2_0_MULTICAMERA_QA.md).**
> The [single-scene CDR→ISIS→DTM-grid ortho probe](reports/p1_9_optical_ortho_probe.md) succeeded,
> but the left camera covers only **42.75% of valid DTM pixels** and independent ground registration is not established.
> Reuse the right-camera CDR to measure union coverage and mosaicking before processing more scenes.
> The [CDR protocol amendment](protocols/p19_cdr_route_amendment_proposal.json) is a **proposal, not an approved replacement**
> for the original EDR preregistration. Gate 3 is **not frozen** and Gate 4 model scoring remains on hold.
> Start with [AGENTS.md](AGENTS.md).
>

> **New P1.9 EDR access lead (2026-10-18):** Official LROC NAC observing-phase [shapefile archives](https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_NAC_EQ_NOMINAL_MISSION) include exact `PDS_VOLUME`, `FILE_SPECI` and `URL` fields for EDRs. Try [docs/P1_9_EDR_URL_DISCOVERY.md](docs/P1_9_EDR_URL_DISCOVERY.md) and `scripts/resolve_lroc_edr_urls.py` **before guessing URLs**. This is a verified official metadata discovery, **not yet a verified downloadable raw IMG**. The held-out evaluation gate is still blocked.


> **Active sprint (2026-10-17):** [P1.9 external LROC DTM heldout QA](docs/P1_9_HELDOUT_BUILD.md).
> LROC index discovery is promising, but the earlier **143 supposedly strict OOD DTMs are provisional metadata candidates only**: source IDs and real polygon clearance were not fully verified. The equal-area-circle proof was invalid and is now withdrawn. See the [research erratum](reports/p1_8_new_heldout_sources.md).
>
> **Latest actual GPU experiments:** [P1.8 fixed-mask ablations](reports/p1_8_fixed_mask.md): 15 runs on RX 7900 XT, 163 tests reported passed, no proven Brier gain on the repeatedly viewed development test set.
> Coding agents should begin with [AGENTS.md](AGENTS.md) → [P1.9 Gate 0/1](docs/P1_9_HELDOUT_BUILD.md) and leave the existing 64-tile product-only freeze unchanged.
>
**Historical P0 scaffold notes below (superseded by [AGENTS.md](AGENTS.md) and the current sprint).**

**Status:** ready-to-run scaffold. The scripts are CPU-tested locally, **not validated on your ROCm host or on the complete lunar dataset**. They are designed to fail clearly when real inputs are missing; nothing silently substitutes synthetic lunar data.

## Hard decision: start with an accessible real dataset

NASA/IBM's pretraining **NAC_HighRes** public Hugging Face viewer currently exposes **6 sample rows**, not the full ~1 million tile corpus. The sample contains NAC, 3m DTM and slope NetCDF files and is enough for a data/physics I/O integration test, **not a generalization paper**. The separate, public **NAC crater detection** repository is ~105MB and contains COCO labels and NAC `.npy` images; use it only to smoke-test the learning, split and risk-coverage pipeline.

Links:
- https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-pretraining-data
- https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-NAC-Crater-Detection
- https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model

## Experiment 0 — HIP training works (30–60 min, no data)

In your working ROCm 7.0 Docker image and existing PyTorch environment (do **not** blindly reinstall PyTorch):

```bash
python - <<'PY'
import torch
print(torch.__version__, 'HIP=', torch.version.hip)
print('GPU=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT DETECTED')
PY
python scripts/rocm_smoke.py --batch 8 --size 192 --steps 60
```

**Pass:** HIP GPU recognized; all forward/backward steps run; `outputs/rocm_smoke.json` has valid samples/sec and VRAM. If failure: inspect `/dev/kfd`, `/dev/dri` and matched ROCm/PyTorch; don't start LFM until repaired. RX 7900 XT is included in AMD ROCm 7.0 Linux hardware list for supported OS releases.

## Experiment 1 — true data audit and slope labels (CPU; half-day)

```bash
pip install huggingface_hub xarray netCDF4 numpy
python scripts/hf_audit.py --dataset sample  # list first; no download
python scripts/hf_audit.py --dataset sample --download
python scripts/make_slope_labels.py data/sample/NAC_HighRes/slope_3m
```

Outputs: file-list, count of physically present assets, `outputs/slope_labels.csv`, raw slope histograms, valid fractions. Open NetCDF metadata and verify variable, units, CRS, grid, and bounds. **Do not train a NAC→DTM model on these ~6 examples.** Do not assume a 256-pixel NAC crop corresponds to a 256-pixel 3m slope crop; geospatial resampling is a separately gated task.

**Stop condition:** only sample data accessible or unable to download NAC+3m DTM+3m slope for a useful group-diverse cohort. Find alternate legally accessible stereo DTM assets / contact authors before dedicating time to a large model.

## Experiment 2 — real lunar ML and risk curve smoke test (RX 7900 XT; ~1–2 h)

The crater set is an **engineering fallback**; its label is number of *annotated* craters, **not terrain safety or full landing hazards**. Standard and large annotation protocols must not be conflated.

```bash
python scripts/hf_audit.py --dataset crater --download
python scripts/crater_pilot.py --data data/crater --epochs 5 --batch 16
```

Artifacts: `outputs/crater_pilot/metrics.json`, `predictions.npz`, `risk_coverage.csv`. Confirm official train/test separation and check site-level overlap before using the results scientifically. This verifies dataloading, basic CNN training, Brier and risk-coverage computation on **real NAC** images, not the main slope hypothesis.

## Experiment 3 — first scientifically useful NAC slope pilot (requires independent paired real data)

Once G0 has **at least dozens of independent parent DTM sites**, and preferably hundreds of grouped tiles, build geographic-aligned packed rows:

- Input: optical NAC ROI, optional 60m coarse DTM and viewing geometry.
- Hidden label: 3m slope empirical distribution (raw area fractions), one roughness and one relief measure.
- Group: DTM source ID / parent EDR + geographic block. Hold out groups, not adjacent windows from the same stereo product.
- Baseline A: constant prevalence / 60m coarse slope only.
- Baseline B: small CNN (train from scratch) or frozen LFM embedding + linear/ridge/MLP head.
- Labels: raw `P(slope>10°)>5%` and 2 additional rule thresholds; definitions fixed on validation before test.
- Output: MAE, event Brier, UAR–Coverage curve, threshold sensitivity, bootstrap by geographic block.

First compare point estimate vs independent Gaussian uncertainty; **joint posterior only when at least two independently observed attributes and enough groups are actually present**. Fit a rank-2 low-rank joint Gaussian only after confirming nontrivial residual correlation. Use the *same* calibration and test samples for every model. If the 3m stereo DTM is an uncertain reconstruction itself, state results as conditional on that reference, not proven physical ground truth.

## Priorities and constraints

1. Do **not** spend paid GPU budget until Step 0 and Step 1 pass.
2. ROCm single GPU 20GB: `batch=4..16`, fp32 or fp16 per tested kernels; avoid FlashAttention CUDA-only packages / Qwen / TiM for now.
3. Keep full raw lunar data and datasets on the 7900 XT box; weekend 4x4090 10GB-constrained node receives only compact experiment packs.
4. Never substitute synthetic labels for real lunar experiment claims.
5. Save dataset commit SHA, package versions, Git commit, exact data split and seed with every result.

## First 24-hour deliverables

- `outputs/rocm_smoke.json`
- `outputs/slope_labels.csv` generated from actual downloadable sample
- Audit note: how many **independent**, correctly georeferenced NAC+stereo-DTM pairs are actually available
- `outputs/crater_pilot/risk_coverage.csv` as real-image pipeline check
- A decision: **GO real stereo slope pilot / BLOCKED dataset / PIVOT to crater geometry risk**
