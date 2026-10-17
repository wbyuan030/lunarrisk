# LunarRisk: RX 7900 XT First Experiments

> **Current Coding Agent sprint (2026-10-17):** [P1.8 continuation — fixed-mask ablation + new independent-data audit](docs/P1_8_CONTINUE.md). Start with P18-04a and a public-source index audit; the old hydration/CSV/statistics fixes **already passed**.
>
> **Evidence:** [P1.8 progress](reports/p1_8_progress.md) reports 148 passing tests for P18-01/02/03; [heldout review](reports/p1_8_heldout_protocol.md) reports **BLOCKED** for strict group-disjoint V2 because all 272 groups were previously touched. The frozen 64-tile, 22-new-NAC-product candidate overlaps 4 previously used groups and is **not** strict geographic/source OOD. [P1.7 ablation](reports/p1_7_ablation.md) remains exploratory.
>
> **Current decision:** proceed with fixed-parameter-count modality masking and a scientifically legal augmentation comparison on the development set. Preserve frozen V2 files and do not claim NAC increment or independent landing safety until external holdout evidence exists.

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
