# Single-card 7900 XT launch plan — preregistered gates

The *science* experiment is monocular NAC + (where available) 60m DEM -> the empirical region-scale 3m DTM slope-exceedance fraction. All other tests are supporting or smoke tests.

## E0 GPU health / throughput
- Data: synthetic random tensors **for hardware testing only**.
- Model: tiny 4-layer CNN, 1-channel input at 192 px, BCE.
- Output: `outputs/rocm_smoke.json`.
- Pass: HIP GPU recognized, forward/backward without driver error, 60 steps, peak VRAM <= 18 GiB, samples/sec recorded.
- Failure action: inspect `rocminfo`, `torch.version.hip`, Docker devices; no full-model install.

## E1 public data availability
- Sources: SomBench official NAC_HighRes public sample; separately, NAC crater benchmark.
- Manifest: enumerate *actual* files, inspect NetCDF data variables, pixel units and WKT/CRS coordinate axes; calculate valid fractions for slope/DTM.
- Output: `outputs/slope_labels.csv` and written table: listed, downloadable, co-registered, split-group-unique counts. Confirm 6-row sample is not full corpus.
- Pass for end-to-end A1: a genuinely obtainable, sufficiently diverse set of NAC + DTM_3M + SLOPE_3M and metadata (do not set a sample count prematurely); DTM source ID known; image/DTM coordinate matching verified visually for >= 10 examples.
- Block A1 if only public demo sample accessible; do not claim 6 samples constitute a valid ML dataset.

## E2 data-path fallback: annotated crater count
- Real data: ~105MB official NAC crater detection repository. Use official split JSON; `standard` protocol only.
- Label: `count(annotated crater)>=median_train_count` as an **engineering proxy**, not physical landing safety.
- Model: tiny CNN scratch-training, optional 5 epochs.
- Output: train logs, test Brier, coverage vs fraction of high crater-count patches erroneously accepted.
- Pass: complete data/fit/evaluation workflow without split mixing or crashes. **NOT** a publication-quality crater result.
- Follow-up: site grouping, annotation coverage audit, more meaningful physical geometry rules.

## E3 first real physical baseline (only after E1 gate)
- Input: geo-aligned NAC ROI and optional 60m coarse DTM; actual valid 3m slope hidden at inference.
- Target: y1=percentage valid pixels with slope>10 deg, y2=percentage slope>15 deg; two bounded continuous attributes to begin. To get rank-2 joint model, add genuinely independently supervised local roughness and regional relief.
- Deterministic event: unsafe if y1 > 0.05 (plus a second threshold in an *unseen rule* split).
- Models: regional mean/coarse DTM baseline; small CNN; frozen LFM features + ridge/MLP (if ROCm compatibility verified); conditional variance head.
- Metric: MAE(y1), RMSE(y1), event Brier, binary NLL, false-acceptance rate at 20%, 40%, 60%, 80% accepted coverage; count independent geographical test sites.
- Partition: parent stereo DTM, parent NAC EDR and contiguous geographic groups—not random crops.
- Checkpoint: only select hyperparameters using val; calibration and independent test remain untouched.

## E4 probability model gate (only if E3 real supervised baseline exists)
- Compare: deterministic point-estimate AST, independent conditional normals, rank-2 covariance joint conditional normal on >=2 attributes.
- Enforce physical support (fractions in [0,1]) when modeling probabilities; never claim a Gaussian in raw physical space fits automatically.
- Evaluate probability of each held-out Boolean rule using approximate joint integration under equal time/sample budget.
- Gate: event probability Brier improves, and UAR at **same** acceptance coverage does not worsen; confidence intervals grouped by site.
- If group count low, report exploratory result only (no Q1 claims).

## Daily cash/time rule
- No paid GPU while local G0/G1 is blocked. Local 7900 XT used for all smoke testing and low-memory pilots; four 4090 GPUs only after dataset and pilot metrics are present.
- Run one experiment + write result note per iteration; code/manifest + hashes preserved.
