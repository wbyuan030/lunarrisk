# P1.5 — Risk Evaluation Repair and Next Experiments

Date: 2026-10-17. This change fixes **evaluation/training code**, not historical experimental output. The numbers in `reports/p1_baseline.md` are **pre-fix**, and its UAR/HMR and g15 CNN event Brier must not be used until rerun.

## What changed
- The risk–coverage evaluator accepts the **lowest P(unsafe)** first. Previously it sorted `1-P(unsafe)` ascending and preferentially accepted high-risk sites.
- Every CNN now predicts **four outputs**: `[g10_fraction, g15_fraction, logit_unsafe_g10, logit_unsafe_g15]`. Each event head receives its own labels, BCE loss, calibration and evaluation scores.
- CNN variants use the same `--seed` regardless of order in `--variants`. For a paired multiseed comparison, run every variant at each seed.
- `coarse_ridge` event probabilities now get Platt calibration on the independent `cal` split, matching CNN's protocol.
- Regression fractions and input modality layout are otherwise unchanged. Existing CNN checkpoint head shapes (3 outputs) **are incompatible** with the revised 4-output model; retrain.

## First task for Coding Agent (7900 XT, no cloud)
1. Pull the latest `main`. Run `python3 -m unittest discover -s tests -v`; report exact pass/fail counts. Fix any issues without altering the semantics of `P(unsafe)`.
2. Run a tiny 1-epoch CPU/GPU fixture through `train_variant` and `evaluate_model`, asserting that event scores have shape `[N,2]` and that identical high-AUC risk predictions yield low UAR at low coverage.
3. Rerun the existing **399 train / 110 test** comparison on identical hydrated data using seeds **42, 43, 44**:
   `python3 scripts/train_baseline.py --seed 42 --out outputs/baseline_fixed_seed42 --force-cache`
   repeat for seeds 43 and 44 (no need to rebuild cache after seed42 when size/content are unchanged).
4. Produce fresh `test_predictions.csv`, `metrics.json`, and `risk_coverage.csv` per seed. Add a summary table with mean/std across seeds; report paired group-bootstrap CI **on the differences** (fusion minus coarse_grid and fusion minus coarse_ridge). State seed count and geographic groups.
5. Audit calibration on both `g10>0.05` and `g15>0.02` events separately: raw vs calibrated Brier, AUC, NLL, risk–coverage. Check g15 predictions differ from g10 in a nontrivial way. Keep `test` untouched for tuning.
6. Update `reports/p1_baseline.md` with a dated **post-fix** section; keep old results clearly labeled as superseded. Include Git commit SHA, exact sample counts, ROCm/PyTorch versions and elapsed GPU time.

## Go / No-Go after the rerun
- **GO small scale-up** only if corrected risk curves, both event heads, split independence and calibration checks pass. Then freeze an additional held-out test protocol and expand training to 3–5k real tiles. Include observation geometry and photometric normalization as controlled ablations.
- **HOLD** LFM, joint posterior, TiM, Qwen, weekend 4x4090 and paid cloud GPUs until the small-scale evidence is reliable.
- The existing rule prevalence (76 unsafe / 110 test) is imbalanced. Do **not** select a new threshold using test. A new safety proxy rule can be pre-registered using train/cal/val, then evaluated on a fresh unseen test region.
- Spatial alignment: current coordinate interpolation is useful but `GEO_JOIN_PENDING` still requires a footprint and projection audit. A 10 km center-distance buffer does not prove source-DTM independence.

## Notes on interpretation
`g10`/ `g15` thresholds are **research proxies** rather than certified landing safety definitions. The earlier claim that the coarse model gives UAR 1.0 at 20% coverage was caused by reversed ranking. The previously reported MAE and AUC are not corrected by this patch; event heads and ridge calibration can change Brier/NLL, so these require retraining and fresh outputs.

## Rerun status (2026-10-17)

Completed on the local RX 7900 XT (torch 2.5.1+rocm6.2, HIP 6.2.41133), git commit
`4638b2e` plus the follow-up export/aggregation commits.

- Tests: **66/66 PASS** (`python3 -m unittest discover -s tests -v`). Added 8 tests:
  a 1-epoch `train_variant` fixture (event scores `[N,2]`, independent heads), a
  high-AUC → low-UAR fixture, and 8 aggregation checks (paired bootstrap, cohort identity,
  split independence).
- Rerun: `--seed 42 --force-cache`, then seeds 43 and 44. Wall time 44 s / 21 s / 21 s
  (86 s total; no cloud, single local GPU).
- Artifacts: `outputs/baseline_fixed_seed{42,43,44}/*`,
  `outputs/baseline_fixed_summary/{summary.json,summary.md}`.
- Report: `reports/p1_baseline.md` §0 is the post-fix section; §5–§7 are labeled pre-fix.

Headline (3 seeds, 110 test tiles / 25 groups, primary = cal-split Platt probability):

- Risk curves are sane again: `coarse_ridge` UAR at 20% coverage 0.136 (was a spurious 1.000).
- Paired group-bootstrap (1000 resamples, differences averaged over seeds):
  `cnn_fusion - cnn_coarse_grid` ΔBrier −0.0171 [−0.0331, −0.0019] for g10>0.05 and
  −0.0277 [−0.0469, −0.0104] for g15>0.02 → the NAC channel gives a small but
  distinguishable event-probability gain; ΔMAE includes 0.
- `cnn_fusion - coarse_ridge`: ΔMAE +0.0069 [−0.0014, +0.0154] (no gain) and for g15
  ΔMAE +0.0201 [+0.0099, +0.0307] (significantly worse).
- Calibration is not universally helpful: `coarse_ridge` and `cnn_coarse_grid` have better
  raw Brier; `cnn_fusion` and `cnn_nac` need the cal-split Platt fit.
- Split independence re-checked on the hydrated labels: 0 of 272 groups and 0 products
  appear in more than one split.
- `g10` and `g15` heads are not identical (mean |Δ| 0.044–0.160, Pearson 0.88–0.99).

Decision: **GO small scale-up (conditional)** — freeze an extra held-out protocol, expand
training to 3–5k tiles, and add observation geometry plus photometric normalization as
controlled ablations. LFM, joint posterior, TiM, Qwen, weekend 4×4090 and paid cloud stay
on HOLD.
