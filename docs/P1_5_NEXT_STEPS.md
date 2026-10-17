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
