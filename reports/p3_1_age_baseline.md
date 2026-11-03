# P3.1 Age baseline

Date: 2026-11-03. **Formal result: BLOCKED_DATA_ACQUISITION.**

The locked design is [p31_baseline_eval_plan.json](../protocols/p31_baseline_eval_plan.json). Age is the official five-class classification task, with unknown prefixes filtered, not continuous age regression. Run independent scratch CNN and Train empirical prior on the stored 11,400 Train / 2,400 Val rows; select only with Val full-five-class Macro-F1. The fixed 50-epoch maximum, patience 5, batch 16, FP32 and seeds 42/43/44 are predeclared. Retaining the last batch, FP32 and HDF5-derived normalization are explicit differences from upstream.

The full pinned HDF5 has not been acquired. Its 74,449,119,347-byte download remains outside the initial cumulative 1 GiB permission and the pending proposal has not been approved. No formal learning curve, multi-seed mean or Test metric exists. The new runner's finite regression on the existing 5 Train + 5 Val pilot is reported separately in `p3_1_runner_validation.md`; it cannot substitute for this baseline. This report adds no accuracy or research gain claim.

When the fixed full asset is available after an explicit acquisition decision, the runner checks full file/statistics/map hashes and ordered split IDs before formal execution:

```bash
python3 -I scripts/p31_train_classifier.py \
  --data "$P31_FULL_HDF5" \
  --stats reports/evidence/p30_20261103/hf_channel_stats.json \
  --class-map reports/evidence/p30_20261103/geologic_units.json \
  --purpose baseline --plan protocols/p31_baseline_eval_plan.json \
  --task age --epochs 50 --batch-size 16 --seed 42 --precision fp32 \
  --outdir outputs/p31/age_seed42
```

Use an existing compatible ROCm runtime. A command here is a prepared execution interface, not a download approval or completed result. Test scoring remains HOLD.
