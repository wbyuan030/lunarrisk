# P3.1 Geology baseline

Updated: 2026-11-04. **Acquisition AUTHORIZED and RUNNING; formal Train/Val result pending verified full asset.**

The core task remains official 49-output classification. [p31_baseline_eval_plan.json](../protocols/p31_baseline_eval_plan.json) fixes independent-task scratch CNN and Train empirical prior, stored split, hashes, training budget and Val model selection before scores. Full 49-class Macro-F1 includes zero F1 for absent classes, matching pinned upstream `downstream_base.py`; supported-class macro is additional, with per-class counts and confusion. Train has 46 represented classes; the three absent classes are also absent from Val/Test. This is not an unseen-class protocol.

The user explicitly authorized the fixed single HDF5 acquisition and advancing baselines on 2026-11-04; see [authorization](../protocols/p31_acquisition_authorization.json). Download is running, with size/SHA256 verification required before training. No formal Geology run, multi-seed aggregate or Test score is reported yet. P3.0's Age-supervised five-example feature cache is preserved as old engineering evidence and is not relabeled scratch Geology or lunar pretraining. The ordinary P3.1 Geology CNN is initialized independently, receiving no extra Age supervision.

Prepared command after the fixed full asset becomes available under an explicit acquisition decision and after first Age training-chain validation:

```bash
python3 -I scripts/p31_train_classifier.py \
  --data "$P31_FULL_HDF5" \
  --stats reports/evidence/p30_20261103/hf_channel_stats.json \
  --class-map reports/evidence/p30_20261103/geologic_units.json \
  --purpose baseline --plan protocols/p31_baseline_eval_plan.json \
  --task geology --epochs 50 --batch-size 16 --seed 42 --precision fp32 \
  --outdir outputs/p31/geology_seed42
```

Use an existing compatible ROCm runtime. Seeds 43/44 follow the locked configuration after first-seed verification. Any candidate training comparison must share encoder, modality, examples and budget with ordinary masked-softmax; ID-head parameter similarity alone does not establish equal compute. Semantic descriptions and any new class holdout require a separate design. Test scoring remains HOLD.
