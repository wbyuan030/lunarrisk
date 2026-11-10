# P3.3 frozen-encoder paired probe: pretrained beats random by +0.116 macro-F1

2026-11-10. **`FROZEN_PROBE_COMPLETE_POSITIVE_TRANSFER_SIGNAL`.** On the fixed 5,135 Train / 1,155 Val cohort, a linear 49-way Geology head trained on mean-pooled tokens from the official pretrained NASA encoder reaches **0.1885 ± 0.0083** Val full-49 macro-F1 (three seeds, sample standard deviation), against **0.0721 ± 0.0018** for an identically initialized random encoder. The paired difference is **+0.1164 ± 0.0090** per seed. This is a first positive transfer signal for frozen features, not a fine-tuning result, not a significance claim and not a Test result.

| Seed | Pretrained macro-F1 | Random macro-F1 | Difference | Pretrained acc. | Random acc. | Best epoch (P/R) |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.186553 | 0.074159 | +0.112394 | 0.4416 | 0.3368 | 10 / 10 |
| 43 | 0.181371 | 0.071343 | +0.110028 | 0.4407 | 0.3385 | 9 / 9 |
| 44 | 0.197607 | 0.070913 | +0.126694 | 0.4476 | 0.3411 | 20 / 8 |
| Mean ± std | 0.188510 ± 0.008293 | 0.072138 ± 0.001763 | +0.116372 ± 0.009017 | - | - | - |

## Design actually executed

Both arms use the official base architecture, the two raw physical inputs (`elevation` metres, `slope` degrees) normalized only by the audited NASA config under `moonstone_native256_v1`, a frozen encoder, an identity input adapter and one `Linear(768, 49)` head with **identical** initialization created by deep-copying a seeded head. Only the encoder weights differ: the verified checkpoint, strictly loaded, versus the same architecture seeded identically. Batch order, epoch permutation `Generator().manual_seed(seed + epoch)`, optimizer (AdamW lr 1e-3, wd 1e-2, betas .9/.999, eps 1e-8), gradient clipping 1.0, 20 epoch maximum, patience 5 and earliest-tie Val selection are identical across arms, seeds and this pre-registered [plan](../protocols/p33_frozen_probe_plan.json).

Features are the encoder token sequence `(512, 768)` mean-pooled to 768 values, exactly the pooling `TerrainTransfer.forward` applies before its head; a smoke batch showed `head(mean(tokens))` and the full forward agree to `0.0`. Extraction ran once per arm over the full cohort with TF32 disabled, then heads trained on the cached vectors in seconds.

## Gates that ran before any score

- Fresh-process feature recomputation differs from the cached snapshot by at most **1.47e-4** (pretrained) and **4.8e-7** (random), against features whose mean absolute magnitude is 0.28 and 0.77. The acceptance bound was set to 1e-3 from that measurement, before training; bitwise identity across processes is not claimed, because ROCm kernels select different paths in different process lifetimes even though they are stable inside one process.
- Two independent head-training runs on the same cache produced **identical** per-arm macro-F1 values, so the reported numbers are reproducible from the archived cache manifest.
- No Test ID is constructed, read or scored; `test_scored` is false everywhere.

## Caveats that limit the claim

- Val was used for early stopping and checkpoint selection, so the absolute numbers are mildly optimistic, equally for both arms.
- Three seeds give a sample standard deviation, never a confidence interval, and the difference is consistent in sign across all three.
- The random arm is unusually strong (34% accuracy on 49 classes) - random ViT features already carry terrain structure - so the baseline is not trivial, and the gap is still large.
- Pretraining-data overlap with the Val geography cannot be excluded; this is a frozen-feature transfer signal under overlap, not a clean generalisation claim.
- Only a linear head on mean-pooled features is trained. No encoder gradient, LoRA, adapter learning, full fine-tuning, Test evaluation or new-class capability is established.
- These numbers are not comparable to the P3.2 context-head matrix, which used a different frozen encoder, different features and a different head.

## Reproduction

The cache holds `(5135, 768)` and `(1155, 768)` float32 vectors per arm with SHA256-pinned manifests; extraction ran in `unshare --net` with the private Python 3.12.9 / torch 2.12.1+rocm7.2 runtime on the RX 7900XT. Evidence: [summary](evidence/p33_20261110/frozen_probe_01/summary.json), [cache manifest](evidence/p33_20261110/frozen_probe_01/cache_manifest.json), [feature check](evidence/p33_20261110/frozen_probe_01/feature_check_replay.json), six [Val prediction files](evidence/p33_20261110/frozen_probe_01/) and [verification](evidence/p33_20261110/frozen_probe_01/verification.json).
