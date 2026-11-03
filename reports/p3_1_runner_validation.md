# P3.1 scalable runner validation

Date: 2026-11-03. **PASS for implementation and finite real-pilot regression. No formal benchmark result.** Classifier/pair/control implementation is committed at `baa97f5`; source-file hashes accompany every run. The regression driver is also identified by its own SHA256 in the archived verification, and is included in this evidence delivery.

## Bounded-memory and split contract

`p31_train_classifier.py` is a new entry point; P3.0 smoke scripts and evidence are unchanged. It streams SHA256 in 4 MiB blocks once per invocation, persists asset verification, and never infers full content validation from mtime. It retains worker-local read-only HDF5 handles, mini-batch training/validation/ordered prediction CSV writing, and the last partial batch. Test is rejected as a requested split; model inputs contain only pixels. Label/ID metadata supports target and split auditing.

The baseline mode requires the prelocked full-data/statistics/class-map hashes, size, ordered Train/Val IDs, and training settings. It does not grant acquisition permission. The current protocol label remains `HDF5_DERIVED_PILOT`; successful sample pairing cannot establish worldwide equivalence.

Each epoch checkpoint includes current and selected-best weights, optimizer, Python/NumPy/torch/GPU RNGs, Train/Val DataLoader generators, fixed total LR horizon, early stopping state/history, config/source/data/split/plan hashes and explicit scaler policy. Current weights resume optimization; best weights produce final predictions. New run directories are exclusive; owned checkpoint updates use `.partial` then atomic replace.

Feature export preallocates chunked HDF5 and writes one batch at a time. It records weights, source/data/stats/map/split/config/selection provenance and channel order, and explicitly states FP32 export with autocast disabled. A feature cache represents that frozen encoder only.

## This execution

Runtime: private host scratch copies of existing container packages, torch `2.5.1+rocm6.2`, HIP `6.2.41133-dd7f95766`, h5py `3.16.0`, NumPy `2.2.6`, host's existing ROCm libraries and RX7900XT. No installation, container filesystem write or shared environment change. A delegated container test had been rejected for temporary writes under its read-only scope; it was not retried. Initial host suite attempts exposed incomplete copied dependencies; completing copies of the existing scientific packages resolved those errors without modifying project behavior. Original failed logs remain local; the final sanitized passing log is archived.

- Full suite: **510 tests, all PASS, zero skips**, 7.420 s. This includes 28 new tests: 19 pairing, 8 classifier and 1 candidate identity. The previous optional hydrated-label check was available in this checkout; it did not skip.
- `python3 -m compileall -q scripts src tests`: PASS.
- New synthetic tests verify a 21-row fixture (11 Train, 9 Val, 1 Test), batch sizes 4/4/3, no Test pixel reads, invalid metadata/IDs/split/target rejection, worker-local handles with two workers, hash boundedness, checkpoint probabilities/RNGs, chunked features, full-K zero-division metrics and metadata-only prior evaluation. Synthetic fixtures do not establish real I/O equivalence.

Real regression reused only the existing 5 Train + 5 Val pilot, SHA256 `cf106933772b7c08ba973cf9c7cfdfdfbcb9e2d038fb87d34f7e05c4cc2b8a35`. No new training examples or Test scoring. Configuration: Age, seed 42, two planned epochs, batch 3 (batches 3+2), FP32, workers 0, same fixed schedule for interrupted and uninterrupted runs.

1. Uninterrupted two epochs, with batched feature export.
2. Two-epoch plan interrupted after epoch one.
3. Resume that checkpoint in a new process, still with planned epochs two.
4. Independent prediction replay in another new process.

**All eleven verification checks passed.** Resume reproduced current/best weights, optimizer state, all RNG states, both loader generators, epoch losses/Val metrics and both prediction files exactly. Independent replay reproduced both prediction files byte for byte. Feature arrays are `(5,256)` per split, chunked `(3,256)`, with correct order/provenance and final partial writes. No Test scoring occurred.

| Evidence | SHA256 |
|---|---|
| Train prediction CSV | `fdc50f536b67938decfb8607bab998a642623b86a3fdbecc5929a0dcd544a457` |
| Val prediction CSV | `09969b6cb782dec2c2109eb686224b86f268ca71bd404bc92b5e97ae55714da1` |
| Selected/current model state fingerprint after two epochs | `b46886554acf38014b69a9f36c6621b379152e84f61b99746040db5ae98028a1` |
| Uninterrupted checkpoint file | `828be50ec84b9facfb78ac11cd31e6fb6d18f3dcde5c0b627a2b9490be6fa683` |
| Resumed checkpoint file | `177950707684d8e9b0b2825fcb2bac242a1dd3274cf8f233b7b07eebc3e86f8f` |

The checkpoint files have different wall-time history and therefore different file hashes; the compared training states themselves are exact. Local checkpoints, feature arrays and all per-run CSVs remain under `outputs/p31/`, outside Git. Public verification, reports, representative prediction CSVs and file hashes live in `reports/evidence/p31_20261103/`.

## Measured resources and limits

| Run | Parent peak RSS bytes | GPU peak allocated bytes | GPU peak reserved bytes | Training window seconds | Training examples/s |
|---|---:|---:|---:|---:|---:|
| Uninterrupted | 2,317,893,632 | 71,321,088 | 98,566,144 | 5.570941 | 1.795029 |
| Interrupted first epoch | 2,027,495,424 | 71,321,088 | 96,468,992 | 0.786505 | 6.357242 |
| Resumed second epoch | 2,042,642,432 | 73,287,168 | 96,468,992 | 0.747128 | 6.692296 |
| Replay only | 1,876,365,312 | 54,801,920 | 71,303,168 | 0 | N/A |

Training windows synchronize the GPU and include loader/normalization/transfer/forward/backward/update. The first process includes cold library/kernel effects; these small-run rates must not be extrapolated to full training. RSS is the parent process; all real runs use workers zero. Allocated/reserved are torch allocator peaks, not whole-card usage. Streamed asset hashing is timed separately in each asset manifest; no full-data scaling claim comes from this pilot.

## Reproduction

Commands below assume a Python runtime with the existing compatible packages available. The actual host execution used `python3 -I` with a trusted task-local bootstrap that explicitly prepended the copied runtime, plus the existing ROCm library path. When using private copies, invoke that bootstrap before the script path; `-I` intentionally ignores `PYTHONPATH`. The regression driver likewise uses isolated child interpreters and explicitly reuses its own trusted torch package directory. Paths below are variables, not private machine paths.

```bash
LUNARRISK_GIT_SHA="$P31_SOURCE_COMMIT" python3 -I scripts/p31_train_classifier.py \
  --data "$P31_EXISTING_PILOT" \
  --stats reports/evidence/p30_20261103/hf_channel_stats.json \
  --class-map reports/evidence/p30_20261103/geologic_units.json \
  --epochs 2 --batch-size 3 --export-features \
  --expected-data-sha256 cf106933772b7c08ba973cf9c7cfdfdfbcb9e2d038fb87d34f7e05c4cc2b8a35 \
  --outdir outputs/p31/runner_uninterrupted_NEW

LUNARRISK_GIT_SHA="$P31_SOURCE_COMMIT" python3 -I scripts/p31_validate_runner.py \
  --data "$P31_EXISTING_PILOT" \
  --stats reports/evidence/p30_20261103/hf_channel_stats.json \
  --class-map reports/evidence/p30_20261103/geologic_units.json \
  --reference outputs/p31/runner_uninterrupted_NEW \
  --outdir outputs/p31/runner_verification_NEW
```

This regression closes runner engineering checks only. Formal Age/Geology reports remain blocked on acquisition of the fixed full asset. It does not add evidence of pretraining, semantic candidate gains, new-class transfer or global performance.
