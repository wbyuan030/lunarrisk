# P3.1 candidate ID control

Date: 2026-11-03. **Structural identity and bounded latency measurement PASS. No trained accuracy or semantic result.** The old `CandidateIDHead` API and all P3.0 checkpoints remain unchanged; the report name is `candidate_id_control`.

The shared scorer computes `s_c=f(x,embedding(c))` independently for every known public ID. Consequently `softmax(head(x,S))` equals masking and renormalizing **that head's own** full-class probabilities. It is not an identity with the separately trained ordinary classifier. The new regression exercises singleton, nonordered subsets and a reversed full list. The GPU control also verified the identity for K=1/5/10/25/49; maximum absolute discrepancy was `1.4901161193847656e-08`.

## Actual compute and latency

Measured on the existing RX7900XT, torch `2.5.1+rocm6.2`, FP32, batch 2, seed-42 random initialization, no training. Fixed first two existing Train crops (IDs 65 and 0); candidate lists are the first K public IDs for every sample, independent of target. No accuracy is calculated. Both heads use the same scratch frontend and the same features/data. For each condition, three warmups; 30 synchronized head measurements and 10 synchronized complete measurements. Measurements are sequential, without concurrent GPU training.

| K | Ordinary head+mask median ms | Candidate ID head median ms | Ordinary complete median ms | Candidate ID complete median ms |
|---:|---:|---:|---:|---:|
| 1 | 0.2048 | 0.3318 | 35.5125 | 36.6403 |
| 5 | 0.1783 | 0.3076 | 36.6707 | 36.5120 |
| 10 | 0.1766 | 0.2910 | 36.7794 | 37.2243 |
| 25 | 0.1720 | 0.2996 | 35.9438 | 37.0986 |
| 49 | 0.1922 | 0.3035 | 36.6220 | 37.0407 |

The complete path includes selected HDF5 reads, normalization, stacking, host-to-device transfer, scratch frontend and head/softmax; filesystem caches are warm and not cleared. Ordinary includes its mask validation/softmax. Candidate includes public-ID input validation, embedding/concatenation, shared MLP and softmax. Frontend including read/normalization/transfer without a head had median **36.0269 ms**. Full distributions and source/input hashes are archived in `reports/evidence/p31_20261103/candidate_control.json`.

Parameter counts are 12,289 ordinary and 12,293 candidate ID. Counting only linear multiply-accumulates per example gives 12,200 ordinary versus `11,466*K` candidate ID, or **561,834 at K=49**, approximately 46 times the ordinary head. This analytic count excludes embeddings, input validation, concatenation, activations, masking/softmax and frontend. Measured head latency does not scale directly with these counts at this tiny batch; validation and launch overhead matter. The ID head is slower in every measured head-only condition, and the complete path is dominated by input/frontend work. Small full-path timing differences do not establish a speed benefit. Similar parameter counts are insufficient evidence of equal compute or OnePass acceleration.

## Scope and command

```bash
python3 -I scripts/p31_candidate_control.py \
  --data "$P31_EXISTING_PILOT" \
  --stats reports/evidence/p30_20261103/hf_channel_stats.json \
  --class-map reports/evidence/p30_20261103/geologic_units.json \
  --outdir outputs/p31/candidate_control_NEW
```

This is an inference structure/timing control on random weights, not a trained candidate comparison. No Test was scored. Stored split is not an unseen-class protocol. A future semantic scorer must first lock public descriptions/text model/cost and use identical representation, modality, sample budget and ordinary masked-softmax control. Per-sample true geological unit names, age prefixes or target-guaranteed candidates cannot enter its inputs under the current protocol. No semantic head is implemented in this delivery.
