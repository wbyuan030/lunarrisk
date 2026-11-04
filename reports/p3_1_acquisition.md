# P3.1 authorized full benchmark acquisition

Updated: 2026-11-04. **Single-file acquisition PAUSED for VPN quota at 33,964,443,553 bytes.** Explicit large-file authorization remains recorded, but the user's later direction is to use existing data without further transfer. Full asset integrity has not been reported PASS. A [separate offline subset track](p3_1_partial_data_baselines.md) uses complete local Train/Val chunks; complete-data formal scores remain pending. Recovery narration below records the earlier running state.

The user instructed: “允许大文件下载 推进基线”. [The new authorization record](../protocols/p31_acquisition_authorization.json) permits the previously recommended fixed HDF5 and locked local Age/Geology Train/Val baselines. The original P3.0 proposal and P3.1 scientific plan remain unchanged as historical artifacts; their old not-authorized fields do not revoke this later explicit instruction. This does not authorize full mmap, new weights, snapshots, paid compute, four GPUs or Test scoring.

| Fixed asset | Value |
|---|---|
| Repository | `ayushprd/Moonstone` |
| Revision | `53099efd0f2db81eee96b28eae87e1efdd281310` |
| File | `benchmark/lunar_patches_v4.h5` |
| Expected size | 74,449,119,347 bytes |
| Expected full SHA256 | `e3e96bccc044d856cd55883c9cda7628f02afbdc1b2d1818b401bbcc590df83f` |
| Data-card license | CC-BY-4.0 |
| Workspace allowance | 80 GiB; single data copy, not a measured peak |
| Local namespace | `outputs/p31/assets/moonstone_53099efd/` |
| Executed source | Initial `39ac6d4`; bounded-window recovery `ce72160` |

Preflight observed approximately **463 GiB free** (`df -h`), and the downloader also checks exact available workspace before starting. It writes only `lunar_patches_v4.h5.partial` until complete size and full streamed SHA256 match; only then does atomic rename create the final HDF5. The sibling manifest and lock are small control files. No HF snapshot/cache copy is made. Source/scripts remain outside the data directory.

The downloader uses 4 MiB bounded reads and HTTPS-only bounded redirects without reading redirect bodies. The initial unbounded-response invocation stopped after three premature EOFs. Recovery now requests exact **64 MiB HTTP range windows**, with at most three interrupted responses per invocation; successful windows do not consume that interruption limit. Any reconnect resumes exactly at the owned partial's current length: nonzero-offset HTTP200, wrong Content-Range, wrong declared length and content encoding are rejected before reading a body. An unfinished or hash-mismatched partial is retained for audit, never silently restarted from zero or renamed as verified data. An existing final asset is fully checked before reuse. Simultaneous acquisition is excluded by a local file lock.

Response-body counts are recorded; HTTP/TLS/redirect overhead and bytes consumed internally before a transport exception are not claimed as measured. The original initial-sample conservative ledger remains **165,257,071 bytes** under its unchanged 1 GiB cap. This authorized complete asset is a separately recorded scope, not a reset or expansion to unrelated files.

Executed command, with private runtime paths represented by variables:

```bash
python3 -I "$JOB_TMP/p31_python.py" scripts/p31_acquire_benchmark.py \
  --outdir outputs/p31/assets/moonstone_53099efd --apply
```

The trusted task-local bootstrap reuses private copies of existing container packages on the host. No shared environment, torch installation or container filesystem write is required. Initial log: `outputs/p31/full_acquisition_01.log`; current recovery log: `outputs/p31/full_acquisition_02.log`; live manifest: `outputs/p31/assets/moonstone_53099efd/download_manifest.json`. The recovery uses the same command plus explicit `--resume` on the same owned directory. Do not run concurrent acquisitions.

### Observed interruption and recovery

The first process ended with `EOFError` after three prematurely terminated responses, retaining **14,590,953,377 bytes** (about 19.6% of the asset). This is a transport failure, not a completed download or checksum PASS. Exact upstream/proxy cause is unknown. Before recovery, all **48 previously verified HDF5 range bodies** lying within that saved prefix were compared against the partial and matched their independent hashes. This checks those selected regions only, not the entire prefix or asset.

The same partial is now being continued using strictly bounded 64 MiB requests, with explicit start/end/total validation before each body and a final short window. No prefix was discarded or re-downloaded, and no second data copy was created. The interrupted manifest and partial-region check inventory are preserved in `reports/evidence/p31_acquisition_recovery_20261104/`. Full-data SHA256 and formal training remain pending.

Recovery validation: **534 repository tests, all PASS, zero skips**, 11.617 seconds; compileall PASS. The three added cases exercise multiple successful windows without consuming the interruption limit, exact final-short-range continuation across a process restart, and HTTP200 rejection for an initial bounded range. The scientific baseline plan and trainer/launcher sources are unchanged.

Offline validation: **7 acquisition tests PASS**, covering bounded reads/hash/atomic rename, exact partial continuation, HTTP200/wrong-range/encoding rejection before body, checksum failure, verified reuse/tampering and disk/unowned-partial preflight. Default allowlist dry-run PASS with zero network; compileall PASS. Synthetic bodies do not certify the full asset. The live checksum, final bytes, elapsed time and download counters will be recorded only after completion.

After verification, execute the already fixed baseline configuration: Age42, Geology42, then remaining seeds43/44 after first-seed numerical/resource checks. Preserve the 50-epoch maximum, batch16, FP32, workers0, patience5 and official stored split; select only on Val full-K Macro-F1. Full Test bytes may exist in storage and be covered by file checksum, but Test samples are not used for training, validation or model scoring.
