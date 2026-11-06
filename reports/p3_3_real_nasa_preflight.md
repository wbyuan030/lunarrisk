# Real NASA Paired Preflight

The fixed NASA-IBM Lunar Foundation Model has now run on the local RX 7900 XT for the first time in this project. One authenticated Train 2705 terrain crop passed **both random-initialized and official-pretrained frozen-encoder forward/backward checks** under `moonstone_native256_v1`. This is an engineering preflight, not training, a task score, or evidence of transfer benefit. The user explicitly authorized execution of the pinned NASA GitHub source, the verified checkpoint and this one paired Train preflight; Test remains HOLD.

[Verification and source/asset hashes](evidence/p33_20261106/nasa_real_01/verification.json) · [actual preflight receipt](evidence/p33_20261106/nasa_real_01/preflight.json) · [independent verification](evidence/p33_20261106/nasa_real_01/independent_verification.json) · [normal NASA import](evidence/p33_20261106/nasa_real_01/nasa_import_receipt.json) · [runner receipt-gate check](evidence/p33_20261106/nasa_real_01/runner_gate_validation.json) · [commands](evidence/p33_20261106/nasa_real_01/commands.json) · [actual log](evidence/p33_20261106/nasa_real_01/preflight.log).

| Check | Actual result |
| --- | --- |
| NASA source revision / normal import | `d54c67aad513cb9daca444afa425cfb278e4fbf8`; PASS in a fresh isolated Python process; checkout clean afterwards |
| NASA checkpoint | Revision `b657bdfe6fc4c0445d918f77a2d7b919805f3bce`; 2,425,755,440 bytes; SHA256 `5843824dfa56f89552b37b5295aa865ed28777c5f2b33286db2ef81b4cf08e5b` |
| Strict selected encoder loading | **116 of 116 selected tensor/buffer keys**, 85,761,024 elements, names/shapes and finite floating values verified; 295 decoder or unused-modality keys excluded and reported |
| Input | Train ID2705, target43, elevation and slope each `(1,1,256,256)`, 65,536 finite valid pixels per channel; original authoritative Train membership/label verified |
| Spatial/normalization | Explicit `moonstone_native256_v1`, original config normalization, no interpolation or crop; prepared-tensor hash matches the earlier independent input-only result |
| Random arm | Forward/backward PASS; logits `(1,49)`; 37,681 trainable head parameters; peak device allocation 421,319,168 bytes |
| Pretrained arm | Forward/backward PASS; logits `(1,49)`; 37,681 trainable head parameters; peak device allocation 454,873,600 bytes |
| Actual resource record | Whole child-process peak RSS 3,324,128 KiB; full launcher elapsed 28.34 s. First-arm timing includes cold GPU initialization and is not an arm-performance comparison |
| Runner receipt validation | `validate_preflight` accepts the actual receipt for frozen adaptation, CUDA, fixed config and native256 policy; no model was rerun for this check |

## Execution and evidence boundaries

Execution used the separately installed private CPython3.12.9 / torch2.12.1+rocm7.2 / torchvision0.27.1+rocm7.2 / HIP7.2.53211 environment on the RX7900XT (`gfx1100`). The launcher enabled deterministic PyTorch algorithms before GPU use, checked GPU0 identity and the exact authorized ID/target, ran a fresh subprocess with `-I -B`, and used an `unshare` network namespace with Hugging Face offline flags. The actual launcher is retained only in ignored `outputs/p33/real_paired_preflight_01/launcher.py`; its SHA256 and a path-sanitized command are in the evidence. No shared ROCm, Python or PyTorch environment was replaced.

`load_verified_checkpoint` rechecked checkpoint size/SHA256 before `torch.load(weights_only=True, mmap=True)`, enforced complete selected-inference key and shape coverage, and loaded the pruned inference target strictly. Both arms used the same architecture, head/adapters initialization and input; encoder parameters remained frozen. Only the ordinary Geology head was trainable. The preflight checked finite loss and every trainable gradient, and verified frozen encoder parameters received no gradients. No optimizer step or score was produced. The complete local receipts and log remain under ignored `outputs/p33/real_paired_preflight_01/`; checked-in evidence contains no weights or raw pixels.

An independent non-model pass rehashed the actual checkpoint, config, batch and provenance; verified ID/target, finite masks and shapes; checked the selected-key receipt and source cleanliness; and compared the prepared-tensor digest with the earlier independently verified native256 input. It also checked the real receipt against the runner's existing preflight gate. Neither the model preflight nor earlier 18/75/701 suites were repeated. Kernel-log screening immediately afterwards found no selected GPU fault/reset/BUG/call-trace messages; this is not a long-duration stability test.

Moonstone's nominal north–south grid spacing is approximately236.9 m and east–west spacing varies with latitude; this is not an assertion of equivalence to the NASA configuration's declared60 m source sampling. The official pretraining dataset's resize/crop/flip pipeline is not applied under this separately declared downstream policy. Raw-raster execution lineage and whole-cohort finite-sentinel behavior remain unverified.

The next scientific gates are to establish valid provenance for the complete fixed Train/Val cohort, scan it without scoring, and lock an actual paired budget before training. The example budget is not an executed or frozen budget. Val has already been reused and remains exploratory; the user-authorized preflight does not open Test or establish geological transfer, generalization or safety performance.
