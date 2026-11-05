# P3.3 CPU-only candidate: direct metadata feasible, not provisioned

2026-11-05, after `d0b4c1f`. Remote main was unchanged at this phase's start. This adds a concrete **CPU-only** candidate and an offline metadata audit tool; it does not install an environment, execute NASA, establish GPU compatibility or select an alternative scientific protocol.

[Candidate plan](../protocols/p33_cpu_feasibility_plan.json) · [CPU pair evidence](evidence/p33_20261105/cpu_candidate_01/cpu_pair.json) · [Direct candidates](evidence/p33_20261105/cpu_candidate_01/direct_candidates.json) · [Transitive samples](evidence/p33_20261105/cpu_candidate_01/transitive_samples.json) · [Verification](evidence/p33_20261105/cpu_candidate_01/verification.json).

## Concrete candidate and bounded positive evidence

Use CPython **3.12**, with exact patch/source still unselected. Official indexes advertise matching Linux x86_64 cp312 **torch2.12.1+cpu / torchvision0.27.1+cpu** wheels. Their small PEP658 metadata files were fetched and verified against the hashes advertised in the indexes; the wheel payloads were not fetched or hashed. TorchVision metadata requires torch==2.12.1, which permits the matched +cpu local build. Torch CPU metadata has no CUDA/ROCm/Triton dependency declaration. This is package metadata, not a binary import audit. [Torch CPU index](https://download.pytorch.org/whl/cpu/torch/), [TorchVision CPU index](https://download.pytorch.org/whl/cpu/torchvision/).

All 28 proposed direct pins satisfy the unchanged NASA requirement ranges. All 26 other direct candidates advertise non-yanked wheels with Python3.12-compatible metadata and tags. The selected native wheel floors are at most manylinux2.28, below the recorded host's glibc2.35; this does not prove native-library/ABI behavior. The 72 applicable direct-to-direct constraints checked all match (69 from the 26 PyPI candidates plus 3 from CPU TorchVision metadata). These are metadata checks, not model or regression tests.

Two non-obvious choices matter: TorchGeo0.9.0 itself requires Python>=3.12, despite NASA's broader >=3.11 declaration; SMP must be0.5.0 rather than NASA's allowed minimum0.4 to satisfy the selected TerraTorch/TorchGeo requirements. The complete proposed direct set and official per-package metadata URLs are in the evidence, not a pip installation file.

HTTP HEAD and PyPI metadata advertise **194,106,305 bytes** for the CPU pair plus **250,752,565 bytes** for the other 26 wheels: **444,858,870 bytes** in total. This excludes the interpreter, transitive dependencies, source builds, unpacking, caches, checkpoints and data. It is **not** a complete download allowance or storage budget. Advertised wheel hashes remain unverified until a separately approved payload acquisition.

## Concrete unresolved dependency gap

OmegaConf2.3.0 requires antlr4-python3-runtime==4.9.*. The examined [4.9.3 release metadata](https://pypi.org/pypi/antlr4-python3-runtime/4.9.3/json) advertises no compatible wheel. Other permitted4.9 releases were not exhaustively checked. Therefore an all-wheel closure is not established; source-build behavior and build dependencies need separate review, not an implicit build fallback.

Limited additional samples found metadata-compatible examples for transformers5.3.0 with hub1.15.0/tokenizers0.22.2 and OpenCV-headless4.13.0.92 with NumPy2.4.4. These are not selected transitive pins or API validation. The remaining graph, optional extras requested transitively and native libraries are unresolved. No dependency resolver, build backend, package code or NASA import was executed during this research.

## Offline installed-metadata check

The standalone `scripts/p33_runtime_metadata.py` entrypoint checks direct installed-distribution metadata without importing torch, torchvision or NASA, contacting the network or opening a GPU. It does not alter the existing model wrapper or relax any guard. Its exact CLI and focused validation are recorded in the verification evidence. It needs the already available packaging library; if missing, it reports an audit error rather than installing it.

A metadata match means only that the checked declarations match. It is not a full dependency-closure, binary-ABI, source-cleanliness, model-construction, checkpoint, GPU or numerical certificate. Its intended local use is a later, already authorized candidate environment; its existence does not authorize creating one or installing packages. The current old ROCm environment already fails the known declarations, so do not repeat its inventory or rerun this new tool there merely to rediscover that blocker. Do not rerun completed75/701 tests or the real native256 input preflight to fill those unrelated gaps.

## Local tool validation completed

At `2f59986`, the local follow-up passes all18 new metadata tests with zero skips and compiles the three new files. Independent checks reproduce28 direct bounds,72 direct-to-direct constraints,26+2 advertised wheel/Python tag checks and the444,858,870-byte direct subtotal against archived evidence. Existing NASA source hashes match. [Local report](p3_3_metadata_local_validation.md), [verification](evidence/p33_20261105/metadata_local_01/verification.json). This is tool/metadata validation, not runtime compatibility or a complete lock. No standalone old-environment inventory, old75/701 suite or native256 input preflight was repeated; no provisioning/model execution occurred.

## Later CPU execution remains gated

1. Resolve the bounded transitive/source-build questions and choose exact interpreter/artifact hashes and a complete download/storage budget.
2. Obtain explicit approval before any new private interpreter/environment or package acquisition/installation. Preserve the working Python3.10/torch2.5.1+rocm6.2 and host driver. Do not inherit system-site packages or old ROCm library paths.
3. Only after the environment and execution scope are approved, verify distribution metadata, then normal pinned-source imports and the CPU base constructor with checkpoint_path=None in fresh processes under agreed resource limits. The constructor transiently allocates decoder modules, so memory fit is not yet known. Avoid editable installation into the clean pinned checkout.
4. Checkpoint acquisition/strict coverage, actual paired forward/backward and full fixed-cohort/budget gates remain separate. No GPU query, driver change, asset download, Test access or training is included here.

The GPU driver-support findings remain unchanged. CPU feasibility preparation does not establish that the old driver is incompatible, nor does it solve the GPU training objective. No transfer result is claimed.
