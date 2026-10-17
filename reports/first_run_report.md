# LunarRisk 7900 XT First Run — Sprint P0

日期：2026-10-17（Asia/Shanghai）
Canonical repo：`/home/wbyuan030/proj/lunarspec`
Runtime workspace：`/data/lunarrisk`（宿主机 `~/rocm_dev/rocm_7_1_ubuntu_2204/data/lunarrisk`，容器内 `/data/lunarrisk`）
Git SHA：本 Sprint 首次提交；精确值见 `outputs/git_sha.txt`（由 `git rev-parse HEAD` 生成）。

## 0. 环境（P0-01）

| 项目 | 实测 |
|---|---|
| Host | Ubuntu 22.04.5 LTS，kernel 6.8.0-138-generic，x86_64 |
| CPU / RAM | AMD Ryzen 7 7840HS，16 threads，30 GiB |
| GPU | AMD Radeon RX 7900 XT，19.984 GiB VRAM，42 CUs，`gfx1100` |
| 第二 GPU | AMD Radeon 780M（iGPU，15.34 GiB，未用于实验） |
| Docker | 28.2.2；容器 `rocm-lab`，镜像 `rocm/dev-ubuntu-22.04:7.1.1-complete` |
| Docker 入口 | 宿主机 system daemon：`DOCKER_HOST=unix:///var/run/docker.sock`（Docker Desktop context 不可用） |
| 设备 | `/dev/kfd`、`/dev/dri/card1,card2`、`renderD128,renderD129` 均可见 |
| Python | 3.10.12 |
| PyTorch | **2.5.1+rocm6.2**（既有环境，未重装/未覆盖） |
| HIP | 6.2.41133-dd7f95766 |
| 磁盘 | `/` 与 `/data` 同一 NVMe，462 GiB 可用（下载前审计通过） |
| 网络 | 容器 `network_mode: host`；HF/S3 经宿主机代理 `http://127.0.0.1:7890` |

**P0-01: PASS**。未安装 CPU/CUDA PyTorch；未使用付费云；未使用周末 4×4090。

## 1. 代码恢复与修复

交接包在解包时发生文件名/内容错位：19 个扁平文件的字节内容整体错位，`scripts/`、`tests/`、`docs/`、`reports/` 目录丢失，5 个脚本与 1 个测试文件缺失。处理：

1. 用 `FILE_INDEX.json` 中的 SHA256 精确还原 17 个文件到原始路径（全部 hash 校验通过）。
2. 缺失文件按 `docs/TASKS.md`/`docs/RUNBOOK.md` 的验收规格重写：
   - `scripts/rocm_smoke.py`
   - `scripts/hf_audit.py`
   - `scripts/make_slope_labels.py`
   - `scripts/crater_pilot.py`
   - `tests/test_slope_labels.py`
   - `requirements-extra.txt`
3. 新增 `scripts/s3_sample.py`：审计公开 S3 100k 发布、抽样下载并验证 NAC+DTM+slope 三元组。
4. 原始错位目录已备份到 `/tmp/lunarspec_backup/`（临时；不进入仓库）。

## 2. P0-02 — RX 7900 XT Smoke

命令：

```bash
python3 scripts/rocm_smoke.py --steps 60 --batch 8 --size 192
```

结果（`outputs/rocm_smoke.json`）：

| 指标 | 值 |
|---|---|
| HIP GPU | AMD Radeon RX 7900 XT (gfx1100) |
| steps | 60 / 60 |
| loss | 0.8368 → 0.0076 |
| elapsed | 3.916 s |
| throughput | 122.57 samples/s |
| peak VRAM | 99.96 MiB |
| dtype | fp32 |
| warning | `hipBLASLt` 不支持 gfx1100，自动回退 hipblas（不影响结果） |

**P0-02: PASS**。仅证明 HIP 前向/反传可用，不代表 LFM 吞吐。

## 3. P0-03 — 数据审计

详见 `reports/data_audit.md`。核心事实：

- HF 公开样例：列出 83 文件 / 142.96 MiB；实际下载 32 文件 / 6.64 MiB；Parquet 索引 **6 行**；NAC/DTM/slope/aspect 各 6 个可解析；完整三元组 6；独立 PRODUCT_ID 6。
- **S3 公开语料 `s3://nasa-lunar-fm-bench` 可无签名访问**：100k 索引 90,786 行，1,090 个 PRODUCT_ID，80 个 LTM zone，90,481 个 slope_3m Tile。抽样 12 个不同 PRODUCT/zone，12/12 可读、CRS 一致、边界对齐（dx=dy=1.0 m）。
- Crater：408 张 256×256 uint8；standard 317 / large 91；official min5px split 219/52/46；磁盘缺失 0；test 与 train/val 的 site 有重叠（不是 site-OOD）。
- split 泄漏：S3 100k 的 train∩val 有 6 个 PRODUCT_ID 重叠；test∩train、test∩val 均为 0。

**P0-03: PASS**。

## 4. P0-04 — 3 m 坡度真实标签

详见 `reports/physics_audit.md`。

- `slope_3m` 变量 `band_data`，171×171，float32；CRS 为 Moon 2015 Sphere / Transverse Mercator；GSD 2.994 m。
- 文件无 `units` 属性；单位由官方 README 声明 + DTM 重算经验验证：12/12 Tile 的 Pearson 0.985–1.000、均值比 0.995–1.025，确认为 **degrees**，并同时验证 slope 与 DTM 同网格。
- `outputs/slope_labels.csv`：HF 6 Tile，每 Tile 29,241 个有效像素，g(10°) 0.008–0.509。
- `outputs/slope_labels_s3.csv`：S3 12 Tile，g(10°) 0.005–0.998，覆盖 12 个独立 zone。
- 边界单测 17/17 PASS：恰好 10° 不计入 `>10°`；11 桶左闭右开；全 NaN；负值/>90°；valid_fraction=0.95；空文件；无/多候选变量；单位缺失/覆盖。
- 所有行标记 `GEO_JOIN_PENDING`；未做 NAC footprint crop/重投影，不能直接作为监督训练对。

**P0-04: PASS**（物理统计与单位验证通过；地理联接仍是 P1 任务）。

## 5. P0-05 — Crater 工程烟测

命令：

```bash
python3 scripts/crater_pilot.py --data data/crater --epochs 5 --batch 16
```

- 数据：official `annotations_min5px_{train,val,test}.json`；仅 standard，排除 large。
- 标签：`count(standard annotations) >= median(train)=114.0`；train 正例率 0.507。
- 模型：4 层 CNN，scratch，Adam，5 epochs，seed 42，GPU。
- test：Brier **0.3363**，NLL 0.8989，AUC 0.6230，accuracy 0.3913；常数基线 Brier 0.2486。
- Risk-Coverage：coverage 0.2 时 UAR 0.333（9 个放行中 3 个高坑密度）；coverage 0.8 时 UAR 0.649。
- site 重叠：train∩test 4、train∩val 4、val∩test 2。

**P0-05: PASS（仅工程管线）**。模型未跑赢常数基线，且 split 非 site-OOD；不得作为科学结论，也不得把“高标注坑数”称为着陆安全。

## 6. 安全与数据泄漏检查

- 未重装/覆盖 ROCm PyTorch；未安装 CUDA/CPU torch。
- 未创建付费云实例；未使用 4×4090；未做 Qwen/TiM/One-Pass/联合高斯。
- 未用 6 个 HF 样例训练任何模型；未生成/复制合成月球标签。
- 未把 Parquet 索引行当作文件；下载前检查磁盘与选择范围。
- crater official split 被原样使用；未用 test 调参；明确记录 site 重叠。
- 所有真实标签来自原生 NetCDF 像素计数；synthetic 只用于单测与 GPU smoke。
- 数据源 revision、索引 SHA256、seed、环境版本均已记录。

## 7. P0-06 — 结论

### 决定：`GO_REAL_A1`

依据：

1. 存在可公开、可下载、可解析的主任务三模态数据：S3 100k 发布 **90,481 个 slope_3m Tile / 1,090 个 NAC PRODUCT / 80 个 zone**。
2. 12/12 抽样三元组 CRS 一致、网格边界对齐；slope 与 DTM 重算逐像素一致（corr ≥ 0.985）。
3. 标签分布覆盖 g(10°) 0.005–0.998，不是常量或少量样例。
4. test 与 train/val 在 PRODUCT_ID 级完全不相交。

### 进入 P1 前必须满足的条件

1. **P1-01**：从 100k 索引生成 manifest（路径、SHA256、split、PRODUCT_ID、LTM、bounds、units provenance、DTM source = null）。
2. **分组切分**：以 PRODUCT_ID + LTM/连续地理块为 group；官方 train/val 的 6 个重叠 PRODUCT 必须合并或重新分配；不得用随机裁块。
3. **地理配准**：实现 footprint 级 NAC→3 m crop/重采样，保留 `co_registration_status`；当前 `GEO_JOIN_PENDING` 不能直接训练。
4. **单位写入 manifest**：`slope_3m` 记录 `units=degrees`、验证方法与证据；禁止自动假定。
5. **DTM source**：官方索引未给每 Tile DTM source ID，manifest 保持 Null，并以地理块作为主要泄漏防线。
6. 只使用 standard crater 协议；crater 结果仅工程参考。
7. 未经明确新指令不启动 P1 训练。

### 下一步最小任务

`P1-01`：对 100k 索引做 manifest + group 统计（不训练），输出可审计的 train/val/test 分组与空间块；随后再决定基线模型规模。

## 8. 复现命令

```bash
# 容器
DOCKER_HOST=unix:///var/run/docker.sock docker exec -it rocm-lab bash
cd /data/lunarrisk

# 环境 + GPU
python3 -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
python3 scripts/rocm_smoke.py --steps 60 --batch 8 --size 192

# 单测
python3 -m unittest discover -s tests -v

# HF 样例与 crater
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  python3 scripts/hf_audit.py --dataset sample --download
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  python3 scripts/hf_audit.py --dataset crater --download

# 坡度标签
python3 scripts/make_slope_labels.py data/sample/NAC_HighRes/slope_3m \
  --units degrees --units-source "SomBench README: slope_3m degrees, range [0,90]"

# S3 审计（下载索引 + 12 Tile）
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  python3 scripts/s3_sample.py --n 12

# S3 坡度标签
python3 scripts/make_slope_labels.py data/s3/sample/slope_3m \
  --units degrees --units-source "SomBench README: slope_3m degrees, range [0,90]" \
  --out outputs/slope_labels_s3.csv --summary outputs/slope_labels_s3_summary.json

# Crater 烟测
python3 scripts/crater_pilot.py --data data/crater --epochs 5 --batch 16
```

产物：`outputs/rocm_smoke.json`、`outputs/hf_audit_sample.json`、`outputs/hf_audit_crater.json`、`outputs/s3_sample_audit.json`、`outputs/s3_sample_overlays.png`、`outputs/slope_labels.csv`、`outputs/slope_labels_s3.csv`、`outputs/crater_pilot/{metrics.json,predictions.npz,risk_coverage.csv}`、本报告与 `reports/data_audit.md`、`reports/physics_audit.md`。
