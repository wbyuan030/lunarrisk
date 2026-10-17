# RUNBOOK — 7900 XT 首批实验操作手册

> 所有命令均在交接包根目录执行，且当前已进入用户现有 ROCm 7.0 Docker/Python 环境。**这里列的是需要 Coding Agent 在用户机器执行的命令，并非本次已经执行的结果。**

## A. P0-01：环境检查，不破坏已安装 PyTorch

```bash
pwd
python --version
python - <<'PY_CHECK_GPU'
import torch
print('Torch:',torch.__version__)
print('HIP:',torch.version.hip)
print('GPU detected:',torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name:',torch.cuda.get_device_name(0))
PY_CHECK_GPU
ls -l /dev/kfd /dev/dri 2>&1 || true
df -h .
```

如果检测失败先检查 Docker 是否挂载 AMD 设备、用户权限、Torch 的 HIP 构建版本；不要直接重装为 CUDA/CPU Torch。

## B. P0-02：真实 RX 7900 XT 计算烟测

```bash
python scripts/rocm_smoke.py --steps 60 --batch 8 --size 192
cat outputs/rocm_smoke.json
```

该脚本无需真实遥感数据，使用随机 Tensor。成功只证明基础 PyTorch GPU 训练与反传可用，不证明 LFM 能跑或科研指标有效。

## C. P0-03：先枚举，再下载 SomBench 样例

```bash
python -m pip install -r requirements-extra.txt
python scripts/hf_audit.py --dataset sample
# 核对列出文件、磁盘配额及下载范围后，再运行：
python scripts/hf_audit.py --dataset sample --download
find data/sample/NAC_HighRes -type f | head -30
find data/sample/NAC_HighRes/slope_3m -name '*.nc' -type f | wc -l
```

如果官方仓库没有公开目标模态，下载的实际文件数可能为零；先核验仓库路径和 `snapshot_download` 的模式，不要生成替代假数据。`NAC_HighRes.parquet` 可能只是目录索引。

### 检查第一份 NetCDF 数据和元信息

```bash
python - <<'PY_INSPECT_NC'
from pathlib import Path
import xarray as xr
files = list(Path('data/sample/NAC_HighRes').rglob('*.nc'))
print('available_nc=',len(files))
if files:
    with xr.open_dataset(files[0]) as ds:
        print(ds)
        for var in ds.data_vars:
            print(var, ds[var].attrs)
PY_INSPECT_NC
```

**重点**：投影/地理边界、坡度单位、nodata、变量维度。没有验证单位不要直接将数值解释为度。

## D. P0-04：原生坡度网格经验统计

```bash
python scripts/make_slope_labels.py data/sample/NAC_HighRes/slope_3m
head -n 3 outputs/slope_labels.csv
```

如目录/变量不同，查实际 xarray 数据变量并使用 `--variable NAME`。成功仅表示真实 NetCDF 坡度网格统计完成，**没有与 NAC 单目影像的地理 ROI 配准**。此时记录 `GEO_JOIN_PENDING`，不能开始 NAC→3m 训练。

## E. P0-05：官方 Crater 真实影像备用训练

```bash
python scripts/hf_audit.py --dataset crater
# 先检查范围再下载：
python scripts/hf_audit.py --dataset crater --download
find data/crater -type f | head -25
python scripts/crater_pilot.py --data data/crater --epochs 5 --batch 16
cat outputs/crater_pilot/metrics.json
head outputs/crater_pilot/risk_coverage.csv
```

若文件在 `data/crater/nac_craters_dataset/` 下，改为 `--data data/crater/nac_craters_dataset`。对照官方 COCO `images[].label_set` 和 `file_name` 及时修正协议识别，不能静默把所有样本过滤掉。官方划分按标注 Box，研究若声称 site OOD 需要另做验证。

运行本地坡度边界单元测试（合成输入，仅测试代码）：

```bash
python -m unittest discover -s tests -v
```

## F. 结果与复现日志

最终填写 `reports/first_run_report.md`，至少包括：

```markdown
# LunarRisk 7900XT First Run
- 日期、Git SHA、机器、GPU、Docker 镜像、Torch/HIP 版本、磁盘:
- P0-01: PASS/FAIL/BLOCKED; evidence:
- P0-02: PASS/FAIL; steps, VRAM, throughput, stdout:
- P0-03: listed / downloaded / readable / complete NAC+DTM+slope / independent groups:
- P0-04: label count, units, CRS, valid fractions, geo_join_pending:
- P0-05: PASS/FAIL/SKIPPED; official split counts, Brier, proxy label note:
- 安全与数据泄漏检查:
- GO / BLOCKED_REAL_A1 / NEEDS_MORE_DATA:
- 需要用户决定的事项与下一步最小任务:
```

## G. 常见失败分流

| 故障 | 优先定位 | 不允许的捷径 |
|---|---|---|
| 没检测到 HIP GPU | `/dev/kfd`, `/dev/dri`, ROCm Torch wheel、Docker 权限 | 安装 CPU Torch 覆盖环境 |
| HF 403、404、缺模态 | 仓库 revision、实际目录、公开权限、文件清单 | 把目录行当文件或生成假数据 |
| slope NetCDF 变量不唯一 | `ds.data_vars`, `attrs['units']`, `--variable` | 任意选第一个数值变量 |
| 仅 6 张可用图 | 数据访问策略、作者沟通、其他合法可用 DTM | 在同源数据裁大量块充样本 |
| 标注 JSON 或 `.npy` 找不到 | 查 `--data` 根目录、仓库命名约定 | 重新随机分割标签 |
| GPU OOM | Batch、Tensor shape、显存峰值 | 在未批准下租 GPU |

## H. 待以后处理

长期代码设计见 `references/PROJECT_CONTEXT.md`，周末 4×4090 计算节点的 10GiB 磁盘保护和 Staging 见 `optional_gpuflow/` 的现有 MVP。**本次不修改 GPUFlow，不用四卡做 P0。**

## I. P1 已落地脚本（2026-10-17 执行）

P1 的报告见 `reports/p1_manifest.md` 与 `reports/p1_baseline.md`。脚本按顺序：

```bash
# 1) 100k 索引 → manifest（分组泄漏审计，不下载 payload）
python3 scripts/build_manifest.py                     # 5 km 块分组
python3 scripts/build_manifest.py --block-m 25000     # 分组敏感性

# 2) 最终切分：train/cal/val/test + 10 km 缓冲（组级零重叠）
python3 scripts/finalize_split.py

# 3) 真实数据下载 + 物理标签（可 --resume 续跑）
python3 scripts/hydrate_tiles.py --train 400 --cal 120 --val 120 --test 120 --workers 12 --resume

# 4) 基线实验：constant / coarse_ridge / cnn_nac / cnn_coarse_grid / cnn_fusion
python3 scripts/train_baseline.py
python3 scripts/train_baseline.py --min-nac-valid 0 --out outputs/baseline_alltiles
python3 scripts/plot_baseline.py
```

约定与坑：

- `outputs/`、`data/` 被 `.gitignore` 排除；产物留在磁盘，报告里给出文件名与哈希。
- `scripts/train_baseline.py` 的 `--min-nac-valid`（默认 0.99）会丢弃 NAC 层无效像素过多的极区 Tile；
  被丢弃 Tile 的标签统计会写入 `metrics.json` 的 `nac_availability`，不要静默忽略。
- 数据缓存为 `outputs/cache_v2.npz`（2 通道：NAC + 60 m 坡度）；布局变更时会自动重建。
- 本机 MIOpen 存在轻微运行间不确定性（CNN MAE 变动约 0.007）；比较小差异要用多 seed 与组级 CI。

## J. P1.6 修复与重跑（2026-10-17 第二轮 review 后）

已修复（均有回归测试）：

1. **缓冲区最近邻遮蔽 Bug**（`scripts/finalize_split.py`）：旧实现先取"唯一最近异集标签"再判定，val/cal 会被更近的
   train 遮蔽而逃过对 test 的距离检查。现改为按优先级（test > val > cal > train）对**仍存活**的集合做 KD-Tree 查询，
   并把 `buffer_contract`（存活集合两两最小距离）写入 `outputs/split_summary.json`。
   *在真实 100k 数据上该修复改变 0 个 Tile 的 split*（drop 仍为 123），但缺陷是真实的。
2. **逐图 1%–99% 拉伸抹杀对比度**（`scripts/train_baseline.py`）：NAC 现为 3 通道
   `nac_contrast = (I - median)/median`（保留绝对明暗对比）、`nac_texture`（局部结构）、`coarse_slope`；
   并把 `log(p99-p1)`、`log(median)` 与观测几何（incidence/emission/phase/太阳方位 sin/cos）拼到 head。
   几何对所有 CNN 变体一视同仁；NAC 派生的对比度标量只给含 NAC 通道的变体
   （`VARIANT_SCALARS`），以保持"固定结构下隔离光学信息"的对照有效。
3. **60 m 粗坡度未按坐标对齐**（同上）：`load_aligned_tile()` 用 `slope_60m` 自身的 x/y 双线性采样到 3 m 网格，
   并把 `coarse_offsets_m`、prefill NaN 比例写入 `metrics.json` 的 `coarse_alignment`。
4. **平衡先验规则**：新增 `g10_gt_0.15` 与复合规则 `g10_gt0.10_or_g15_gt0.03`（阈值只由 train/cal 固定，
   未接触 test）。两条原规则仍走专用事件头；平衡规则用统一的回归 margin 打分 + cal Platt 标定。
5. **DTM 多属性标签**：`hydrate_tiles.py --enrich` 在已下载数据上补出 `dtm3m_roughness_9m_m`
   （3×3 平面去趋势 RMS）与 `dtm3m_relief_p95_p5_m`，无需重新下载。
6. `plot_baseline.py` 删除了未使用的 `--reference` 读取；`hydrate_tiles.py --resume` 在全量已下载时不再报错。

重跑命令：

```bash
python3 scripts/build_manifest.py            # 新增观测几何列
python3 scripts/finalize_split.py            # 修复后的缓冲区（含 buffer_contract）
python3 scripts/hydrate_tiles.py --enrich    # 在本地数据上补几何 + 粗糙度/极差
for s in 42 43 44; do
  extra=""; [ "$s" = "42" ] && extra="--force-cache"
  python3 scripts/train_baseline.py --seed $s --out outputs/baseline_v3_seed$s $extra
  echo $? > /dev/null
done
python3 scripts/aggregate_seeds.py           # 默认读取 outputs/baseline_v3_seed{seed}
```

数据缓存新版本为 `outputs/cache_v3.npz`（3 通道），旧 `cache_v2.npz` 不再使用。

## K. P1.8 数据治理与 held-out 协议（2026-10-17）

新增/修改：

- `scripts/hydrate_tiles.py`：`run_hydration()` 可注入 driver；`--resume` 全量已存在时 **exit 0**，
  `todo` 为空与全部下载失败是两种不同情形（后者 exit 2）；计数器写入 summary；
  已存在行与 manifest 的 split/group/产品/zone/block 不符、`sample_id` 重复、本地源文件缺失、
  旧 cohort 不在新 selection 内 → 一律拒绝。
- 标签写入改为**原子合并**：读既有 → 按 `sample_id` 合并 → 列并集（原顺序保留、新列追加）→ 临时文件 + fsync
  → `os.replace`。`--enrich` 同样原子且要求 `one_to_one`。
- `scripts/aggregate_seeds.py` / `scripts/compare_ablations.py`：新增真值/事件/分组/zone 一致性、
  概率质量、seed/config 协议与 provenance 校验（缺失记 UNKNOWN，不猜测）；NLL 拆分为
  `NLL(primary)` 与 `NLL(raw)` 并做 Brier 交叉校验。
- `scripts/freeze_heldout.py`：`--exclusion {group,product}`、`--max-per-product`、`--min-tiles`、
  `--out-prefix`。严格模式在本 100k 语料上返回 `BLOCKED_NEW_HELDOUT`（272/272 组已被覆盖）。

常用命令：

```bash
python3 -m unittest discover -s tests -v          # 148 tests
# 断点续下（全量已存在时 exit 0，不重写 labels）
python3 scripts/hydrate_tiles.py --resume
# 在已下载数据上补几何 + DTM 属性（原子写回）
python3 scripts/hydrate_tiles.py --enrich
# 冻结严格 held-out（当前 BLOCKED）与产品级候选
python3 scripts/freeze_heldout.py
python3 scripts/freeze_heldout.py --exclusion product --out-prefix heldout_v2_product_level --n 120
# 统计复检（不覆盖旧结果）
for cfg in p17_full p17_no_geo p17_no_photo p17_no_both; do
  python3 scripts/aggregate_seeds.py --pattern "outputs/${cfg}_seed{seed}" \
    --out "outputs/p18_recheck/${cfg}_summary"
done
python3 scripts/compare_ablations.py --out outputs/p18_recheck/cross_config
```

注意：`outputs/` 与 `data/` 被 `.gitignore` 排除；`protocols/` 入库。**V2 冻结后不得因结果好坏更换站点。**

## L. P1.8 续作：fixed-mask 消融与严格 held-out 调研（2026-10-17）

```bash
# P18-04a：等容量消融（每变体 3 通道 + 完整标量 head，缺失模态置 0）
python3 scripts/train_baseline.py --ablation-mode fixed-mask --seed 42 --no-augment \
  --out outputs/p18_full_seed42
# 消融组合：--no-geometry / --no-nac-photometry / 两者
# 合法增强对照（仅无几何条件）：--augment --no-geometry
for config in p18_full p18_no_geo p18_no_photo p18_no_both; do
  python3 scripts/aggregate_seeds.py --pattern "outputs/${config}_seed{seed}" --out "outputs/${config}_summary"
done
python3 scripts/compare_ablations.py --configs p18_full,p18_no_geo,p18_no_photo,p18_no_both \
  --reference p18_full --pattern "outputs/{config}_seed{seed}" --out outputs/p18_fixed_mask_summary

# P18-05b：语料可行性（1M 索引 + LROC NAC DTM 全量清单）
python3 scripts/audit_heldout_sources.py     # 1M release 是否有未触及的组
python3 scripts/audit_lroc_dtm.py            # LROC DTM 档案与 SomBench 产品重叠 + 空间隔离
```

约束与坑：

- `fixed-mask` 与 `legacy` 的指标**不可混比**：聚合脚本会拒绝混用模式或不同参数量的运行。
- 同配置重复运行 MAE 有 ~0.005 的运行间波动（MIOpen）；小于该值的差异不要解释。
- LROC DTM 清单 4.3 MB，来自 PDS；`pyshp` 已加入 `requirements-extra.txt`。
- 已冻结的 `protocols/heldout_v2_product_level_*` 不得重写；其 `exclusions` 字段的历史勘误见
  `reports/p1_8_heldout_protocol.md` §7。
