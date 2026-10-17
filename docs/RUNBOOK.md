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
