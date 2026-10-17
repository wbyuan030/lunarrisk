# P1.7 — 评估硬化、可复现消融与独立地理验证

状态：**代码已就绪，实验尚未运行**（2026-10-17）。
基线：P1.6 v3，commit `29d4b73`；399 train / 110 test，25 个测试地理 Group。参阅 `reports/p1_baseline.md` 顶部的 P1.6 结果。

> 给 Pi / Codex 的可执行交接：先修指标口径与输入消融，后审计新测试区域，再决定是否扩到 3–5k Tile。
> 未经用户另行批准，不使用 4×4090、付费云、LFM/Qwen 或大型联合后验。

## A. 这次提交已完成的代码工作（需要在 ROCm 主机实测）

| 项 | 位置 | 行为 |
|---|---|---|
| P1.7-A1 几何安全增强 | `scripts/train_baseline.py` | 默认不做翻转/rot90；`--augment` 必须显式与 `--no-geometry` 联用，因为静态太阳方位标量没有随旋转同步变换 |
| P1.7-A2 控制变量消融 | 同上 | `--no-geometry` 去除所有变体的观测/光照几何标量；`--no-nac-photometry` 去掉 NAC 绝对对比度通道和 NAC 亮度/对比度标量，保留归一化纹理通道 |
| P1.7-A3 风险排序并列值 | 同上 | 截断点出现并列分数时，按随机打破并列的期望危险数量计算 UAR/HMR；避免常数模型结果受 CSV 行序影响；`unsafe_accepted_n` 在并列时可为小数（期望值） |
| P1.7-A4 来源独立性 | `scripts/aggregate_seeds.py` | 聚合前发现任何跨 split 的源组/产品立即退出，不再只是记录 |
| P1.7-A5 双事件头审计 | `scripts/train_baseline.py` + `scripts/aggregate_seeds.py` | 导出 `cnn_*_raw_event_prob_g10_gt_0.05` 与 `cnn_*_raw_event_prob_g15_gt_0.02`，比较真正的**危险事件分类**输出而非两路回归 fraction；缺列或完全相同时退出 |
| P1.7-A6 单元测试 | `tests/` | 补充并列分数、消融特征配置、非法增强、泄漏断言和真实事件头检查 |

以上修改改变训练和评估协议，**P1.6 的旧模型数字不可直接与 P1.7 横向比较**。必须在新的固定配置上重跑所有对照。旧结果仍可作为历史探索性结果。

## B. 首先运行（本地 RX 7900 XT；无需新下载）

在仓库根目录、原有 ROCm Docker/PyTorch 环境中执行：

```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 scripts/train_baseline.py --help
```

异常应按实际原因修复后再训练，不得把 synthetic fixture 冒充真实地形实验。
特别检查 `--augment` 单独使用时是否报错；`--no-geometry --augment` 应当允许运行。

## C. 受控消融：**先不追加新数据**

固定：相同缓存 `outputs/cache_v3.npz`、可用 Tile 列表、train/cal/val/test 分组、训练参数和 Seed。运行每组 42/43/44 共 **12 次训练脚本**，每次含 NAC / coarse / fusion 三个 CNN（Ridge 与常数基线用于参照）：

| run | 观测几何 | NAC 绝对对比度 | 图像空间增强 | 目的 |
|---|---|---|---|---|
| `p17_full` | 是 | 是 | 关闭 | 一致几何条件下的基线 |
| `p17_no_geo` | 否 | 是 | 关闭 | 几何信息的贡献 |
| `p17_no_photo` | 是 | 否 | 关闭 | NAC 绝对对比度与亮度信息的贡献 |
| `p17_no_both` | 否 | 否 | 关闭 | 仅保留 NAC 纹理与 60m 坡度 |

```bash
set -euo pipefail
for config in p17_full p17_no_geo p17_no_photo p17_no_both; do
  case "$config" in
    p17_full)     flags="" ;;
    p17_no_geo)   flags="--no-geometry" ;;
    p17_no_photo) flags="--no-nac-photometry" ;;
    p17_no_both)  flags="--no-geometry --no-nac-photometry" ;;
  esac
  for seed in 42 43 44; do
    # Cache 仅用于统一数据输入；不同模型/消融使用各自独立输出目录。
    python3 scripts/train_baseline.py --seed "$seed" --no-augment \
      --out "outputs/${config}_seed${seed}" $flags
  done
  python3 scripts/aggregate_seeds.py \
    --pattern "outputs/${config}_seed{seed}" \
    --out "outputs/${config}_summary"
done
```

重要：上面 shell 对 `flags` 的单词拆分是预期行为，禁止把包含多个参数的 `flags` 整串包成一个 shell 参数。

**验收指标**：同一规则的 MAE、raw/标定 Brier、NLL、AUC、UAR@20/40%，各 Seed 均值/标准差与地理组配对 Bootstrap 差值。每组结果带 `metrics.json.protocol.ablation`、Git SHA、缓存版本。报告四种配置中的 `cnn_fusion` vs `cnn_coarse_grid` 的配对差异，以及 `cnn_nac` 的单模态表现。

**特别约束**：这 110 张旧 Test 已用于多轮查看，消融结果只能称为探索性；不能反过来据此选定“最终最优模型”并声称在它上面获得独立 Test 性能。

可选稳定性对照：`--no-geometry --augment`，**仅在无几何标量的配置**中增加一组 3 Seed 对照；不能把它和关闭增强的结果当作仅改变某一个因素之外的可直接比较实验。

## D. 独立测试与数据扩容：在 C 审计后进行

**P1.7-B1：在创建更多训练实验之前冻结新的 Held-out V2 测试清单。**

- 从 90,786 索引中根据源 NAC EDR、地理块及投影区选择**此前未用于训练、验证、标定、或上面模型决策**的新站点；记录 sample_id、PRODUCT_ID、LTM_CODE、分组、种子、空间缓冲分布、Manifest SHA256。
- 对可能复用的 NAC/立体 DTM 同源区域做人工 footprint 审核。当前 `DTM_source_ID` 缺失，**不能声称已经完成严格的 DTM 源隔离**。
- 冻结清单前不看这些区域的模型指标；一旦冻结不得因表现不好而换测试集。旧测试集保留为 `development_test`，新集命名 `heldout_v2`。
- 不要通过移动旧 Test/Cal Tile 到 Train 来扩容；在不改变固定验证/标定/测试集的前提下追加训练组。

**P1.7-B2：训练扩至 3–5k 真 Tile（RX 7900 XT 先试）。**

- 扩容仅从合格 `train` 组抽样；记录有效 NAC 比例和被排除样本的区域偏差；先检查本机可用磁盘，再 `hydrate_tiles.py --resume`。
- 必须先确认当前 `hydrate_tiles.py` 的 `--resume` / `--enrich` 对“从 400 扩到 3000+”的追加语义，避免重选样本或覆盖既有 split。
- 对 V2 评估前固定一个小型候选配置列表（例如 full 与 no_geo、以及强 Ridge），严禁在 V2 上调阈值/模型/校准。

**P1.7-B3：报告与 GO/NO-GO。**

- `reports/p1_7_ablation.md`：12 次实验的准确命令、环境、Git SHA、表格、误差条、旧数据探索性说明。
- `reports/p1_7_heldout_protocol.md`：新测试集选择算法、数据来源、冻结 ID/hash、源级和空间审计。
- `reports/p1_7_scaleup.md`：训练规模、独立测试集结果、每组负例/危险率与置信区间。

只有在 V2 独立集上观察到光学增量、标签/地理来源真实有效时，再考虑 LFM 冻结特征、更多地形属性的独立/相关后验及 4×4090。原定实验 `g10>0.05`、`g15>0.02` 与探索性的平衡/复合规则必须明确区分，复合规则目前仍是工程代理，不是真实着陆安全认证。

## E. 当前已知限制与止损

- P1.6 报告已验证的 81 项测试和结果属于**旧 commit**；当前新增修改未经 ROCm 主机回归实测。
- 新的平均 UAR 并列规则会改变 `constant` 的历史曲线；同分条件下 `unsafe_accepted_n` 是期望数，不要解释成整数真实被放行危险样本数。
- 原始 3m DTM 的准确 stereo 源 ID 未提供；仅坐标重采样与 10km 中心间隔不构成独立空间 OOD 的完全证明。
- 几何标量是切片级汇总，不是物理渲染的逐像素光照模型；不要把融合网络当作已解决单目形状-光照歧义。
- 仅本地 AMD，不能以推送了脚本代表 GPU 实验已经完成。
