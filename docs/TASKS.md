# P0/P1 任务清单 — 给 Coding Agent

## Sprint P0：现在执行

| ID | 任务 | 交付物 | 验收/停止条件 |
|---|---|---|---|
| P0-01 | 环境、权限、磁盘、代码审计 | `reports/first_run_report.md` 的环境节 | 列明 ROCm/PyTorch/驱动/项目路径及可用磁盘；不覆盖环境 |
| P0-02 | RX 7900 XT 60-step Forward/Backward Smoke | `outputs/rocm_smoke.json` | `torch.version.hip` 存在；GPU 可见、训练 60 步完成、记录峰值显存与速度 |
| P0-03 | SomBench 官方目录与真实文件审计 | `reports/data_audit.md` | 分别报告 listed/downloaded/readable/co-registered 和独立源组数量；绝不将 Parquet 索引行当真实数据 |
| P0-04 | 3m 坡度真实标签与边界单测 | `outputs/slope_labels.csv`, `reports/physics_audit.md` | 检查变量/单位/无效值/网格；未实现地理配准就标 `GEO_JOIN_PENDING` |
| P0-05 | 官方 NAC crater 训练/指标烟测（备用） | `outputs/crater_pilot/metrics.json`, `risk_coverage.csv` | 用真实 `.npy`+COCO，保持官方 split，限定 standard，工程验证而非风险科研 |
| P0-06 | 汇总证据、止损与建议 | `reports/first_run_report.md` | 是否准许进入 P1、还缺什么数据、脚本改了什么、每步复现命令 |

### 必做的单元测试

- `slope>10` 边界：恰好 10° 不应计入大于 10°；当前代码以 10 个阈值形成 **11 个区间桶**（含 `>30°`），与旧 Plan 的“10 桶”并不一致，进入概率模型前统一维度；直方图 `[a,b)` 与严格 `>` 在阈值相等时不同。
- 输入全 NaN、负坡度、超过 90°、有效比例刚好 0.95、空文件、无坡度变量、有多个候选变量。
- 真实安全标签必须基于未平滑计数，不能先做 ILR/ALR 零值替换。
- Crater `standard` 与 `large` 标注协议分离；找不到文件时清晰报错而不是空训练。
- 不假定不同分辨率的 NAC 与 slope 可以按相同像素坐标裁剪。

### P0 的实际 GO / No-Go

- **GO**：拥有可追溯到不同 source/site 的、足够多且地理配准可验证的 NAC+DTM+Slope 监督；报告实际可用独立组数后再由实验计划确定训练规模。
- **BLOCKED**：只有公开少量样例，或者模态缺失/投影不清/无法按 footprint 匹配。允许继续做 Crater smoke，不许包装为地形论文结果。
- **PARTIAL**：环境已通但数据不充分，下一项就是补数据来源，而不是启动大型模型。

## Sprint P1：仅在 G0 通过后立项（本次不自动执行）

1. P1-01：实现基于原始 georeference 的 NAC 和 3m stereo 地形联接，输出 manifest、源 ID、sha256、空间 overlay 样例。
2. P1-02：以 DTM 源/父 NAC EDR/地理块做无泄漏的 train/val/cal/test 划分。
3. P1-03：常数/粗 DTM 基线和小 CNN 或冻结 LFM 特征回归，目标 `g(10°)` 与 `g(15°)`；只使用原始真实标签。
4. P1-04：固定同一地理测试集，报告物理 MAE、事件 Brier、UAR-Coverage 曲线，以及重复站点分组 CI。
5. P1-05：如果多属性真实标签和有效相关误差成立，才探索独立后验与联合后验；不要让联合高斯成为拿到第一张图的前置条件。

**当前研究优先级**：数据完整性 > 物理标签正确性 > 点估计基线 > 风险曲线 > 联合不确定性 > One-Pass。
