# P2.2 P22-00 — 三处评分修复的实测验证（2026-10-23）

**状态：PASS（代码与离线测试）/ 部分受限（ROCm 确定性需显式开启）。**
本报告只验证 P2.2 之前提交的三处 P0 修复，不重复 P2.1 的历史数字。
所有命令与输出来自本轮在本地 RX 7900 XT + ROCm 容器 `rocm-lab` 内的真实运行。

## 1. 代码状态与环境

| 项 | 值 |
|---|---|
| 起始提交 | `aadec1c`（`docs(readme): make P2.2 corrected posterior evaluation the current sprint`） |
| 工作树 | P2.2 修改未提交时运行；最终提交见 `git log` |
| Python | 3.10.12（容器 `/data/lunarrisk`） |
| torch | 2.5.1+rocm6.2，`torch.version.hip = 6.2.41133-dd7f95766`，`torch.cuda.is_available() = True` |
| GPU | AMD Radeon RX 7900 XT（单卡，未使用四卡/云卡） |
| 依赖 | numpy 2.2.6、pandas 2.3.3、scikit-learn 1.7.2 |
| 数据 | `outputs/labels_hydrated.csv`（760 行）+ `outputs/cache_v3.npz` |

## 2. 本轮新增的验证代码

| 改动 | 位置 |
|---|---|
| 点模型温度真实缩放残差标准差 | `scripts/train_posterior_p21.py::predict`（P2.2 前提交） |
| `U=0` 默认共享 Cal-T，并记录 `temperature_policy` | `train_posterior_p21.py::main` |
| `--ablate-u-calibration-mode {shared,refit,both}` | 同上 |
| `--matched-marginals`（保留 `diag(Σ_joint)`） | `posterior.py::covariance_from_params`、`predict(covariance_mode=...)` |
| 物理 NLL 使用真实 `z` 与正号 Jacobian | `posterior.py::nll` / `jacobian_term`（P2.2 前提交） |
| 可加载 `*.pt` checkpoint 与 round-trip 推理 | `save_checkpoint` / `load_checkpoint` / `predict_from_checkpoint` |
| 逐样本概率导出（所有 split，含 U 消融分支） | `--predictions-out` |
| `--deterministic` | `torch.use_deterministic_algorithms(True)` |

## 3. 复现命令与真实结果

```bash
docker exec rocm-lab bash -lc 'cd /data/lunarrisk && python3 -m compileall -q scripts src tests'
python3 -m unittest discover -s tests -p 'test_train_posterior_p21.py' -v
python3 -m unittest discover -s tests -p 'test_posterior.py' -v
python3 -m unittest discover -s tests          # 全仓
```

| 命令 | 结果 |
|---|---|
| `compileall scripts src tests` | 无错误 |
| `test_train_posterior_p21.py` | **15 tests OK** |
| `test_posterior.py` | **18 tests OK** |
| 全仓 `unittest discover` | **431 tests OK**（P2.1 结束时为 319 项历史记录；上一次记录 416 项，本轮新增 15 项后 431 项） |

> 说明：历史 “412/319 项通过” 属于旧代码；本轮以 431 项真实输出为准。

## 4. 修复一：Point 温度必须作用于采样

- 单测 `test_temperature_scales_point_diagonal_and_joint_noise_not_mean` 对
  `point/diagonal/joint` 断言 `z_T − mu = T (z_1 − mu)`（`T = 1.8`），
  `T = 1` 时与未校准完全一致，`T = 0` 被拒绝。
- 真实运行（joint, seed 42, 40 epoch）导出 Cal-T `1.0012`，并用于逐样本概率；
  三次 joint 运行的 Cal-T 为 `1.0012 / 0.9757 / 1.0019`，Point 为
  `0.9876 / 1.1592 / 0.9740`，均有限且为正。

## 5. 修复二：同 checkpoint 的 `U=0` 校准口径

单测 `test_same_checkpoint_u_zero_ablation_preserves_shared_temperature` 验证：
同一 joint checkpoint、同一 Monte-Carlo 随机数下，`U=0` 与完整 joint 的差
只来自协方差（`mu`、encoder、T 全部相同）。

真实 joint 运行（`--ablate-u-calibration-mode both`）：

| seed | Cal-T(full joint) | shared U=0 T | 独立 refit U=0 T |
|---|---|---|---|
| 42 | 1.0012 | 1.0012 | 2.1178 |
| 43 | 0.9757 | 0.9757 | 1.3911 |
| 44 | 1.0019 | 1.0019 | 1.5460 |

**结论**：`shared_T` 与 `refit_T_on_cal` 是截然不同的实验口径（refit 温度约为 1.4–2.1），
二者不得混入同一张表。`U=0` 同时缩小边缘方差，因此另外实现了
`--matched-marginals`：保留 `diag(Σ_joint)`，只去掉 off-diagonal。
一个结构性的实现事实（由 Cholesky 下三角决定）：**`P(g10>t)` 在 matched-marginals
下与完整 joint 完全相同**（第一行只用到 `noise[0]`）；只有涉及第二及以后属性的
事件（`g15`、OR、AND、NOT）才携带相关性信息。因此 matched-marginals 的归因只能
在这些规则上做。

## 6. 修复三：物理密度 NLL

- `test_physical_nll_equals_internal_plus_observed_target_jacobian`：断言
  `NLL_phys = NLL_internal + log|det J_f(z_true)|`；
- `test_jacobian_matches_finite_differences`：对 `logit` 链与 `exp` 变换做数值差分；
- `test_physical_nll_differs_by_the_jacobian`：替换 `z` 后物理 NLL 改变，而内部 NLL 不变。

**边界说明（必须保留）**：`g10=0/1`、`g15=0` 等端点由 epsilon 截断，连续变量变换
在端点没有正确的点质量。本轮所有 P2.2 事件指标均为**内部连续区间**的规则概率，
未把被截断的物理密度当作最终论文指标。

## 7. 额外发现：ROCm 上同 Seed 并不可复现

同一命令（joint, seed 42, 5 epoch）连续两次运行：

```
weights_sha256: a8dadd28… ≠ 14c252da…     train_nll 完全相同（8.452755）
```

即损失打印一致但权重逐位不同（MIOpen/归约顺序）。加入
`torch.use_deterministic_algorithms(True)` 后两次运行
`weights_sha256 = bf3e978d… == bf3e978d…`，完全一致。

因此 P2.2 所有正式运行使用 `--deterministic`；报告中的 “3 Seed” 差异是真正的
训练种子差异，不混入未受控的运行噪声。未开启该开关的历史数字需要重新运行才可比较。

## 8. Checkpoint 落盘与 round-trip

`--save-checkpoint` 写出 `torch.save` 的 `*.pt`，包含：

- `state_dict`、`variant/rank/width/in_channels/n_scalars`；
- 训练集拟合的 `scalar_mean/std` 与列名；
- `point_sigma`、`temperature`、`seeds`、`target_eps`；
- `labels_sha256`、`cache_sha256`、`git_sha`、`weights_sha256`。

实测：新进程调用 `predict_from_checkpoint` 重新计算 joint seed42 的 5 条规则概率，
与导出 CSV 的 `max|diff| = 0.00e+00`（全部 120 个 test 行）。
单测 `test_reload_reproduces_per_rule_probabilities` 覆盖该路径。

## 9. 限制

1. 所有指标来自旧的 `development_test`（110 Tile / 25 Group），已被反复查看，
   **不是外部 OOD**；Gate 3 仍为 NOT_FROZEN，Gate 4 仍为 HOLD。
2. 物理 NLL 的边界点质量未建模，见第 6 节。
3. 确定性开关只在本 ROCm 版本上验证；若换环境需重新验证。
4. 本轮未在外部 heldout 上运行任何模型，也未改写任何冻结清单。
