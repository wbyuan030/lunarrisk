# P2.1 — 旧 External Held-out 分组的保守冻结前预检（2026-10-22）

**状态：P0 WARNING / Gate 3 不可签署。** 此记录只依据仓库已提交的旧分组池和预注册 v1 选择表，不涉及模型评分，也没有修改任何候选 ID 或历史文件。

## 输入（只读）

- `protocols/heldout_v2_group_pool.csv` — 51 行 Group；其 `center_lat`、`center_lon` 均为各组成员中心的平均经纬度。
- `protocols/heldout_v2_external_protocol.json` — `PENDING_SIGNOFF`、`NOT_FROZEN`；选中 15 Group / 18 Scene。
- 代码 `scripts/check_group_pool_collision.py`（检测器）与 `scripts/audit_groups_p20.py`（原 nearest-edge 聚类）。

## 当前远端数据的非几何预检

对全部 `51×50/2=1,275` 个不同 Group 的中心点，在月球半径 `R=1737.4km` 的球面上计算
haversine 距离并按 `<10km` 筛选（经度反经线归一）：

- **17 对**跨 Group 中心距 <10km；
- **8 对**至少一侧属于当前 15 个拟选 Group；
- `SPARIM1` 和 `SPARIM2` 各自为单景、分在不同拟选 Group，
  中心均为 `(-51°,171°)`，中心距为 `0km`，并且原池为两者均记录了
  `min_candidate_distance_km=0.0`。

**结论口径：** 一组场景的平均中心接近**既不能证明 DTM 多边形重叠，也不能证明独立**；
然而“零距离且都拟入选”的案例必须通过真实 DTM polygon 严格排除。
不能用中心距预检的「无报警」代替地理独立性证明。当前检测器若报警应输出
`BLOCKED_NEEDS_FULL_PAIRWISE_POLYGON_AUDIT`，本次尚未在 Coding Agent 容器跑这一命令；
上述数字由读取远端池/协议后的独立同公式扫描得出。

## 导致问题的代码逻辑

`scripts/audit_groups_p20.py::cluster` 只合并 `nearest_candidate_DTM` 一条边，
不保证加入**所有**符合 `polygon_distance<10km` 的 pair；而最近邻是在一个比最终
69 可用景更大的候选集上预先计算的，过滤后可能丢失边。故仓库报告的 51 Group
尚不足以作为**经过两两证明的** 10km Single-linkage 分组数。

## 下一步验收

1. Coding Agent 使用原始 69 个可用 DTM 的真实 polygon，在一致月球 CRS 上**遍历 2,346 个 pair**并建立所有 `distance<10km` 的边；不得继续用最近邻边近似。
2. 输出有 SHA256 的 pair-edge CSV、分组全集、跨 Group 实际最小多边形距离；
   要特别给出 `SPARIM1/SPARIM2` 真正的多边形间距与是否合并。
3. 在原 v1 草案不变前提下新建 **v2** 候选池及选样协议，给出 v1→v2 组数/成员/哈希的差异；
   按 `sha256(group_id)` 和组内上限重新抽组。
4. 单测必须覆盖“不是任何点最近邻、但仍距另一场景 <10km”的桥接案例；
   验证 **所有不同 Group 之间的实际 polygon 距离 ≥10km**。
5. 全过程不读取模型评分，不运行新 OOD，不把 PENDING_SIGNOFF 自动改成 FROZEN。

参见 [`docs/P2_1_JOINT_RISK_RESEARCH.md`](../docs/P2_1_JOINT_RISK_RESEARCH.md) 的 B 工作流。
