# P1.8 — `heldout_v2` 冻结协议与结果

日期：2026-10-17；代码状态 `a53937e`。
相关脚本：`scripts/freeze_heldout.py`；测试：`tests/test_freeze_heldout.py`（9 项，纯离线）。
产物：`protocols/heldout_v2_protocol.json`（严格模式：BLOCKED）、
`protocols/heldout_v2_product_level_{protocol.json,manifest.csv}`（产品级候选）。

## 1. 结论先行

| 定义 | 状态 | 原因 |
|---|---|---|
| 严格：组级 / 产品级 / 空间独立 | **`BLOCKED_NEW_HELDOUT`** | 100k release 的 **272 个泄漏 group 全部**已被 760 Tile 覆盖；剩余 754 个未用产品全部落在已用 group 内 |
| 放宽：产品级 + 空间独立（组可复用） | `FROZEN_PRODUCT_LEVEL_NOT_GROUP_DISJOINT`，64 Tile | 未用产品 22 个，距所有已用 Tile ≥ 10.078 km，但仅来自 4 个已用 group |

**结论：在本 100k 语料内无法构造组级独立的 held-out 集。** 任何用这批 Tile 得到的"独立泛化"结论都
只能表述为"samples/products 未见过"，不能表述为"地图/源组未见"。要获得真正的组级独立 V2，
必须引入本语料之外的 NAC 立体产品（或明确接受产品级迁移声明）。

## 2. 证据

```
manifest groups        272
covered by the 760     272   (unused groups = 0)
manifest products      1090
unused products        754
test-split tiles with unused product & >=10 km   2505  (4 groups, all used)
```

覆盖来源：`scripts/hydrate_tiles.py` 的组内轮转选样在 400/120/120/120 的配额下恰好遍历了每个 split 的全部 group
（train 195、cal 26、val 26、test 25 = 272）。

## 3. 严格模式（BLOCKED）算法与产物

```
candidates = manifest[split_final == "test" & has_slope3m]
           − used sample_id − used NAC_EDR_ID − used group_id
           − (center distance < 10 km from any used tile)
```
候选数 = 0 → 不冻结任何 Tile，写入 `protocols/heldout_v2_protocol.json`（status `BLOCKED_NEW_HELDOUT`，
coverage 与 next_options 一并记录），进程退出码 2。

## 4. 产品级候选（已冻结，带告警）

| 项 | 值 |
|---|---|
| 状态 | `FROZEN_PRODUCT_LEVEL_NOT_GROUP_DISJOINT` |
| Tile 数 | 64（请求 120，受 group/间距约束） |
| NAC 产品 | 22 个（全部未用过） |
| 泄漏 group | 4 个（**均为已用 group**） |
| LTM zone | 4 个（LTM_44N 46、LTM_36N 16、LPS_S 1、LTM_27S 1） |
| 空间缓冲 | 距已用 Tile 最近 10.078 km（最远 51.133 km） |
| 内部间距 | V2 内部同样要求 ≥ 10 km，避免近重复 |
| 与已用数据重叠 | sample_id 0、产品 0、group 4 |
| Manifest SHA256 | `33f0898fca6b4a6c95cf5bda7a19c8181674723f386bae88fd2366184963ed97` |

选样算法：按 LTM zone 轮转 + seeded shuffle + 每产品最多 5 个 Tile + V2 内部 ≥10 km 间距，
全部来自 `split_final == "test"`（后续 train 扩容只从 `split_final == "train"` 抽，不会污染该集）。
冻结前**未下载任何 payload**，**未计算任何模型指标**。

## 5. 该候选集能/不能支持什么

- 能：检测"新 Tile、新 NAC 产品"上的误差/Brier 退化；用于发现开发集过拟合。
- 不能：声称跨地图/跨源组的 OOD 泛化；4 个已用 group 意味着空间/源组相关性无法排除。
- `DTM_source_ID` 仍为 UNKNOWN：3 m DTM 的立体像对来源无法核验，**不能**声称 DTM 源级独立。
- footprint 级投影审计仍未完成（`GEO_JOIN_PENDING`）。

## 6. 决策点（需要用户/协议方选择）

1. **引入语料外数据**：从其它公开 NAC 立体 DTM 产品（或官方 1M release 中不同的 NAC 产品，
   若确认其 group 未覆盖）构造真正组级独立的 V2。这是唯一能满足严格定义的方式。
2. **接受产品级迁移声明**：以上述 64 Tile 为 `heldout_v2_product_level`，在论文/报告中把结论限定为
   "unseen samples and unused NAC products, within the same leakage groups"。
3. **暂不评估**：保持 P18-07 未执行，直到 1 或 2 被明确。

无论选哪条，**冻结的 ID 清单与 SHA256 不得因后续表现而更换**；旧 110 Tile 继续作为 `development_test`。
