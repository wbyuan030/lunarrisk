# P3.0 固定 ID mmap 配对探针

2026-11-03；起点 `4f3ce7e`，真实运行源码 `bfe37b3b8219749750a41c2a92a264674dc5854d`。固定行探针已完成，状态 **PARTIAL_ROW_MATCH_NOT_EQUIVALENCE**；28 通道的既定 HF 路径在探针像素上完全相同，不能称完整裁片等价性 PASS。下面方案在网络读取前已提交锁定。

## 预先固定的读取与判定

沿用已有 pilot ID `0,1,2,3 / 20,21,22,23`、全部 28 通道、HF revision `53099efd0f2db81eee96b28eae87e1efdd281310`。不新增样本、不读 Test、不训练模型。精确 URL、文件字节数、offset/length 见 [机器可读 dry-run allowlist](../protocols/p30_mmap_probe_plan.json)。

headerless C-order float32 网格 `(23040,46080)`，每 Tile 为 256px；offset 为 `((tile_row*256+y)*46080+tile_col*256)*4`。按已有上游写入/读取定义，在本机 little-endian float32 解释；文件大小必须严格等于网格字节数，不接受带 .npy header 的偏移。只合并相邻且已选中、同 split 的 Tile，不跨过未选列。

完整 8×28 裁片仅 58,720,256 字节，但需要 14,336 个严格单段 Range 请求。先对每个 crop 的固定行 `0,128,255` 做可证伪探针：**168 次请求、688,128 字节（672 KiB）**。请求必须 206 且 Content-Range/长度/编码精确匹配；不自动重试。任何部分读取失败保留已接收字节与局部结果。

沿用先前锁定的 `rtol=1e-5, atol=1e-6`；远端 mmap 为参照。非有限 mmap 值按官方 loader 归零；像素 presence 以原 mmap finite 且 nonzero 判断。像素容差允许的微小零附近差异仍必须通过 mask 精确比较，不能借容差抹平 presence 不同。完整 channel/group presence 未读取，不能由三行探针宣称验证通过。

HF 统计是当前 HDF5-derived pilot 的既定路径；Git 统计仅用于来源诊断，不能观察结果后换成更接近的处理并改称原协议 PASS。分别保留每行每通道误差、mask 差异、原始 Range cache 的 SHA256 与输入/源码 hash。

- 任一既定 HF 路径像素或 mask 不符：`HDF5_MMAP_EQUIVALENCE_FAILED`。
- 所有探针行相符：`PARTIAL_ROW_MATCH_NOT_EQUIVALENCE`，仍不等于完整裁片等价性 PASS。
- HTTP/文件读取失败：`BLOCKED_PARTIAL_DATA_ACCESS`，记录实际已接收字节。

## 初次预算延续

本轮承接此前保守累计 **43,982,703 字节**（实际已接收 39,016,735）；不是重新获得 1 GiB。真实完成 168 次请求、新增 **688,128 字节**；保守累计 **44,670,831**、实际累计 **39,704,863**，均在初次上限内。HTTP/TLS 开销未测量，未自动重试。

默认无网络；真实执行需 `--apply`。原始 probe cache 只在新本地输出目录保存，不提交 Git；公开归档只保存结果、静态来源 URL、字节计数和 hash。

## 既有 pilot 的补充质量审计

只读取已下载的 8 patch、没有追加网络传输：17/28 通道在该批全部为 nonfinite（roughness、4 thermal、8 M3、4 Hapke）；其余通道中 radar 部分覆盖、Clementine 少量缺失。LPGRS TiO2 有 446,749/524,288 个有限 raw zero，不能把它们自行改为 NoData。这个北极单类 batch 既不代表五类任务，也不代表全模态覆盖；此前 CNN smoke 的含义仅为工程链路。

## 已验证的离线实现

新增 `mmap_pair.py`、`p30_probe_mmap.py` 与 5 个 synthetic 测试：精确偏移/预算、相邻合并不越 gap、Test/错误ID/重复位置/文件大小/浮动来源拒绝、nonfinite/mask/容差差异。完整现有 ROCm 环境回归 **476 tests，475 PASS、1 旧 labels 可选检查 SKIP**；compileall exit 0。

## 真实配对结果

168 个精确 Range 均成功，共比较 **172,032 像素**。全部 28 通道使用既定 HF stats 时，探针行最大绝对误差 **0.0**、超容差像素 **0**、mask 差异 **0**。这包含 radar 的可见/缺失部分与 TiO2 的有限 raw zero；支持在这些位置保留现有 HF 处理，不需要因先前的来源不确定性擅改流水线。

Git stats 诊断在 11 个有数据的通道中产生超容差差异（其中 WAC morphology 只有 1 像素超容差，其余有些是全部有效像素）。例如 elevation 最大差 0.02782、slope 0.10124、Mini-RF CPR 1.74804、S1 3.68041、Clementine 5.56001。Git 路径没有替换既定 HF 路径。完整数值见 [逐通道 CSV](evidence/p30_mmap_20261103/channel_comparison.csv)。

17 个全缺失通道的零一致性只验证了该区域缺失处理，没有验证其有效数据归一化。三行覆盖不是完整裁片检查，完整 channel/group presence 也未读全；不将 `PARTIAL_ROW_MATCH_NOT_EQUIVALENCE` 升级为官方 loader PASS。也不能从这批北极样本推断全球或全部 radar 来源处理等价。

168 个本地原始 Range cache 的长度与 SHA256、运行源码与提交内容全部核对。公开证据：[summary](evidence/p30_mmap_20261103/summary.json)、[完整 report](evidence/p30_mmap_20261103/probe_report.json)、[预算账本](evidence/p30_mmap_20261103/transfer_ledger.json)、[raw quality](evidence/p30_mmap_20261103/pilot_raw_quality.json)、[测试与运行日志](evidence/p30_mmap_20261103/validation.json)。原始 cache 不入 Git。

实际命令参数为 `--prior-transfer-bytes 43982703 --apply`，其余固定输入与 [dry-run allowlist](../protocols/p30_mmap_probe_plan.json) 相同；从上述源码提交的全新快照执行。没有模型推理、Test 像素、全量数据/权重下载或环境修改。
