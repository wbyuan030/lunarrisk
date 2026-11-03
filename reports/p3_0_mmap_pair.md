# P3.0 固定 ID mmap 配对探针

2026-11-03；起点 `4f3ce7e`。本文件先记录读取前锁定的方案；此提交尚未执行真实 mmap 请求，不报告配对结果。

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

本轮承接此前保守累计 **43,982,703 字节**（实际已接收 39,016,735）；不是重新获得 1 GiB。计划成功后保守累计应为 44,670,831、实际累计应为 39,704,863，均在初次上限内；这些是计划值，运行前不冒充实测。

默认无网络；真实执行需 `--apply`。原始 probe cache 只在新本地输出目录保存，不提交 Git；公开归档只保存结果、静态来源 URL、字节计数和 hash。

## 已验证的离线实现

新增 `mmap_pair.py`、`p30_probe_mmap.py` 与 5 个 synthetic 测试：精确偏移/预算、相邻合并不越 gap、Test/错误ID/重复位置/文件大小/浮动来源拒绝、nonfinite/mask/容差差异。完整现有 ROCm 环境回归 **476 tests，475 PASS、1 旧 labels 可选检查 SKIP**；compileall exit 0。真实 mmap I/O 尚未在本提交执行。
