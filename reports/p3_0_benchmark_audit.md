# P3.0 公开资产与权重审计

日期：2026-11-03。代码基线：`6dea815`。P30-00 审计完成；真实部分数据访问 PASS，预训练接入仍 BLOCKED。无正式 benchmark 成绩。

## 固定来源与许可证

| 来源 | 实际固定版本 | 许可与验证范围 |
|---|---|---|
| [Moonstone Git](https://github.com/ayushprd/Moonstone/tree/cbd4720653eb05afd975d2a36280858c5c56a275) | `cbd4720653eb05afd975d2a36280858c5c56a275` | MIT；读取 22 个源码/许可/配置文件，核对 Git blob 与 SHA256；未执行上游代码 |
| [Moonstone HF](https://huggingface.co/datasets/ayushprd/Moonstone/tree/53099efd0f2db81eee96b28eae87e1efdd281310) | `53099efd0f2db81eee96b28eae87e1efdd281310` | 数据卡声明 CC-BY-4.0；读取小文件、HDF5 全部元数据及 8 个 Train/Val patch |
| [NASA LFM Git](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/tree/d54c67aad513cb9daca444afa425cfb278e4fbf8) | `d54c67aad513cb9daca444afa425cfb278e4fbf8` | Apache-2.0；只读 README/LICENSE/pyproject 与 tree |
| [NASA LFM HF](https://huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model/tree/b657bdfe6fc4c0445d918f77a2d7b919805f3bce) | `b657bdfe6fc4c0445d918f77a2d7b919805f3bce` | 模型卡声明 Apache-2.0；读取配置与 checkpoint 前 16 字节，未载入权重 |

完整文件大小、URL、SHA256、Git blob 与远端 LFS 声明见 [证据目录](evidence/p30_20261103/)、[精确 allowlist](evidence/p30_20261103/download_allowlist.json)。完整文件未下载时，LFS hash 仅为远端声明，不是本地完整校验。

Moonstone 数据归作者发布，数据再分发/派生元数据按 CC-BY-4.0 署名；本仓库的 `metadata_rows.csv.gz` 是其元数据的派生表，加入官方纬度分带与 Tile 边界，没有重新生成真值。代码遵循独立 MIT 许可，见 [许可声明](../licenses/p30_moonstone_MIT.txt)。不能以代码许可替代数据许可。

## 本地环境与初次传输

在项目目录、GPU 数据根与用户 HF cache 的限定搜索中未找到现成 Moonstone/MG-MAE/NASA LFM 资产；这不是全机器不存在的证明。实际运行环境：Python 3.10.12、torch `2.5.1+rocm6.2`、HIP `6.2.41133-dd7f95766`、AMD Radeon RX 7900 XT；已有 h5py 3.16.0。没有安装依赖或替换 PyTorch。最终审计时可用磁盘 `515057623040` 字节（约 479.7 GiB）。

[传输账本](evidence/p30_20261103/transfer_ledger.json) 记录实际响应体 **39,016,735 字节（约 37.2 MiB）**，保守累计计数 **43,982,703 字节**，均低于 1 GiB。区别来自最初为并行源码审计保留 5 MiB，实际仅使用 276,912 字节。HTTP/TLS 头及网络开销未测量。`ByteBudget` 为进程内共享对象，跨脚本显式传递 `--prior-transfer-bytes`；它不提供持久化跨进程锁，不能把多个独立调用各自的 1 GiB 当作总预算。

HTTP Range 必须返回 206、精确 Content-Range 与一致长度/编码，否则读取响应体前终止。HDF5 metadata 读取为 333 次 Range、21,823,488 字节；8 patch 提取为 16 次 Range、16,777,216 字节。第一次容器请求超时，响应体新增 0 字节；转发既有 HTTPS proxy 后成功，未记录代理值。没有全量 snapshot、权重下载或 Test 像素请求。

## 真实像素可用，官方 loader 等价性未完成

`benchmark/lunar_patches_v4.h5` 实际大小 **74,449,119,347 字节**，头部通过 Range 读取，HDF5 含 `(16200,28,256,256)` float32 像素，按 `(1,28,256,256)` gzip 分块。读取路径为：固定 HF HDF5 → 严格 Range file → h5py → 原始 Train/Val patch 与标签 → 本地 `MoonstoneHDF5Pilot`。不是只有标签的空壳文件。

固定选择原始顺序最前 4 个有效 Train 与 4 个有效 Val：ID `0,1,2,3 / 20,21,22,23`。全部像素 hash 与 pilot 文件 hash 见 [提取审计](evidence/p30_20261103/pilot_extraction.json)。源数据与缓存留在本地，Git 只保存元数据、hash、代码和日志。

官方下游从全局 mmap 读像素；文件虽以 `.npy` 结尾，源码使用 headerless float32 memmap，不应直接 `np.load`。本轮未获取官方 mmap 的配对像素，因此接口明确为 **HDF5_DERIVED_PILOT**。HF 和 Git 的统计文件有 27/28 通道不同；radar 专门修复流水线与通用标准化的 raw-zero 策略不同。不能将 generic z-score 声称为已证明发布 mmap 的全部处理。

## 权重与输入阻塞

**MG-MAE：BLOCKED_PRETRAINED_WEIGHTS。** 固定完整 Git tree 未列出权重，GitHub releases/tags 为空；HF 作者模型列表为空，有限 Moonstone 模型搜索未核实官方 MG-MAE checkpoint。证据仅覆盖这些来源，不推断互联网上没有权重。README 的 `checkpoints/latest.pt` 示例不算可加载资产。没有从头预训练。

**NASA LFM：公开文件可达，接入 BLOCKED。** `backbone/checkpoint.pt` 为 **2,425,755,440 字节**；精确 16-byte Range 显示 ZIP 前缀，未做完整 hash/架构反序列化验证。单个 tokenizer 权重约 1.15GB，同样超过初次上限。代码声明 Python `>=3.11,<3.13`，现有环境为 3.10.12，需要另行隔离兼容性方案。

NASA 原生 NAC 约 1m、WAC 约 100m，需要对应光学几何/静态上下文；Moonstone 为 28 通道、约 237m/px。已读取 WAC_vis 配置含 415/566/604/**643**/689nm 五通道，而 Moonstone Hapke 为四通道且物理产品不同。缺失 643nm 和观测上下文不能任意补零冒充已测信息。尚无经验证 input mapping，不运行 NASA encoder 或 SomBench 扩展。

## 下一轮最小获取方案（未执行、未授权全量下载）

- HDF5 全量路径至少需 74,449,119,347 字节（约 69.34 GiB），外加小型 class map/stats。建议固定 URL/hash、直接 `.part` 下载后校验并原子改名，避免 HF cache 与工作文件双份；实现前不能保证峰值空间。它仍不能单独证明 mmap 等价性。
- 28 个全球 float32 mmap 共 118,908,518,400 字节（约 110.74 GiB），另加 packed masks、元数据；与 HDF5 同存已超过 180 GiB，不属于首轮下载范围。更小的固定 ID mmap Range 配对可作为下一步设计，需提前锁定数值容差、通道和 radar 规则。
- MG-MAE 需作者可核实权重；NASA 需独立 Python 兼容方案及真实模态/单位/尺度/上下文映射，再审阅权重预算。

现有免费 8 patch 足以进行有限普通 Age CNN 工程 smoke。正式 Age/Geology 矩阵、预训练收益、候选条件化收益和测试评分均未获得证据。旧 LROC Gate 3 NOT_FROZEN、Gate 4 HOLD；P2.2 STOP_OR_REFRAME/NOT ESTABLISHED 保持不变。
