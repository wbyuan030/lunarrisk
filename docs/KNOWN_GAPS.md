# 代码审阅与明确缺口

- `rocm_smoke.py`：仅测试小 CNN 在随机 Tensor 上的 HIP 训练，不能代表 LFM/BF16/LoRA/大模型性能。
- `hf_audit.py`：能列目录并按模式下载，不会检查逐条 Parquet 的 NAC/DTM join、NetCDF 是否完整、地理投影与 SHA256，下载实际仓库路径可能需适配。
- `make_slope_labels.py`：有 slope 网格计数和面积比例；**10 个阈值 + 区间端点实际生成 11 桶**（`bin_0` 到 `bin_10`），旧 Plan 的“10 桶”与代码不一致。脚本不验证单位、无 NAC–DTM crop/重投影、没有 source-level group，可能以文件 stem 作为不唯一 sample ID。
- `crater_pilot.py`：有真实 NAC `.npy` 的小 CNN 与 Brier/Coverage；使用 `__standard_` 文件名判断协议，需与实际 COCO metadata 核对；路径层级不一致时需显式指定 `--data`。不是 landing safety 指标。
- `optional_gpuflow/`：是已有调度 MVP，不具有已验证的 10GiB DiskGuard、可靠 Staging/checksum、四卡任务队列和回传归档；不作为今天的前置任务。
- 没有 NAC→3m 训练器；没有真正的组级地理切分、联合属性后验、AST 通用求值器；不预先声称实现。
- 未在本机实测真实 7900 XT 与数据，不要把这份文档的计划指标写成训练结果。
