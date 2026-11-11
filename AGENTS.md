# 当前任务：P3.3 — 真实NASA Train2705配对GPU预检PASS，尚无训练或迁移分数（更新2026-11-06）
**2026-11-11 特征来源/头设计消融：** 同一预算（20epoch/patience5/AdamW1e-3/batch8/seeds42-44/Val选模）下三组特征×三种头×三seed共27格：NASA pretrained线性 **0.188510±0.008293**、MLP 0.190067、类平衡CE 0.196045；NASA random线性 0.072138、MLP 0.059356、平衡 0.072525；**Moonstone域内pooled线性 0.235176±0.004210、平衡CE 0.248381±0.009623**、MLP 0.225787。结论：**域内Moonstone特征仍优于跨域NASA特征**（约+0.047），上一轮+0.116只是相对随机初始化的增益；类平衡CE是最便宜的提升（NASA+0.0075/Moonstone+0.0132），加宽MLP无益。**尺寸：** NASA编码器85,348,608参数（fp32约341MB）；上游checkpoint 2,425,755,440字节（含被剪掉的生成解码器）；实际载入116键85,761,024元素；head 37,681（线性）/209,457（MLP）；Moonstone pooled 256维，head 12,593/78,385（其编码器checkpoint本地缺失）。**效率：** NASA抽特征12,580次前向287秒≈43.8次/秒（含HDF5解码），单头20epoch训练3.4-8.6秒，27格矩阵总耗时123.4秒。111项P33聚焦测试零跳过PASS。见[报告](reports/p3_3_feature_ablation.md)、[证据](reports/evidence/p33_20261111/feature_ablation_01/verification.json)、[预注册协议](protocols/p33_feature_ablation_plan.json)。Test未评分。

**2026-11-10 冻结编码器配对探针（首个正向迁移信号）：** 固定Train 5,135/Val 1,155、`moonstone_native256_v1`、官方NASA base编码器冻结、仅训一个49类线性头（输入为编码器512×768 token的均值池化，维度768）：pretrained Val full49 Macro-F1 **0.188510±0.008293**，同种子同初始化random **0.072138±0.001763**，逐seed差 **+0.116372±0.009017**（三seed样本std，非置信区间）；Val accuracy约0.443 vs 0.339。门禁：特征缓存跨进程重算最大绝对差1.47e-4（pretrained）/4.8e-7（random），低于预先设定的1e-3（TF32已关闭；不声称跨进程逐位一致）；两次独立头训练指标与预测hash完全一致；`head(mean(tokens))`与整模型forward差0.0。**边界：** 只用Val早停与选模（两臂同等乐观）、仅线性头、无编码器梯度/LoRA/微调、未评分Test；预训练数据与Val地理重叠不能排除，故这是“重叠下的冻结特征迁移信号”，非泛化或显著性结论。见[报告](reports/p3_3_frozen_probe.md)、[证据](reports/evidence/p33_20261110/frozen_probe_01/verification.json)、[预注册协议](protocols/p33_frozen_probe_plan.json)。107项P33聚焦测试零跳过PASS，compileall PASS；未重跑全套。


**2026-11-10 地形产品身份PASS（第二版口径）：** 固定Train 5,135/Val 1,155共6,290个裁片的高程/坡度与固定revision发布`pretraining/mmap`数组按发布z-score重新推导后**按float32位型逐位相同**（每通道412,221,440像素，exact/bitwise_exact/within全等于像素数，最大绝对差0.0，容差外0，原始非有限像素0）；两个打包mask分别下载并各自校验，全域0非零字节、0置位（声明无任何非有限地形像素）。历史metadata导出`3b3ed8…`本地已不存在，改用受guard的HDF5 metadata组重建并用16,200行与固定归档逐字段核对（ID/split/unit/row/col/中心经纬PASS）后才使用；6,290个选区tile唯一、⊂13,800 Train/Val tiles、与2,400 Test tiles交集为空。新增下载四个文件8,758,886,656字节、零重试；原始LOLA/SLDEM栅格、aligned TIFF字节比对及15步流水线重执行均未做；`derived/slope.tif`本身未发布。**测试口径：** 新增23项、P33聚焦101项零跳过PASS；全套727项720 PASS，余7项为NumPy2.4/pandas与私有环境缺`shapefile`的既有环境失败（未修）。见[报告](reports/p3_3_mmap_terrain_linkage.md)、[证据](reports/evidence/p33_20261110/mmap_linkage_01/verification.json)和[协议](protocols/p33_mmap_terrain_linkage_plan.json)（plan在字节下载之后、任何像素比对之前写成，已注明）。**Test字节口径：** 未选择/比对/评分任何Test tile（`test_tiles_compared=0`），但全文件SHA256与全域mask计数必然流过覆盖Test纬度的字节，不得写“未收到任何Test相关字节”。第一版较宽口径receipt已被[独立复核](reports/p3_3_mmap_linkage_review.md)与同日在已持有数组上的重执行(elevation/slope各412,221,440像素`bitwise_exact`全等于像素数)取代，全部保留在git历史。**这关闭“cohort是否等于发布产品”，不关闭上游原始栅格执行谱系；实际预算仍未冻结，训练/优化步/分数仍NOT_RUN，Test HOLD。** 下一步是科学决策（残余上游限制下的paired实验与冻结预算），需人工review后再动。

**2026-11-07 本地谱系及发布栅格头部：** 本地限定范围清点未找到原始/中间地形文件或实际构建日志，见[清点](reports/p3_3_local_terrain_lineage_inventory.md)。固定revision三个发布aligned TIFF头部均声明float32、NaN NoData及相同网格，但无法证明实际HDF5谱系。**范围偏差：** 三个各1MiB请求及此前USGS 1MiB请求均包含TIFF头后的压缩或原始栅格条带，可能涉及Test地理区域；未解码、分析或评分，已停止后续栅格请求，详见[记录](reports/p3_3_aligned_header_observations.md)。不要声称未收到Test相关字节；Test评分仍HOLD。有限哨兵、实际谱系及预算仍未建立。

**2026-11-07 源元数据后续：** 当前USGS LOLA TIFF有界头部显示int16、NoData −32768、scale0.5；固定SLDEM标签无显式NULL/MISSING/INVALID/SPECIAL字段，不能据此断言无特殊值。源码缩放将声明NoData写为NaN但未明确同步更新profile的nodata，重投影如何处理需结合实际中间文件和版本核实。原样−16384m不在已有cohort范围内，但不能排除插值/混合影响或证明历史来源身份。见[报告](reports/p3_3_terrain_source_metadata.md)及[证据](reports/evidence/p33_20261107/terrain_source_01/source_observations.json)。未重复扫描或预检，来源有效性及实际预算仍未锁定；该阶段提出的后续栅格头部/窗口请求已被上方停止指令取代，不得继续；当前仅审查已有文字记录、源码和独立文本元数据。

**2026-11-07 全cohort数值诊断：** 固定Train 5,135/Val 1,155全部原始高程及坡度像素有限，坡度在0–90度内；保留有限零值。认证历史资产和物理chunk，未读取Test或运行模型；四个探索候选哨兵无匹配并不证明其他有限哨兵不存在。源栅格实际执行谱系仍未独立复现，`finite_only_no_finite_sentinels`未获证实，实际预算未锁定。78项P33测试零跳过PASS。见[报告](reports/p3_3_cohort_terrain_diagnostic.md)和[验证](reports/evidence/p33_20261107/cohort_diagnostic_01/verification.json)。不要重复扫描，下一步核实源有效性后再考虑锁预算；训练须另行授权。

**2026-11-07 云端来源有效性跟进：** 已独立核对6,290行诊断记录及9个文件/源码hash，未重扫像素或重跑测试。精确SLDEM公开标签未声明缺测常数；固定上游缩放对独立mask未显式处理，scaled缓存复用缺少实际身份链。均是待核事项，不是已证实的数据错误。见[来源核验交接](reports/p3_3_terrain_source_validity_followup.md)：下一步先盘点已有源/中间产品headers、mask/scale/offset和构建复用记录，不下载原栅格、不再盲扫有限值。来源guard、实际预算未锁、训练另行授权及Test HOLD不变。

**最新真实NASA预检：** 用户明确授权执行固定GitHub源码、加载固定checkpoint并对Train2705进行一次配对前后向。正常NASA导入PASS；在私有Python3.12.9/torch2.12.1+rocm7.2/HIP7.2.53211、RX7900XT gfx1100、确定性算法和无网络命名空间下，`moonstone_native256_v1`真实双模态输入与random/pretrained两臂前后向均PASS；严格加载全部116个选定encoder张量/缓冲键、85,761,024元素，295个decoder或未选模态键按规则排除。两臂均(1,49)输出、37,681可训练头参数；设备峰值分别421,319,168/454,873,600字节，进程max RSS 3,324,128 KiB。首次臂含冷GPU初始化，不作性能比较。未做optimizer step、训练或得分；checkpoint/config/batch/provenance重哈希、原input-only prepared hash一致、来源checkout clean、runner preflight gate及事后GPU故障筛查PASS。见[实际报告](reports/p3_3_real_nasa_preflight.md)、[验证](reports/evidence/p33_20261106/nasa_real_01/verification.json)。该一次预检不重跑，不代表全cohort或迁移收益。下文较早的“NOT_RUN/阻塞”等保留为对应历史阶段，不覆盖这一最新结果。

**最新本地交付：** `f74af37`安全拉取后，现有私有runtime执行75项P33及完整701项测试，零跳过PASS，compileall/diff PASS。先查明旧LROC缓存测试缺DELTA产品页，补齐合成夹具并断言无urlopen调用后才运行完整回归；云端旧partial日志仍保留为INCOMPLETE。既有Train2705/target43原始裁片、batch和固定NASA配置SHA再次认证，新provenance显式选择native256，真实input-only CLI exit0/PASS；独立逐元素归一化及prepared hash完全相等。见[本地报告](reports/p3_3_native256_local_validation.md)、[22个源码hash及验证](reports/evidence/p33_20261105/native256_local_01/verification.json)。这只确认单裁片输入准备，`official_model_execution=NOT_RUN`，历史strict失败、原资产和P3.2结果不改写。

**最新云端交接：** 先读[`docs/P3_3_MOONSTONE_NATIVE256.md`](docs/P3_3_MOONSTONE_NATIVE256.md)和[`reports/p3_3_native256_policy.md`](reports/p3_3_native256_policy.md)。在任何迁移分数前新增独立`moonstone_native256_v1`协议：保留Moonstone原256网格、原NASA配置归一化和256/16模型位置几何，不复现NASA数据resize512/crop/flip，不声称60m物理尺度等价；默认strict路径仍拒绝原差异。必须显式选择协议、匹配固定配置SHA和新provenance，贯通preflight/dataset/model/manifestV2/run/replay。独立源码/diff审查通过；本轮75项P33中59PASS/16SKIP，compileall/diff检查PASS；完整套件因旧LROC网络请求被停止，只有部分日志，绝非全套PASS。本地旧690PASS不能延伸到新代码。

**最新runtime方案：** 见[`docs/P3_3_PRIVATE_RUNTIME_PLAN.md`](docs/P3_3_PRIVATE_RUNTIME_PLAN.md)、[`protocols/p33_private_runtime_plan.json`](protocols/p33_private_runtime_plan.json)和[研究报告](reports/p3_3_private_runtime_plan.md)。已只读核实27个固定源码文件及官方发行信息：独立Python3.12 + torch2.12.1/torchvision0.27.1的ROCm7.2轮子是候选，绝非已安装/已兼容环境。正常包初始化会导入训练/数据依赖，不能只补torch/einops。完整依赖解析/大小/hash仍缺；宿主版本及驱动release映射已补齐；候选ROCm7.2不在已安装driver6.2.x的官方支持行中，不能靠venv或容器绕过。

**宿主只读核实（历史记录）：** `7784886`后已确认Ubuntu22.04.5/kernel6.8.0-138/glibc2.35、已加载及磁盘amdgpu模块6.8.5和DKMS包`1:6.8.5.60204-2070768.22.04`；目标卷可用462,071,136,256字节。当时已知模块与包身份，但尚未查清驱动release映射；此缺口现已由下段官方包目录证据解决。不能把模块6.8.5当ROCm版本。见[实际宿主报告](reports/p3_3_host_runtime_audit.md)、[证据](reports/evidence/p33_20261105/host_runtime_01/host_audit.json)。未安装、下载权重或改驱动；75/701与input-only均未重跑。

**最新官方兼容结论：** `69763c5`之后只读核实AMD包目录，已安装版本对应driver6.2.4/family6.2.x；其官方支持行不包含候选ROCm7.2。这是支持矩阵不覆盖，不是实测不能运行。官方ROCm6.4/7.0索引最高torch2.9.1/2.10.0均低于NASA要求；NASA初始官方提交已要求torch≥2.12，回退祖先也无解。见[兼容路径报告](reports/p3_3_driver_runtime_compatibility.md)及[新证据](reports/evidence/p33_20261105/driver_support_01/public_evidence.json)。历史宿主审计保持原样，原“映射未知”现已解决为具体支持边界。

**最新CPU候选准备：** 已核实Python3.12的官方torch2.12.1+cpu/vision0.27.1+cpu元数据，以及28个直接候选约束和72个直接包间约束；28个直接轮子公布大小合计444,858,870字节，不是完整预算。OmegaConf依赖的已查ANTLR4.9.3没有兼容轮子，完整transitive/build闭包仍缺。见[CPU报告](reports/p3_3_cpu_feasibility.md)、[候选计划](protocols/p33_cpu_feasibility_plan.json)。新增独立离线metadata检查入口不导入torch/NASA、不联网，不等于运行兼容性PASS，也不批准新建/安装环境；模型与原守卫不改。新增18项离线测试PASS，独立审查及三文件compile检查通过；不延伸旧701证据。

**新增metadata工具本地验证：** `2f59986`后仅运行新增18项离线测试，零跳过PASS/0.425s，三文件编译和diff PASS；独立复核28直接约束、72直接包间约束、26+2轮子标签/Python及444,858,870字节公布subtotal，11上游证据hash/22既有源码hash一致。见[本地报告](reports/p3_3_metadata_local_validation.md)、[证据](reports/evidence/p33_20261105/metadata_local_01/verification.json)。仅新测试的CLI夹具读取metadata；未额外运行旧环境inventory，不代表候选runtime已建立，未重跑75/701或真实input-only。

**最新CPU闭包与容量研究：** `f3d82c9`后已合入本地18项验证，不再重跑。元数据候选图共128包（127轮子＋ANTLR源码例外），307有效边/2093 marker排除边（含获准源码检查新增的1条typing排除边）、全部28固定pin及extras已独立核验；127份轮子metadata哈希有索引核验。选定Transformers5.15.1避开最新版对hub/tokenizers的冲突，未放宽NASA约束。最高manylinux要求2.31；含构建工具去重后的公布归档量649,568,457字节。ZIP目录数据给出保留归档及两环境成员的约2.84GB计账subtotal，不是安装峰值/预算。见[闭包报告](reports/p3_3_cpu_dependency_closure.md)、[候选计划](protocols/p33_cpu_dependency_candidate.json)。所有ANTLR4.9版本都仅源码包；获准一次读取的117,034字节4.9.3归档已核哈希；78成员/68文件合计477,312字节，setup.py与官方tag字节一致、setup.cfg仅egg_info设置。未解包到目录、执行或安装；动态构建/产物仍未知，详见[源码证据](reports/evidence/p33_20261105/cpu_closure_01/antlr_source_inspection.json)。pip25.3/setuptools81/wheel0.45.1仅静态方案，未安装/构建，完整闭包及运行兼容性不能写PASS。

**最新驱动维护结果：** 用户授权“step by step地修”，实际完成driver30.10.2两包安装、双内核DKMS、initrd重建和有人值守重启。普通权限备份因`/boot/vmlinuz*`为root:root0600失败，改由用户交互sudo执行；自写校验器另有一次tar类型位/权限位误比较，均已修正并保留原失败记录。最终归档16,088项/13,628普通文件逐内容对live源核验，SHA256`0901f4def8ddd02cdbca1e012e785ff945caf9302d28270a5cd791af81c93f1c`，664,326,134字节。stale48维护记录先修为124/138；APT 2.4.15首次因`Pathname to install is not absolute`在解包前失败且包清单未变，随后用相同认证绝对路径本地包`dpkg --install`成功，仅变更两目标包，无新增/移除，audit空。138在包配置时、124另行编译；两内核均`6.14.14`且vermagic匹配。旧6.8.5悬空DKMS登记未删除，保留于root0700`/var/lib/p33-driver-recovery/`。两实际initrd解包核验：每内核8模块SHA/vermagic、660个现存声明固件、依赖选择和blacklist均PASS；8个包内缺失固件按实卡GC11.0.0/11.0.1、非SRIOV、默认PSP/内存发现和无启动覆盖核实为不适用路径。重启后`6.8.0-138-generic`加载amdgpu6.14.14，两GPU(03:00.0 7900XT、c9:00.0核显)均绑定amdgpu，rocminfo枚举gfx1100/gfx1103，无XID/reset/call trace。旧torch2.5.1+rocm6.2有限FP32 GPU matmul精确PASS。见[报告](reports/p3_3_driver_maintenance.md)、[证据](reports/evidence/p33_20261105/driver_maintenance_01/verification.json)。仍是：备份未做实际恢复启动；SMU版本提示、DCN314 REG_WAIT超时和可选ucode缺失已记录但非稳定性证明。

**ROCm7.2最新进展：** 用户要求继续，允许必要时使用Docker，并授权上述具名PyPI工具的隔离安装、构建以及NASA模型下载。已创建私有Python3.12.9运行/构建环境，完整固定NASA源码检出并核验27个已审文件；125个非框架候选轮子（453,494,002字节）、官方ROCm torchvision（3,816,468字节）及triton-rocm3.7.1（311,741,985字节）均直连下载并验哈希。torch2.12.1+rocm7.2轮子6,177,014,207字节，下载在该阶段仍在后台进行；有限范围元数据表明需要triton-rocm3.7.1，原307个活跃候选约束替换后仍满足，但完整wheel与最终安装锁未确认。真实验收仍未运行。独立构建工具及ANTLR源码（3,030,171字节）验哈希后，自动审批两次拦截安装，第二次发生在用户已回复“我授权”之后，理由仍为需具名外部来源授权；不能绕过。用户授权的固定NASA checkpoint直连huggingface.co三次超时，未下载；尝试同一官方域名的显式IP直连诊断亦被自动审批拦截，不得绕过。见[进度报告](reports/p3_3_rocm72_provisioning.md)及[证据](reports/evidence/p33_20261106/rocm72_01/verification.json)。上述下载中的状态是报告生成时快照，继续前查本地receipt与进程，别重复下载。

**最新离线构建：** 用户随后明确授权从`https://files.pythonhosted.org`安装pip25.3、setuptools81.0.0、wheel0.45.1并隔离构建ANTLR4.9.3；先前自动审批拦截已解决。三个验哈希工具已安装到独立build_venv，ANTLR使用`unshare --user --map-root-user --net`无网络构建成功；生成wheel 144,590字节，SHA256`ed1325d05a227d3a21d12795e9faa737ddc4da022d19ea76caf9173a33e883a1`，元数据及ZIP完整性PASS。125个非框架轮子的内部metadata与已审官方哈希、Python3.12/Linux标签均核验通过。候选清单共129轮子、总6,946,211,252字节；128轮子已验证，torch完整轮子仍待下载完成/核验，尚非最终锁或目标运行时PASS。见[构建证据](reports/evidence/p33_20261106/rocm72_build_01/verification.json)。

**NASA checkpoint最新进展：** 用户建议`hf-fast`类方案后，从第三方AI FastHub镜像无本地代理仅获取固定revision的backbone checkpoint。实际大小2,425,755,440字节和SHA256`5843824dfa56f89552b37b5295aa865ed28777c5f2b33286db2ef81b4cf08e5b`与已有官方LFS记录完全一致；仅静态读取ZIP目录395成员、无重复或不安全路径，未反序列化。见[checkpoint证据](reports/evidence/p33_20261106/rocm72_checkpoint_01/verification.json)。官方Hugging Face直接超时及显式IP诊断被拦截为历史事实，镜像获取未使用该诊断；模型加载/执行仍NOT_RUN。

**最新GPU依赖锁：** 完整torch2.12.1+rocm7.2轮子6,177,014,207字节已取得，官方SHA256 `7a349a5475607ae64d284db6733a2a0255cad9f3e40f31f6bbf02b69946928a2`及整轮内部metadata与之前范围读取一致；首个下载尝试收到4,762,733,516字节后超时，自动重试成功，实际总流量未计量。129个轮子全部核哈希，在`unshare --net`内以`uv --offline --no-index --no-build --generate-hashes`解析成功；对全部129项名称、版本与hash独立对照，且同条件dry-run计划安装129项。ZIP目录声明解压数据16,727,655,660字节，非实测安装峰值。真实安装命令被自动审批拒绝，理由是用户具名授权仅涵盖三个构建工具，额外依赖必须明确点名来源；目标venv仍为空。见[锁与来源清单](reports/evidence/p33_20261106/rocm72_lock_01/verification.json)。

**最新私有GPU runtime：** 用户随后明确点名授权`https://files.pythonhosted.org`、`https://download-r2.pytorch.org`及本地ANTLR构建轮子，129项依赖已按锁文件在无网络隔离的私有Python3.12.9环境离线安装，`uv pip check`及全部129版本/锁对照PASS。实测私有venv占用16,857,640,960字节，剩余435,139,547,136字节。新torch2.12.1+rocm7.2/torchvision0.27.1+rocm7.2/HIP7.2.53211在RX7900XT gfx1100上进行开启确定性算法的64×64 FP32矩阵前向和反向，数值/梯度精确PASS，峰值设备内存67,191,808字节；所观察GPU库均来自该私有环境，未混入旧`/opt/rocm`，日志筛查无新GPU fault。此为微型runtime检查，非NASA模型执行，也不代表AMD已认证此torch版本。尝试准备正常导入固定NASA源码的脚本时，自动审批拦截了脚本创建，要求明确授权执行`https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model`的外部代码；脚本未创建，NASA导入/构造/预训练权重加载均NOT_RUN。见[运行时证据](reports/evidence/p33_20261106/rocm72_runtime_01/verification.json)。

**本地下一步：** 真实配对预检已完成并通过现有runner preflight gate，不重复。下一阶段先核实全固定Train/Val cohort的有效性及有限sentinel/来源证据，并在任何task分数前锁定独立实际配对预算；现有example仅示例，尚非实际预算。若需改变cohort、缺失值或空间策略，另立协议后执行。物理尺度差异、原strict守卫及旧Python/torch/ROCm不变；真实训练、完整回放、迁移收益仍未建立，Test HOLD。

**首先阅读 [`reports/p3_3_local_readiness.md`](reports/p3_3_local_readiness.md)、[`docs/P3_3_NASA_TRANSFER_HANDOFF.md`](docs/P3_3_NASA_TRANSFER_HANDOFF.md) 和 [`protocols/p33_nasa_transfer_plan.json`](protocols/p33_nasa_transfer_plan.json)。** 上游`8eeab47`已安全拉取，并合并并发`ce8b1ef`的固定根Train身份核验；此前pre-runner本地新增28项及完整654项测试零跳过PASS，compileall PASS。身份/标签认证通过后，真实裁片仍在空间策略处被拒绝；[最新合并验证](reports/evidence/p33_20261105/local_02/verification.json)。固定NASA配置已从既有缓存复核大小/SHA256/Git blob；直连SSH取得两个固定revision的10个所需源码文件并审计，未执行上游Python。原始Train2705裁片已核验，但真实输入CLI因官方`pre_resize512/input_size256`被当前空间策略守卫拒绝，exit1及日志保留，没有成功receipt。

**并行云端更新：已实现 [`scripts/p33_train_pair.py`](scripts/p33_train_pair.py) 的 lock/run/replay 配对训练入口，见 [`docs/P3_3_NASA_PAIRED_RUNNER.md`](docs/P3_3_NASA_PAIRED_RUNNER.md)。** 原始地形直接经认证prefix读取，固定Train/Val，显式预算/源码/资产/runtime锁定，双臂选模和独立进程回放。云端48项P33测试通过、16项因torch/h5py缺失跳过；这不延伸上述本地654项证据到新runner。未执行官方训练或冻结实际预算。保留本地所有审计、pre_resize512/input_size256和runtime阻碍，不静默放宽守卫；自动resume尚未实现。

**新增runner本地验证：** `423005e`安全拉取后，在现有私有runtime执行全部64项P33及完整690项测试，零跳过PASS，compileall PASS；[实际报告](reports/p3_3_runner_local_validation.md)、[日志/18个源码hash](reports/evidence/p33_20261105/runner_local_01/verification.json)。云端16项跳过已在本地执行，但torch训练/重建与HDF5仍是小型合成夹具，mock输出的LOCKED不是实际预算。没有运行官方NASA模型、完整真实cohort扫描、GPU训练或正式独立进程回放；下述空间/runtime阻塞保持，科学源码、旧证据/预算均未改。

## P3.3下一项与真实边界

- 原主机/ROCm容器的Python3.10.12、torch2.5.1保持不变；另立私有Python3.12.9/torch2.12.1+rocm7.2环境完成官方模型单裁片GPU前后向，但并非AMD对该框架版本的认证或完整训练稳定性验证。
- 已显式声明Moonstone-native256配对空间策略，单个真实Train裁片输入及真实模型配对前后向PASS。官方dtm/slope的Untok数据transform先resize512、再归一化及crop/flip；native256独立下游协议保留原网格，backbone接收张量。默认strict仍拒绝旧差异，不得修改YAML或静默跳过策略检查。
- 已核实源码声明的高程米/坡度度、58–62°SLDEM/LOLA混合、NaN处理与HDF5原通道传递。原始栅格header、完整执行谱系、逐样本fallback未独立复现。236.901m是名义南北/赤道间距，东西地面间距随纬度变化；presence不是像素mask，finite zero保留。
- 固定checkpoint已核验并严格加载，真实Train2705配对forward/backward已PASS；使用runner在任何task分数前仍需扫描固定Train/Val cohort并锁定实际预算。若需改变cohort或空间策略，另立明确协议后再执行。官方训练、全cohort结果和迁移收益仍NOT_RUN/NOT_ESTABLISHED；不把合成wrapper测试当模型执行。
- 保留P3.2负结果与18run，不重跑。完整地质语义来源核实仍是独立后续项，不与NASA预训练迁移混写。只用Train/Val，Test HOLD；必要数据/权重直连授权及下文其他资源边界不变。

## P3.2已完成阶段与独立语义后续项

**阅读 [`reports/p3_2_progress.md`](reports/p3_2_progress.md)、[`reports/p3_2_lunarjev_mvp.md`](reports/p3_2_lunarjev_mvp.md) 和 [`reports/p3_2_decision.md`](reports/p3_2_decision.md)。** 原设计 [`docs/P3_2_LUNARJEV_MVP.md`](docs/P3_2_LUNARJEV_MVP.md)、[`protocols/p32_lunarjev_design.json`](protocols/p32_lunarjev_design.json) 保留为实施前交接；其未实施措辞不代表最新状态。实际运行清单是 [`protocols/p32_head_run_manifest_v2.json`](protocols/p32_head_run_manifest_v2.json)。P3.1入口可在 [edffc0a](https://github.com/wbyuan030/lunarrisk/blob/edffc0a7c274efb4c1cee03e6c0f3203d87ebba7/AGENTS.md) 追溯。

## 本轮已完成，不能反复重跑

- P32-01：三个原Geology最佳checkpoint/receipt核验，冻结参数和BN，Train5,135/Val1,155各导出16×256 FP32 token及同token均值pooled控制。实际每cache110,118,080字节，留在`outputs/p32/context_03/`；[缓存证据](reports/p3_2_context_cache.md)。
- P32-02：统一公共49+5候选表、120维catalog-only TF-IDF，明确`TAXONOMY_ONLY_LOCAL`。只有官方符号和Age映射，含身份性质literal特征，不冒充完整地质语言定义或预训练模型。逐样本真实unit name/age prefix绝不作为输入。
- P32-03/04：B0/B1/B2/J四头×seeds42/43/44及固定描述置换B2/J×三seed，共18run完成，全部独立进程精确回放六份预测CSV。旧Geology encoder冻结、所有头同cache/ID/任务监督/预算，未改P3.1设置。
- P32-05：J1/J2/J3已如实报告。Val full49 Macro-F1均值±样本std：**B0 .255488±.014813，B1 .212279±.009432，B2 .221628±.014017，J .202774±.020098**。J−B2 −.018854±.029749；正确描述相对固定乱序也无增益。共享上下文工程可行，语义/空间收益未建立。
- 验证：**626项测试零跳过PASS**、compileall PASS、真实GPU有限前后向PASS；最终162个本地/180个归档hash、3个cache hash、108份逐行预测ID/target/概率/指标/选模/停止/聚合与精确回放PASS。[最终核验](reports/evidence/p32_20261105/runs/sequence_verification.json)。首次冷GPU统计初始化和旧schema适配失败保留，不是额外超参试验。

原六组P3.1普通scratch结果和所有源码/预算/evidence保留。完整上游文件hash未核实；Val已反复使用，仅为探索性北半球开发轨。没有独立确认性Test、全月、新类或着陆安全结论。

## 下一项限定工作

1. **核实可溯源完整地质类定义及一个冻结文本表示。** 优先现有公开资产；缺项明确，不编造岩性/年代/风险描述。已有本地文本权重不等于已核验可用，核对模型卡/许可/revision/权重hash/runtime。缺口只阻塞对应完整语义科学主张，不回到反复smoke。
2. **另立完整语义tier和实际manifest再实验。** 当前lexical负结果保留；沿用同特征B0/B1/B2/J、普通masked-softmax、同描述pooled与Geology概率→Age、固定描述置换。所有样本共享公共描述，语义头不加绕过描述的可学习class-ID embedding/class-specific bias。
3. 不为“救”当前J延长预算、挑encoder/seed、事后选prompt、扩大模型或重复旧18run。少数类阈值已由Train≤50预先定义。n=3样本std不能写成置信区间；候选排列/子集恒等式不是语义/新类能力证据。
4. 后续实现用`src/lunarrisk/lunarjev/`、`scripts/p32_*`及独立`outputs/p32/`命名空间。尚未实现、核验或运行的项目不写PASS；更新进度、真实命令和源码/data/cache/text hash。

## 最新授权与资源边界

**用户最新明确指示：拉远端继续推进，若需模型权重或更多数据，不走代理即可下载。** 此指示更新旧的全面暂停边界。必要获取固定来源/revision/大小/hash，显式禁用代理，记录实际直连与许可；不反复索取已获同类授权。当前首轮既有数据足够，未新增下载、未恢复完整HDF5。现保留33,964,443,553字节，尚缺40,484,675,794字节；不做无必要40GB传输或全snapshot。

仅本地RX7900XT/现有ROCm；不替换torch或共享环境、不买云、不启动四卡；共享四卡机器个人存储≤10GiB约束不变。数据、cache、checkpoint、权重、凭据和私有路径不入Git。只用Train/Val，公开Test cache/评分HOLD。

MG-MAE权重未在审计来源核实，不从头预训练冒充；NASA LFM物理模态/尺度/几何/Python兼容性另验，不伪造通道。当前不恢复Qwen/完整OnePass联训、TiM、主动路由、全球预训练或低秩后验扩容。冻结文本表示核实与LLM联训是不同范围。

P2.2负结果和P3.0/P3.1全部旧资产不改写，H1/H2增益仍NOT_ESTABLISHED。旧LROC Gate3 NOT_FROZEN/Gate4 HOLD，不批准EDR/CDR变更、不覆盖heldout/provenance或原始NAC/DTM/CDR、旧labels/cache。

## 验证与提交

云端完整套件曾在legacy LROC网络请求处被停止，历史只有部分日志。本地查明cached-only测试缺少DELTA产品页，补齐合成缓存并断言无网络调用，随后完整701项零跳过PASS；这不改写云端历史失败。后续新增P33优先用`python3 -m unittest discover -s tests -p 'test_p33_nasa*.py' -v`验证；新增网络用例仍须先核实不会触发禁止的数据获取。

```bash
git pull --ff-only origin main
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts src tests
```

需要时沿用已验收私有runtime/现有ROCm，不安装替换共享包。626项是本轮实现证据，不自动涵盖未来代码。每阶段完成后保存PASS/FAIL/BLOCKED、实际资源、checkpoint和逐样本回放，按习惯直接push main，不force-push、不覆盖并发提交。超出下载/付费/环境边界时准确报告，不隐式扩大权限。
