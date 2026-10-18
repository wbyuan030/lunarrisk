# P1.9：用 LROC 官方观测索引解析 EDR 文件 URL（2026-10-18）

**状态：已提交离线解析器；真实 IMG 目前仍没有在线文件级验证。** 本方案不会修改 168 条立体观测请求、现有冻结名单、尺度政策 A，亦不运行 Gate 4。

## 发现的正式档案

LROC 官方的 NAC 分期观测 Shapefile 不只是几何信息，还直接携带四个重要字段：
**PRODUCT_ID**（带 LE/RE 的相机影像产品号）、**PDS_VOLUME**（卷号）、**FILE_SPECI**（卷内真实相对路径）和 **URL**（发布时指向 IMG 的链接）。

与先前暴力猜 PDS 卷号和日期目录相比，这是一条更可靠的**路径发现路线**。需要注意：URL 可能指向已经迁移的老主机，字段存在不等于今天还能下载。

以下官方页面均明确给出字段解释、ZIP/CSV 下载入口：
- [NAC EQ NOMINAL](https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_NAC_EQ_NOMINAL_MISSION)：2009-09 至 2010-09，纬度 ±60°
- [NAC NP NOMINAL](https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_NAC_NP_NOMINAL_MISSION)：60–90°N
- [NAC SP NOMINAL](https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_NAC_SP_NOMINAL_MISSION)：60–90°S
- [NAC EQ SCIENCE](https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_NAC_EQ_SCIENCE_MISSION)：2010-09 至 2012-09
- [全部 NAC 分期观测索引](https://data.lroc.im-ldi.com/lroc/rdr_product_select?commit=Search&filter%5Beast%5D=&filter%5Blat%5D=&filter%5Blon%5D=&filter%5Brad%5D=&filter%5Bsouth%5D=&filter%5Btext%5D=&filter%5Btopographic%5D=either&filter%5Bwest%5D=&page=41&per_page=100&show_thumbs=1&sort=time_reverse)：Commissioning / Nominal / Science / Extended × EQ/NP/SP
- [NAC EQ NOMINAL 官方 PDS 目录](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/SHAPEFILE/NAC_EQ_NOMINAL_MISSION/)：ZIP 约 22 MB、CSV 约 99 MB，优先 ZIP。

本次新增 scripts/resolve_lroc_edr_urls.py：
读取官方 ZIP 内 DBF、单独 DBF 或 CSV，根据 outputs/p19_edr_acquisition_list.csv 的 edr_id 列自动展开 LE/RE，抽取上面四个字段。
重名路径有歧义则拒绝猜测，可选少量 HTTP HEAD 验证而不下载大影像。
输出 outputs/p19_edr_resolved_urls.csv 和同名 summary JSON。
注意 168 个 acquisition IDs 最多对应 336 个相机 EDR 文件。

## 第一轮 Pilot — 先取小型索引，避免直接取整批影像

~~~bash
git pull --ff-only origin main
mkdir -p data/lroc_edr_index
ROOT='https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/SHAPEFILE'
curl -fL --retry 2 --connect-timeout 15 \
  "$ROOT/NAC_EQ_NOMINAL_MISSION/NAC_EQ_NOMINAL_MISSION_360.ZIP" \
  -o data/lroc_edr_index/NAC_EQ_NOMINAL_MISSION_360.ZIP
python3 -m zipfile -t data/lroc_edr_index/NAC_EQ_NOMINAL_MISSION_360.ZIP
python3 scripts/resolve_lroc_edr_urls.py \
  --requests outputs/p19_edr_acquisition_list.csv \
  --index data/lroc_edr_index/NAC_EQ_NOMINAL_MISSION_360.ZIP \
  --out outputs/p19_edr_nominal_pilot.csv \
  --verify-head --max-head 5
~~~

此索引只覆盖 NOMINAL 阶段，后期产品出现在 SCIENCE/EXTENDED、极区 NP/SP 等索引中；
对这些产品显示 NOT_IN_INDEX 是预期行为。可重复使用 --index 传入多个 ZIP/CSV，不必解压 ZIP。

一条可以用于交叉核对路径结构的历史样例（**不代表今日 HTTP 可达，也未必属于这 168 个 ID**）：
LROLRC_0005/DATA/SCI/2010267/NAC/M139939938LE.IMG。
出处：[NASA Ames Stereo Pipeline 的 LRO NAC SfS 文档](https://stereopipeline.readthedocs.io/en/stable/sfs_usage.html)。

结果 status 可能为：NOT_IN_INDEX、UNVERIFIED_URL、AMBIGUOUS_INDEX、HTML_OR_BROWSER_REDIRECT、HTTP_404、HEAD_BINARY_CANDIDATE。
HEAD_BINARY_CANDIDATE 仅表示服务器 HEAD 表现像二进制文件，之后还需文件小范围 Range/IMG+LBL 验证。

若下载元数据 URL 重定向，先记录 curl -IL 的最终 Location 和主机；不可把返回的 HTML 当作索引。
若所有旧 URL 均不通，利用 PDS_VOLUME + FILE_SPECI 便可精确知道实际文件所在卷内路径，
后续只需调查卷根目录的映射，而不必猜日期/目录。

## 同步修复 ISIS 命令名

原 docs/P1_9_EDR_ORTHO_RUNBOOK.md 错写为 lrolc2isis；
LROC NAC 正确应用是 **lronac2isis**。
典型流程：lronac2isis → spiceinit → lronaccal → lronacecho → mapprojection。
[官方 Ames Stereo Pipeline 教程](https://stereopipeline.readthedocs.io/en/stable/examples/lronac.html)。

## 验收和研究纪律

1. 取得至少一条属于项目 168 个 acquisition 的卷号、FILE_SPECI 与官方 URL（官方索引匹配）。
2. 实测至少一条对应 IMG 的文件级访问（不是 HTML/JS），连同 LBL 证据与 SHA256；先做一景，不批量下载。
3. 只有影像和 DTM 配准、物理 QA、既有 ≥10 km / 源级独立审计都通过，才进入正式 heldout 冻结。
4. 不挪用已经多次看过的 110 Tile，也不把 NAC 缩略图或 DTM 派生阴影充当正式光学输入。

**核查范围：** 新脚本在助手离线容器通过 5 项 synthetic 单测和编译检查；
本容器对 PDS 域名无法 DNS 解析，因此**尚未直接验证真实 IMG 内容，也不能宣布 Gate 2 解锁**。
