# weights/ 权重清单（2026-08-01 整理）

> 部署用哪套、哪些已判死，以本表为准；分数出处：`deploy/README.md` 的 A/B 实测表、
> 各 `*_progress.json` 评分快照、git log 提交说明。部署默认基座/阈值的单一真源在
> `src/phasepicker/defaults.py`（当前 diting + P=0.2 / S=0.15）。
> 「留出集」= 对应训练集自留 holdout 的官方计分；「真题」= 去年两轮真题包 ab_compare/全量评分（满分 2 分/文件）。

| 文件 | 基座 / 训练数据 / 关键配置 | 留出集或真题分数 | 状态 |
|---|---|---|---|
| `phasenet_diting_weights.tar.gz` | seisbench 官方 diting 预训练缓存包（USTC 271 万条中国 DiTing；原生 50Hz、窗 60.02s），供 `deploy_api.sh`/`setup.sh` 离线恢复，零微调 | 真题第1轮 1000 条 **1.629** / 第2轮 915 条 **1.500**（阈值 0.3）；新默认阈值 0.2/0.15 下 **1.717 / 1.654** | **部署候选（当前基座）** |
| `phasenet_stead_weights.tar.gz` | seisbench 官方 stead 预训练缓存包（100Hz），同上离线恢复 | 真题第1轮前 200 条 1.496 | 留档（应急 fallback） |
| `phasenet_iquique_bootstrap_best.pt` / `_epoch5_best.pt` | **diting 基座**微调（50Hz 新管线）；Iquique 445MB 池，10/5 epochs（testttt1 仓库产出，2026-08-01 收编） | Iquique 自留出集 1.512→**1.643**(e10)/1.618(e5)；**真题全量倒退**：两轮合计 1.6264(e10)/1.6150(e5) vs diting 1.7019；扫遍 16 组阈值最高 1.894 仍 < diting 同 300 条 1.922 | 留档（**Iquique 域增益不迁移中国真题的第二次铁证**——微调只认中国域/官方样例数据） |
| `phasenet_iquique_full_best.pt` | stead 微调；Iquique 全量 13327 条，lr 3e-5 × 5 epochs，batch 16，按 key 85/15 切分 | 留出集 1.3113→**1.4285**（P 满分率 72→90%、S 64→69%）；真题第1轮前 200 条 1.678 | 留档（stead 系整线被 diting 1.899 压制） |
| `phasenet_iquique_full_sw8_best.pt` | 同上 + `--s-weight 8` 实验 | 留出集 1.4294（与不加权持平，S 满分率 69→68%） | **已判无效**（S 瓶颈不在 loss 权重，勿再试加权） |
| `phasenet_iquique_8k_best.pt` | stead 微调；Iquique 8000 条 | 留出集 1.2012→1.3273（P 满分率 55→70%，S 61→56% 退步） | 留档（被全量版取代） |
| `phasenet_iquique_refined_best.pt` | stead 微调；Iquique 1000 条（850 train / 150 holdout），commit edcf273 | 早期小样本，无可比留出分 | 留档（早期实验） |
| `phasenet_iquique_ft_best.pt` | stead 微调；Iquique 首轮 1000 条，commit 17ecd12 | 早期小样本，无可比留出分 | 留档（早期实验） |
| `phasenet_ethz_ft1_best.pt` | stead 微调；ETHZ（欧洲）临时子集（取数未沉淀成脚本） | 无留出/真题记录 | 留档（早期实验） |
| `phasenet_geonet_ft1_best.pt` | stead 微调；GeoNet（新西兰）FDSN 取数，10 epochs | 合成 5 条 2.0，无真题增益记录 | 留档（GeoNet 路线已归档） |
| `phasenet_geonet_quick_best.pt` | stead 微调；GeoNet 83 条快速验证，commit 758d84d | 无留出/真题记录 | 留档（GeoNet 路线已归档） |
| `phasenet_ft_diting_demo_best.pt` / `_last.pt` | stead 微调；DiTing demo 小样本（训崩修复后的链路验证品） | 合成 5 条 2.0 | 留档（链路验证品，非 diting 基座微调） |
| `official_r1_to_r2/`（t2/t3 joblib + manifest） | T2 震级 / T3 事件分类树模型 baseline（第1轮训练→第2轮留出，commit 4c5c394） | 见 manifest.json | 留档（去年 .an 工具链用，今年 API 赛制不涉及） |

补充说明：

- 同名 `*_progress.json` 是对应 `.pt` 的训练/评分快照（loss、合成分、留出分），与权重成对保留。
- Iquique 系结论（数据量是主瓶颈、S 加权无效、best 挑选 bug 修复史）见 git log
  `6c3ad57` / `5dde5a9` / `b67a404` 与 `训练与提分计划.md`。
- P2 微调产出的新权重（diting 基座）入库时：在本表加一行、附 progress.json，
  并先过 `scripts/ab_compare.py` 真题闸门再考虑部署。
