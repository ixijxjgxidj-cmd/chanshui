# 第 9 轮论文证据：软标签锐化、召回天花板与赛区域迁移

## 0. 检索与核验通道声明（如实标注）

**本轮 anysearch 与 Playwright MCP 均不可用**（`list_mcp_resources` 返回空，无任何 MCP 资源）。
因此本轮改用 HTTP 直连通道，必须按其真实身份记录，不得声称使用了未调用的工具：

| 通道 | 用途 | 脚本 |
|---|---|---|
| arXiv API `export.arxiv.org/api/query` | 关键词检索 | `scripts/lit_search.py:search_arxiv` |
| Crossref `api.crossref.org/works` | 关键词检索 + **书目级核验** | `search_crossref` / `verify_doi_metadata` |
| OpenAlex `api.openalex.org/works` | 关键词检索 | `search_openalex` |
| Semantic Scholar Graph API | 关键词检索 | `search_s2` |
| 直接 GET arXiv `/abs/` 与出版商 DOI 落地页 | **原文页核验** | `verify_record` |

**检索日期**：2026-08-13。**9 条查询轴**（`memory/papers/_raw/round09_q_*.json`）：
`sigma`、`sigma2`、`label`、`heatmap`、`pose`、`kd_jepa`、`domain`、`recall`、`unc`。
去重后 900 条记录，主题打分后候选 134 条（`round09_scored.json`），策展 25 篇。

**核验结果分层**（`round09_verified.json` + `round09_crossref.json`）：

- **14 篇原文页核验通过**：arXiv `/abs/` 页面标题 token 命中率 1.0。
- **11 篇原文页抓取失败但 Crossref 书目核验通过**：Nature/AGU/SSA/Springer 等出版商页面返回
  `Client Challenge`（Cloudflare/JS 墙），`title_token_match_ratio=0` 属**抓取失败**而非论文不存在；
  改用 Crossref `/works/{doi}` 权威元数据，11/11 标题命中率 **1.0**，年份、期刊、被引数一并记录。
- **书目级核验 ≠ 全文核验**。下表 `核验` 列如实区分两者；标注为「书目」的条目，其关键实验结论来自摘要与本项目既有引用链，**未逐句核对全文**，这是本轮明确的证据缺口。

## 1. 证据矩阵（25 篇，其中原始实验论文 ≥10）

| # | 标签 | 标题 | 年 | 出处 | 核验 | 与本项目候选的直接关系 |
|---:|---|---|---:|---|---|---|
| 1 | A1 | PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method | 2018 | GJI, arXiv 1803.03211 | 原文页 | 给出 `sigma=label_width/5=6` 采样点、且**不按相位区分 sigma** 的参考实现，是本轮 S sigma 收紧的直接口径依据 |
| 2 | A2 | Earthquake Transformer: an attentive deep-learning model for simultaneous earthquake detection and phase picking | 2020 | Nature Communications, cited 957 | 书目 | 高斯软标签 + 注意力检测的标准做法；说明软标签宽度是被广泛沿用而非本仓独创 |
| 3 | A3 | Learning Earthquake Wave Arrival Time Picking from Labels with Inaccuracies | 2026 | arXiv 2606.15377 | 原文页 | 标签本身含到时误差时的学习策略，对应本项目「误差是方差不是偏置」的实测结论 |
| 4 | A4 | Improving Deep Learning-Based Seismic Phase Picking by Addressing Label Imbalance | 2025 | Research Square rs-10439246/v1 | 原文页 | P/S/N 三通道类不平衡；对应 `phase_weight` 与 P/S 分权 |
| 5 | A5 | Bayesian 不确定性拾取（BSSA） | 2023 | BSSA 10.1785/0120230068, cited 11 | 书目 | 给出把拾取不确定性显式建模的路径，可作为压方差的替代机制 |
| 6 | A6 | 软标签在火山地震分类中的应用 | 2025 | Bulletin of Volcanology | 书目 | 软标签宽度与类间混淆的权衡 |
| 7 | A7 | ARRU phase picker | 2021 | SRL 10.1785/0220200382, cited 43 | 书目 | 概率峰形与不确定性输出；与集成概率平均的峰锐度相关 |
| 8 | A8 | Which picker fits my data? | 2022 | JGR Solid Earth, cited 185 | 书目 | **跨区域拾取器选择的系统评测**，直接支持本轮「跨区域最坏表现不退化」的取舍准则 |
| 9 | A9 | SeisLM: a Foundation Model for Seismic Waveforms | 2024 | arXiv 2410.15765 | 原文页 | 自监督地震基座；JEPA 路线的最接近参照 |
| 10 | A10 | Nabro 火山序列深度学习拾取 | 2021 | JGR Solid Earth, cited 67 | 书目 | 域外台网迁移的失败模式 |
| 11 | A11 | Bridging scales: 深度学习拾取跨尺度迁移 | 2020 | GRL, cited 138 | 书目 | 训练域与测试域尺度差异导致的系统偏移 |
| 12 | A12 | 实验室地震深度学习拾取 | 2024 | JGR Machine Learning & Computation | 书目 | 极端域外条件下的可迁移性反例 |
| 13 | A13 | 用本地数据定制拾取模型 | 2023 | Frontiers in Earth Science 10.3389/feart.2023.1306488 | 原文页 | **少量本地数据微调优于通用模型**，支持华南域先验成员 `huanan` 入集成 |
| 14 | A14 | PickBlue: 海底地震仪拾取 | 2023 | Earth and Space Science, cited 41 | 书目 | 换台网类型的迁移代价 |
| 15 | A15 | OBSTransformer | 2024 | GJI, arXiv 2306.04753 | 原文页 | 迁移学习到 OBS 的完整实验流程 |
| 16 | A16 | DAS 半监督拾取 | 2023 | Nature Communications, arXiv 2302.08747 | 原文页 | 半监督扩召回，对应「漏检天花板」路线 |
| 17 | A17 | CSESnet | 2022 | Frontiers in Earth Science 10.3389/feart.2022.1032839 | 原文页 | **中国区域台网**训练的拾取网络，与 USTC 省级权重同源背景 |
| 18 | A18 | DASFormer | 2025 | Visual Intelligence | 书目 | 自监督时序结构学习 |
| 19 | A19 | 多事件震相拾取（EGU 2026） | 2026 | EGU 摘要 10.5194/egusphere-egu26-6122 | 原文页 | 官方规则允许「多个 P/S 或没有」，与 `cap_max_p=1` 的尾部风险直接相关 |
| 20 | A20 | PhaseLink | 2019 | JGR Solid Earth, cited 197 | 书目 | 拾取到事件关联，长记录过检治理的上游思路 |
| 21 | B1 | Distribution-Aware Coordinate Representation for Human Pose Estimation (DarkPose) | 2019 | arXiv 1910.06278 | 原文页 | **热图解码偏差的定量分析**；本项目亚采样解码的方法学来源 |
| 22 | B2 | The Devil is in the Details: Delving into Unbiased Data Processing (UDP) | 2019 | arXiv 1911.07524 | 原文页 | 编解码一致性；软标签宽度与解码器必须配套 |
| 23 | B3 | Rethinking the Heatmap Regression for Bottom-up Human Pose Estimation | 2020 | arXiv 2012.15175 | 原文页 | **按目标尺度自适应 sigma**；反过来支持「不该给 S 无依据地放宽 sigma」 |
| 24 | B4 | Soft labels 的偏差—方差分解 | 2021 | arXiv 2102.00650 | 原文页 | 软标签宽度直接控制偏差—方差权衡，是 sigma 收紧的理论依据 |
| 25 | B5 | Revisiting Knowledge Distillation via Label Smoothing Regularization | 2019 | arXiv 1909.11723 | 原文页 | KD 与标签平滑的等价视角；解释为何本项目历史 KD 扫参已饱和 |

## 2. 从文献到本轮唯一变量

三条独立证据链指向同一处：

1. **口径证据**（A1、B3、B2）：PhaseNet 与 SeisBench 的参考实现都**不按相位区分 sigma**；姿态估计文献里 sigma 随目标尺度自适应是有依据的，而本仓给 S 单方面放宽到 0.3s 没有对应依据。
2. **理论证据**（B4）：软标签宽度即偏差—方差旋钮。S 的容差（0.2s）比 P（0.1s）宽，但**评分容差不是标签宽度**——把 sigma 设成 0.3s 已超过 S 的满分容差，等于主动把概率质量摊到扣分区。
3. **实测证据**（本轮 `_signed_residual_r1r2.json`）：最优常数平移只值 +1.7pp，说明失分是**方差**而非偏置，应当锐化峰形而不是平移。

因此第 014 号实验的唯一变量定为 `--sigma-s` 的第二个分量：`0.3 → 0.2`。

## 3. 可迁移性风险与反例

- A10、A11、A12、A14 一致显示：换台网、换尺度、换仪器类型时深度拾取器退化明显。本届由**广西地震局**举办，去年为四川地震局，去年数据仅供参考 → R1/R2/08 均为**四川域**，对新赛季属域外。这是本轮把 R1/R2 从「优化目标」降级为「不回归检查」的文献支撑。
- A8 提供了正面方法论：拾取器选择应按**跨区域评测**而非单一数据集均分。本轮据此把取舍准则改为「跨区域最坏表现不退化」。
- **反例警示**：A13 支持本地数据微调，但本项目可用的「本地数据」只有封存包，不得用于训练；因此只能通过**公开多区域数据 + 华南域先验权重**近似，无法真正获得广西域内监督。这一缺口本轮无法闭合，必须在结论中显式声明。

## 4. 未能核验的声明

- 11 篇出版商页面被 Cloudflare/JS 墙拦截，仅完成 Crossref 书目核验，**关键实验数字未逐句核对全文**。
- A9（SeisLM）与 A18（DASFormer）指向的 JEPA/自监督路线，本轮**未做任何复现实验**，不能据此声称该路线在本项目可行或不可行。
- 官方规则 PPT 与公告 PDF 均**未写明数据区域**；公告 PDF 为扫描件（提取仅 91 字符，无文本层），赛区推断来自用户口述而非官方文档。