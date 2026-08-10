# 研究轮次 01：短时 T2/T3 波形是否应在摄入层被拒绝

- 日期：2026-08-11
- 候选改动：保留 T1 的 5 秒下限；对不依赖 PhaseNet picks、且自身具有固定窗补齐逻辑的 T2/T3 模型允许任意正长度波形。
- 直接触发证据：第 1 轮官方 T3 有 6 个 1.5–3.5 秒合法样本，旧 API 在模型前全部丢弃。

## 检索与核验协议

AnySearch 检索式：

1. `SeismicXM foundation model seismic waveform paper variable length padding 10240 event classification`
2. `deep learning seismic event classification earthquake explosion collapse landslide waveform CNN paper`
3. `self-supervised pretrained seismic waveform representation event classification paper`
4. `short duration seismic waveform event classification zero padding neural network paper`
5. `seismic foundation model classification benchmark SeisLM SeisCLIP SeismicBERT paper`
6. 若干精确标题检索，用于核对 DOI、任务和摘要。

Playwright 原文/代码核验：

- `https://github.com/cangyeone/seismicxm`
- `https://github.com/cangyeone/seismicxm/blob/main/makejit.picker.py`

核验到的关键上游事实：

- SeismicXM README 明确模型输入是 `[N, C, T]`，示例 `T=10240`，并直接输出 event type 与 hidden representation。
- README 明确推荐用 `hidden[:, :, 0]` 接外部分类头；本项目当前 T2/T3 正是这条路径。
- 上游 `makejit.picker.py` 对长度 `T` 建立 10240 点索引并执行 `idx.clamp(..., max=T-1)`；短输入通过边界重复进入模型，而不是设置 5 秒拒绝线。
- 本项目的 `tasks/seismicxm_features.py` 对不足 10240 点的通道零填充。固定窗补齐是模型预处理契约，5 秒下限不是 SeismicXM 契约。

## 15 篇直接相关原始实验论文证据矩阵

本轮 15 篇均为提出数据、模型或实验评估的原始研究；综述仅用于发现线索，不计入 15 篇。

| # | 论文 | 年份 / 标识 | 实验任务 | 对本轮决策的直接意义 | 边界 |
|---:|---|---|---|---|---|
| 1 | SeismicXM: A Cross-Task Foundation Model for Single-Station Seismic Waveform Processing | 2026, DOI `10.1785/0220250290` | 单台站波形的多任务表征、震相/极性/事件类型 | 当前编码器的直接来源；固定 10240 点输入与事件分类头说明长度适配应由模型预处理承担 | 论文页面受 403，输入细节由作者 GitHub 原文代码交叉核验 |
| 2 | SeisLM: a Foundation Model for Seismic Waveforms | 2024, arXiv `2410.15765` | 自监督波形预训练与多下游任务 | 支持把长度/任务适配放在表征模型与下游头，而不是在通用读入器用单一任务阈值删样本 | 不是当前部署模型，不能直接证明本模型分数 |
| 3 | SeisCLIP: A Seismology Foundation Model Pre-Trained by Multimodal Data for Multipurpose Seismic Feature Extraction | 2024, DOI `10.1109/TGRS.2024.3354456` | 事件分类、定位、机制等迁移任务 | 证明预训练表征 + 轻量下游头是成熟路线；不应在下游头前无证据丢弃稀有短样本 | 使用时频谱与多模态信息，预处理不同 |
| 4 | Exploring Foundation Models for Seismic Event Processing | 2023, DOI `10.31223/X58D7P` | 自监督时序表征与事件判别 | 事件判别受时序上下文影响，但论文没有提出统一的最短秒数；模型适配优于硬编码删样本 | 预印本，性能未达 SOTA |
| 5 | Deep Learning Models Augment Analyst Decisions for Event Discrimination | 2019, DOI `10.1029/2018GL081119` | 地震事件源判别 | 原始波形模型可补充人工分析，强调保存可判别波形证据；与“短样本直接空响应”相冲突 | 区域与类别体系不同 |
| 6 | Automatic Classification of Volcano Seismic Signatures | 2018, DOI `10.1029/2018JB015470` | 火山地震信号自动分类 | 事件持续时间本身是类别信息；按统一下限删除短类会造成类别相关缺失 | 火山场景，不等同于比赛五类 |
| 7 | Discrimination of earthquakes, explosions, and collapses based on the deep learning: Applications to DiTing 2.0 dataset | 2024, DOI `10.1016/j.cageo.2024.105830` | 地震/爆破/塌陷三分类 | 与比赛来源和类别最接近，支持直接从波形判别自然/非自然事件；不能把 T1 拾取上下文假设套到分类 | 摘要未给出短窗专门消融 |
| 8 | A Deep Active Learning Approach to the Automatic Classification of Volcano-Seismic Events | 2022, DOI `10.3389/feart.2022.807926` | 少标签火山事件分类 | 稀有类样本具有高标注价值；规则性删除少数短样本会恶化长尾覆盖 | 不研究 SeismicXM |
| 9 | Comparative Performance Assessments of Machine-Learning Methods for Artificial Seismic Sources Discrimination | 2021, DOI `10.1109/ACCESS.2021.3076119` | 人工震源判别 | 说明源类型分类需要比较多种特征/模型，输入保留应以任务模型能力为准 | 传统/深度模型混合，数据分布不同 |
| 10 | Classifying small earthquakes, explosions and collapses in the western United States using physics-based features and machine learning | 2024, DOI `10.1093/gji/ggae316` | 小地震/爆破/塌陷 | 直接覆盖“小事件”与多源类型；短小/低能事件不能在分类前按 T1 规则淘汰 | 使用物理特征，不是端到端编码器 |
| 11 | Generalization of Deep-Learning Models for Classification of Local Distance Earthquakes and Explosions across Various Geologic Settings | 2024, DOI `10.1785/0220230267` | 跨地质区地震/爆破泛化 | 强调跨域泛化；长度相关的选择偏差会成为额外域偏移 | 二分类，类别少于比赛 |
| 12 | Deep learning and transfer learning of earthquake and quarry-blast discrimination: applications to southern California and eastern Kentucky | 2023, DOI `10.1093/gji/ggad463` | 跨区迁移的地震/采石爆破判别 | 支持使用预训练/迁移表征处理区域差异；保留原始样本比统一拒绝更符合迁移假设 | 主要是二分类 |
| 13 | Automatic Classification of Microseismic Records in Underground Mining: A Deep Learning Approach | 2020, DOI `10.1109/ACCESS.2020.2967121` | 矿山微震记录分类 | 微震记录天然可能短，深度模型仍可分类；证明“短”不是普适无效条件 | 传感器与类别不同 |
| 14 | Using Deep Convolutional Neural Networks for Earthquake and Explosion Classification | 2025, DOI `10.1109/ACCESS.2025.3552127` | 地震/爆炸 CNN 分类 | 再次支持从固定窗波形做源类型分类；是否有效应由模型和验证决定 | 新论文、外部复现证据有限 |
| 15 | Uncertainty-aware deep learning methods for robust discrimination of earthquakes and explosions | 2025, DOI `10.1093/gji/ggaf303` | 带不确定性的源类型判别 | 对异常/边界样本应保留并输出模型判断或不确定性，而不是在共同摄入层静默变空 | 重点是不确定性，不直接规定 padding |

## 证据综合

论文集合支持三点：

1. 事件分类是独立于震相拾取的波形任务，常由固定窗、裁剪/补齐、预训练表征和下游头完成。
2. 短小、低能和稀有事件恰是源类型判别的重要长尾；按 T1 的上下文需求统一删掉会引入类别相关选择偏差。
3. 没有一篇证据支持“所有 T2/T3 输入少于 5 秒必然无效”。相反，当前模型作者代码明确允许短于 10240 点的输入通过索引边界适配。

论文证据本身不证明具体分数增益。录取仍以官方历史包上的任务级 A/B 为准。

## 实验假设

- H1：T1 保持默认 5 秒下限，预测行为不变。
- H2：`needs_picks=False` 的 T2/T3 模型使用 `min_duration_s=0` 后，官方第 1 轮 6 个短 T3 文件不再空响应。
- H3：这 6 个文件经当前生产 SeismicXM bundle 均预测为正确的第 5 类；在一文件一题、空响应记错的口径下，整体 T3 从有效 `182/200=91.0%` 恢复到离线 `188/200=94.0%`，第 5 类召回从 `4/10=40%` 恢复到 `10/10=100%`。
- H4：依赖 picks 的 T2/T3 估计器仍使用 5 秒下限，避免把短窗送进需要 PhaseNet 上下文的路径。
