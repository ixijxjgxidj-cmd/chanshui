# 第 13 轮文献复核：JEPA 方差、跨数据集验证与后续准入

检索与阅读日期：2026-08-15。

## 方法与边界

本轮紧随 T2 第 39D 轮公开 STEAD 方差分解。检索候选来自远程、可复现的两份机器产物：

- `outputs/round39_lit/readlist.json`：OpenAlex、Crossref，部分条目另有 arXiv，30 条候选；
- `outputs/t2_round39c_literature_crossref_arxiv/readlist_crossref_arxiv.json`：Crossref 发现且以 arXiv 题名和摘要核验，12 条候选。

本会话未提供 AnySearch 或 Playwright/浏览器工具，故未做浏览器全文抓取，也不把此记录伪称为浏览器精读。以下是对摘要、出版元数据和项目适用性的**摘要级定向精读**。这些论文仅用于制定公开数据实验的先验与否证边界；不用于任何比赛包、R1、R2 或 08 的模型选择。

## 15 篇直接相关文献与可执行结论

| # | 文献 | 核心证据（摘要级） | 对本项目的约束/动作 |
|---:|---|---|---|
| 1 | Assran et al. (2023), *I-JEPA*, DOI `10.1109/CVPR52729.2023.01499` | 以 context 表征预测大尺度 target 表征；遮蔽策略与信息充分的上下文是关键。 | 支持非生成式表征预测，但不支持在单一公开域事后扫遮蔽率。第 39D 已显示配置差异被方差淹没，冻结 `p8_m32_b10`。 |
| 2 | Bardes et al. (2024), *V-JEPA*, arXiv `2404.08471` | 仅靠特征预测目标即可获得可迁移表示，评估覆盖运动与外观任务。 | 将“跨域一次性评估”而非训练内分数作为 JEPA 的下一道证据门槛。 |
| 3 | Baevski et al. (2022), *data2vec*, arXiv `2202.03555` | 掩蔽输入预测全局上下文化的教师 latent，跨语音、视觉和文本。 | 支持 teacher-target 表征预测这一架构方向；若 39B 失败，不以更换 target/教师动量作为同一数据上的补救性网格搜索。 |
| 4 | He et al. (2022), *Masked Autoencoders Are Scalable Vision Learners*, arXiv `2111.06377` | 高遮蔽率和不对称编码器/解码器可扩展，但结论来自大规模视觉预训练。 | 可作为遮蔽学习的机制参考，不能外推为本地 3C 地震短窗中高遮蔽率必优；39D 的直接实验优先级更高。 |
| 5 | Nie et al. (2023), *PatchTST*, DOI `10.1609/AAAI.V37I9.26317` | 时间序列 patch 化保留局部语义并降注意力成本；有跨数据预训练转移结果。 | 保留 patch=8 及 Transformer 时序表征的工程选择，后续以跨数据源而非 patch 网格验证。 |
| 6 | Yue et al. (2022), *TS2Vec*, DOI `10.1609/AAAI.V36I8.20881` | 多尺度对比式时间序列表征适配不同语义粒度。 | 若 JEPA 不能跨 INSTANCE 再现，只能把多尺度对比目标作为**新的、预注册的机制**，不能与 39C 的遮蔽消融混为一轮。 |
| 7 | Mou & Zhu (2024), *TS-MAE*, DOI `10.1016/J.INS.2024.121576` | 掩蔽自动编码器面向时间序列表征。 | 证明时序掩蔽学习具有合理性，但并未取代跨区域和跨仪器外域验证。 |
| 8 | Mousavi et al. (2019), *STEAD*, DOI `10.1109/ACCESS.2019.2947848` | 全球公开三分量地震波形数据集，天然包含区域和台网异质性。 | STEAD 是合法训练源；继续按事件与自然区域隔离，不把随机记录切分当泛化证据。 |
| 9 | Michelini et al. (2021), *INSTANCE*, DOI `10.5194/ESSD-13-5509-2021` | 意大利公开机器学习地震数据，台网、区域与 STEAD 明显不同。 | 作为 39B 的独立外域；只使用官方 `train/dev`，官方 `test` 加载器级拒绝。 |
| 10 | Woollam et al. (2022), *SeisBench*, DOI `10.1785/0220210324` | 标准化地震 ML 数据、任务与评估接口。 | 采用可复现数据来源、哈希、事件 split 和固定加载器审计，而非临时手工筛样。 |
| 11 | Mousavi et al. (2020), *Earthquake Transformer*, DOI `10.1038/S41467-020-17591-W` | 地震检测与相位拾取的注意力模型，说明波形模型必须在不同网络条件下验证。 | 不将相位拾取论文的架构指标直接换算为本项目震级收益；只借鉴严格跨网络验证思想。 |
| 12 | Münchmeyer et al. (2021), *Which Picker Fits My Data?*, DOI `10.1029/2021JB023499` | 不同训练数据与应用区域间的表现存在显著迁移差异。 | 39B 的双向 STEAD↔INSTANCE、平均与最差域双门槛是必要而不是锦上添花。 |
| 13 | Koh et al. (2021), *WILDS*, arXiv `2012.07421` | 真实分布偏移需利用有意义的域分组评估。 | 禁止以随机记录切分替代自然区域/来源留出；报告最差域，不能只报均值。 |
| 14 | Gulrajani & Lopez-Paz (2021), *In Search of Lost Domain Generalization*, arXiv `2007.01434` | 域泛化方法在严格比较中不稳定，强 ERM 基线不可省略。 | 39B 四臂必须有同架构 scratch 对照；JEPA 平均升高但最差域下降仍不录取。 |
| 15 | Kumar et al. (2022), *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*, arXiv `2202.10054` | 微调可能破坏预训练特征并降低分布外表现。 | 固定监督微调配方；禁止查看 C 域后按 checkpoint、epoch 或冻结层数回选。若未来研究微调策略，应另行预注册并双向验证。 |

补充但尚不进入下一轮的两项方向：Ovadia et al. (2019), *Deep Ensembles*，为公开域上的多成员集成提供理论和经验依据；Beyer et al. (2022), *Knowledge distillation: A good teacher is patient and consistent*，指出蒸馏强依赖教师一致性与训练配方。二者都只能在 JEPA 先通过 39B 双向门槛后作为新的公开数据预注册实验，不能用于绕过该门槛。

## 综合判断

文献与第 39D 数据证据共同排除了四类不当下一步：

1. 在同一 STEAD 上继续扫描遮蔽率、连续块、patch 尺寸、context 比例；
2. 以训练侧相关性、单一地区或单次随机切分作为 JEPA 录取依据；
3. 用 C 域选择 λ、模型种子、checkpoint、成员或蒸馏温度；
4. 在跨数据集基本泛化尚未证明时直接启动混合训练、7 成员集成或蒸馏。

唯一证据链最强且算力性价比合理的下一轮为既有的第 39B 预注册协议：在公开 INSTANCE 完整性闸门全部绿灯后，比较 STEAD scratch/JEPA 和 INSTANCE scratch/JEPA，并进行 STEAD→INSTANCE 与 INSTANCE→STEAD 双向一次性评估。JEPA 必须在两个方向都使平均分严格提高且最差域不下降，才允许试验公开混合训练或 3–7 成员集成/蒸馏。

## 数据与合规

本轮没有下载数据集、没有在本地训练、没有读取 08、R1、R2 或其任何衍生产物。下一轮启动条件保持不变：压缩包精确字节数、`bzip2 -t`、解压退出及稳定、400 条随机 trace 的键/形状/有限值、E/N/Z 通道审计均通过，并写出 `ready_for_cache=true`。
