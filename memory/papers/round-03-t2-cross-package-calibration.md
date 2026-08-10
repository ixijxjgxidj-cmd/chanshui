# 研究轮次 03：T2 跨包震级偏差与源包残差校准

- 日期：2026-08-11
- 状态：论文研究完成；实验尚未运行
- 对应预注册：`memory/experiments/003-t2-cross-package-residual-calibration.md`
- 本轮问题：冻结的 SeismicXM `1024` 维特征与 `StandardScaler + Ridge(alpha=30)` 在包内 5 折 OOF 上表现正常，但跨包出现近似相反的全局偏差。只使用源包标签和源包 OOF 残差，能否学到对目标单样本安全、且不依赖目标批次统计的低维修正？

## 1. 当前失败现象

冻结缓存 `outputs/seismicxm_t3/features_tta.npz` 中，T2 数据形状均为 `200 × 1024`：

| 训练→评测 | MAE | RMSE | `prediction - truth` 均值 | 预测均值 | 真值均值 |
|---|---:|---:|---:|---:|---:|
| R1→R2 | `0.621046` | `0.742015` | `-0.589283` | `4.594217` | `5.183500` |
| R2→R1 | `0.660726` | `0.756001` | `+0.539461` | `5.003461` | `4.464000` |

与之相对，包内 5 折 OOF 为：

| 源包 | OOF MAE | OOF 有符号误差 |
|---|---:|---:|
| R1 | `0.281552` | `-0.00848` |
| R2 | `0.214581` | `-0.00811` |

这不是单纯的包内欠拟合。R1 标签均值/标准差为 `4.4640/0.4932`，R2 为 `5.1835/0.3652`；跨包误差主要呈现为与源包均值相关的回归收缩和标签/条件分布漂移。

先验诊断还显示，最近邻余弦距离、10 邻居平均距离和 Ridge 多 alpha 分歧与跨包绝对误差的 Spearman 相关都很弱：

| 风险信号 | R1→R2 | R2→R1 |
|---|---:|---:|
| 最近邻余弦距离 | `0.0555` | `0.0905` |
| 10 邻居平均距离 | `0.0535` | `0.0934` |
| Ridge 多 alpha 分歧 | `-0.0130` | `-0.2050` |

因此，本轮不把“距离大”直接等同于“误差大”，也不允许用目标包误差反向选择距离阈值。

## 2. 检索与核验协议

### AnySearch

本轮同时使用普通检索、`academic.search`、`academic.preprint` 和全文抽取。主要检索式包括：

1. `single station earthquake magnitude estimation cross region transfer learning waveform`
2. `earthquake magnitude regression domain shift calibration transfer learning target region`
3. `covariate shift regression importance weighting kernel mean matching target unlabeled`
4. `domain generalization regression invariant risk minimization group DRO anchor regression`
5. `conformal regression covariate shift uncertainty deep ensembles calibration`
6. `CORAL domain adaptation regression covariance alignment target batch`
7. 16 个候选标题与 DOI/arXiv 的精确检索，用于核对作者、年份、正式页面和实验结论。

### Playwright 实际打开并读取的页面

本轮有效核验 16 篇新的原始研究，不重复计入前两轮的正式计数：

1. `https://ar5iv.labs.arxiv.org/html/1911.05975`
2. `https://ar5iv.labs.arxiv.org/html/2101.02010`
3. `https://ar5iv.labs.arxiv.org/html/2204.02924`
4. `https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2023.1070010/full`
5. `https://www.nature.com/articles/s43247-024-01718-8`
6. `https://earth-planets-space.springeropen.com/articles/10.1186/s40623-024-02005-8`
7. `https://proceedings.neurips.cc/paper/2006/hash/a2186aa7c086b46ad4e8bf81e2a3a19b-Abstract.html`
8. `https://www.jmlr.org/papers/v8/sugiyama07a.html`
9. `https://ar5iv.labs.arxiv.org/html/1801.06229`
10. `https://ar5iv.labs.arxiv.org/html/1907.02893`
11. `https://ar5iv.labs.arxiv.org/html/2007.01434`
12. `https://ar5iv.labs.arxiv.org/html/1911.08731`
13. `https://ar5iv.labs.arxiv.org/html/1904.06019`
14. `https://ar5iv.labs.arxiv.org/html/1905.03222`
15. `https://ar5iv.labs.arxiv.org/html/1612.01474`
16. `https://ojs.aaai.org/index.php/AAAI/article/view/10306`

Wiley 的 MagNet 正文页与 OUP 的 TEAM-LM 正文页触发 Cloudflare，未把安全验证页计作“已读正文”；两篇均改用公开 arXiv/ar5iv 全文核验。Project Euclid 的局部回归补充页面未返回可用正文，不计入本轮 16 篇。

## 3. 地震震级模型证据

### 1. A Machine-Learning Approach for Earthquake Magnitude Estimation（MagNet）

- 作者/年份：S. Mostafa Mousavi、Gregory C. Beroza；2019 预印本，2020 刊发。
- 标识：DOI `10.1029/2019GL085976`；arXiv `1911.05975`。
- 方法与数据：用 CNN/LSTM 从单台原始波形直接估计震级，使用约 30 万条 STEAD 波形。论文特别保留绝对振幅信息，而不是把每条波形归一化到相同峰值。
- 关键实验结果：在大规模独立测试集上，深度模型相对传统单台经验关系和较浅模型给出更稳定的震级估计；误差随可用 P 波后时长和训练样本覆盖改善。
- 局限：训练规模比本项目两个包合计 400 条标签大约三个数量级；STEAD 的区域、仪器和震级定义也不等同于比赛。
- 本项目关系：当前 SeismicXM 路径做 max 归一化，可能已丢失 MagNet 明确保留的绝对振幅载体。此前把相同手工/幅值特征直接拼接到 deep1024 已失败，不能原样重做。
- 决策：采用“绝对幅值缺失是结构性风险”的解释；本轮不重训编码器，不重新拼接已失败特征。

### 2. Earthquake Magnitude and Location Estimation from Real Time Seismic Waveforms with a Transformer Network（TEAM-LM）

- 作者/年份：Jannes Münchmeyer、Dino Bindi、Ulf Leser、Frederik Tilmann；2021。
- 标识：DOI `10.1093/gji/ggab139`；arXiv `2101.02010`。
- 方法与数据：Transformer 聚合多台站波形、站点位置和触发时间，联合输出震级与位置分布；跨欧洲、意大利和日本等区域训练/迁移。
- 关键实验结果：扩大训练数据约 4 倍可显著降低误差；论文明确观察到大震系统性低估，跨区 transfer learning 比直接使用源区模型更能缓解。
- 局限：核心优势来自多站、站点坐标、区域数据量与目标区微调；当前比赛 T2 单请求没有站点位置和目标区标签。
- 本项目关系：直接支持“震级低估不是换一个轻量回归头就必然消失”，也说明只有源包的包内验证不能证明跨包可用。
- 决策：采用跨区域验证思想；延期多站/目标区迁移。

### 3. CREIME: A Convolutional Recurrent Model for Earthquake Identification and Magnitude Estimation

- 作者/年份：Megha Chakraborty、Darius Fenner、Wei Li、Johannes Faber、Kai Zhou、Georg Rümpker、Horst Stöcker、Nishtha Srivastava；2022。
- 标识：arXiv `2204.02924`。
- 方法与数据：原始三分量卷积循环网络，联合判断地震/噪声并估计震级，以多任务约束代替单一回归头。
- 关键实验结果：论文独立测试的震级 RMSE 约 `0.65`，并同时报告事件识别能力。
- 局限：需要端到端训练与大量波形；其独立集 RMSE 与当前跨包 `0.74–0.76` 同量级，并不证明小样本轻量头可把误差压到比赛理想区间。
- 本项目关系：支持多任务与原始幅值信息，但当前没有安全重训所需数据。
- 决策：延期到 DiTing 数据到位后的端到端阶段。

### 4. Rapid Earthquake Magnitude Estimation Combining a Neural Network and Transfer Learning in China

- 作者/年份：Jingbao Zhu、Shuilong Li、Shanyou Li、Yongxiang Wei、Jindong Song；2023。
- 标识：DOI `10.3389/fphy.2023.1070010`。
- 方法与数据：先在较大源区数据上训练 CNN，再用中国目标区少量标注数据迁移；应用到 2022 年芦山 `M6.1` 地震。
- 关键实验结果：目标区微调相对直接迁移和从头训练更快、更稳，并能在短 P 波窗内快速估计震级。
- 局限：收益依赖目标区带标签样本；不是纯 source-only domain generalization。
- 本项目关系：当前没有新的目标区标签，本周不能把论文的 transfer learning 收益外推为源包残差校准收益。
- 决策：延期；只采用“目标区微调是关键条件”的风险结论。

### 5. Universal Neural Networks for Real-Time Earthquake Early Warning Trained with Generalized Earthquakes

- 作者/年份：Xiong Zhang、Miao Zhang；2024。
- 标识：DOI `10.1038/s43247-024-01718-8`；预印本 arXiv `2312.15218`。
- 方法与数据：通过数据重组构造任意震源和台网分布的“广义地震”，训练可跨台网应用的实时检测、定位和震级网络。
- 关键实验结果：在日本和加州多次地震序列上，首台触发约 4 秒后可报告结果；位置平均误差约 `2.6–6.3 km`，震级平均误差随序列/时间约 `0.05–0.32`。
- 局限：需要大规模波形、站点几何和专门的数据重组；不是冻结单台表示上的后处理。
- 本项目关系：说明域泛化可通过训练分布设计获得，但当前 400 条标签和无位置输入不具备同等条件。
- 决策：延期数据重组训练；采用“训练分布覆盖比复杂头更重要”的结论。

### 6. Real-Time Earthquake Magnitude Estimation via a Deep Learning Network Based on Waveform and Text Mixed Modal

- 作者/年份：Baorui Hou、Yueyong Zhou、Shanyou Li、Yongxiang Wei、Jindong Song；2024。
- 标识：DOI `10.1186/s40623-024-02005-8`。
- 方法与数据：把波形与文本化站点/震相信息混合，显式使用差分 P 到时和站点位置，随时间更新震级。
- 关键实验结果：约 3 秒输入时 MAE 低于 `0.29`，约 14 秒时接近 `0.15`。
- 局限：使用本项目当前 API 没有的到时差和位置特征；数据量和任务设置不同。
- 本项目关系：再次说明低 MAE 与额外物理信息直接相关，不能把结果归功于通用回归校准。
- 决策：延期；不把不可用特征偷偷替换为包级标签均值。

## 4. 分布偏移、鲁棒学习与不确定性证据

### 7. Correcting Sample Selection Bias by Unlabeled Data（Kernel Mean Matching）

- 作者/年份：Jiayuan Huang、Alexander J. Smola、Arthur Gretton、Karsten M. Borgwardt、Bernhard Schölkopf；NIPS 2006 / 论文集 2007。
- 标识：DOI `10.7551/mitpress/7503.003.0080`。
- 方法与数据：在 RKHS 中匹配源/目标特征均值，直接估计源样本权重，不显式估计密度；实验显示对选择偏差有效。
- 局限：必须先拿到一批无标签目标样本；核心假设仍接近 `P(y|x)` 稳定、主要变化在 `P(x)`。
- 本项目关系：当前双向约 `±0.55` 的截距漂移可能包含条件/标签漂移；逐请求 API 也没有稳定目标批次。
- 决策：本轮拒绝 KMM；未来若赛前能合法积累无标签目标批次，再单独预注册。

### 8. Covariate Shift Adaptation by Importance Weighted Cross Validation

- 作者/年份：Masashi Sugiyama、Matthias Krauledat、Klaus-Robert Müller；2007。
- 标识：JMLR 8:985–1005；正式页 `https://www.jmlr.org/papers/v8/sugiyama07a.html`。
- 方法与数据：用目标/源输入密度比加权交叉验证，使模型选择估计目标风险；论文在模拟与真实数据中展示普通 CV 在 covariate shift 下可选错模型。
- 局限：同样需要无标签目标样本和可靠密度比；权重方差在高维小样本下会很大。
- 本项目关系：1024 维、每包 200 条正是密度比估计最不稳定的区域。
- 决策：拒绝本轮使用；保留其“模型选择协议必须匹配目标分布”的原则。

### 9. Anchor Regression: Heterogeneous Data Meet Causality

- 作者/年份：Dominik Rothenhäusler、Nicolai Meinshausen、Peter Bühlmann、Jonas Peters；2018 预印本，2021 刊发。
- 标识：arXiv `1801.06229`；DOI `10.1111/rssb.12398`。
- 方法与数据：用可观测 anchor 变量惩罚与环境相关的残差方向，在结构移位下获得鲁棒性；包含模拟和真实数据实验。
- 局限：需要有意义的 anchor/环境变量，并且其干预含义可解释。
- 本项目关系：只有 R1/R2 包 ID，两种环境不足以识别稳定因果结构；包 ID 也不能在正式单样本请求中直接作为已知校正量。
- 决策：拒绝当前轮；未来有站点、区域、仪器等元数据时重评。

### 10. Invariant Risk Minimization

- 作者/年份：Martin Arjovsky、Léon Bottou、Ishaan Gulrajani、David Lopez-Paz；2019。
- 标识：arXiv `1907.02893`。
- 方法与数据：要求同一预测器在多个训练环境上同时最优，Colored MNIST 等实验用于展示环境不变特征。
- 局限：依赖多个足够多样且正确划分的训练环境；优化和模型选择对实现敏感。
- 本项目关系：两个包不能支撑可靠环境不变表示训练，且本轮冻结编码器。
- 决策：拒绝实现 IRM；采用“不要从单环境相关性推断域外稳定性”的原则。

### 11. In Search of Lost Domain Generalization（DomainBed）

- 作者/年份：Ishaan Gulrajani、David Lopez-Paz；2020。
- 标识：arXiv `2007.01434`。
- 方法与数据：在统一训练、验证和模型选择协议下比较多种 domain generalization 算法与 ERM，覆盖多个标准数据集和算法。
- 关键实验结果：很多复杂方法在公平模型选择下并未稳定超过经过认真实现的 ERM；模型选择协议本身是主要差异来源之一。
- 局限：主要是视觉分类，不是地震回归。
- 本项目关系：直接支持保留 Ridge 强基线、使用源包嵌套 OOF 选择、再做双向跨包硬门槛。
- 决策：采用验证协议；不因方法名称“域泛化”就放宽准入线。

### 12. Distributionally Robust Neural Networks for Group Shifts

- 作者/年份：Shiori Sagawa、Pang Wei Koh、Tatsunori B. Hashimoto、Percy Liang；2019/2020。
- 标识：arXiv `1911.08731`。
- 方法与数据：Group DRO 直接优化最坏预定义组损失；在 Waterbirds、CelebA、MultiNLI 等数据上证明强正则化和早停对最坏组泛化关键。
- 局限：必须知道组标签；只有两个包且每包内部没有可靠站点/区域分组。
- 本项目关系：可以把 R1/R2 当评测组，但不能在只训练一个源包时把目标包组损失用于选型。
- 决策：不实现训练算法；采用“双向最坏方向不得恶化”的录取规则。

### 13. Conformal Prediction Under Covariate Shift

- 作者/年份：Ryan J. Tibshirani、Rina Foygel Barber、Emmanuel J. Candès、Aaditya Ramdas；2019。
- 标识：arXiv `1904.06019`。
- 方法与数据：对目标/源似然比进行加权，使 conformal prediction 在 covariate shift 下保持覆盖保证；包含理论与实验。
- 局限：需要已知或估计目标权重，主要保证区间覆盖而不是降低点预测 MAE。
- 本项目关系：比赛按点预测绝对误差计分，且没有稳定目标批次。
- 决策：拒绝作为本轮提分方法；可在未来 API 需要可信区间时复用。

### 14. Conformalized Quantile Regression

- 作者/年份：Yaniv Romano、Evan Patterson、Emmanuel J. Candès；2019。
- 标识：arXiv `1905.03222`。
- 方法与数据：把 quantile regression 的自适应区间与 split conformal 校准结合，在多个回归数据集上获得接近标称覆盖且区间更自适应。
- 局限：核心产物是区间；不能保证点预测 MAE。当前 QuantileRegressor/GBR quantile 的 T2 历史实验已跨包失败。
- 本项目关系：不能用“区间校准”包装已失败的分位数点预测再重跑。
- 决策：本轮拒绝；不原样重启 quantile 路线。

### 15. Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles

- 作者/年份：Balaji Lakshminarayanan、Alexander Pritzel、Charles Blundell；2016/2017。
- 标识：arXiv `1612.01474`。
- 方法与数据：独立初始化多个网络并聚合预测，在分类、回归和 OOD 实验中改善负对数似然、校准与不确定性质量。
- 局限：需要多次完整训练/推理；不确定性与绝对误差仍需在当前域验证。
- 本项目关系：当前 Ridge 多 alpha 分歧与跨包绝对误差相关为 `-0.013/-0.205`，没有形成有效风险信号。
- 决策：拒绝当前轮的模型分歧门控；只输出诊断，不增加编码器调用。

### 16. Return of Frustratingly Easy Domain Adaptation（CORAL）

- 作者/年份：Baochen Sun、Jiashi Feng、Kate Saenko；2016。
- 标识：DOI `10.1609/aaai.v30i1.10306`。
- 方法与数据：通过线性变换对齐源/目标二阶统计，在标准视觉 domain adaptation 数据集上以很低复杂度获得有竞争力的结果。
- 局限：显式需要目标批次协方差；高维、小目标批次下协方差估计不稳定，并可能抹掉与震级相关的方向。
- 本项目关系：第 2 轮 T3 已证明训练包中心化会放大跨包漂移；当前 API 还是逐请求，不能默认存在可代表正式分布的目标批次。
- 决策：本轮拒绝 CORAL；未来只能在合法、冻结的无标签目标批次上独立预注册。

## 5. 方法决策矩阵

评分为 `1`（差/高风险）到 `5`（好/低风险）；“数据现成”与“API 兼容”越高越好。

| 方向 | 预期 MAE 收益 | 数据现成 | 本周可完成 | 单请求 API | 过拟合安全 | 推理成本 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| 源包 OOF 残差 + 低维邻域强收缩 | 3 | 5 | 5 | 5 | 3 | 5 | 采用为最小反证 |
| 仅按源包 OOF 预测值做收缩校准 | 3 | 5 | 5 | 5 | 3 | 5 | 并入同一低维候选空间 |
| KMM / 重要性加权 CV | 3 | 2 | 3 | 1 | 2 | 3 | 拒绝当前轮 |
| CORAL 目标协方差对齐 | 3 | 2 | 4 | 1 | 2 | 4 | 拒绝当前轮 |
| Anchor / IRM / Group DRO | 2 | 1 | 2 | 4 | 2 | 3 | 延期到多环境数据 |
| CQR / 加权 conformal | 1（点 MAE） | 3 | 4 | 2 | 4 | 4 | 不作为点预测提分 |
| Deep ensemble 不确定性门控 | 2 | 4 | 3 | 5 | 2 | 2 | 现有分歧信号已否证 |
| 端到端震级网络/目标区迁移 | 5 | 1 | 1 | 4 | 2 | 2 | 等 DiTing/目标区数据 |

## 6. 本轮采用的最小机制

本轮不声称解决一般域适应，只检验一个较窄假设：Ridge 的一部分跨包误差是否来自可由源包 OOF 观察到的“回归到均值”残差结构。

采用机制：

1. 冻结 SeismicXM 特征、Ridge `alpha=30` 和现有缓存。
2. 在源包内部交叉拟合，得到每个样本未见过自身标签的 OOF 基线预测与 residual。
3. 校准空间只包含 OOF 基线预测值和最多 16 个、仅由源包拟合的 PCA 分量。
4. 用余弦局部邻域估计 residual，修正幅度强收缩并截断。
5. 邻域距离闸门只由源包内部距离分布确定；不使用目标标签、目标均值或目标批次协方差。
6. 配置只由源包嵌套 OOF 选择；随后一次性做 R1→R2、R2→R1 双向开发验证。
7. 只有双向门槛全部通过，才允许一次性读取冻结 08 T2 特征。

## 7. 明确拒绝的解释与边界

- 如果候选失败，不能据此证明所有域适应无效；只能否证当前数据量下的 source-only 低维局部残差机制。
- 如果候选在某一方向上涨、另一方向下跌，按预注册拒绝，不能用 08 结果救回。
- 如果开发阶段失败，禁止读取 08 T2 特征；已有 08 缓存的存在不构成查看许可。
- 不使用固定包均值差、目标标签均值、目标批次协方差或根据 08 有符号误差硬加常数。
- 不重复 GBR quantile、QuantileRegressor、Huber、简单双 Ridge 平均或相同的手工幅值拼接。
- 即使开发通过，08 仍须比冻结 MAE `0.523197455` 至少改善 `0.010`，否则不进入生产。

## 8. 对下一阶段的含义

若本轮失败，现有证据将共同指向一个更强结论：当前最大限制来自输入中缺少绝对幅值/位置/距离、训练分布覆盖不足或真实 `P(y|x)` 漂移，而不是 Ridge 后面缺一个小校准头。届时最高价值方向应转到不改模型的生产鲁棒性/文档漂移，或等待 DiTing 数据到位后重新建立端到端、按包留出的震级训练阶段；不再继续扩大同类 kNN/PCA 网格。
