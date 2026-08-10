# 研究轮次 02：T3 长尾类别与跨包表征几何

- 日期：2026-08-11
- 状态：研究完成；对应开发实验因跨包域移被拒绝，未进入 08 终检
- 对应预注册：`memory/experiments/002-t3-long-tail-domain-generalization.md`
- 具体失败：生产 SeismicXM + cosine kNN 在第 1 轮重复分层验证中 balanced accuracy 为 `0.8656`，第 2/3 类 recall 为 `0.80/0.65`；08 包第 4 类 recall 仅 `1/5=0.20`，而第 1 轮同类包内 recall 约 `0.88`，显示少数类与跨包几何漂移共同作用。
- 本轮问题：在不重训 SeismicXM 编码器、不使用尚未到手的 DiTing 原始数据、也不反复调 08 答案的条件下，类平衡原型/局部邻域和置信门控是否是本周最高收益路径？

## 检索与核验协议

### AnySearch 普通检索

1. `seismic event source classification few-shot class imbalance prototype network domain generalization earthquake explosion collapse landslide`
2. `small and imbalanced seismic event classification ghost attention network`
3. `seismic source type classification domain shift transfer learning earthquake explosion waveform experimental`
4. 15 个候选标题的精确检索，用于核对正式页面、DOI、作者代码和摘要。

### AnySearch 学术检索

先调用 `academic` 垂直域发现，使用 `academic.search` 与 `academic.preprint`；主要检索式为：

1. `seismic event classification class imbalance few-shot learning prototypical network waveform`
2. `earthquake explosion discrimination domain adaptation domain generalization learned embeddings seismic`
3. `seismic event classification cross regional validation imbalance earthquakes explosions collapses waveform`
4. `long-tailed recognition nearest class mean prototype classifier calibrated k nearest neighbors class balanced embeddings`
5. `imbalanced few-shot classification class balanced prototypes cosine nearest neighbor confidence gating`

AnySearch 还对可访问的 ar5iv/arXiv HTML 执行全文抽取，核对实验、结论与局限；没有只依赖搜索摘要。

### Playwright 实际打开记录

成功打开并读取的 15 个原文、正式出版页或作者正式代码页如下：

1. `https://arxiv.org/html/2510.23795v1`
2. `https://github.com/mfllwz/China-Earthquake-Classification-Model`
3. `https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2025.1708136/full`
4. `https://link.springer.com/article/10.1007/s10950-022-10109-5`
5. `https://pure.psu.edu/en/publications/convolutional-neural-networks-versus-ps-amplitude-ratios-in-low-y/`
6. `https://esurf.copernicus.org/articles/7/171/2019/`
7. `https://arxiv.org/abs/1703.05175`
8. `https://arxiv.org/abs/1911.04623`
9. `https://arxiv.org/abs/1901.05555`
10. `https://arxiv.org/abs/1906.07413`
11. `https://arxiv.org/abs/1910.09217`
12. `https://arxiv.org/abs/2007.07314`
13. `https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Balanced_Contrastive_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.html`
14. `https://arxiv.org/abs/1803.04765`
15. `https://arxiv.org/abs/2104.00466`

其中 OUP 的 EQTypeNet 正文页触发 Cloudflare，因此用论文作者公开的正式模型仓库核验模型、类别、资产和迁移建议，并用 AnySearch 学术结果核对论文实验数值。OUP 的 2026 Siamese 论文、IEEE Ghost-Attention 和 ACM 页面也被安全页拦截；它们只作为补充搜索证据，不计入上述 15 篇。没有把安全验证页当作论文正文。

## 15 篇原始实验论文证据矩阵

本轮计数的 15 篇全部是提出方法并报告实验的原始研究；没有用综述凑数，也没有重复计入轮次 01 的 15 篇。

### 1. Exploration of Machine Learning Methods to Seismic Event Discrimination in the Pacific Northwest

- 作者/年份：Akash Kharita, Marine Denolle, Alexander R. Hutko, J. Renate Hartog, Stephen D. Malone；2025 预印本、2026 Seismica。
- 标识：arXiv `2510.23795`；Playwright/全文页：`https://arxiv.org/html/2510.23795v1`。
- 问题与方法：四分类地震、爆破、地表事件和噪声；比较物理/TSFEL/散射特征的随机森林与 1D/2D CNN。
- 数据与实验：约 20 万条三分量波形、超过 7 万事件；同时使用平衡公共测试集、1 万事件网络集、全球地表事件和近场爆破等域外集，并按事件 ID 隔离训练/测试。
- 关键结果：2D CNN 在域内和域外评估中超过 92%，最佳随机森林约 89%；轻量 QuakeXNet-2D 约 7 万参数、约 1.2 MB，普通硬件扫描一天 100 Hz 三分量数据约 9 秒。
- 局限：大规模、多区域训练数据远多于本项目当前 389 个带标签 T3 特征；其类别定义也不是比赛 1–5 的一一映射。
- 项目关系与决策：强烈支持跨包/域外测试必须独立于包内验证，也说明噪声应作为真实类别处理；不直接移植新 CNN。本周采用其验证协议思想，延期模型替换。

### 2. EQTypeNet: Deep Learning Tri-Branch Earthquake Automatic Classification Model and Its Application in China

- 作者/年份：Tianran Lu, Lianqing Zhou, Na Zhang, Mengqiao Duan, Ziyi Li, Ming Zhao, Li Sun；2025。
- 标识：DOI `10.1093/gji/ggaf263`；Playwright 实际核验作者仓库：`https://github.com/mfllwz/China-Earthquake-Classification-Model`。
- 问题与方法：地震/爆破/塌陷分类；三分支输入波形、频谱图和 P/S 比值，并通过补充非自然事件与小区域迁移缓解不平衡和地域差异。
- 数据与实验：论文在中国多区域数据上做二分类、三分类和小区域迁移；仓库发布 binary/ternary HDF5 与 `demo.npz`。
- 关键结果：论文页面摘要报告二分类单台站 macro-F1 约 0.99；三分类 DT 测试中单台站 precision 为地震 94.8%、爆破 87.8%、塌陷 87.5%，macro-F1 0.90，网络级 macro-F1 0.95；小区域迁移 F1 约 0.98/0.97。
- 局限：依赖 P/S 信息、频谱分支和带标签目标区域迁移；当前比赛输入没有位置，且本周没有新的目标区标签。
- 项目关系与决策：直接证明中国源类型分类的少数类与地域变化需要专门处理；当前不换编码器，部分采用“冻结通用表征、校准区域头”的思想。

### 3. Machine Learning-Based Classification of Seismic Events: A Case Study of Seismic Events in Jilin Province, NE China

- 作者/年份：Fangyu Ren 等；2025。
- 标识：DOI `10.3389/feart.2025.1708136`；Playwright：Frontiers 正文页。
- 问题与方法：898 个 `1.5≤ML≤3.5` 地震、爆破、矿山塌陷事件；87 维时域、频域、时频物理特征，比较 SVM、XGBoost、BPNN。
- 关键结果：区域内三种模型均超过 94%；跨区域验证最佳 SVM 仅 84%。P/S 频谱幅值和最大 P/S 幅值比最重要。
- 局限：物理特征需要稳定震相或窗选；跨区只有 84%，说明高区域内分数不能外推。
- 项目关系与决策：与本项目 08 第 4 类崩落高度一致。采用“跨包是硬门槛”；拒绝根据第 1 轮包内 rare-class 分数直接上线静态加权。

### 4. Rapid Classification of Local Seismic Events Using Machine Learning

- 作者/年份：Luozhao Jia, Hongfeng Chen, Kang Xing；2022。
- 标识：DOI `10.1007/s10950-022-10109-5`；Playwright：Springer 正文预览页。
- 问题与方法：参考 VGG/ResNet/Inception 构建地震、爆破、矿山塌陷分类器，输入三分量全波形时间序列或频谱。
- 数据与实验：河南区域网 47 个宽频带台站、约 6.4k 观测，`ML 0.6–4.5`；使用 60 秒波形与首波对齐。
- 关键结果：recall 和 accuracy 均超过 90%，最低可处理至 `ML 0.6`。
- 局限：需要首波对齐且训练数据量、区域信息明显多于本项目；同域结果不能证明跨包泛化。
- 项目关系与决策：支持完整三分量波形有足够分类信息，但不支持新增重模型。本轮仅保留为中国三分类可行性证据。

### 5. Convolutional Neural Networks Versus P/S Amplitude Ratios in Low-Yield Seismic Event Discrimination

- 作者/年份：Sampath Rathnayaka, Ross Maguire, Andrew Nyblade, Björn Lund, Brandon Schmandt；2025。
- 标识：DOI `10.1785/0220240417`；Playwright：Penn State 正式研究输出页。
- 问题与方法：用 scalogram CNN 区分瑞典 Kiruna 的地震、矿爆和矿山相关事件，并测试美国训练 CNN 的区域可迁移性。
- 关键结果：本地区二分类 CNN 可达 90% 以上并优于 P/S 比；美国训练模型迁移至 Kiruna 时表现不足 90%。
- 局限：作者无法完全分离地质路径、爆破方式和震源深度分别造成的迁移损失。
- 项目关系与决策：这是“同类语义、不同地区仍会失效”的直接实验。采用保守门控和跨包准入；不把静态类先验视为域移解决方案。

### 6. Systematic Identification of External Influences in Multi-Year Microseismic Recordings Using Convolutional Neural Networks

- 作者/年份：Matthias Meyer, Samuel Weber, Jan Beutel, Lothar Thiele；2019。
- 标识：DOI `10.5194/esurf-7-171-2019`；Playwright：Copernicus 正文页。
- 问题与方法：Matterhorn 多传感器多年微震记录中识别人类活动等外部影响；CNN 与图像/微震集成分类。
- 关键结果：微震 CNN 错误率低于 1%，比对照算法低约 3 倍；集成错误率 0.79%、F1 0.9383；约四分之一未经清洗的检测事件来自非地震活动。
- 局限：目标是外部影响/噪声识别，不是比赛五种源类型；含多传感器信息。
- 项目关系与决策：支持把噪声/外部活动当成真实分布而非后处理异常，也支持置信门控。只采用评估思想，不引入额外模态。

### 7. Prototypical Networks for Few-Shot Learning

- 作者/年份：Jake Snell, Kevin Swersky, Richard S. Zemel；2017。
- 标识：arXiv `1703.05175`；Playwright：arXiv 摘要页；AnySearch 全文：ar5iv HTML。
- 问题与方法：在学习的嵌入空间中以每类样本均值作为原型，按距离执行少样本分类；使用 episodic training。
- 数据与实验：Omniglot、miniImageNet、CUB-200；报告少样本与零样本实验并达到当时先进结果。
- 关键结果：简单类均值原型是强少样本归纳偏置；距离选择与 episode 构造显著影响结果。作者尝试每维类方差但没有获得收益。
- 局限：其表征为少样本任务专门训练，本项目是冻结的 SeismicXM 通用表征；单质心可能忽略源类型的多模态地域结构。
- 项目关系与决策：原型头可在现有 1024 维缓存上零成本实现。采用为最便宜反证，但同时测试多原型，不能默认单质心更好。

### 8. SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning

- 作者/年份：Yan Wang, Wei-Lun Chao, Kilian Q. Weinberger, Laurens van der Maaten；2019。
- 标识：arXiv `1911.04623`；Playwright：arXiv；AnySearch：ar5iv 全文。
- 问题与方法：不做复杂 meta-learning，研究冻结特征上的最近邻/类均值；比较原始、L2 归一化、先中心化再 L2 归一化。
- 关键结果：简单特征变换即可得到竞争性少样本结果；中心化 + L2 归一化在 miniImageNet 的五种设置中三种超过已有结果。
- 局限：视觉预训练特征与地震 Transformer 表征不同；中心向量必须严格由训练折估计，否则会泄漏。
- 项目关系与决策：当前生产只有 L2 `Normalizer`，没有训练折中心化。将“训练折中心化 + 类原型/局部邻域”列为本轮首个最小实验。

### 9. Class-Balanced Loss Based on Effective Number of Samples

- 作者/年份：Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie；2019。
- 标识：arXiv `1901.05555`；Playwright：arXiv；AnySearch：ar5iv 全文。
- 问题与方法：用 `(1-β^n)/(1-β)` 表示样本有效数量，以其倒数重新加权损失。
- 数据与实验：长尾 CIFAR、ImageNet、iNaturalist；多种损失和不平衡因子。
- 关键结果：有效数量权重优于简单逆频率的多个长尾基准；论文也明确简单逆频率在高不平衡真实数据上可能表现差。
- 局限：主要改变端到端训练损失；本轮不重训编码器，且 10–110 条/类的规模远小于论文大数据场景。
- 项目关系与决策：不实现训练损失；只把“不要让类别票数按样本数机械累积”转化为等类先验的邻域分数。

### 10. Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss

- 作者/年份：Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, Tengyu Ma；2019。
- 标识：arXiv `1906.07413`；Playwright：arXiv；AnySearch：ar5iv 全文。
- 问题与方法：LDAM 为少数类设置更大 margin；DRW 延迟到表征学好后再重加权。
- 数据与实验：不平衡 CIFAR、Tiny ImageNet、iNaturalist 2018；组合优于单独方法。
- 关键结果：论文分析显示普通 ERM 学到的中间表征可能优于从头 reweight/resample，后期再调边界更稳。
- 局限：需要重新训练神经网络，且作者承认 DRW 成功的精确理论解释仍不完整。
- 项目关系与决策：支持保留冻结 SeismicXM 表征、只调头；拒绝本周进行 LDAM 编码器训练。

### 11. Decoupling Representation and Classifier for Long-Tailed Recognition

- 作者/年份：Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, Yannis Kalantidis；2020。
- 标识：arXiv `1910.09217`，ICLR 2020；Playwright：arXiv；AnySearch：ar5iv 全文。
- 问题与方法：系统分离表征学习与分类器学习，比较 cRT、最近类均值 NCM、权重归一化和 LWS。
- 数据与实验：ImageNet-LT、Places-LT、iNaturalist；报告 head/medium/few-shot 分组。
- 关键结果：自然采样可学习更通用的表征，随后仅重平衡分类器即可超过复杂损失/记忆模块；NCM 是有竞争力的零训练头。
- 局限：视觉长尾的类频率与比赛包标签先验不一定一致；类均值也可能受域偏移影响。
- 项目关系与决策：与现状高度匹配——已有冻结编码器和小样本头。采用 decoupled 原则，优先比较 NCM、多原型和局部等先验评分。

### 12. Long-Tail Learning via Logit Adjustment

- 作者/年份：Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, Sanjiv Kumar；2021。
- 标识：arXiv `2007.07314`，ICLR 2021；Playwright：arXiv；AnySearch：ar5iv 全文。
- 问题与方法：按类别训练先验进行 post-hoc 或训练期 logit 调整，对 balanced error 给出 Fisher consistency。
- 数据与实验：合成二分类与 CIFAR-10/100-LT、ImageNet-LT、iNaturalist 等；post-hoc 调整优于多种权重归一化。
- 关键结果：只改轻量输出分数即可提升长尾 balanced error，是低成本候选。
- 局限：要求训练先验与目标评价目标的关系合理；本项目未知正式赛类别先验，08 又缺第 5 类，直接按第 1 轮频率校正可能过补偿。
- 项目关系与决策：只允许在嵌套 CV 中检验温和的等类先验/温度调整；不能把 08 类别分布当正式先验。

### 13. Balanced Contrastive Learning for Long-Tailed Visual Recognition

- 作者/年份：Jianggang Zhu, Zheng Wang, Jingjing Chen, Yi-Ping Phoebe Chen, Yu-Gang Jiang；2022。
- 标识：CVPR 2022, pp. 6908–6917；Playwright：CVF Open Access 正式页。
- 问题与方法：指出普通 supervised contrastive learning 在长尾下无法形成理想规则单纯形；通过 class-averaging 与 class-complement 平衡负类梯度和 batch 类覆盖。
- 数据与实验：CIFAR-10/100-LT、ImageNet-LT、iNaturalist 2018；两分支框架达到竞争性结果。
- 局限：核心收益来自重新学习表征；当前没有足够独立数据安全重训 SeismicXM。
- 项目关系与决策：其“每类对几何贡献应等权”支持等类邻域分数，但不实现 BCL 训练，延期到 DiTing 数据到位后的严格阶段。

### 14. Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning

- 作者/年份：Nicolas Papernot, Patrick McDaniel；2018。
- 标识：arXiv `1803.04765`；Playwright：arXiv；AnySearch：ar5iv 全文。
- 问题与方法：在深层表征中检索训练邻居，用 nonconformity、confidence 与 credibility 衡量预测是否有训练数据支持。
- 数据与实验：多个图像任务及对抗/分布外输入；邻居支持能识别训练流形外样本并提供实例级解释。
- 关键结果：单纯 softmax 对分布外输入可能高置信，邻居距离与标签一致性提供额外安全信号。
- 局限：原论文跨多层检索，成本高于本项目只保留一个 1024 维层；少量训练样本时可信度估计方差大。
- 项目关系与决策：当前生产本就是末层 kNN，可零额外编码器成本导出最近支持距离、类间 margin 和邻居一致性。采用为置信门控特征，不声称完整复刻 DkNN。

### 15. Improving Calibration for Long-Tailed Recognition (MiSLAS)

- 作者/年份：Zhisheng Zhong, Jiequan Cui, Shu Liu, Jiaya Jia；2021。
- 标识：arXiv `2104.00466`，CVPR 2021；Playwright：arXiv；AnySearch：ar5iv 全文。
- 问题与方法：两阶段长尾模型易过度自信；使用 label-aware smoothing、mixup 和 shifted batch normalization 改善分类与校准。
- 数据与实验：CIFAR-LT、ImageNet-LT、Places-LT、iNaturalist 2018；同时报告 accuracy 和 ECE，并建立新结果。
- 关键结果：长尾模型的误校准与类别样本数相关；只看 top-1 不足以判断安全切换。
- 局限：需要神经网络第二阶段与 batch norm 参数；当前 sklearn kNN 没有 logits/BN，不能照搬。
- 项目关系与决策：采用“按类校准 + 不把 margin 当概率”的诊断思想；拒绝完整 MiSLAS，实现轻量 CV 校准或门控即可。

## 补充但不计数的搜索证据

- **A Ghost-Attention Network for Discriminating Tectonic and Non-Tectonic Events on a Small and Imbalanced Dataset**（IEEE Access 2024）：AnySearch 核到其针对小型不平衡数据提出 GA-Net；IEEE Playwright 页面为空，未计入 15 篇。
- **Robust Classification of Blasts, Collapses and Natural Earthquakes via Siamese Neural Network**（DOI `10.1093/gji/ggag247`, 2026）：AnySearch 摘要报告在目标区 anchors 可用时，域外 AUPRC 相对 CNN 与 CNN+transfer 分别提高约 7% 和 4%；本项目没有目标区标签，且 OUP 页面被 Cloudflare 拦截，因此只作“目标锚点有效但当前不可用”的补充证据。
- **Classification of Natural and Non-natural Seismic Events Based on Convolutional Neural Networks**（DOI `10.1145/3804601.3804634`）：AnySearch 摘要报告三分类 accuracy 87.63%、爆破 recall 73.17%；ACM 页面被安全验证页拦截，未计入 15 篇。

## 方法决策矩阵

评分采用 1–5，5 为最好；“风险”列 5 表示风险最低。

| 候选 | 预期收益 | 数据现成 | 本周可做 | 延迟/内存 | 跨包证据 | 过拟合风险 | 可回滚 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 训练折中心化 + NCM/多原型余弦头 | 4 | 5 | 5 | 5 | 3 | 4 | 5 | 首选最小反证 |
| 每类 top-m 温度化局部邻域、等类先验 | 4 | 5 | 5 | 5 | 3 | 3 | 5 | 与原型头同一网格比较 |
| 距离/margin/邻居一致性置信门控 | 3 | 5 | 4 | 5 | 4 | 4 | 5 | 候选头有增益后再启用 |
| post-hoc 类先验/logit adjustment | 3 | 5 | 5 | 5 | 2 | 2 | 5 | 仅嵌套 CV，小范围温度 |
| LDAM/BCL/MiSLAS 端到端重训编码器 | 3 | 1 | 1 | 2 | 3 | 1 | 3 | DiTing 数据到位后延期 |
| EQTypeNet/新 CNN 替换 SeismicXM | 3 | 1 | 1 | 2 | 3 | 1 | 3 | 本周拒绝 |

## 证据综合与本轮实现边界

1. 地震论文的共同结论不是“某种网络普遍更强”，而是地域、爆破方式、路径和类别组成会造成显著迁移损失；本项目必须让第 1→2 轮跨包结果成为硬门槛，不能只优化第 1 轮宏召回。
2. 长尾文献最一致、且与现状最匹配的结论是：通用表征可以保持冻结，先用轻量、可回滚的分类头校正类别边界。当前 SeismicXM 已经是昂贵且验证过的表征，不应在 389 条历史 T3 上重新训练。
3. 最值得首先证否的不是复杂损失，而是当前生产头尚未测试的两个简单因素：训练折中心化，以及每类等先验的原型/局部邻域聚合。
4. 单质心可能掩盖不同区域/震源机制的类内多模态，因此必须与少量多原型和局部 top-m 对照；若不同折的原型数极不稳定，则按预注册判定证据不足。
5. 置信门控的作用是避免候选在训练流形外替换已经很强的生产 kNN；距离与 margin 不是校准概率，必须在外层训练折内确定门槛。
6. 08 只做一次终检，且没有第 5 类。若开发门槛未过，本轮不得提取/查看 08 特征来反向挑参数。

本轮据此进入最小可证伪实验：固定 SeismicXM 特征与嵌套分层划分，仅比较生产 kNN、中心化 NCM、多原型和每类局部邻域，并在第 1→2 轮跨包门槛下决定是否允许进行 08 终检。

## 实验反馈

论文证据正确预警了“同域高分不能外推”，而本轮实验给出了更直接的项目证据：候选在第 1 轮 5×5 嵌套验证上把 balanced accuracy 从 `0.865636` 提高到 `0.908364`，却在第 1→2 轮从生产基线 `187/189` 崩至 `121/189`；R1 OOF margin/support 门控也只恢复到 `130/189`。25 个外层折中 23 个选择中心化家族，说明 SimpleShot 风格中心化在同包内有稳定诱惑，但不具备本项目所需的跨包稳健性。

事后成对诊断显示，移除中心化可将对应 hybrid 从 `121/189` 恢复到 `178/189`、top-m 从 `133/189` 恢复到 `182/189`，仍未达到预注册 `186/189`。因此本轮同时拒绝训练折中心化和当前等类局部/原型头，不把包内宏召回增益误报为可上线收益，也不消耗 08 终检。
