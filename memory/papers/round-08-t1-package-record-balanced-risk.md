# 研究轮次 08：T1 包—记录层级等权风险

- 日期：2026-08-12
- 状态：AnySearch MCP 发现、Playwright 原始页面核验和实验预注册已完成
- 对应实验：`memory/experiments/013-t1-package-record-balanced-distillation.md`
- 父提交：`fcfac3d6fd8b596892c98f10eafbbd0157f6c39d`
- 本轮问题：基础 `KD+hard` 在 R2、08 和 7/7 长记录上改善，却在 R1 四种完整包口径全部回退。训练目标当前按窗口平均，使包、记录和窗口长度共同决定风险贡献。只把训练风险改成“两个训练包等权、每包内记录等权、记录内窗口等权”，能否消除该层级偏差并在三包 × 四口径 12 单元中稳健替换 `g7`？

## 1. 检索、去重与浏览协议

先通过本机 AnySearch MCP 的 `academic.search`、`academic.preprint` 和普通检索发现候选，再用本机 Playwright `1.62.1` 驱动 Chrome 串行打开 arXiv、PMLR、NeurIPS、CVF、JMLR、Frontiers 或出版社原始页面。Cloudflare Browser MCP 当前鉴权返回 `401`，因此本轮不把它写成成功核验工具。搜索摘要只用于发现；标题、年份、方法和主要实验结论至少从原始论文页、正式出版页或论文正文之一复核。

检索式覆盖：

```text
seismic phase picking cross dataset generalization transfer learning regional datasets
domain balanced sampling equal domain risk empirical domain generalization
Fishr SWAD MixStyle SagNet RSC risk extrapolation JTT
hierarchical sampling records windows equal weighting time series empirical risk
cross dataset evaluation seismic phase picking transfer learning
```

去重规则：第 3 轮已计数 IRM、DomainBed、GroupDRO 和 CORAL；第 4–6 轮已计数 Which Picker Fits My Data、SeisBench、OBSTransformer、PickBlue、区域 picker 和 Nabro transfer learning；第 7 轮已计数全部蒸馏论文。这些论文可作为背景，但不计入本轮 15 篇。以下 15 篇均为此前未计数的原始实验研究。

## 2. 十五篇原始实验论文

| # | 论文与标识 | 数据/主要指标 | 与本项目的关系 | 不采用或外推边界 |
|---:|---|---|---|---|
| 1 | **Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization**，ICML 2022，arXiv [`2109.02934`](https://arxiv.org/abs/2109.02934) | DomainBed 多个图像域；以 OOD accuracy 比较 ERM 和 DG 方法，匹配各训练域的梯度方差 | 直接说明多源域不应只被无结构汇总，域级风险/梯度需要显式对待 | 新增梯度协方差正则和强度超参数；当前只有两个训练包，首轮不引入 |
| 2 | **SWAD: Domain Generalization by Seeking Flat Minima**，NeurIPS 2021，arXiv [`2102.08604`](https://arxiv.org/abs/2102.08604) | PACS、VLCS、OfficeHome、TerraIncognita、DomainNet；报告跨域 accuracy，受控协议下优于普通 ERM | 证明简单 ERM 的优化轨迹也会影响 OOD，而复杂 DG 并非唯一方向 | 会改变 checkpoint 聚合、早停区间和训练阶段；违反固定 10 epoch 单变量边界 |
| 3 | **Domain Generalization with MixStyle**，ICLR 2021，arXiv [`2104.02008`](https://arxiv.org/abs/2104.02008) | PACS、OfficeHome、DomainNet 等；跨域分类和 ReID accuracy/mAP | 跨训练域混合特征统计可扩大源域覆盖 | 图像“风格”与地震幅值/频谱未建立等价性；会改变特征分布，不采用 |
| 4 | **Reducing Domain Gap by Reducing Style Bias (SagNet)**，CVPR 2020，arXiv [`1910.11645`](https://arxiv.org/abs/1910.11645) | PACS、OfficeHome、DomainNet 等；跨域 accuracy | 支持把采集风格和任务内容分开考虑 | 需要 style/content 分支和对抗目标；不是风险测度的最小修正 |
| 5 | **Self-Challenging Improves Cross-Domain Generalization**，ECCV 2020，arXiv [`2007.02454`](https://arxiv.org/abs/2007.02454) | PACS、VLCS、OfficeHome、TerraIncognita；跨域 accuracy | 说明 ERM 容易依赖源域主导特征 | 会按梯度丢弃特征并引入比例参数；无法隔离层级权重效应 |
| 6 | **Out-of-Distribution Generalization via Risk Extrapolation (REx)**，ICML 2021，arXiv [`2003.00688`](https://arxiv.org/abs/2003.00688) | 合成因果任务及跨域分类；比较平均风险、域风险方差和 OOD error | 与本轮最直接：先定义可比的域风险，再讨论域间差异 | V-REx 还要惩罚域风险方差并选择系数；本轮只建立等权域风险，不加惩罚 |
| 7 | **Just Train Twice: Improving Group Robustness without Training Group Information**，ICML 2022，arXiv [`2107.09044`](https://arxiv.org/abs/2107.09044) | Waterbirds、CelebA、MultiNLI、CivilComments；worst-group accuracy | 证明平均性能可掩盖少数群组回退，和本项目 12 单元准入一致 | 两阶段训练并依赖首阶段错误重加权；会使用标签结果形成新自由度，不采用 |
| 8 | **Learning to Generalize: Meta-Learning for Domain Generalization (MLDG)**，AAAI 2018，arXiv [`1710.03463`](https://arxiv.org/abs/1710.03463) | VLCS、PACS 等；模拟 source/meta-test 域并报告跨域 accuracy | 说明训练包应保留域身份，不能只视作同分布样本池 | 需要内外层优化和元学习步长；两个源包下方差大，超出单变量范围 |
| 9 | **Generalizing Across Domains via Cross-Gradient Training**，ICLR 2018，arXiv [`1804.10745`](https://arxiv.org/abs/1804.10745) | Digits、字符和语音/图像多域任务；跨域 classification accuracy | 说明域标签可用于训练时构造域扰动 | 需要域分类器和输入梯度增强；地震波形扰动的物理合法性未验证 |
| 10 | **Episodic Training for Domain Generalization**，ICCV 2019，arXiv [`1902.00113`](https://arxiv.org/abs/1902.00113) | PACS、VLCS、OfficeHome；跨域 accuracy | 强调聚合源域 ERM 是强基线，同时可通过模拟域错配训练 | 会改变网络分解和训练 episode；当前先修正更基础的贡献不等问题 |
| 11 | **SelfReg: Self-supervised Contrastive Regularization for Domain Generalization**，ICCV 2021，arXiv [`2104.09841`](https://arxiv.org/abs/2104.09841) | PACS、VLCS、OfficeHome、TerraIncognita；跨域 accuracy | 说明类内表征约束可能提升域稳健性 | 引入表征混合、对比正则和额外权重；与稠密 N/P/S loss 混杂过大 |
| 12 | **Learning Explanations that are Hard to Vary (AND-mask)**，ICLR 2021，arXiv [`2009.00329`](https://arxiv.org/abs/2009.00329) | 合成不变性、语言和图像任务；OOD/generalization accuracy | 直接警示跨样本简单平均梯度可能形成拼接式解 | AND 阈值会稀疏梯度且需新超参数；本轮不改变梯度选择规则 |
| 13 | **Domain-Adversarial Training of Neural Networks (DANN)**，JMLR 2016，arXiv [`1505.07818`](https://arxiv.org/abs/1505.07818) | MNIST/USPS/SVHN、Office 等；target accuracy | 证明域可辨识表征会伤害迁移 | 原方法使用未标注目标域数据，违反 held-out 包完全关闭；明确拒绝 |
| 14 | **MetaReg: Towards Domain Generalization using Meta-Regularization**，NeurIPS 2018，[正式页](https://proceedings.neurips.cc/paper/2018/hash/647bba344396e7c8170902bcf2e15551-Abstract.html) | PACS、VLCS 等；跨域 classification accuracy | 支持从多个源域学习能跨域泛化的正则 | 需要独立 meta-train/meta-test 域和正则网络；两个源包不足以稳定展开 |
| 15 | **Using a Deep Neural Network and Transfer Learning to Bridge Scales for Seismic Phase Picking**，GRL 2020，DOI [`10.1029/2020GL088651`](https://doi.org/10.1029/2020GL088651) | EGS Collab 与原 PhaseNet 尺度差异数据；比较 off-the-shelf 与重训练后的检出/拾取表现 | 地震原始实验直接证明域和采集尺度会改变 picker 表现，训练组成不能被窗口数偶然主导 | 论文使用目标域标注迁移；本项目 held-out 包不可参与训练，只能做 source-only DG |

Playwright 实际成功打开并读取 13 个 arXiv 原始页、NeurIPS MetaReg 正式页和 Frontiers 本地 picker 正式页；Wiley 的 GRL 正文被 Cloudflare 挡在“请稍候”页，因此该篇身份与结论用 DOI 正式元数据、AnySearch 原文索引片段和论文公开摘要交叉核验，不冒充成功读取 Wiley 全文。

## 3. 证据综合

1. **多源训练必须先定义目标风险。** Fishr、REx、MLDG、CrossGrad 和 Episodic Training 都保留域身份；直接按全部窗口平均等价于让窗口更多的包和记录拥有更大先验权重。
2. **复杂 DG 并没有在受控协议下稳定压倒强 ERM。** SWAD 和 DomainBed 背景证据都要求先把基线、模型选择和风险计算做对。本项目已观察到具体的层级窗口偏差，因此先修正测度比叠加表征正则更可解释。
3. **地震拾取确实存在强域移。** 跨尺度 transfer learning、本地定制和既有跨数据集 benchmark 都表明 off-the-shelf picker 的表现受区域、仪器、事件尺度和训练组成影响；但目标域标签迁移不能用于 LOPO held-out。
4. **最坏域和最坏群组不能被均值覆盖。** JTT、REx 和本项目 `KD+hard` 的 8/12 正向共同说明平均上涨不是稳健录取证据。
5. **本轮只修正一个可观测偏差。** 不使用 held-out 数据，不改变模型、KD/hard 比例、epoch、优化器、阈值、teacher、窗口集合或后处理。

## 4. 冻结候选

训练两包记作 `p`，包内记录记作 `r`，记录窗口记作 `w`。保留全部既有窗口，仅给每窗赋权：

```text
weight(p, r, w) = total_training_windows
                  / (number_of_training_packages
                     * records_in_package(p)
                     * windows_in_record(p, r))
```

由此得到：

- 全体窗口平均权重严格为 `1`；
- 两个训练包总权重相等；
- 每包内所有记录总权重相等；
- 每条记录内部所有窗口等权；
- mini-batch 内先计算逐样本 dense CE，再乘窗口权重并取 batch mean；
- `KD+hard` 的 KD 与 hard 两项使用同一个窗口权重；
- `window-erm` 保持全 1 权重，作为实现回归检查，不重新训练为候选。

## 5. 实验边界与反证条件

只训练一个新候选：`loss=kd-hard, risk=package-record-balanced`。固定三折 LOPO、`PhaseNet(diting)`、10 epoch、AdamW `1e-4`、batch 2、30 秒 stride、`0.7 KD + 0.3 hard`、P/S sigma `0.2/0.3s`、冻结 BatchNorm/Dropout 和完整生产后处理。

准入仍为三包 × 四数量罚 12 单元相对 `g7` 全部不下降且至少一项严格上升。必须同时报告完整覆盖、逐包 FP/FN、P/S 时差分、受损文件和七条长记录。若失败，不得在同一三包上继续调包权、记录权、KD/hard 比例、epoch、阈值或后处理；下一轮必须重新研究并预注册实质不同机制。
