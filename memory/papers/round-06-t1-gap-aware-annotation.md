# 研究轮次 06：T1 gap-aware annotation 与显式缺失掩码

- 日期：2026-08-11
- 状态：AnySearch 发现、Playwright 逐篇核验和方法决策已完成；尚未运行第 6 轮模型实验
- 对应预注册：`memory/experiments/006-t1-gap-aware-annotation.md`
- 基准提交：`6d0f50baf33f37175c832e2391b714e083396b07`
- 本轮问题：第 5 轮证明最终 `Pick` 列表固定 margin 删除无法修复零填充缺口造成的远程诱发与丢失。本轮只研究一个更上游、可证伪的问题：在正常阈值、条件式 force-pair 和 dedup 之前消费 `Waveform.gaps`，局部屏蔽 P/S annotation，是否足以阻止缺口候选进入后处理，同时保持物理缺口 10 秒外概率与最终输出不变？

## 1. 检索与浏览协议

本轮先用 AnySearch 普通检索，再使用 `academic.search` 与 `academic.preprint` 两个学术子域，围绕以下关键词发现候选：

1. seismic waveform gap、channel dropout、missing channel、phase picking augmentation；
2. explicit observation mask、partial/gated convolution、zero-filled hole artifact；
3. block/blackout missing time series、irregular sampling、mask-aware sequence model；
4. probabilistic/diffusion imputation 与缺失模式对下游任务的影响。

随后用 Playwright 逐篇实际打开原论文页、arXiv/arXiv HTML、Nature/OSTI/PMLR/AAAI 正式页或作者正式代码仓。共核验 17 篇新的原始实验研究，没有综述计数，也没有把第 5 轮已经完整计数的 17 篇再次计入本轮。

访问边界：

- Nabro 论文 DOI 跳转到 Wiley 后返回 Cloudflare `403`；已用 AnySearch DOI 元数据、论文摘要和作者正式代码仓 `https://github.com/sachalapins/U-GPD` 交叉核验，阅读状态标为“摘要 + 正式代码仓”，不冒充出版社全文。
- Raindrop 的 OpenReview 页面停在浏览器验证；已改用原始 arXiv `2110.05357` 和作者实验室正式论文页核验。
- 其余 15 篇的原始页面均由 Playwright 返回可读内容；其中 Xi-Net、PhaseNet-DAS、GRU-D、SAITS、GP-VAE 和 PrimeNet 还读取了正文或正式全文页面。

## 2. 地震与相邻传感器任务的直接证据

### 1. Seismic signal augmentation to improve generalization of deep neural networks

- 作者/年份：Weiqiang Zhu、S. Mostafa Mousavi、Gregory C. Beroza；2020。
- 标识：DOI `10.1016/bs.agph.2020.07.003`。
- Playwright 实际打开：`https://www.osti.gov/biblio/1774650`。
- 阅读状态：OSTI 正式摘要页；接受稿目录和可访问文本确认了增强类型。
- 问题与方法：针对地震深度模型训练数据不完整与域外泛化，系统整理并实验验证 random shift、事件叠加、噪声叠加、false-positive noise、channel dropout、resampling 等增强。
- 数据与结果：论文报告这些地震专用增强能减少偏差、改善未见样本表现；其中 channel dropout 会把一个或两个分量置零，使模型在训练阶段学会区分“缺失观测”与有效信号。
- 局限：这是训练机制，不是对一个已经冻结的 PhaseNet 在推理后把概率局部置零；也没有证明时间块缺口 10 秒外的概率严格不变。
- 本项目关系：支持未来 gap augmentation/蒸馏，但反对把未训练过的零值区间当作天然中性输入。
- 本周决策：部分采用问题定义；训练方案延期，不进入本轮候选。

### 2. Xi-Net: Transformer Based Seismic Waveform Reconstructor

- 作者/年份：Anshuman Gaharwar、Parth Parag Kulkarni、Joshua Dickey、Mubarak Shah；ICIP 2023，arXiv 页面 2024。
- 标识：DOI `10.1109/ICIP49359.2023.10222465`；arXiv `2406.16932`。
- Playwright 实际打开：`https://arxiv.org/html/2406.16932`。
- 阅读状态：arXiv HTML 全文。
- 问题与方法：双编码器分别处理时间域与 DTFT 实/虚部，1D shifted-window Transformer 解码重建 120 秒波形中的随机缺口。
- 数据与实验：13,696 条 AFTAC/USArray 信号，11,000 训练、2,696 测试；随机 0.5–1 秒缺口；V100 单卡训练 80 epoch、约 7 小时。
- 已核验结果：相对未填充输入，Xi-Net 的 DFD `0.196→0.182`、MRD `0.3467→0.1861`、MAE `0.0955→0.0813`、RMSE `0.135→0.116`；略优于训练 27 天的 21 层 1D CNN 基线。作者仍明确展示“只识别模式但未完整重建”和“完全失败”两类样本，并把 gap 边缘陡坡抖动列为未来工作。
- 局限：只评价波形重建距离，没有验证 P/S 到时、数量罚、T2 幅值或 T3 分类；还包含滤波、上采样和专门训练。
- 本项目关系：说明重建可行但不是无风险捷径；当前比赛链路不能用波形相似指标替代官方到时安全性。
- 本周决策：拒绝本轮接入；仅作为未来独立重建基线。

### 3. Acoustic Scene Classification Using Multichannel Observation with Partially Missing Channels

- 作者/年份：Keisuke Imoto；2021。
- 标识：DOI `10.23919/EUSIPCO54536.2021.9616170`；arXiv `2105.01836`。
- Playwright 实际打开：`https://arxiv.org/abs/2105.01836`。
- 阅读状态：原始摘要页。
- 问题与方法：研究分布式麦克风因设备故障或网络丢包产生整通道或局部块缺失时，声场分类遭受的信息损失与训练/测试失配；提出简单的缺失通道数据增强。
- 数据与结果：在多通道声景分类实验中比较缺失模式，并验证与缺失模式匹配的增强能改善分类鲁棒性。
- 局限：目标是窗口级分类，不要求 0.1/0.2 秒峰位；增强需要重训。
- 本项目关系：与三分量单分量 gap 最接近的相邻传感器证据，说明缺失模式应在训练或模型输入中显式表示，而非仅靠零值代替。
- 本周决策：采用“显式缺失语义”原则；训练延期。

### 4. Seismic arrival-time picking on distributed acoustic sensing data using semi-supervised learning

- 作者/年份：Weiqiang Zhu、Ettore Biondi、Jiaxuan Li 等；2023。
- 标识：DOI `10.1038/s41467-023-43355-3`。
- Playwright 实际打开：`https://www.nature.com/articles/s41467-023-43355-3`。
- 阅读状态：Nature 开放全文。
- 问题与方法：先用预训练 PhaseNet 生成噪声伪标签，再用 GaMMA 跨通道关联清洗，训练消费二维时空上下文的 PhaseNet-DAS；第二轮用 PhaseNet-DAS v1 再生成伪标签。
- 数据与实验：Long Valley 与 Ridgecrest 四条 DAS 光缆；正文还对 180 小时、10,000 通道连续数据进行实验。
- 已核验结果：可关联 picks 比率从 PhaseNet 的 `59%–69%` 提高到 `89%–92%`；v1 检出事件数为 PhaseNet 的 2–5 倍，v2 又比 v1 增加 25%–50%；P/S 差分到时误差均值约 `0.001/0.005s`、标准差约 `0.06/0.25s`；连续数据中检出事件约为传统台网目录的 2–3 倍。
- 局限：收益来自数千同步通道、伪标签、关联物理约束和训练；单站三分量 API 没有这些条件。
- 本项目关系：说明真正可靠的伪触发抑制要在模型/关联层利用额外上下文；一个局部 annotation mask 只能阻止被遮区峰参与后处理，不能修复模型已经在远处产生的概率漂移。
- 本周决策：采用其上游一致性思想；模型本身拒绝当前接入。

### 5. A Little Data Goes a Long Way: Automating Seismic Phase Arrival Picking at Nabro Volcano With Transfer Learning

- 作者/年份：Sacha Lapins、Berhe Goitom、J.-M. Kendall 等；2021。
- 标识：DOI `10.1029/2021JB021910`。
- Playwright 实际打开：DOI/Wiley 入口（403）与作者正式代码仓 `https://github.com/sachalapins/U-GPD`。
- 阅读状态：可靠摘要 + 正式代码、权重和数据说明。
- 问题与方法：以既有 picker 的特征层为基础，用少量 Nabro 域内数据迁移得到 U-GPD。
- 数据与实验：35 天的 2,498 个事件波形；800 条事件/噪声测试；另处理七台站 14 个月连续数据。
- 已核验结果：相对从头训练，迁移在 500 条等小训练集上更少过拟合；U-GPD 比两个域外现成模型有更高分类准确率和更小到时残差；连续数据检出 31,387 个满足至少 4P+1S 的事件，高于原基座 26,808 和人工目录 2,926，单 GPU 少于 4 小时完成。
- 局限：仍需域内标签和训练，且使用多台站事件门槛。
- 本项目关系：再次否定“冻结域外模型 + 一条推理时启发式”能够等价替代缺失模式训练。
- 本周决策：延期到真实 gap/DiTing 数据阶段。

## 3. 显式 mask 的结构性证据

### 6. Image Inpainting for Irregular Holes Using Partial Convolutions

- 作者/年份：Guilin Liu、Fitsum A. Reda、Kevin J. Shih、Ting-Chun Wang、Andrew Tao、Bryan Catanzaro；2018。
- 标识：DOI `10.1007/978-3-030-01252-6_6`；arXiv `1804.07723`。
- Playwright 实际打开：`https://arxiv.org/abs/1804.07723`。
- 阅读状态：原始摘要页；NVIDIA 正式项目页交叉核验。
- 问题与方法：普通卷积会把洞内替代值当成有效输入并产生颜色/边界 artifact；partial convolution 只在 mask 标为有效的位置卷积并按有效数归一化，每层同步更新 mask。
- 数据与实验：Places2 等图像；正式项目提供 12,000 个覆盖六档缺失面积、含/不含边界约束的测试 mask，并报告定量和定性优于既有不规则洞修复方法。
- 局限：模型从训练起每一层都消费 mask；不能把它简化成现有 PhaseNet 输出端一次置零。
- 本项目关系：这是本轮最关键的结构证据：要阻止零填充值污染特征，mask 必须进入卷积计算；annotation 层只能做候选 veto，无法回溯改写特征。
- 本周决策：采用为硬失败条件依据；不实现新网络。

### 7. Free-Form Image Inpainting With Gated Convolution

- 作者/年份：Jiahui Yu、Zhe Lin、Jimei Yang、Xiaohui Shen、Xin Lu、Thomas Huang；2019。
- 标识：DOI `10.1109/ICCV.2019.00457`；arXiv `1806.03589`。
- Playwright 实际打开：`https://arxiv.org/abs/1806.03589`。
- 阅读状态：原始摘要页。
- 问题与方法：gated convolution 为每个位置、每个通道学习动态特征选择，推广只由二值 mask 决定的 partial convolution；配合 SN-PatchGAN 处理任意形状缺失区。
- 数据与结果：在数百万图像上训练，对自动和交互式自由形状修复报告比先前方法更高质量、更灵活的结果，并发布代码和模型。
- 局限：门控是训练得到的，不存在可直接移植到冻结 PhaseNet annotation 的参数。
- 本项目关系：说明“该屏蔽多少上下文”本身是可学习问题，事后固定 guard 不具有同等能力。
- 本周决策：拒绝本周重构；采用其对固定局部规则能力边界的解释。

## 4. 显式缺失模式与不规则时序证据

### 8. Recurrent Neural Networks for Multivariate Time Series with Missing Values（GRU-D）

- 作者/年份：Zhengping Che、Sanjay Purushotham、Kyunghyun Cho、David Sontag、Yan Liu；2018。
- 标识：DOI `10.1038/s41598-018-24271-9`。
- Playwright 实际打开：`https://www.nature.com/articles/s41598-018-24271-9`。
- 阅读状态：Scientific Reports 全文。
- 问题与方法：同时输入 observation mask 和距上次观测的时间间隔，并用可训练衰减作用于输入与隐状态；利用 informative missingness，而不是先填补再假装完整。
- 数据与结果：MIMIC-III、PhysioNet 和合成分类数据上达到当时 SOTA，并优于均值/前向填充及普通 GRU 基线。
- 局限：医疗缺失模式可能与标签相关；当前 MiniSEED gap 是传输/仪器缺失，不应把“是否缺失”当作地震相位证据。
- 本项目关系：支持保留 `Waveform.gaps` 元数据；也说明真正 mask-aware 需要模型架构和训练，局部概率置零只是一项安全 veto。
- 本周决策：部分采用；不替换生产模型。

### 9. BRITS: Bidirectional Recurrent Imputation for Time Series

- 作者/年份：Wei Cao、Dong Wang、Jian Li、Hao Zhou、Lei Li、Yitan Li；2018。
- 标识：NeurIPS 2018；arXiv `1805.10572`。
- Playwright 实际打开：`https://arxiv.org/abs/1805.10572`。
- 阅读状态：原始摘要页；NeurIPS 正式论文文本交叉核验。
- 问题与方法：把缺失值作为双向 RNN 图中的可学习变量，同时联合分类/回归，不预设线性或平滑生成过程。
- 数据与结果：空气质量、医疗和人体活动定位三套真实数据上，imputation 与下游分类/回归均优于当时基线。
- 局限：训练与双向递归成本高；输出是重建值，不保证地震相位尖峰的亚采样到时不偏。
- 本项目关系：支持“缺失处理必须由观测两侧共同约束”，但不支持本周临时重建。
- 本周决策：延期。

### 10. SAITS: Self-Attention-based Imputation for Time Series

- 作者/年份：Wenjie Du、David Côté、Yan Liu；2023。
- 标识：DOI `10.1016/j.eswa.2023.119619`；arXiv `2202.08516`。
- Playwright 实际打开：`https://arxiv.org/abs/2202.08516`。
- 阅读状态：arXiv 摘要与 HTML 正文。
- 问题与方法：两个 diagonally-masked self-attention block，利用 missingness 和 attention map 动态加权；联合 Masked Imputation Task 与 Observed Reconstruction Task，避免 Transformer 只复原已观测值而忽略缺失值。
- 数据与结果：四套公开真实时序，按 MCAR 人工缺失；论文报告 imputation 精度达到新 SOTA、训练/推理更高效，并改善不完整数据上的下游模式识别。
- 局限：训练缺失为均匀 MCAR，而本项目是连续 blackout gap；目标是平均重建误差，不是到时峰安全。
- 本项目关系：说明“仅把缺失位置置零”不是训练目标；模型必须同时看到 mask 并对缺失位置承担损失。
- 本周决策：拒绝当前接入；未来 gap 蒸馏可参考双任务损失。

### 11. CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation

- 作者/年份：Yusuke Tashiro、Jiaming Song、Yang Song、Stefano Ermon；2021。
- 标识：NeurIPS 2021；arXiv `2107.03502`。
- Playwright 实际打开：`https://arxiv.org/abs/2107.03502`。
- 阅读状态：原始摘要页。
- 问题与方法：条件扩散模型显式以已观测数据为条件，学习缺失部分的概率分布，并可输出多种合理补全。
- 数据与结果：医疗和环境数据上，概率 imputation 指标比既有方法改善 40%–65%；确定性误差改善 5%–20%，也支持插值与预测。
- 局限：多步采样和专门训练成本高；“合理样本”不等于震相时间/幅值无偏。
- 本项目关系：说明缺失重建存在不确定性；比赛 API 不能从生成分布任取一条波形再假定拾取安全。
- 本周决策：拒绝本周生产，保留为长期研究。

### 12. Neural Controlled Differential Equations for Irregular Time Series

- 作者/年份：Patrick Kidger、James Morrill、James Foster、Terry Lyons；2020。
- 标识：NeurIPS 2020；arXiv `2005.08926`。
- Playwright 实际打开：`https://arxiv.org/abs/2005.08926`。
- 阅读状态：原始摘要页。
- 问题与方法：用 controlled differential equation 让连续时间隐藏状态在每次新观测到来时受控更新，直接处理部分观测、不规则采样和变长时序。
- 数据与结果：多套实证数据上相对 ODE/RNN 基线达到 SOTA，并给出通用逼近性质。
- 局限：需要训练新的连续时间模型；它解决不规则观测表示，不是冻结卷积 picker 的输出修补。
- 本项目关系：证明“跳过无观测区、只在观测到来时更新”是可行架构，但不能由 annotation 层零值 veto 等价实现。
- 本周决策：延期。

### 13. Graph-Guided Network for Irregularly Sampled Multivariate Time Series（RAINDROP）

- 作者/年份：Xiang Zhang、Marko Zeman、Theodoros Tsiligkaridis、Marinka Zitnik；2022。
- 标识：ICLR 2022；arXiv `2110.05357`。
- Playwright 实际打开：OpenReview 验证页与 `https://arxiv.org/abs/2110.05357`。
- 阅读状态：arXiv 原始页；作者实验室全文/项目页交叉核验。
- 问题与方法：每个样本建立动态传感器图，用消息传递和时间注意力直接建模不对齐观测及变化的传感器依赖，不先统一插值。
- 数据与结果：三套医疗/人体活动数据，F1 最多提高 11.4 个绝对百分点；在 leave-sensor-out 故障场景仍占优，论文正文报告活动识别准确率约提高 9.3 个百分点。
- 局限：收益依赖跨传感器图和训练；本项目只有三个分量且没有独立站间图。
- 本项目关系：支持对单分量缺失做显式结构处理；也提示单分量 gap 与三分量全 gap 应分别验证。
- 本周决策：采用验证分组思想；模型拒绝当前接入。

### 14. GP-VAE: Deep Probabilistic Time Series Imputation

- 作者/年份：Vincent Fortuin、Dmitry Baranchuk、Gunnar Rätsch、Stephan Mandt；2020。
- 标识：AISTATS 2020，PMLR 108:1651–1661。
- Playwright 实际打开：`https://proceedings.mlr.press/v108/fortuin20a.html`。
- 阅读状态：PMLR 正式全文入口和摘要。
- 问题与方法：VAE 的低维 latent 随时间按 Gaussian Process 平滑演化，以结构化变分近似处理缺失并提供不确定性。
- 数据与结果：计算机视觉和医疗高维数据上优于经典和当时深度 imputation 方法。
- 局限：核心先验是低维 latent 平滑；地震 P/S onset 是尖锐非平稳结构，错误平滑可能直接移动到时。
- 本项目关系：支持记录不确定性，不支持平滑重建作为默认输入。
- 本周决策：拒绝本周接入。

### 15. NRTSI: Non-Recurrent Time Series Imputation

- 作者/年份：Siyuan Shan、Yang Li、Junier B. Oliva；2021。
- 标识：arXiv `2102.03340`。
- Playwright 实际打开：`https://arxiv.org/abs/2102.03340`。
- 阅读状态：原始摘要页。
- 问题与方法：把 `(time, data)` 视为 permutation-equivariant set，采用分层、非递归的缺失补全；支持不规则采样、多模态随机补全和部分维度观测。
- 数据与结果：广泛时序 imputation benchmark 上报告 SOTA，尤其针对稀疏观测下传统方法退化的问题。
- 局限：仍是训练后的重建器，且多模态补全会把比赛输出变成随机或需额外选择的链路。
- 本项目关系：可作为未来 blackout 重建对照，但不是当前固定 annotation mask 的证据。
- 本周决策：延期。

### 16. Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models（SSSD）

- 作者/年份：Juan Miguel Lopez Alcaraz、Nils Strodthoff；2022/2023 版本。
- 标识：TMLR 2022；arXiv `2208.09399`。
- Playwright 实际打开：`https://arxiv.org/abs/2208.09399`。
- 阅读状态：原始摘要页。
- 问题与方法：条件扩散结合 structured state space model，利用长程依赖进行 imputation 与预测。
- 数据与结果：多套数据和多种缺失情形达到或超过概率 imputation SOTA；特别报告在 prior methods 失效的连续 blackout missing 场景仍能给出有意义结果。
- 局限：这是最贴近连续 gap 的通用时序证据，但仍需训练、多步扩散和任务专用验证，没有地震到时/幅值保真结论。
- 本项目关系：若未来独立重建轮启动，应把 blackout 而非 MCAR 作为主验证；当前不用于事后救援。
- 本周决策：延期。

### 17. PrimeNet: Pre-training for Irregular Multivariate Time Series

- 作者/年份：Ranak Roy Chowdhury、Jiacheng Li、Xiyuan Zhang、Dezhi Hong、Rajesh K. Gupta、Jingbo Shang；2023。
- 标识：DOI `10.1609/aaai.v37i6.25876`。
- Playwright 实际打开：`https://ojs.aaai.org/index.php/AAAI/article/view/25876`。
- 阅读状态：AAAI 正式页与摘要。
- 问题与方法：时间敏感对比学习 + 数据重建预训练；按原始采样密度生成 triplet，并始终遮住固定时间长度而非固定点数，使不同局部密度下的重建难度更一致。
- 数据与结果：医疗和 IoT 自然不规则/异步数据上的分类、插值和回归显著优于 SOTA。
- 局限：需要无标签预训练和下游微调；“固定时间块 mask”适合未来增强设计，但本周没有可冻结训练划分。
- 本项目关系：直接支持未来按秒而非按样点设计 gap augmentation；不支持修改当前冻结权重。
- 本周决策：部分采用实验单位；训练延期。

## 5. 证据综合

17 篇新证据形成七条与本轮直接相关的结论：

1. **零值替代不是显式缺失语义。** Partial/gated convolution、GRU-D、SAITS 都把 mask 放进模型计算或训练目标；只把缺失样本写成零会让普通卷积把替代值当观测。
2. **annotation 层 mask 的能力是单向的。** 它能让 gap 区间 P/S 概率不再越过正常阈值或 `0.03` force-pair floor，但区间外值与 raw gapped annotation 完全相同，不能修复上游卷积已造成的远程漂移。
3. **连续 blackout 与随机 MCAR 不同。** SSSD 和 Xi-Net 专门强调连续块缺失更难；SAITS 的均匀 MCAR 结果不能直接外推到 MiniSEED gap。
4. **可靠缺失处理普遍需要训练或额外上下文。** channel dropout、U-GPD、PhaseNet-DAS、RAINDROP、PrimeNet 的增益都来自匹配数据、显式 mask、站间/分量上下文或重新训练。
5. **重建指标不能替代比赛指标。** Xi-Net 即使在 MAE/RMSE 上改善，仍有失败样本和边缘抖动；CSDI/GP-VAE/NRTSI 的概率或平均重建也没有证明 P/S 峰、幅值与类别无偏。
6. **本轮最便宜的反证在概率层。** 若 raw gapped annotation 在物理 gap 10 秒外已经出现超过数值容差的改变、阈值/floor 穿越或峰集合改变，任何局部 annotation mask 都不可能满足“远处概率与最终 picks 不变”。
7. **即使本轮通过，也只是一条安全 veto。** 它不等价于 partial convolution、GRU-D 或训练后 channel dropout，只能进入生产实现评审，不能宣称模型已具备真正 gap-aware 表征。

## 6. 方法决策矩阵

评分为 `1`（差/高风险）到 `5`（好/低风险）。“收益”仅指未来含 gap 输入的鲁棒性，不虚构三套无 gap 历史包分数。

| 方法 | 预期鲁棒收益 | 现有数据 | 本周可完成 | 训练/实现成本 | 延迟/内存 | 到时/噪声安全 | 跨包与过拟合 | 回滚/API 兼容 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P/S annotation 物理区间及固定 guard 置零 | 3 | 5 | 5 | 5 | 5 | 3 | 5 | 5 | **唯一最小反证候选** |
| annotation 置 NaN/invalid | 2 | 5 | 5 | 5 | 5 | 1 | 4 | 4 | `find_peaks`/聚合语义不稳定，拒绝 |
| 含 gap 的模型滑窗全部 veto | 2–3 | 4 | 2 | 2 | 3 | 2 | 4 | 2 | 会丢掉宽上下文，映射复杂，拒绝本轮 |
| 仅禁止 gap 相关 force-pair | 2 | 5 | 5 | 5 | 5 | 2 | 5 | 5 | 只修一个分支，正常阈值/远程漂移仍在，拒绝单独候选 |
| channel/gap augmentation 后微调或蒸馏 | 4 | 2 | 2 | 2 | 4 | 4 | 2 | 3 | 等真实 gap/可冻结训练划分 |
| Xi-Net/SAITS/CSDI/SSSD 等重建 | 3 | 1 | 1 | 1 | 1–2 | 1–2 | 1 | 2 | 独立轮延期 |
| 什么都不做 | 1 | 5 | 5 | 5 | 5 | 1 | 5 | 5 | `OFF` 对照 |

关键取舍：

- “P/S 置零”直接消费已经存在的 `Waveform.gaps`，不改波形、不增加模型调用；低于正常阈值和 force-pair floor，因此能从机制上阻止区间内峰进入两个挑峰分支。
- 它无法改变 mask 之外的 raw gapped probability。本轮因此先做远程概率硬审计；这不是附加指标，而是候选可行性的逻辑必要条件。
- NaN 可能改变 SciPy/SeisBench 连续峰段、比较和排序语义；没有必要用未定义行为替代确定性的零概率。
- 整窗 veto 需要追踪每个 annotation 点由哪些滑窗贡献，且一个 gap 可能使约 60 秒窗口全部失效；在现有证据下比局部置零更容易制造漏检。
- 重建和训练方法有论文依据，但它们是新模型实验，不能在本轮局部规则失败后事后切换。

## 7. 本轮冻结的方法选择

唯一候选：

```text
mask_phase_annotations(annotations, gaps, guard_s)
```

只处理通道后缀为 `_P`/`_S` 的 annotation：合法 gap 按 UTC 排序合并，扩展固定 guard 后，把闭区间内概率精确置为 `0.0`；其它采样点、其它 annotation 通道、trace metadata 和 raw 输入不变。固定 guard 只允许 `0/0.5/1/2/5/10s`。

实验顺序不可反转：

1. 先导出无 gap 与 raw gapped 的七成员 P/S 概率曲线；
2. 检查 gap 10 秒外最大概率差、`1e-6` 数值容差、正常阈值/`0.03` floor 穿越、峰位与正常/低阈值候选变化；
3. 再在已经得到的 raw gapped annotation 上离线应用固定 guard，不增加模型调用；
4. 若远程概率或远程候选不满足预注册条件，所有 guard 直接不可录取，08 锁住；不得用更宽 guard、插值、taper 或重建补救。

## 8. 与第 5 轮的实质区别

第 5 轮在正常阈值、force-pair、亚采样精化、标准 dedup、长 SNR 和 20 秒 dedup **全部完成后**删除最终 `Pick`，因此 gap 内假峰可能已改变远程配对或簇代表。

本轮候选位于：

```text
ensemble annotation
→ gap-aware P/S zero mask
→ normal thresholds
→ conditional force-pair floor
→ refine/dedup/SNR/long dedup
```

它能阻止被遮峰参与后续逻辑，属于不同作用点和不同因果机制；但若模型前向本身已改变远程 annotation，本轮仍必须拒绝。这一硬边界正来自本轮 partial/gated convolution、GRU-D、PhaseNet-DAS 和 blackout-imputation 证据，而不是重复扩大第 5 轮删除窗。
