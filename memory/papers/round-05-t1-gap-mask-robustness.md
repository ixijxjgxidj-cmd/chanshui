# 研究轮次 05：T1 零填充缺口屏蔽鲁棒性

- 日期：2026-08-11
- 状态：AnySearch/Playwright 研究、方法决策与隔离实验均已完成；最终列表固定 margin mask 被拒绝
- 对应预注册：`memory/experiments/005-t1-gap-mask-robustness.md`
- 基准提交：`f8d180488176b85a20bec9f6b224dd5d36c4dde1`
- 本轮问题：读取层把 MiniSEED 分段缺口合并为精确零值并保存 `Waveform.gaps`，但生产 picker 尚未消费该元数据。一个只删除物理缺口及固定 margin 内最终拾取的稳定子序列过滤器，能否消除缺口诱发误报，同时不伤害缺口外稳定拾取、不留下远距离 force-pair/dedup 副作用？

## 1. 检索与核验协议

本轮围绕四组关键词检索：

1. zero padding、recording gap、missing segment 与神经震相拾取伪触发；
2. phase-picker prediction inconsistency、窗口边界和输入微扰；
3. gap-aware augmentation、缺失数据标签屏蔽与域适配；
4. seismic missing-data reconstruction、interpolation、denoising 及其下游偏差。

使用 AnySearch 普通检索、学术检索、预印本检索和 DOI 精确查询发现候选，并用 Crossref DOI 元数据复核标题、作者和年份。随后用 Playwright 实际打开原论文页、arXiv/ar5iv、政府归档、Zenodo、正式代码仓或出版社入口。共记录 17 篇，其中至少 16 篇是带实验结果的原始方法研究；没有用综述凑数，也没有重复计入第 4 轮完整计数的 EQTransformer、GPD 等论文。

访问边界如实记录：

- EdgePhase 的 Wiley 正文页返回 Cloudflare `403`，因此用 DOI 摘要、ADS 元数据和作者正式 GitHub 代码仓交叉核验，阅读状态标为“摘要 + 代码仓”，不冒充全文。
- IEEE Xplore 的 DSDL-SOF 页面返回 `418 Unusual Traffic`；Earthdoc 的无监督重建页停在 Cloudflare 验证。两篇只按 DOI/摘要与可访问的原始元数据记录。
- Springer、Frontiers、Nature、arXiv/ar5iv、CDC、OSTI、Zenodo 页面均实际读取到正文、摘要或资料页内容。

## 2. 最直接的论文证据

### 1. Application of a Convolutional Neural Network for Seismic Phase Picking of Mining-Induced Seismicity

- 作者/年份：Sean W. Johnson、Derrick J. A. Chambers、Michael S. Boltz、Keith D. Koper；2020 在线、2021 卷期。
- 标识：DOI `10.1093/gji/ggaa449`；正式 URL `https://doi.org/10.1093/gji/ggaa449`。
- Playwright 实际打开：`https://stacks.cdc.gov/view/cdc/215370`；并定位到 CDC 托管全文 PDF。
- 阅读状态：政府归档全文与出版社/DOI 元数据。
- 问题与方法：把原本在南加州训练的 CNN 迁移到矿山诱发地震 P 波拾取；比较域外直接使用、少量矿山数据微调、人工作业和经典 picker。
- 数据与实验：一个月 2,345 个定位事件、23,942 个手工 P 拾取，网络包含 8 个井下与 11 个地表台站；另比较多个矿山、采样率和网络尺度。
- 已核验结果：域外模型不经再训练表现不佳；少量域内数据迁移后接近人工分析员。正文还明确记载一次 zero padding 尝试会在“第一个非零数据点”产生 spurious picks。
- 局限：矿山数据以 P 波和特定台网为主，不能直接给出比赛三分量 P/S 的最佳 margin。
- 本项目关系：这是本轮最直接的实验证据——零填充边界确实可能被神经 picker 解释为到时；同时说明不能把域外重建或模型替换直接上线。
- 本周可实现性与决策：采用“必须显式处理缺口边界”的问题定义；不采用其重新训练方案。

### 2. A Mitigation Strategy for the Prediction Inconsistency of Neural Phase Pickers

- 作者/年份：Yongsoo Park、Gregory C. Beroza、William L. Ellsworth；2023。
- 标识：DOI `10.1785/0220230003`；正式 URL `https://doi.org/10.1785/0220230003`。
- Playwright 实际打开：`https://www.osti.gov/pages/biblio/1963658`。
- 阅读状态：OSTI 文章页与可下载接受稿元数据，正文级摘要和结论可读。
- 问题与方法：研究神经震相 picker 在输入波形发生很小扰动或滑窗位置改变时，输出可能显著变化的问题；测试更小滑窗步长和聚合策略。
- 数据与实验：在连续波形目录构建场景中比较不同滑窗/训练数据设置。
- 已核验结果：小输入扰动能造成实质性预测不一致并降低目录完整性；减小滑窗步长并聚合结果可明显缓解，训练数据同样重要。
- 局限：论文处理滑窗平移，不是零填充缺口，也没有证明最终列表删除能修复上游概率变化。
- 本项目关系：直接支持预注册中的“远距离副作用”硬门槛。只检查 gap 内 pick 是否删除是不够的，还必须检查缺口是否改变 10 秒外的触发、丢失、force-pair 或 dedup 代表。
- 本周可实现性与决策：采用其一致性审计思想；不重开历史已失败的 overlap 扫描。

### 3. SeisBench—A Toolbox for Machine Learning in Seismology

- 作者/年份：Jack Woollam、Jannes Münchmeyer、Frederik Tilmann、Andreas Rietbrock、Dietrich Lange 等；2022。
- 标识：DOI `10.1785/0220210324`；arXiv `2111.00786`。
- Playwright 实际打开：`https://ar5iv.labs.arxiv.org/html/2111.00786`。
- 阅读状态：ar5iv 全文。
- 问题与方法：统一地震机器学习数据、模型、训练和基准接口；为不同来源数据建立一致 metadata、固定划分和可组合增强。
- 数据与实验：集成多个公开地震数据集、已有 picker 和统一 benchmark；数据格式包含噪声、长度、采样率与完整性等元数据。
- 已核验结果：格式显式区分缺少元数据、噪声样本和变长记录；SeisBench 的数据/增强语义允许用零值生成 gap，并把深入 gap 的标签清理为 NaN，而不是把缺口当正常波形监督。
- 局限：工具箱定义安全的数据语义，但不替本项目证明某个最终 margin 足够，也不解决已进入 force-pair/dedup 的伪拾取。
- 本项目关系：支持使用现有 `Waveform.gaps` 元数据和 gap-aware 标签思路；也说明若最终层失败，下一轮应研究 annotation/标签层，而不是静默插值。
- 本周可实现性与决策：部分采用；本轮只验证最终稳定子序列 mask，训练增强延期。

## 3. 拾取架构、域适配与多台站证据

### 4. PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method

- 作者/年份：Weiqiang Zhu、Gregory C. Beroza；2018。
- 标识：DOI `10.1093/gji/ggy423`；arXiv `1803.03211`。
- Playwright 实际打开：`https://arxiv.org/abs/1803.03211`。
- 阅读状态：原始摘要页与论文元数据。
- 问题与方法：三分量 U-Net 类网络逐样点输出 P、S、noise 概率，概率峰值用于到时。
- 数据与实验：北加州三十多年分析员标签，超过 700 万个波形样本。
- 已核验结果：论文报告相对既有方法更高拾取精度和召回，并特别提高 S 波观测能力。
- 局限：训练分布和连续滑窗决定概率背景；逐样点输出并不保证零填充边界不会形成峰。
- 本项目关系：当前 DiTing/PhaseNet 基座与该机制同源，因此边界阶跃会直接作用于 P/S/noise 概率曲线；最终 mask 是输出层的最小保护，不是模型修复。
- 本周可实现性与决策：保留当前基座；采用概率峰可能受局部缺失影响的风险判断。

### 5. EdgePhase: A Deep Learning Model for Multi-Station Seismic Phase Picking

- 作者/年份：Tian Feng、Saeed Mohanna、Lingsen Meng；2022。
- 标识：DOI `10.1029/2022GC010453`；正式 URL `https://doi.org/10.1029/2022GC010453`。
- Playwright 实际打开：Wiley 正文入口 `https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022GC010453`（403）和作者代码仓 `https://github.com/lovelytt0/EdgePhase`。
- 阅读状态：DOI/ADS 摘要 + 正式代码仓；出版社正文不可读。
- 问题与方法：在 EQTransformer 上加入 Edge Convolution/图神经网络，让相邻台站交换与震相相关的信息。
- 数据与实验：代码仓包含 SCSN2021 训练/验证构建、station metadata、edge index、训练/测试脚本、模型和希腊地震案例。
- 已核验结果：摘要报告相对单台站 EQTransformer 提高 F1，并在案例中利用台网一致性改善事件与相位结果。
- 局限：依赖多台站拓扑、坐标和同步上下文；比赛 API 对每条记录不保证提供可关联台网。
- 本项目关系：说明真正降低局部伪触发可借助跨台站证据，但当前输入契约不具备。不能用单台站最终 mask 冒充 EdgePhase。
- 本周可实现性与决策：当前拒绝；若未来官方明确多台站同步输入，再重审。

### 6. Phase Neural Operator for Multi-Station Picking of Seismic Arrivals

- 作者/年份：Hongyu Sun、Zachary E. Ross、Weiqiang Zhu、Kamyar Azizzadenesheli；2023。
- 标识：DOI `10.1029/2023GL106434`；arXiv `2305.03269`。
- Playwright 实际打开：`https://arxiv.org/abs/2305.03269`。
- 阅读状态：原始摘要页。
- 问题与方法：用神经算子同时建模任意台网几何的时空上下文，联合拾取多个台站。
- 数据与实验：在网络级波形上与主流单台站/多台站基线比较。
- 已核验结果：摘要报告检测到更多事件与相位，并改善拾取精度。
- 局限：需要台网上下文、训练和新的模型资产；不能解决当前单文件局部缺口的即时上线问题。
- 本项目关系：再次说明可靠的上游消歧需要额外上下文。本轮若发现 10 秒外副作用，只能拒绝最终 mask，而不能宣称一个更宽 margin 可替代时空模型。
- 本周可实现性与决策：延期，不进入本轮候选。

### 7. OBSTransformer: A Deep-Learning Seismic Phase Picker for OBS Data Using Automated Labelling and Transfer Learning

- 作者/年份：Alireza Niksejel、Miao Zhang；2024。
- 标识：DOI `10.1093/gji/ggae049`；arXiv `2306.04753`。
- Playwright 实际打开：`https://arxiv.org/abs/2306.04753`。
- 阅读状态：原始摘要页。
- 问题与方法：对高噪声、仪器响应和域差异明显的海底地震仪数据自动标注，并从陆地 EQTransformer 迁移学习。
- 数据与实验：11 次部署、423 台 OBS、约 36,000 个地震样本；另提取 25,000 个噪声样本，在次全球、区域和局部测试集验证。
- 已核验结果：相对原 EQTransformer，距离大于 200 km 时 P/S recall 分别提高 68%/76%。
- 局限：收益来自大量域内事件和噪声、重新训练与模型选择，不是无训练后处理。
- 本项目关系：支持“输入异常/仪器域必须用针对性数据证明”，反对把探索地震重建器直接迁入比赛生产。
- 本周可实现性与决策：延期到有真实 gap/DiTing 数据后；不用于本轮。

### 8. Deep Learning Models for Regional Phase Detection on Seismic Stations in Northern Europe and the European Arctic

- 作者/年份：Erik B. Myklebust、Andreas Köhler；2024。
- 标识：DOI `10.1093/gji/ggae298`；正式模型资料 `https://zenodo.org/records/11231543`。
- Playwright 实际打开：Zenodo 模型/训练数据页；ADS 原始摘要入口也已导航核验。
- 阅读状态：正式数据与模型资料页；论文正文未完整读取。
- 问题与方法：面向北欧和欧洲北极区域台站训练/比较区域相位检测模型。
- 数据与实验：Zenodo 发布训练数据、模型和事件/台站资料，文件总量约 26.7 GB，模型包约 614 MB。
- 已核验结果：论文与资产共同证明区域化 picker 可复现，但实现依赖多年区域资料和较大的训练资产。
- 局限：区域、传感器、噪声和标注与中国比赛包不匹配；资料规模不适合本周临时迁移。
- 本项目关系：支持严格的域外风险控制和当前“先做不训练的最小反证”。
- 本周可实现性与决策：拒绝本周迁移；保留为域适配方法证据。

### 9. PickBlue: Seismic Phase Picking for Ocean Bottom Seismometers With Deep Learning

- 作者/年份：Thomas Bornstein、Dietrich Lange、Jannes Münchmeyer、Jack Woollam、Andreas Rietbrock 等；2023。
- 标识：DOI `10.1029/2023EA003332`；arXiv `2304.06635`。
- Playwright 实际打开：`https://arxiv.org/abs/2304.06635`。
- 阅读状态：原始摘要页。
- 问题与方法：把 PhaseNet/EQTransformer 适配到 OBS，并联合三分量地震计和水听器通道。
- 数据与实验：15 次部署、355 个台站、13,190 个事件，约 90,000 个 P 和 63,000 个 S 人工拾取。
- 已核验结果：域内训练和四通道输入改善 OBS 拾取，说明特定缺失/噪声结构需要匹配的数据与传感器设计。
- 局限：新增通道和大规模人工标签不可用于当前三分量比赛 API。
- 本项目关系：不支持无训练重建；支持把缺口看作需要显式元数据和验证的分布变化。
- 本周可实现性与决策：延期，不替换生产模型。

## 4. 缺失数据重建与去噪证据

### 10. Deep Learning for Irregularly and Regularly Missing Data Reconstruction

- 作者/年份：Xintao Chai、Hanming Gu、Feng Li、Hongyou Duan、Xiaobo Hu、Kai Lin；2020。
- 标识：DOI `10.1038/s41598-020-59801-x`。
- Playwright 实际打开：`https://www.nature.com/articles/s41598-020-59801-x`。
- 阅读状态：Nature 全文。
- 问题与方法：用编码器—解码器/U-Net 类图像到图像网络，从不完整二维勘探地震剖面重建规则和不规则缺失道；损失包含像素与结构信息。
- 数据与实验：在有完整参考的二维勘探地震数据上人为生成缺失掩码，训练后补齐规则/不规则空间空洞。
- 已核验结果：论文展示模型可填补被破坏剖面的空洞，并在其数据和缺失率设置下保持结构。
- 局限：任务是多道空间重建，不是单台三分量时间缺口；需要成对完整/缺失样本，重建到时相位可能产生不可量化偏移。
- 本项目关系：只能证明深度重建可作为独立研究方向，不能证明它比删除缺口邻域拾取更安全。
- 本周可实现性与决策：拒绝当前生产；待获得真实 gap 数据和到时保持指标后另轮研究。

### 11. Can Learning from Natural Image Denoising Be Used for Seismic Data Interpolation?

- 作者/年份：Hao Zhang、Xiuyan Yang、Jianwei Ma；2020。
- 标识：DOI `10.1190/geo2019-0243.1`；arXiv `1902.10379`。
- Playwright 实际打开：`https://arxiv.org/abs/1902.10379`。
- 阅读状态：原始摘要页。
- 问题与方法：把自然图像 CNN denoiser 作为 plug-and-play 先验嵌入 POCS 迭代，实现 CNN-POCS 地震道插值，无需针对每种缺失模式重新训练。
- 数据与实验：二维合成与现场勘探数据，对比 `f-x` 预测滤波和 curvelet POCS。
- 已核验结果：论文报告在 SNR、去混叠和弱特征重建上优于对照。
- 局限：仍依赖多道空间连续性、迭代优化和自然图像先验；单站时间 gap 没有邻道约束。
- 本项目关系：计算和假设均远大于一个区间 mask，且不能保证 P/S 到时不偏移。
- 本周可实现性与决策：拒绝本轮；若未来有阵列数据可重新评估。

### 12. Deep Neural Networks Based Denoising of Regional Seismic Waveforms and Impact on Analysis of North Korean Nuclear Tests

- 作者/年份：Andreas Steinberg、Peter Gaebler、Gernot Hartmann、Johanna Lehr、Christoph Pilger；2024 在线、2025 卷期。
- 标识：DOI `10.1007/s00024-024-03491-3`。
- Playwright 实际打开：`https://link.springer.com/article/10.1007/s00024-024-03491-3`。
- 阅读状态：Springer 开放全文。
- 问题与方法：把深度去噪应用到区域距离地震波形，并检查对朝鲜核试验下游分析的影响，而不只报告视觉 SNR。
- 数据与实验：区域真实波形、合成噪声/信号组合与下游分析；正文讨论短窗边界处理。
- 已核验结果：去噪可改善部分弱信号，但短窗内 zero padding 会产生边缘 artifact；下游矩张量等分析出现轻微 bias，说明“波形看起来更干净”不等于到时和物理量无偏。
- 局限：目标主要是信噪比与源分析，模型和区域都不同于比赛 picker。
- 本项目关系：这是拒绝本周插值/去噪的关键安全证据。任何重建方案都必须先证明 P/S 到时、数量罚和 T2/T3 不偏；当前没有这样的证据。
- 本周可实现性与决策：采用其下游验证纪律；拒绝本轮去噪生产接入。

### 13. Seismic Signal Denoising and Decomposition Using Deep Neural Networks（DeepDenoiser）

- 作者/年份：Weiqiang Zhu、S. Mostafa Mousavi、Gregory C. Beroza；2019。
- 标识：DOI `10.1109/TGRS.2019.2926772`；arXiv `1811.02695`。
- Playwright 实际打开：`https://arxiv.org/abs/1811.02695`。
- 阅读状态：原始摘要页。
- 问题与方法：在时频域学习稀疏表示和 signal/noise mask，把输入分解为地震信号与噪声。
- 数据与实验：构造的地震信号—噪声组合和真实连续记录，评价去噪与后续检测能力。
- 已核验结果：论文报告优于传统滤波的去噪/分解能力，并可恢复被强噪声掩盖的信号。
- 局限：需要专门训练和时频重构；其“noise”语义不等于遥测缺口，重构可能改变峰位和幅值。
- 本项目关系：若缺口 mask 失败，DeepDenoiser 也不能作为同轮事后补救；必须另设到时保真与跨包门槛。
- 本周可实现性与决策：延期。

### 14. Depthwise Separable Convolution U-Net for 3D Seismic Data Interpolation

- 作者/年份：Zhenhui Jin、Xinze Li、Hui Yang、Bangyu Wu、Xu Zhu；2023。
- 标识：DOI `10.3389/feart.2022.1005505`。
- Playwright 实际打开：`https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2022.1005505/full`。
- 阅读状态：Frontiers 全文。
- 问题与方法：用 depthwise separable 3D convolution 替换部分标准 U-Net 卷积，配合 SSIM/混合损失和 switchable normalization，降低三维缺失道插值的参数与计算量。
- 数据与实验：SEG C3 合成数据和现场三维勘探数据；比较标准 U-Net 与 DS-U-Net。
- 已核验结果：重建切片视觉和 SNR 接近标准 U-Net，同时显著减少参数和计算；论文也展示不同配置 SNR 仍有差距。
- 局限：三维空间道缺失、有完整标签、需训练；即使轻量化也远重于确定性后处理。
- 本项目关系：证明“可以更便宜地重建”不等于“适合单站时间 gap 且到时无偏”。
- 本周可实现性与决策：拒绝本轮，保留为未来重建基线。

### 15. Deep Preconditioners and Their Application to Seismic Wavefield Processing

- 作者/年份：Matteo Ravasi；2022。
- 标识：DOI `10.3389/feart.2022.997788`。
- Playwright 实际打开：`https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2022.997788/full`。
- 阅读状态：Frontiers 全文。
- 问题与方法：训练自动编码器，把 decoder 作为物理反问题的非线性预条件器，用于 deghosting、wavefield separation 和规则/不规则欠采样重建。
- 数据与实验：从逐渐复杂的合成数据到 marine field data；与固定基变换和不同训练设计比较。
- 已核验结果：学习预条件器能加快收敛并改善多种波场任务；训练改进相对初始方案带来约 2–3 dB SNR 改善。
- 局限：依赖代表性 latent manifold、反演算子和迭代优化；论文也指出弱事件可能难以重建。其阵列波场先验不适合独立单站时间缺口。
- 本项目关系：如果连高成本重建都依赖强先验，就更不能把简单 taper/interpolation 当无害默认值。
- 本周可实现性与决策：拒绝当前生产，延期到独立研究轮。

### 16. Seismic Data Reconstruction Based on Double Sparsity Dictionary Learning With Structure Oriented Filtering

- 作者/年份：Lakshmi Kuruguntla、Vineela Chandra Dodda、Anup Kumar Mandpura、Sunil Chinnadurai、Karthikeyan Elumalai；2023。
- 标识：DOI `10.1109/JSTARS.2023.3323362`。
- Playwright 实际打开：`https://ieeexplore.ieee.org/document/10274884/`，页面返回 IEEE `418`。
- 阅读状态：仅 DOI/摘要和搜索元数据；未读取 IEEE 正文。
- 问题与方法：级联 double-sparsity dictionary learning 与 structure-oriented filtering；前者去噪，后者削弱残余噪声并填充缺失点。
- 数据与实验：二维合成和现场地震剖面；与既有重建方法比较 SNR 和 MSE。
- 已核验结果：摘要报告 DSDL-SOF 在其数据上获得更好的噪声衰减和重建指标。
- 局限：依赖多道结构方向、字典学习和滤波；正文不可访问，且没有单站 P/S 到时保真证据。
- 本项目关系：不满足生产证据等级，不能用于比赛 API。
- 本周可实现性与决策：拒绝。

### 17. Seismic Noise Attenuation by Signal Reconstruction: An Unsupervised Machine Learning Approach

- 作者/年份：Yang Gao、Pingqi Zhao、Guofa Li、Hao Li；2021。
- 标识：DOI `10.1111/1365-2478.13070`。
- Playwright 实际打开：`https://www.earthdoc.org/content/journals/10.1111/1365-2478.13070`，页面停在 Cloudflare 验证。
- 阅读状态：AnySearch 可读摘要 + DOI 元数据；浏览器正文不可读。
- 问题与方法：训练时随机 mute 一部分输入，让网络从剩余上下文重建信号，同时学习随机噪声衰减，不要求成对 clean 标签。
- 数据与实验：二维合成和现场勘探地震数据。
- 已核验结果：摘要报告合成数据优于传统方法，现场数据也显示可行性和实用性。
- 局限：无监督仍然需要训练分布和空间结构；随机 mute 的训练目标不保证比赛 P/S 峰位、数量或幅值无偏。
- 本项目关系：是未来 gap augmentation/自监督重建的候选证据，不是本轮最终列表 mask 的替代品。
- 本周可实现性与决策：延期到有可冻结训练/验证划分之后。

## 5. 证据综合

17 篇证据形成六条稳定结论：

1. **零填充不是中性操作。** Johnson 等人的正文直接观察到第一个非零样点处的伪拾取；区域去噪论文也观察到短窗 zero padding 的边界 artifact。
2. **缺口影响可能不局限于物理区间。** Prediction Inconsistency 证明很小输入变化就能让神经 picker 输出显著变化。因此本轮必须检查 10 秒外的 induced/lost picks，不能用扩 margin 掩盖上游副作用。
3. **数据/标签层 gap awareness 有合理依据，但需要训练。** SeisBench 的 gap 增强会同步清理 gap 内标签；OBSTransformer、PickBlue 和区域 picker 共同说明域适配收益来自大量匹配数据，而不是一条无验证预处理规则。
4. **大多数“缺失数据重建”证据来自多道勘探地震的空间缺失。** U-Net、CNN-POCS、DS-U-Net、deep preconditioner、DSDL-SOF 和无监督重建利用相邻道、波场结构或代表性流形，不能直接外推到单站三分量时间断档。
5. **去噪/重建可能改善 SNR，却仍给下游物理量引入偏差。** 因而比赛准入必须以 P/S 到时、数量罚和 T2/T3 回归为准，不能以波形视觉或 SNR 代替。
6. **最终列表 mask 是当前最便宜、最可回滚的反证，但能力边界很窄。** 它只能删除最终仍落在 gap 邻域的 pick；如果 gap 已改变概率、force-pair、SNR 或 dedup 的远处结果，本机制必须失败，而不是继续扩大 margin。

## 6. 方法决策矩阵

评分为 `1`（差/高风险）到 `5`（好/低风险）。“官方收益”在本轮指未来 gap 输入的数量罚与鲁棒性收益，不是对三套无 gap 历史包虚构分数提升。

| 方法 | 预期官方/鲁棒收益 | 数据现成 | 本周可完成 | 实现成本 | 训练成本 | 延迟/内存 | 噪声/多事件/长记录安全 | 速度/加速度输入安全 | 跨包与过拟合安全 | 可复现/回滚/API 兼容 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 最终 Pick 列表固定 margin mask | 3 | 5 | 5 | 5 | 5 | 5 | 3 | 5 | 5 | 5 | **采用为最小反证** |
| 阈值/force-pair 前 annotation 或概率级 gap-aware mask | 4 | 4 | 3 | 3 | 5 | 4 | 4 | 5 | 4 | 4 | 若最终层失败，另轮预注册 |
| gap augmentation + 标签屏蔽后微调/蒸馏 | 4 | 2 | 2 | 2 | 2 | 4 | 4 | 3 | 2 | 3 | 等真实 gap/DiTing 数据 |
| taper、线性/样条插值或局部滤波 | 2 | 5 | 5 | 4 | 5 | 5 | 2 | 3 | 2 | 5 | 缺少到时保真证据，拒绝 |
| U-Net/CNN-POCS/字典/生成式重建或去噪 | 3 | 1 | 1 | 1 | 1–2 | 1–3 | 2 | 2 | 1 | 2 | 本周拒绝，独立研究 |
| 什么都不做 | 1 | 5 | 5 | 5 | 5 | 5 | 1 | 5 | 5 | 5 | 仅作为 `OFF` 对照 |

关键取舍：

- 最终 mask 不增加模型调用、权重、成员、TTA 或主要内存，能同时用于短/长、速度/加速度和噪声输入；无 gap 时可严格返回原列表对象。
- 它的安全分只有 3，而不是 5，因为上游模型与后处理可能已经产生不可逆远程变化。本轮预注册用 `remote_induced_new=0`、`remote_lost_reference=0`、`collateral_deleted=0` 把这个缺陷变成可证伪门槛。
- annotation/probability 级 mask 理论上更能阻止 force-pair 与 dedup 被 gap pick 污染，但会改变模型后处理顺序，必须根据本轮失败位置另行预注册，不能事后切换。
- 重建方法的论文结果多数来自空间剖面、合成/现场勘探数据和专门训练；当前既没有真实时间 gap 标签，也没有证明到时、数量、幅值和分类同时安全的证据。

## 7. 本轮冻结的方法选择

论文矩阵完成后，仍维持预注册的唯一候选，不扩充机制：

```text
mask_gap_picks(picks, gaps, margin_s)
```

固定 margin 仅为 `0/0.5/1/2/5/10s`，开发只使用 R1/R2，08 只在开发选出最小合格 margin 后以冻结值终检。任何以下现象都直接否决最终层方案：

- gap 诱发的新 pick 在 active margin 后仍残留；
- 物理 gap 10 秒外出现 induced 或 lost pick；
- 删除原始与 gapped 波形中都稳定存在、且实际位于物理 gap 外的参考 pick；
- 单条与 batch 不一致、无 gap 不保持对象身份、或性能 P95 达不到 5 ms；
- 需要扩大网格、按相位/文件自适应或接入重建才能解释结果。

本轮即使通过，也只能说明该确定性过滤器具备进入生产实现评审的资格；历史三包没有真实 gap，因此不得宣称历史官方分数提高。生产接入仍需默认关闭开关、三包无 gap 预测哈希零变化、全量测试与 API/长记录/噪声回归。

## 8. 若失败，下一轮研究边界

若最终 mask 失败，允许进入下一轮的问题只有：

> 能否在阈值触发、条件式 force-pair 和 dedup 之前使用 `Waveform.gaps` 屏蔽 annotation/probability 区间，从源头阻止 gap pick 参与后处理，同时保持缺口外概率和无 gap 路径逐位不变？

届时必须重新预注册作用位置、滑窗边界、probability mask/taper 语义和失败条件。不得在第 5 轮结果出来后顺手改成插值、生成式修复、重新扫 overlap 或扩大 margin。

## 9. 实验对论文决策的回证

隔离实验结果与论文矩阵中的核心风险一致：77 个 R1/R2/噪声变体产生 31 个 induced 和 36 个 lost，其中 13 个 induced、2 个 lost 位于物理 gap 10 秒之外。最终区间过滤的 `0s` margin 只清掉 8/31 个 induced；扩大到 `10s` 仍留下全部 13 个远程 induced，却误删 37 个稳定 reference picks。

因此本轮不是因为实现成本或性能拒绝：七成员、输入身份、注入一致性、single/batch、重复性全部通过，后处理 P95 仅 `2.1463 ms`。真正的拒绝原因是 Prediction Inconsistency 所提示的上游上下文变化已经发生，最终删除层无法恢复；Johnson 等人观察到的零填充边界伪拾取也不只表现为恰好落在 gap 内的一种局部错误。

论文决策最终落地为：

- 最终 Pick 列表固定 margin mask：**拒绝**；
- taper/interpolation/深度重建：仍然拒绝本周生产，不能用本轮负结果作为事后切换理由；
- annotation/probability 级 gap awareness：保留为独立下一轮候选，必须重新检索新证据和预注册；
- gap augmentation/重训练：等待真实 gap 或可冻结的训练/验证数据，不依赖尚未取得的 DiTing 原始数据。
