# 研究轮次 07：T1 七成员教师—学生 PhaseNet 蒸馏

- 日期：2026-08-12
- 状态：AnySearch MCP 发现、Browser MCP 串行核验和 CPU 可行性基准均已完成；正式三折 LOPO 蒸馏进入预注册
- 对应实验：`memory/experiments/011-t1-cpu-distillation-feasibility.md`
- 基准提交：`a846c26118195f8c63cb8d26f0cfe5ebac3bd7bc`
- 本轮问题：已有七成员教师在三套历史 T1 包上优于所有留存单模型/旧候选，但生产推理成本高、成员训练谱系不完整。能否把冻结教师的稠密 N/P/S 概率蒸馏到单个 DiTing PhaseNet，并严格按整包 leave-one-package-out（LOPO）验证，使学生最终输出在三包 × 四数量罚 12 个单元中不下降？

## 1. 检索与浏览协议

本轮先用 AnySearch MCP 的普通与学术检索能力发现候选，再用 Browser MCP 串行打开原始论文页、arXiv、CVF、ACL Anthology、AAAI、Springer、MDPI、PMC 或作者正式代码仓。Browser MCP 容易在高并发下返回 `429`，因此按每批 3–6 页串行核验，没有并发轰炸，也没有把搜索摘要冒充全文。

检索主题分为四组：

1. 地震震相拾取的轻量化、嵌入式模型与物理约束；
2. 单教师、同容量学生、序列级和稠密预测知识蒸馏；
3. 多教师选择、在线/互学、teacher assistant 与温度策略；
4. 稀疏标签、边界和结构关系蒸馏对稠密时序输出的可迁移性。

本轮计入 17 篇此前六轮未完整计数的原始实验研究。访问边界如下：

- `SeismicSense` 的 ACM 正文入口在 Browser MCP 中停留在安全页；AnySearch MCP 取得正式摘要，Browser MCP 实际读取作者正式仓库，因此阅读状态只写“正式摘要 + 作者代码仓”，不宣称读到 ACM 全文。
- IEEE 上的 `A Lightweight Network for Seismic Phase Picking on Embedded Systems`、China Seismic Network picker benchmark 和 Research Square/Sciety 的 `PhaseNeXt` 正文入口未成功读取；它们不计入 17 篇正式核验数量。
- `LEQNet` 已在第 4 轮完整计数，本轮不重复计数。

## 2. 地震拾取与端侧模型的直接证据

### 1. SeismicSense: Phase Picking of Seismic Events with Embedded Machine Learning

- 标识：DOI [`10.1145/3672608.3707845`](https://doi.org/10.1145/3672608.3707845)。
- 实际访问：ACM 正文入口的安全页；作者正式仓库 [`ds-kiel/SeismicSense`](https://github.com/ds-kiel/SeismicSense)。
- 阅读状态：正式摘要 + 作者代码仓，不是出版社全文。
- 方法与结果：面向资源受限微控制器压缩震相检测/拾取；正式摘要报告模型比所比较 SOTA 小约 20 倍，只需约 186 KB RAM，事件/P/S 的 F1 约为 `99.4%/98%/96%`，整数化 MCU 推理约加速 18 倍。
- 局限：模型目标、数据划分、后处理与本比赛七成员 PhaseNet 不同；端侧 F1 不能替代三包四口径官方评分。
- 本项目关系：证明单模型大幅压缩在地震拾取上可行，但不能证明直接换一个小架构会保留当前七成员的峰形、force-pair 与长记录行为。
- 本轮决策：保留为第二阶段压缩方向；第一阶段先用同架构 PhaseNet 隔离“蒸馏是否有效”这一变量。

### 2. High-Precision Coal Mine Microseismic P-Wave Arrival Picking via Physics-Constrained Deep Learning

- 标识：DOI [`10.3390/s25237103`](https://doi.org/10.3390/s25237103)。
- 实际访问：MDPI/Sensors 正式全文页。
- 阅读状态：开放全文。
- 方法与结果：使用 U-Net 型网络并加入物理约束进行煤矿微震 P 波初至拾取；论文报告 precision `96.60%`、recall `90.59%`、F1 `93.50%`、平均误差 `5.49 ms`。
- 局限：只处理 P 波，任务分布、标签与本项目 P/S 联合输出不同；物理约束的收益不能直接外推到冻结七教师平均。
- 本项目关系：支持保留真实到时硬标签和任务约束，而不是只拟合教师 soft target；学生若复制教师错误，必须由硬标签损失和最终评分暴露。
- 本轮决策：采用“soft + hard”对照，不在首轮另加新的物理损失。

## 3. 蒸馏的基础与时序证据

### 3. LightTS: Lightweight Time Series Classification with Adaptive Ensemble Distillation

- 标识：arXiv [`2302.12721`](https://arxiv.org/abs/2302.12721)，SIGMOD 2023。
- 实际访问：arXiv 原始页面。
- 方法与结果：在 128 个真实时间序列数据集上，自适应加权多个教师并构建准确率/模型大小的 Pareto 选择；说明多教师时序知识可以被压缩到轻量学生。
- 局限：目标是整窗分类，不是逐采样点 N/P/S 概率和亚秒峰位；自适应教师权重还会增加选型自由度。
- 本项目关系：支持 ensemble distillation，但不支持第一轮用三套历史包标签事后学习七成员权重。
- 本轮决策：教师严格冻结为生产平均；只有该最小方案通过后才考虑动态选择。

### 4. Distilling the Knowledge in a Neural Network

- 标识：arXiv [`1503.02531`](https://arxiv.org/abs/1503.02531)。
- 实际访问：arXiv 原始页面。
- 方法与结果：Hinton、Vinyals、Dean 证明 ensemble 的 soft targets 可压缩进单模型，并在 MNIST 与商业语音模型上验证；温度化输出能传递类别间“暗知识”。
- 局限：经典分类 KL/交叉熵没有解决长时序滑窗、峰形、标签时间偏差和数量罚。
- 本项目关系：给出七教师平均概率到单学生的直接理论/实验起点。
- 本轮决策：首轮使用最简单的 dense response cross-entropy，不做温度网格。

### 5. Born Again Neural Networks

- 标识：arXiv [`1805.04770`](https://arxiv.org/abs/1805.04770)，ICML 2018。
- 实际访问：arXiv 原始页面。
- 方法与结果：学生可以与教师同容量、同架构，仍通过蒸馏超过教师或标准训练；论文报告 CIFAR-10 error `3.5%`、CIFAR-100 error `15.5%`，最佳 ensemble 约 `14.9%`。
- 局限：图像分类结果不能保证同架构 PhaseNet 在跨包到时评分中提升。
- 本项目关系：直接支持首个学生仍采用 `PhaseNet(diting)`，无需先设计更小网络；这样能把实验重点放在七成员概率知识而非架构变化。
- 本轮决策：采用同架构学生。

### 6. Sequence-Level Knowledge Distillation

- 标识：ACL Anthology [`D16-1139`](https://aclanthology.org/D16-1139/)，EMNLP 2016。
- 实际访问：ACL Anthology 正式页面。
- 方法与结果：把教师生成的序列级目标用于神经机器翻译；最佳学生约快 10 倍，结合剪枝可少约 13 倍参数而只损失约 0.4 BLEU，相对无蒸馏小模型提高约 `4.2/1.7 BLEU`。
- 局限：离散翻译序列与连续概率时间轴不同；不能直接把最终 pick 列表当唯一蒸馏目标，因为列表丢失峰宽、噪声概率和次峰信息。
- 本项目关系：支持最终仍以完整序列行为验证，但训练目标应优先保留稠密概率，而不是只模仿离散 picks。
- 本轮决策：缓存 N/P/S 概率曲线，最终通过完整生产后处理评分。

## 4. 稠密预测与结构知识蒸馏

### 7. Structured Knowledge Distillation for Semantic Segmentation

- 标识：CVPR 2019，[CVF 正式页面](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.html)。
- 实际访问：CVF 正式页面；AnySearch MCP 读取原论文 PDF 内容。
- 方法与结果：在 Cityscapes、CamVid、ADE20K 上结合 point-wise、pair-wise 和 holistic 蒸馏；正式代码记录的 Cityscapes baseline 约 `69.10 mIoU`，组合蒸馏可到约 `74.08`，另一配置约 `75.3`。
- 局限：二维语义分割的像素关系与一维地震概率并不等价；结构损失会显著增加首轮变量。
- 本项目关系：证明稠密输出不能只看最终离散标签，但并不要求第一轮一次加入所有结构项。
- 本轮决策：首轮只做 point-wise dense response KD；pair/holistic 仅在最小方案结果可解释后重开。

### 8. Channel-wise Knowledge Distillation for Dense Prediction

- 标识：arXiv [`2011.13256`](https://arxiv.org/abs/2011.13256)，ICCV 2021，[CVF 正式页面](https://openaccess.thecvf.com/content/ICCV2021/html/Shu_Channel-Wise_Knowledge_Distillation_for_Dense_Prediction_ICCV_2021_paper.html)。
- 实际访问：CVF 正式页面。
- 方法与结果：对每个通道的空间分布归一化后蒸馏，在 Cityscapes、Pascal VOC、ADE20K 上验证；多种 ResNet18/MobileNetV2 学生常见提升约 3–4 mIoU。
- 局限：通道归一化会改变 N/P/S 概率的绝对校准，可能影响固定阈值和 `0.03` force-pair floor。
- 本项目关系：提示未来可蒸馏“每相位峰形分布”，但本项目生产规则依赖绝对概率，因此首轮不能只保留通道内相对形状。
- 本轮决策：保留原始 N/P/S 概率标尺，暂不使用 channel-wise 归一化损失。

### 9. Intra-class Feature Variation Distillation for Semantic Segmentation

- 标识：DOI [`10.1007/978-3-030-58571-6_21`](https://doi.org/10.1007/978-3-030-58571-6_21)，ECCV 2020。
- 实际访问：Springer 正式页面。
- 方法与结果：用类别原型与类内特征变化向学生传递结构知识，在语义分割上改善小模型。
- 局限：跨图像像素的类别原型迁移到地震时序时，噪声、P、S 的局部上下文和峰宽含义不同。
- 本项目关系：支持“逐点 KL 可能不是终点”，但不能在第一轮把未验证的类内结构正则与基础 KD 混在一起。
- 本轮决策：延期到基础 KD 明确失败模式后。

### 10. SimCVD: Simple Contrastive Voxel-Wise Representation Distillation for Semi-Supervised Medical Image Segmentation

- 标识：arXiv [`2108.06227`](https://arxiv.org/abs/2108.06227)，DOI [`10.1109/TMI.2022.3161829`](https://doi.org/10.1109/TMI.2022.3161829)。
- 实际访问：PMC 开放全文；论文身份同时由 arXiv 与 DOI 交叉核验。
- 方法与结果：在稀缺标签的医学分割中结合 voxel-wise、边界感知和结构关系蒸馏，说明局部边界与表征关系可补充逐点监督。
- 局限：三维体素边界与 P/S 峰的时间误差容忍不同，且 contrastive memory/采样会增加计算和超参数。
- 本项目关系：为未来局部峰形、边界与相位关系损失提供依据。
- 本轮决策：不进入首轮；先判断 dense probability + hard label 是否已经足够。

## 5. 多教师、在线学习与容量差证据

### 11. Knowledge Distillation by On-the-Fly Native Ensemble

- 标识：arXiv [`1806.04606`](https://arxiv.org/abs/1806.04606)，NeurIPS 2018。
- 实际访问：arXiv 原始页面。
- 方法与结果：在单阶段训练中构造 native ensemble 并在线向分支蒸馏，不要求先训练固定教师。
- 局限：当前七教师已经冻结，CPU 实测生成全量平均概率只需约 2 分 12 秒；在线共训反而引入同步、随机性和教师漂移。
- 本项目关系：是没有冻结教师时的备选，不是当前成本最小路线。
- 本轮决策：拒绝第一轮在线教师共训。

### 12. Deep Mutual Learning

- 标识：CVPR 2018，[CVF 正式页面](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.html)。
- 实际访问：CVF 正式页面。
- 方法与结果：多个学生从头互相学习，无需预训练强教师，并在分类和行人重识别实验中提升单模型。
- 局限：它没有利用当前已经验证过的冻结七成员教师，还会把训练随机性扩展到多个网络。
- 本项目关系：只有在教师不可用或需要共同探索时才有吸引力。
- 本轮决策：不作为首选。

### 13. Online Knowledge Distillation with Diverse Peers

- 标识：arXiv [`1912.00350`](https://arxiv.org/abs/1912.00350)，DOI [`10.1609/aaai.v34i04.5746`](https://doi.org/10.1609/aaai.v34i04.5746)，AAAI 2020。
- 实际访问：AAAI 正式页面。
- 方法与结果：以两级蒸馏保持 peer diversity，并让 group leader 最终单独推理。
- 局限：需要同时训练多个 peer；当前实验目标是把固定 ensemble 压缩为一个学生，而不是重建新的在线 ensemble。
- 本项目关系：提醒不能因蒸馏让多个教师快速同质化，但现有教师权重已冻结，不存在在线塌缩。
- 本轮决策：不增加 peer 变量。

### 14. Reinforced Multi-Teacher Selection for Knowledge Distillation

- 标识：DOI [`10.1609/aaai.v35i16.17680`](https://doi.org/10.1609/aaai.v35i16.17680)，AAAI 2021。
- 实际访问：AAAI 正式页面。
- 方法与结果：按实例动态选择教师；论文实验强调“更强教师不一定产生更强学生”，多教师选择需匹配样本和学生状态。
- 局限：动态选择策略本身需要训练/验证，历史三包又长期参与选型，容易把成员权重学成包特化规则。
- 本项目关系：七成员并非越强越该无限加权；但第一轮若学习权重，就无法区分蒸馏有效性和教师选择过拟合。
- 本轮决策：冻结生产平均；动态选择只在新独立证据下重开。

### 15. Improved Knowledge Distillation via Teacher Assistant

- 标识：arXiv [`1902.03393`](https://arxiv.org/abs/1902.03393)，DOI [`10.1609/aaai.v34i04.5963`](https://doi.org/10.1609/aaai.v34i04.5963)，AAAI 2020。
- 实际访问：AAAI 正式页面。
- 方法与结果：在 CIFAR-10/100 和 ImageNet 上证明教师—学生容量差过大时，直接蒸馏会退化，中间容量 teacher assistant 可以桥接。
- 局限：本项目学生与教师成员同为 PhaseNet；容量差主要来自 ensemble，而不是单体网络深度差。
- 本项目关系：若未来换成 186 KB 级极小学生，teacher assistant 才更可能必要。
- 本轮决策：同架构首轮不增加 assistant。

### 16. Curriculum Temperature for Knowledge Distillation

- 标识：arXiv [`2211.16231`](https://arxiv.org/abs/2211.16231)，DOI [`10.1609/aaai.v37i2.25236`](https://doi.org/10.1609/aaai.v37i2.25236)，AAAI 2023。
- 实际访问：AAAI 正式页面。
- 方法与结果：动态、可学习温度按训练过程调整蒸馏难度，额外计算较小。
- 局限：动态温度增加候选自由度；当前三套历史包不是盲测，事后温度搜索会放大选择偏差。
- 本项目关系：温度可能优化后续训练，但不是验证最小蒸馏假设所必需。
- 本轮决策：首轮固定原始概率标尺，不做温度/课程网格。

### 17. Knowledge Distillation from A Stronger Teacher

- 标识：arXiv [`2205.10536`](https://arxiv.org/abs/2205.10536)，NeurIPS 2022。
- 实际访问：arXiv 原始页面。
- 方法与结果：当强教师与学生预测差异过大时，精确 KL 匹配可能扰乱学生；论文用预测关系/correlation 目标缓解能力差距。
- 局限：本项目同架构学生的能力差距尚未实测，不能预先假定需要 correlation loss。
- 本项目关系：明确反对纯粹、无条件地复制教师；首轮必须保留硬标签对照，并记录教师—真值冲突、学生新增回归和长文件行为。
- 本轮决策：只预注册 `KD-only` 与 `KD+hard` 两个候选；若两者都失败，不在同一历史包上追温度或关系损失。

## 6. 证据综合

17 篇论文与服务器 CPU 实测共同形成以下结论：

1. **继续 PhaseNet 蒸馏，而不是切换到轻量概率特征分类头。** 多教师/同架构蒸馏有直接实验依据；CPU 教师缓存和学生训练成本又远低于预设上限。
2. **学生首选同架构 DiTing PhaseNet。** Born Again Networks 说明同容量学生仍可受益；同架构能减少架构和蒸馏机制的混杂。
3. **缓存稠密 N/P/S soft targets，而不是只缓存最终 picks。** 经典 KD、序列级和稠密预测研究都表明中间概率/结构包含离散标签未保留的信息；最终 picks 仍作为官方评分终点。
4. **第一轮冻结生产平均，不学习教师权重。** LightTS 与 reinforced selection 说明多教师选择可能有效，但当前三包长期参与选型，新增权重自由度会使结果难以解释。
5. **保留真实硬标签候选。** 物理约束拾取与 stronger-teacher 研究都表明强教师不等于无误教师；`KD+hard` 用于检查纯模仿是否复制教师错误。
6. **首轮不叠加 feature relation、contrastive、动态温度、teacher assistant 或在线 mutual learning。** 这些方向有实验依据，但一次引入会破坏可证伪性。
7. **必须整包 LOPO。** 同一历史包随机拆分会泄漏采集条件、站点风格和既有调参痕迹；三套包都只能作为历史回归集，不能冒充盲测。
8. **最终判据是完整生产后处理和四种数量罚。** 训练 loss、概率相似度或单个包上涨都不能替代三包 × 四口径 12 单元、逐文件 FP/FN、P/S 时差分和长记录检查。

## 7. 方法决策矩阵

评分为 `1`（差/高风险）到 `5`（好/低风险）。

| 方法 | 教师信息保留 | 当前数据适配 | 首轮可证伪 | 训练成本 | 超参数风险 | 生产兼容 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| 同架构 PhaseNet、冻结平均概率、dense KD | 4 | 5 | 5 | 5 | 5 | 5 | **首选** |
| 同上 + 真实 Gaussian hard labels | 4 | 5 | 5 | 5 | 4 | 5 | **唯一并列候选** |
| 动态多教师选择/学习权重 | 5 | 2 | 2 | 3 | 1 | 4 | 新独立数据后再开 |
| channel/pair/holistic/contrastive 结构蒸馏 | 5 | 3 | 2 | 2–3 | 2 | 3 | 基础 KD 失败结构明确后再开 |
| 动态温度/curriculum | 4 | 3 | 3 | 4 | 2 | 5 | 首轮不扫网格 |
| teacher assistant | 4 | 2 | 3 | 2 | 3 | 4 | 极小学生阶段再开 |
| 在线 native ensemble / mutual learning | 4 | 2 | 2 | 1–2 | 2 | 3 | 当前冻结教师下拒绝 |
| LogisticRegression/小 MLP 概率特征蒸馏 | 1–2 | 3 | 4 | 5 | 4 | 2 | 仅当 PhaseNet 成本超门槛时备用；实测未触发 |

## 8. 正式实验预注册

固定学生与训练配置：

```text
student          = PhaseNet.from_pretrained("diting")
sampling_rate    = 50 Hz
window_samples   = 3001
window_stride    = 30 s
batch_size       = 2
epochs           = 10
optimizer        = AdamW
learning_rate    = 1e-4
weight_decay     = 0
grad_clip        = 1.0
BatchNorm        = frozen/eval
Dropout          = eval
teacher_target   = frozen g7 averaged N/P/S float16
teacher_long     = first 5 members for records >300s
```

只比较两个候选：

```text
KD-only:
loss = CE(teacher_probability, student_probability)

KD+hard:
loss = 0.7 * CE(teacher_probability, student_probability)
     + 0.3 * CE(hard_gaussian_label, student_probability)
```

硬标签复用项目现有定义：P `sigma=0.2s`，S `sigma=0.3s`。不做温度、alpha、学习率、epoch、成员权重或后处理阈值的事后网格。

三折固定为：

```text
fold 1: train round2 + final08, hold out round1
fold 2: train round1 + final08, hold out round2
fold 3: train round1 + round2, hold out final08
```

泄漏边界：

- held-out 包的概率、答案、loss、epoch 指标和最终分数都不能参与训练、早停或候选选择；
- 固定训练 10 epochs，不使用 held-out early stopping；若需要运行健康检查，只能在训练两包内部固定文件级划分；
- 每折训练完成后才允许对 held-out 包推理；三折 out-of-package（OOP）预测拼成完整三包候选；
- 最终运行与 `g7` 相同的 cap、conditional force-pair、SNR、长记录五成员对应逻辑和四种数量罚；
- 录取要求为 12 单元全部不下降且至少一个严格提升，并同时报告逐文件回归数、FP/FN、P/S 时差分和七个长文件；
- 若 `KD-only` 与 `KD+hard` 都失败，不得围绕同一三包继续调温度、alpha、epoch、成员权重或阈值后宣称通过。

## 9. 本轮结论

本轮不是“论文证明蒸馏一定提分”，而是把下一步从宽泛探索收敛成一个成本已知、变量有限、可严格否证的实验。CPU 权威基准显示全量教师概率缓存预计约 `0.0367h`、平均 float16 N/P/S 约 `62.1 MiB`、最慢单折 10 epoch 约 `0.0553h`，因此 PhaseNet 蒸馏没有计算阻塞。

下一步是在训练服务器生成真实冻结教师缓存，训练三折 `KD-only` 与 `KD+hard` 学生，并只用整包 OOP 预测做最终比较。该服务器只承担训练、特征生成与实验计算，不作为最终生产部署目标；本轮及后续实验不得修改、重启或接管任何生产服务。
