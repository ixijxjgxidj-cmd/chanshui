# 第 10 轮论文：远场粗大错时、长记录过检，与 T2/T3 合规重建

## 0. 通道与核验口径（如实标注）

**anysearch 与 Playwright MCP 在本会话依然不可用**（`list_mcp_resources` 返回空列表）。
本轮走 HTTP 通道：arXiv Atom API、Crossref REST、OpenAlex works API（Semantic Scholar
全程返回 0 条，疑似限流，已在检索日志中留痕）。脚本为 `scripts/lit_search.py` +
本轮新增 `scripts/round10_shortlist.py`。

检索规模：5 个主题、10 条检索式，命中 862 条，去重后 762 条，主题打分后 326 条进入
候选池，按主题各取 6 条共 **30 篇**进入精读与核验。

核验分两级，**分别标注，不混为一谈**：

| 级别 | 含义 | 本轮数量 |
|---|---|---:|
| direct | 直接抓取 arXiv abs 页或 DOI landing page，HTTP 200 且标题 token 命中率 ≥0.6 | **20** |
| crossref | 出版商页面 403/JS 墙，改用 Crossref `/works/{doi}` 权威元数据核验 | **10** |

30 篇的标题 token 命中率**全部为 1.0**。原始转储 `memory/papers/_raw/round10_q_*.json`、
`round10_scored.json` 已 gitignore；`round10_verified.json`、`round10_crossref.json` 提交入库。

## 1. 本轮要回答的三个问题

1. 实验 015 诊断出的**远场欠拟合**（粗大错时窗 S-P 中位 26.10s，训练池 73% 是 S-P<10s），
   除了按 S-P 分箱重采样，文献还有没有更根本的处方？
2. R2 两个长文件贡献 23.3% 失分、`T1.A.Q0001` 单文件 `extra_p=43/extra_s=46`，
   **长连续记录过检**该怎么治？
3. 本轮新发现的**合规空洞**：T2/T3 现有全部资产都由第 1/2 轮真题拟合而来
   （含 `psdelta` 的最小二乘系数），公开数据能不能重建？

## 2. 主题 FARFIELD：远场/区域距离（6 篇）

| # | 标题 | 年 | 核验 | 标识 |
|---:|---|---:|---|---|
| 15 | Picking Regional Seismic Phase Arrival Times with Deep Learning | 2025 | direct | 10.26443/seismica.v4i1.1431 |
| 16 | Generalization Across Tectonic Regions for a Deep-Learning P-Polarity Picker | 2026 | crossref | 10.22541/essoar.177307752.25366010/v1 |
| 18 | Regional phase picking on single stations using deep learning | 2023 | direct | 10.5194/egusphere-egu23-1831 |
| 21 | Effects of Network-Specific Training and Waveform Denoising on ML-Based Seismic Phase Picking | 2026 | direct | 10.21203/rs.3.rs-9268145/v1 |
| 23 | Test-Time Augmentations and Quality Controls for Improving Regional Seismic Phase Picking | 2025 | crossref | 10.3390/s25237238 |
| 25 | Picking Induced Seismicity with Deep Learning (piSDL) | 2025 | direct | 10.26443/seismica.v4i2.1579 |

### 2.1 关键发现：文献把「窗长」而不是「采样比例」当作远场第一处方

三篇独立工作同时指出**输入窗太短**是区域/远场拾取退化的主因，而不是（或不只是）
训练样本的距离分布：

- **#15（CREW，Seismica 2025）**：面向 ≤20° 的区域相位，训练数据是
  **5 分钟长三分量记录**，1.6M 波形 / 3.2M 标注到时；明确对比了「在局部地震记录上
  训练的既有 ML 模型」并指出其在区域距离上不适用。
- **#18（EGU 2023，Scandinavia 200–2000km）**：P 与 S 间隔**可达 3 分钟**，
  因此把训练片段取到 **324 s** "to capture the multiple arrivals"。
- **#24（GJI 2024，北欧/北极，见 §3）**：`most studies focus on local events and
  short time windows as the input to the detection models`，并因此改造 PhaseNet。

这对本项目是**直接的机制解释**：我们窗长 60.02s，而粗大错时窗 S-P 中位 26.10s、
p90 37.30s——S 已被推到窗尾，可用上下文严重不足。实验 015 原设计（按 S-P 分箱
上采样）治的是**样本占比**，而文献指向的是**感受野/窗长**。两者不冲突，但优先级
应当反过来。

### 2.2 但窗长在本仓 PhaseNet 上不是自由参数（本轮实测否证）

真实实测（CPU，仅前向，不训练）：SeisBench `PhaseNet` 的解码器有为 3001 点
写死的 pad 与 crop（`F.pad(x,(2,3))` / `x[:, :, 1:-2]`），改窗长即触发
`_merge_skip` 尺寸不匹配：

```text
ns=3001   missing=0 unexpected=0 out=(1, 3, 3001)
ns=6001   FAIL RuntimeError: Expected size 1501 but got size 1
ns=12001  FAIL RuntimeError: Expected size 750 but got size 1
```

穷举 2900–3200 与若干长度后，仅 **95 个近 3001 的长度**与 2 的幂对齐长度
（3072/4096/6144/8192/12288/16384）能前向。**结论：把窗长从 3001 拉长到 5 分钟
不是改一个参数，而是要换模型族或改架构。** 这是本轮最重要的可执行性约束，
必须写进下一步方案，避免按文献直觉盲目开工。

### 2.3 piSDL（#25）给出的另一条线：噪声样本进训练集压误报

piSDL 用 171,182 条波形 / 40,576 事件训练，并明确
`Noise samples were added in the training data set to reduce the number of false picks`。
这对问题 2（长记录过检）是**训练侧**处方，与我们已证否的推理侧兜底/SNR 闸互补，
且不在 `failed-experiments.md` 的禁止列表内。

### 2.4 #23 的 TTA 与本仓已证否 TTA 不是同一件事

#23 在 Pn（200–2000km）上做 test-time augmentation + 质量控制。注意本仓
`failed-experiments.md` 已证否的是**极性翻转 TTA**；#23 的增强是移位/尺度类，
属于不同操作。但鉴于本轮 §2.2 的窗长约束与既有 overlap 扫描（0.75/0.9 已证否），
此路优先级低。

## 3. 主题 FP_LONG：长记录过检与关联（6 篇）

| # | 标题 | 年 | 核验 | 标识 |
|---:|---|---:|---|---|
| 7 | Comparison of seismic phase association algorithms and their performance | 2023 | direct | 10.5194/egusphere-egu23-6806 |
| 11 | Application of Neural Network Automatic Event Detection for Reservoir-Triggered Seismicity | 2026 | crossref | 10.3390/s26030783 |
| 12 | Micro-seismic events detection and its tectonic implications in NE Hainan | 2023 | direct | 10.3389/feart.2023.1169877 |
| 17 | Improving Deep Learning-Based Seismic Phase Picking by Addressing Label Imbalance | 2026 | direct | 10.21203/rs.3.rs-10439246/v1 |
| 19 | Earthquake transformer (EQTransformer) | 2020 | crossref | 10.1038/s41467-020-17591-w |
| 24 | Deep learning models for regional phase detection in N Europe / European Arctic | 2024 | crossref | 10.1093/gji/ggae298 |

### 3.1 #17：面积加权软交叉熵——直接对应我们的失分结构

#17 指出语义分割式拾取存在**标签面积失衡**（P/S 窄标签被背景标签淹没），
且**在单条记录含多个相位时（余震序列）检测性能显著退化**——这正是 R2 长文件
`T1.A.Q0001`（extra_p=43/extra_s=46）的形态。处方是
`area-weighted soft cross-entropy loss`，**不改架构**，在 SegPhase 与 PhaseNet
两个架构上都验证有效。

对本项目意义：这是一个**只改损失函数**的改动，不触碰 §2.2 的窗长约束，
可直接接到 `scripts/finetune_phasenet.py`。且我们已证否的是
「S sigma 0.3→0.2」（标签宽度）与「annotation gap mask」，**面积加权损失是不同的量**
（权重而非宽度），不属于重复实验。

### 3.2 #19 EQTransformer：检测头与拾取头联合

EQT 的 detection 通道给出"这里到底有没有事件"的显式监督，理论上正是长记录过检
的对症药。但本仓 `failed-experiments.md` 已证否
「T1 长记录 event confidence」，说明**推理侧**用检测置信度过滤没能生效。
文献支持的是**训练侧**联合监督，两者不同；不过这需要换到 EQT 架构（本仓集成是
PhaseNet 族，概率曲线逐点平均），代价高，列为备选而非首选。

### 3.3 #7 / #12：关联（association）在单台站场景不可用

#7 比较三种关联算法、#12 用 REAL+VELEST+HypoDD，都依赖**多台站**几何。
本赛题按文件/台站独立评分，**没有可用的台网关联信息**，因此这条路不适用。
如实记录以免后续重复检索。
## 4. 主题 T2_MAG：震级估计（6 篇）

| # | 标题 | 年 | 核验 | 标识 |
|---:|---|---:|---|---|
| 20 | Earthquake Magnitude Estimation Using ML from Single Station Seismic Data in Rwanda | 2026 | direct | 10.21203/rs.3.rs-9532090/v1 |
| 26 | End-to-end DL for epicentral distance and magnitude from single station waveforms | 2026 | crossref | 10.2139/ssrn.6512019 |
| 27 | Earthquake magnitude and location estimation with a transformer network | 2021 | direct | 10.1093/gji/ggab139 |
| 28 | Real-time magnitude prediction with ML ensemble + CTGAN synthetic data | 2025 | direct | 10.1016/j.geog.2024.10.001 |
| 29 | Rapid magnitude estimation for Taiwan EEW via frequency-domain multi-station DL | 2026 | crossref | 10.1007/s44195-026-00121-4 |
| 30 | Exploring a CNN Model for Earthquake Magnitude Estimation using HR-GNSS data | 2023 | direct | 2304.09912v1 |

### 4.1 #26 的伪归一化：解决"归一化杀掉幅值"这个核心矛盾

震级本质依赖**绝对幅值**，而神经网络输入必须归一化——#26 明确点出这个冲突
（`the conflict between neural network input normalization and the loss of
amplitude information required for magnitude scaling`），处方是
**pseudo-normalization**：归一化输入但**把归一化因子作为额外输入保留**，
让网络把"波形形状"与"幅值标定"解耦。同时输出震级与震中距。

这对我们直接可用：本仓 `PSDeltaMagnitude` 的特征恰好是
`(log10 A_max, log10 Δt_SP)`，其中 `log10 A_max` 就是幅值标定项、
`log10 Δt_SP` 是距离代理。也就是说**我们的手工公式与 #26 的结构同构**，
只是用 3 参线性回归代替了网络。因此合规重建**不需要换方法，只需要换拟合数据**。

### 4.2 #20 的特征表：给合规重建提供现成特征清单

#20 用 P 波到时附近 **3 秒窗** + 物理特征：log 幅值、累积位移、信号能量、
RMS 幅值、SNR、主频、频谱重心、1–10Hz 带限能量，比较 RF / CNN / 混合。
这份清单可以直接扩充我们 T2 的 2 维特征（当前只有幅值与 S-P）。

### 4.3 #28 的类不平衡警告

#28 指出震级数据集在**大震段严重不平衡**，导致高震级预测误差大，用 CTGAN 合成
补齐。我们的 T2 若用公开数据重建，必须检查震级直方图并报告分段 MAE，
**不能只报总体 MAE**——否则会重复去年 baseline"总体 MAE 0.817 但大震段崩"的风险。

### 4.4 不适用的部分

#27（多台站动态集合 transformer）、#29（多台站频域 FNO+LSTM）都需要**台网同时输入**，
本赛题单文件/单台站评分，不适用。#30 用 HR-GNSS，本赛题只有测震三分量，不适用。
如实记录以免重复检索。

## 5. 主题 T3_CLS：事件类型分类（6 篇）

| # | 标题 | 年 | 核验 | 标识 |
|---:|---|---:|---|---|
| 1 | Severe Class Imbalance versus Near-Balanced Regimes in Seismic Hazard Classification | 2026 | crossref | 10.2139/ssrn.7153278 |
| 2 | Interpretable Physics-Informed Multi-Stream DL for Discrimination | 2026 | direct | arXiv 2602.15993v1 |
| 3 | DL and transfer learning of earthquake / quarry-blast discrimination (S. California, E. Kentucky) | 2023 | crossref | 10.1093/gji/ggad463 |
| 4 | ML-based seismic event classification, Czech Regional Seismic Network | 2026 | direct | 10.5194/egusphere-egu26-13017 |
| 5 | Applied research of DL in classification of earthquake and blasting event | 2023 | direct | 10.21203/rs.3.rs-3024143/v1 |
| 6 | Exploration of ML Methods to Seismic Event Discrimination in the Pacific Northwest | 2026 | direct | 10.26443/seismica.v5i1.2068 |

### 5.1 #6 与 #2 共同锁定 PNW 数据集为 T3 合规重建的落点

- **#6（Seismica 2026）**：用 AI-curated PNW 数据集 **~200k 三分量波形 / >70k 事件**，
  做**四分类**：earthquakes / explosions / surface events / noise，比较 RF（TSFEL、
  物理特征、scattering 特征）与 CNN（1D 时序、2D 频谱图），并在平衡测试集上基准化。
- **#2（arXiv 2602.15993）**：同样在 Curated PNW AI-ready 数据集上，物理信息多流架构
  （SincNet 时域 + 多分辨率频谱 + 物理分支 + BiLSTM）达到 **97.56%**，并超过
  经典 P/S 幅值比方法与标准 CRNN。

这两篇给出**明确可执行的合规路径**：PNW 是公开数据集，含 `source_type` 类别标签，
四类语义与本仓 T3 的 1..5 体系可建立映射（天然地震 / 爆破 / 其它）。
**这解决了交接摘要中"T3 需合成多事件记录、工作量大、需用户裁决"的难题**——
不必合成，公开数据本身就有事件类型标签。

### 5.2 #3 的类不平衡处方与"按事件而非按台站"切分

#3 在数据丰富的南加州直接训 CNN、对爆破做**重复采样增广**以缓解类不平衡，
单台站 F1 >83.5%、台网平均 >98.1%；数据稀少的东肯塔基改用**迁移学习**。
#5 特别强调切分纪律：**同一事件的不同台站波形不得跨训练/验证/测试集**——
与本仓 `--split-mode event-hash` 的事件散列切分一致，可直接复用。

### 5.3 #1 的警告：不平衡与近平衡两种口径必须都报

#1 系统比较严重不平衡与近平衡两种数据体制下的分类表现。落到本项目：
T3 重建必须同时报告**原始不平衡分布**与**平衡测试集**两种准确率，
否则"整体 accuracy 高"可能只是多数类占比高的假象。

## 6. 主题 DATA_XFER：迁移与基准（6 篇）

| # | 标题 | 年 | 核验 | 标识 |
|---:|---|---:|---|---|
| 8 | Which picker fits my data? | 2021 | direct | 10.1029/2021JB023499 |
| 9 | SeisBench (EGU 2021) | 2021 | direct | 10.5194/egusphere-egu21-12218 |
| 10 | A fine-tuning workflow for automatic first-break picking | 2024 | direct | arXiv 2404.07400v1 |
| 13 | OBSTransformer: automated labelling + transfer learning | 2023 | direct | arXiv 2306.04753v1 |
| 14 | SeisBench—A Toolbox for ML in Seismology (SRL) | 2022 | crossref | 10.1785/0220210324 |
| 22 | Seismic Arrival-time Picking on DAS using Semi-supervised Learning | 2023 | direct | arXiv 2302.08747v2 |

### 6.1 #8 是本项目跨域取舍的方法论依据（第 9 轮已引，本轮复核）

#8 系统研究**跨域**场景（训练域与应用域特性不同）下各拾取器的表现差异。
本届广西、去年四川，R1/R2/08 全属域外，因此我们的取舍准则
「跨区域最坏表现不退化」有文献支撑，而不是「R1/R2 均分最大化」。

### 6.2 #13 OBSTransformer 的自动标注思路

#13 面对"目标域无人工标注"，用既有模型在目标域产出**自动标签**再迁移微调。
本项目若要用广西本地公开数据但缺到时标注，这是可行范式。但需注意：
自动标签的误差会被学进去，且我们**不能**用任何封存包做自动标注源。

## 7. 本轮结论与下一轮方案（按可执行性排序）

### 7.1 已被本轮实测否证的方向

**加长窗到 5 分钟不可行**（§2.2 实测）：SeisBench PhaseNet 解码器为 3001 写死
pad/crop，非自由参数。文献 #15/#18/#24 的"长窗"处方要么换架构、要么改 SeisBench
源码，代价与风险都高于收益，**本轮不采纳**，并写入负结果备忘以免重复。

### 7.2 首选：T2/T3 合规重建（解决合规空洞，且是记分板独立列）

依据 §4.1（伪归一化/幅值-距离解耦，与本仓 psdelta 同构）、§4.2（物理特征清单）、
§5.1（PNW 有 source_type 标签，四分类）、§5.2（按事件切分 + 少数类重采样）。

- T2：用公开数据（ETHZ `source_magnitude`、CWA `source_magnitude`、PNW）
  重新拟合 psdelta 三系数，并按 §4.3 报分段 MAE。
- T3：用 PNW `source_type` 四分类重建，按 §5.3 同时报不平衡与平衡两口径。
- **价值**：这是唯一能把 T2/T3 从"全部资产不合规"变成"全部资产合规"的路径，
  且记分板上震级估计与地震分类是独立列。

### 7.3 次选：面积加权软交叉熵（§3.1，#17）

只改损失函数、不动架构、不受窗长约束，且与已证否项（sigma 收紧、gap mask）
是不同的量。需 GPU。

### 7.4 备选：训练集掺噪声窗压误报（§2.3，piSDL #25）

训练侧治长记录过检，与已证否的推理侧兜底/SNR 闸互补。需 GPU。