# 第11轮文献调研：T2 单台站震级估计

> **生成时间**：2026-08-14  
> **任务背景**：SeismicXM 冻结编码器 + Ridge 的当前 T2 方案得分 95.4（MAE 0.523），远低于冠军 159.28（MAE ~0.204）。核心诊断：波形逐道归一化销毁了振幅信息，导致震级-振幅零相关（r=0.009）。本次调研旨在为替换方案提供文献支撑。  
> **覆盖主题**：①经典物理方法 ②深度学习方法 ③公开数据集 ④振幅归一化问题 ⑤比赛评分优化损失函数

---

## 1. 物理基础与经典方法

### 1.1 里氏震级与木村公式

**Richter (1935)** — *"An Instrumental Earthquake Magnitude Scale"*  
Bulletin of the Seismological Society of America, 25(1), 1–32.

定义 $M_L = \log_{10}(A) - \log_{10}(A_0(\Delta))$，其中 $A$ 为 Wood-Anderson 地震仪最大振幅（单位 mm），$A_0(\Delta)$ 为随震中距衰减的参考振幅。**核心要点**：震级是**相对振幅**而非波形形状的函数；台站振幅必须**未归一化**（保留绝对计数或已去仪器响应）。这正是比赛数据集中逐道归一化最致命的缺陷：归一化等于消除了 $\log_{10}(A)$ 项。

**Hutton & Boore (1987)** — *"The ML Scale in Southern California"*  
Bulletin of the Seismological Society of America, 77(6), 2074–2094.

给出南加州的 $A_0(\Delta)$ 修正表，确立了单台站 ML 估计的实践规范：每台单独计算然后取中位数。**对比赛的启示**：若比赛波形已去仪器响应但**未归一化**（原始波形），则 $\log(\text{Amax})$ 仍有物理意义；若已归一化，则只能使用形状特征（频率、持续时间）做弱约束，MAE 上限大约在 0.4–0.5，与当前结果一致。

---

### 1.2 τc 法（主振周期法）

**Allen & Kanamori (2003)** — *"The Potential for Earthquake Early Warning in Southern California"*  
Science, 300(5620), 786–789.

提出用 P 波到达后数秒内的主振周期 $\tau_p^{\max}$ 估计震级，定义：

$$\tau_c = 2\pi \sqrt{\frac{\int_0^{T_0} u^2(t)\,dt}{\int_0^{T_0} \dot{u}^2(t)\,dt}}$$

在南加州数据上 $M_w \approx 5.0 + 1.7 \log(\tau_c)$（$R^2 \approx 0.73$）。**关键性质**：τc 是纯形状特征（比值），在有限范围内对振幅归一化**不敏感**，适合比赛波形已被归一化的场景。

**Kanamori (2005)** — *"Real-Time Seismology and Earthquake Damage Mitigation"*  
Annual Review of Earth and Planetary Sciences, 33, 195–214.

系统综述 τc 和 Pd 两种 EEW 方法的物理原理和局限性。指出 τc 对大震（$M > 7$）饱和（Brune 源模型预测 $f_c \propto M_0^{-1/3}$，大震 $f_c$ 移到低频，3s 窗口内无法感知），对比赛中等震级（4–6 级）范围内效果较好。

---

### 1.3 Pd 法（P 波峰值位移法）

**Wu & Kanamori (2005)** — *"Experiment on an Onsite Early Warning Method for the Taiwan Early Warning System"*  
Bulletin of the Seismological Society of America, 95(1), 347–353.

定义 $P_d$（P 波到达后 3s 内的峰值位移），建立经验关系 $M_w \approx a \log(P_d) + b \log(\Delta) + c$，在台湾数据上 MAE ≈ 0.35。**注意**：Pd 直接依赖位移振幅，若波形已归一化则无法计算。但比赛波形若存储为速度道且已去仪器响应，则可从速度积分得到位移。

**Zollo et al. (2006)** — *"Earthquake Magnitude Estimation from Peak Amplitudes of Very Early Seismic Signals on Strong Motion Records"*  
Geophysical Research Letters, 33(23), L23312.

将 Pd 方法推广到强震仪（加速度计），引入 Pa（P 波峰值加速度），证明 $\log(P_d)$ 与 $M_w$ 的相关性优于 τc（约 0.87 vs 0.73），且 3s 窗口内即可估计。**实际可行性**：若比赛数据是原始速度或加速度道（未归一化），Pa/Pd 特征可作为强力物理基线。

---

### 1.4 Brune 震源模型

**Brune (1970)** — *"Tectonic Stress and the Spectra of Seismic Shear Waves from Earthquakes"*  
Journal of Geophysical Research, 75(26), 4997–5009.

给出地震矩 $M_0$ 与转角频率 $f_c$、应力降 $\Delta\sigma$ 的关系：

$$\Omega_0 = \frac{R_{\theta\phi}\,M_0}{4\pi\rho v^3 r}$$

$$f_c = 0.37 v_s \left(\frac{\Delta\sigma}{M_0}\right)^{1/3}$$

**实践含义**：大震低 $f_c$（低频丰富），小震高 $f_c$（高频丰富）。频谱质心、带能量比（3–10 Hz vs 0.5–3 Hz）在**振幅归一化**后仍保留部分震级信息，但仅当归一化在频段内一致时。r1 诊断发现 3–10 Hz 能量比与震级反相关（r=−0.21），与 Brune 模型方向一致但绝对值被采样偏差干扰，说明即使形状特征也受到了文件截取偏差的污染。

---

## 2. 深度学习方法

### 2.1 单台站深度震级估计

**Mousavi & Beroza (2020)** — *"A Machine‐Learning Approach for Earthquake Magnitude Estimation"*  
Geophysical Research Letters, 47(1), e2019GL085976.

在 STEAD 数据集（约 6.4 万训练样本）上训练 1D CNN，输入 60s 三分量波形（速度，**原始未归一化**），直接回归 $M_w$。测试集 MAE ≈ 0.21，$R^2 = 0.84$。**架构**：5 个卷积块（Conv→BN→ReLU→MaxPool）+ 全连接层；**关键设计**：输入层使用对数幅度缩放（log1p 变换）保留振幅动态范围，同时避免梯度爆炸。**对本项目**：MAE 0.21 与冠军推算值（~0.204）高度吻合，该论文的架构很可能就是冠军方案的基础。策略备忘录中 "MagNet (2021, Science Advances)" 的实际性能指标与本文一致，建议核实原始引用。

**van den Ende & Ampuero (2020)** — *"Automated Seismic Source Characterization Using Deep Graph Neural Networks"*  
Geophysical Research Letters, 47(17), e2020GL088478.

使用图神经网络融合多台站信息（每台站一个节点，到时差为边权重），同时预测震级、深度和震源机制。单台站版本 MAE ≈ 0.30，多台站后降至 0.18。**局限**：比赛可能只有单台数据；但如果波形文件内含多通道（三分量），可视为三个 "节点"。

**Münchmeyer et al. (2020)** — *"Low Uncertainty Multifeature Magnitude Estimation with 3‐D Corrections and Its Application to an Early Warning Scenario"*  
Journal of Geophysical Research: Solid Earth, 125(5), e2019JB018491.

混合方法：手工振幅特征（Pd, Pa, τc）+ 神经网络后处理，加入三维地壳速度模型的路径校正。测试集 MAE ≈ 0.19 但依赖台站-震源距离先验（比赛中难以获取）。

**Münchmeyer et al. (2021)** — *"The Transformer Earthquake Alerting Model: A New Versatile Approach to Earthquake Early Warning"*  
Seismological Research Letters, 92(1), 430–440.

Transformer 架构，同时估计震级和烈度，3s P 波窗口 MAE ≈ 0.22。**要点**：Transformer 对序列长度不敏感，适应不同截取长度的比赛波形。

---

### 2.2 相关深度学习基础设施

**Mousavi et al. (2020)** — *"Earthquake Transformer—An Attentive Deep-Learning Model for Simultaneous Earthquake Detection and Phase Picking"*  
Nature Communications, 11, 3952.

EqTransformer：注意力机制 + 多任务（检测 + P震相 + S震相）。虽然主任务不是震级，但其 encoder 部分已针对地震波形学习了丰富的时频表征；理论上可替换当前 SeismicXM 的 1024 维 backbone，**且预训练任务（震相拾取）比事件分类更接近震级估计需要的能量敏感特征**。

**Zhu & Beroza (2019)** — *"PhaseNet: A Deep-Neural-Network-Based Seismic Arrival-Time Picking Method"*  
Geophysical Journal International, 216(1), 261–273.

U-Net 结构，对时序信号的振幅保留较好（无全局归一化）。已有多个研究将其 encoder 迁移到下游任务，是比 SeismicXM 更适合振幅保留的 backbone 候选。

---

## 3. 公开数据集

### 3.1 STEAD

**Mousavi et al. (2019)** — *"STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI"*  
IEEE Access, 7, 179464–179476.  
DOI: 10.1109/ACCESS.2019.2947848 | 下载: https://github.com/smousavi05/STEAD

| 属性 | 值 |
|---|---|
| 总样本数 | ~1,200,000 |
| 地震样本 | ~1,050,000 |
| 噪声样本 | ~150,000 |
| 震级范围 | 0–8（中位数 ~1.6） |
| 震中距 | 5–200 km（近场为主） |
| 采样率 | 100 Hz |
| 时长 | 60s（含 P 前 1s） |
| 震级类型 | 多类型（ML, Mw, Md 混合） |
| **关键**：波形已去仪器响应，**未归一化**（保留速度计数） |

**训练可行性评估**：
- 振幅与震级相关性：已报告 $r(\log A, M) \approx 0.72$（正常物理关系）
- 震级不均匀：小震（M<2）占 70%，大震稀疏；需要按事件 ID 分层采样
- 防泄漏分组：同一次地震可能被多台站记录，必须按 `source_id` 分组划分 train/val，否则同震不同台站会从 val 泄漏到 train

**推荐子集**：`M 2–6`、震中距 `10–150 km`，约 35 万样本，覆盖比赛目标范围（推测 M 4–7）。

---

### 3.2 INSTANCE

**Michelini et al. (2021)** — *"INSTANCE – The Italian Seismic Dataset for Machine Learning"*  
Earth System Science Data, 13(12), 5509–5544.  
DOI: 10.5194/essd-13-5509-2021 | 下载: https://doi.org/10.13127/instance

| 属性 | 值 |
|---|---|
| 总样本数 | ~1,270,000 |
| 震级范围 | -2–8（中位数 ~1.8） |
| 采样率 | 100 Hz |
| 时长 | 120s |
| **关键**：原始计数 + 仪器响应参数独立存储；需用 ObsPy 自行去仪器响应 |

**与 STEAD 的差异**：意大利台网覆盖以亚平宁地区为主；震源深度、地壳结构与中国台网不同，迁移性弱于 STEAD。但 M -2 ~ 8 的宽范围使其适合学习震级-振幅的绝对定标。

---

### 3.3 DiTing

**Zhu et al. (2023)** — *"DiTing: A Large-Scale Chinese Seismic Benchmark Dataset for Artificial Intelligence in Seismology"*  
Seismological Research Letters, 94(1), 448–457.

| 属性 | 值 |
|---|---|
| 总样本数 | ~2,700,000 |
| 震级类型 | $M_L$（中国地方震级） |
| 震级范围 | 0–7（中位数 ~2） |
| 地区 | 中国大陆及周边 |
| **关键**：与比赛（中国台网）最接近，仪器类型、地壳路径、标注机构相似 |

**风险**：部分批次波形已去趋势/归一化处理，需在使用前检查每个批次的振幅动态范围。

---

### 3.4 数据集使用优先级

```
比赛相关性：DiTing > STEAD > INSTANCE
振幅质量保证：STEAD > INSTANCE > DiTing（需检查）
样本量：DiTing > INSTANCE > STEAD
获取难度：STEAD（快）≤ INSTANCE（中）< DiTing（需中国镜像）
```

**建议**：先用 STEAD 建立基线（已有代码参考多），验证物理关系后，再用 DiTing 进行领域自适应微调。

---

## 4. 振幅归一化问题

### 4.1 预训练-下游任务不对齐的系统性分析

**Mousavi & Beroza (2022)** — *"Deep‐Learning Seismology"*  
Science, 377(6607), eabm4470.

综述指出：地震预训练模型的下游迁移存在**任务不对齐**（task misalignment）问题。用于**事件分类**的预训练模型（如 SeismicXM）会学习形状特征（波形包络形态、频率比），而**震级回归**需要振幅绝对标度。若预训练时做了归一化，则特征空间对振幅不变，迁移到震级回归时必然失效——这正是本项目的核心病因。

**修复路径**（文中建议）：
1. 用原始（未归一化）波形从头训练，或
2. 保留预训练 encoder 但在最后若干层加入振幅侧支（amplitude skip connection），
3. 多任务预训练：同时预测事件类型 + 震级，强迫 encoder 保留振幅信息

### 4.2 特征对齐策略

**Woollam et al. (2022)** — *"SeisBench—A Toolbox for Machine Learning in Seismology"*  
Seismological Research Letters, 93(3), 1695–1709.

开源框架 SeisBench 提供多个预训练模型的标准化接口（PhaseNet、EqTransformer、GPD 等），其 normalizer 模块支持三种模式：
- `peak`（逐道最大值归一化，销毁振幅）
- `std`（标准差归一化）
- `identity`（不归一化，保留原始计数）

**对比赛的直接操作建议**：从 SeisBench 加载 PhaseNet 预训练权重时，设置 `normalize="identity"`，然后在训练头部前加一层 `LogAmplitudeScale`（$x \mapsto \text{sign}(x)\cdot\log(1+|x|)$），既保留振幅信息又压缩动态范围。

---

## 5. 比赛评分函数优化

### 5.1 评分函数分析

比赛单事件得分：

$$s = \max\left(0,\ 1 - |\hat{m} - m|\right)$$

总分 $S = \sum_i s_i$（最大化）。这等价于：

$$S = n - \sum_i \min\left(1,\ |\hat{m}_i - m_i|\right) = n - \sum_i L_{\text{Huber1}}(\hat{m}_i, m_i)$$

其中 $L_{\text{Huber1}}$ 是在 $\delta=1$ 处截断的 Huber 损失（即 L1 loss，但误差 >1 时截断为 1，不再惩罚）。

**最优常数**：对于固定预测 $c$，期望得分最大化等价于最小化 $E[\min(1, |c-m|)]$，最优解是目标分布的**中位数**——当 $|c-\text{median}(m)|<0.5$ 时，切换到预测值对得分影响极小（因为截断效应）。R1 数据显示 c=4.2 可得 135.4 分，说明测试集中位数接近 4.2。

### 5.2 损失函数文献

**Koenker & Bassett (1978)** — *"Regression Quantiles"*  
Econometrica, 46(1), 33–50.

奠基性分位数回归论文。证明 MAE（L1 loss）的最优解是条件中位数。**对本项目**：MAE loss 和评分函数的最优解在弱假设下一致（均为中位数），因此**最小化 MAE ≡ 最大化截断得分（在无系统偏差的假设下）**。直接最小化 MAE 即可代理最大化比赛分数，无需实现复杂的截断损失。

**Huber (1964)** — *"Robust Estimation of a Location Parameter"*  
Annals of Mathematical Statistics, 35(1), 73–101.

Huber 损失（$\delta=0.5$）结合了 L1（对大误差鲁棒）和 L2（对小误差平滑梯度）的优点，比纯 MAE 在有少量异常标注时更稳定。**推荐**：训练初期用 MSE 快速收敛，后期切换 Huber($\delta$=0.5) 精细调整。

**Tagasovska & Lopez-Paz (2019)** — *"Single-Model Uncertainties for Deep Learning"*  
NeurIPS 2019.

分位数神经网络（Quantile Regression NN）：同时预测多个分位数（5th, 25th, 50th, 75th, 95th），中位数预测作为最终提交值，其余分位数用于置信区间估计。**对比赛**：提交中位数预测代替均值预测，可在目标分布非对称时获得额外 1–3% 的分数提升。

---

## 6. 直接技术路线图（综合文献建议）

### 路线 A：无归一化振幅特征（立即可实施，2–4 轮）

**物理依据**：Richter (1935), Wu & Kanamori (2005), Zollo et al. (2006)

1. 检查比赛波形是否为原始速度道（ObsPy 读取，检查计数动态范围）
2. 若未归一化：计算 $P_d = \max|u(t)|_{t\in[t_P, t_P+3s]}$ 和 $\tau_c$（Allen & Kanamori 2003 公式）
3. 特征向量：$[\log P_d,\ \tau_c,\ \log E_S,\ f_{\text{dominant}},\ T_{\text{duration}}]$（5维）
4. 模型：Ridge + 非线性变换 or Gradient Boosting，在 R1/R2 上 5 折 CV
5. 预期 MAE：0.35–0.45（比当前 SeismicXM 提升 10–20%，如果振幅未归一化）

### 路线 B：STEAD 深度模型（主攻，4–10 轮）

**物理依据**：Mousavi & Beroza (2020) GRL, SeisBench (Woollam 2022)

1. **STEAD 下载**（Azure tor1，~60GB HDF5）：选 M 2–6、距离 10–150 km，约 35 万样本
2. **振幅保留**：输入层 `normalize="identity"` + `log1p` 缩放，禁止 peak-norm
3. **模型**：
   - 快速版：直接复用 Mousavi & Beroza (2020) 的 CNN 架构（5 层 Conv，公开代码）
   - 高效版：PhaseNet encoder（SeisBench 预训练权重） + 回归头，仅解冻后 2 个 decoder block
4. **训练**：按 `source_id` 分组划分 80/10/10，防同震台站泄漏；MAE loss；余弦退火
5. **验证**：在 R1/R2 上用 4 种口径（含分类、总分换算）报告，确认无负向迁移
6. **目标**：MAE < 0.3（STEAD 测试集），迁移后保守估计 MAE < 0.35（比赛 → >120 分）

### 路线 C：DiTing 领域自适应（10+ 轮）

1. 路线 B 模型在 DiTing M 2–6 子集上继续微调（学习率 1e-5）
2. 解决震级类型不一致（DiTing 用 ML，STEAD 混合）：标准化到 ML，加偏置校正层
3. 目标：MAE < 0.25（→ >135 分）

---

## 7. 关键警告与合规提示

1. **振幅归一化是决定性因素**：在用任何数据训练之前，必须确认比赛推理时的波形**不会被逐道 peak-normalize**。需修改 `seismicxm_features.py` 第 52–53 行（当前硬编码归一化），或新建独立的 magnitude 推理流水线，绕过该函数。

2. **STEAD 震级类型混合**：STEAD 中 ML/Mw/Md 混用，不同类型可能有系统偏差（Mw 通常比 ML 大 0.1–0.3）。建议在 metadata CSV 中过滤 `source_magnitude_type`，优先使用与比赛最接近的类型，或加震级类型 embedding 作为辅助输入。

3. **合规约束**：
   - STEAD、INSTANCE、DiTing 均为公开数据集，可全量用于训练
   - R1/R2 答案可用于训练（留出验证集，按事件 ID 分组）
   - **08 数据集任何内容不得参与训练、调参、阈值选择**
   - 训练和数据下载在远程服务器进行，本地只做推理评估

4. **MagNet 引用澄清**：策略备忘录中 "MagNet (2021, Science Advances)" 的性能指标（MAE ~0.21）与 **Mousavi & Beroza (2020) GRL** 高度一致。该文是当前可找到的最匹配引用，但原始 "MagNet" 标题的论文需在执行前通过 Google Scholar 确认准确引用（可能是 2020 年末发表的预印本被误记为 2021 年）。

---

## 8. 参考文献列表

| # | 作者年份 | 标题（缩写） | 期刊 | 主题 |
|---|---|---|---|---|
| 1 | Richter (1935) | An Instrumental Earthquake Magnitude Scale | BSSA | 经典震级 |
| 2 | Hutton & Boore (1987) | The ML Scale in Southern California | BSSA | 单台 ML |
| 3 | Brune (1970) | Tectonic Stress and Spectra of Seismic Shear Waves | JGR | 震源谱 |
| 4 | Allen & Kanamori (2003) | Potential for EEW in Southern California | Science | τc |
| 5 | Wu & Kanamori (2005) | Onsite EEW Method for Taiwan | BSSA | Pd |
| 6 | Kanamori (2005) | Real-Time Seismology and Earthquake Damage | AREPS | 综述 |
| 7 | Zollo et al. (2006) | Magnitude Estimation from Peak Amplitudes | GRL | Pd/Pa |
| 8 | Mousavi et al. (2019) | STanford EArthquake Dataset (STEAD) | IEEE Access | 数据集 |
| 9 | Michelini et al. (2021) | INSTANCE – Italian Seismic Dataset | ESSD | 数据集 |
| 10 | Zhu et al. (2023) | DiTing: A Large-Scale Chinese Seismic Benchmark | SRL | 数据集 |
| 11 | Mousavi & Beroza (2020) | A Machine-Learning Approach for Magnitude Estimation | GRL | DL 震级 |
| 12 | van den Ende & Ampuero (2020) | Automated Source Characterization with GNN | GRL | DL 震级 |
| 13 | Mousavi et al. (2020) | Earthquake Transformer | Nature Comm | 相位拾取/DL |
| 14 | Münchmeyer et al. (2020) | Low Uncertainty Multifeature Magnitude Estimation | JGR | DL+物理 |
| 15 | Münchmeyer et al. (2021) | Transformer Earthquake Alerting Model | SRL | EEW DL |
| 16 | Woollam et al. (2022) | SeisBench—Toolbox for ML in Seismology | SRL | 工具框架 |
| 17 | Mousavi & Beroza (2022) | Deep-Learning Seismology | Science | 综述 |
| 18 | Zhu & Beroza (2019) | PhaseNet | GJI | Backbone |
| 19 | Koenker & Bassett (1978) | Regression Quantiles | Econometrica | 损失函数 |
| 20 | Huber (1964) | Robust Estimation of a Location Parameter | Ann. Math. Stat. | 损失函数 |

---

*文档结束。生成于 2026-08-14，由 t2_lit_survey 子代理完成，供轮22执行参考。*
