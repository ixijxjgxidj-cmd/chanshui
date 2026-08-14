# 轮 22 文献精读：无距离单台站震级估计、截窗饱和与跨域迁移

> 日期：2026-08-15  
> 检索：OpenAlex 102 篇初筛 + Crossref 30 篇 DOI 核验；随后沿 MagNet、TEAM、GNN、STEAD、Pd/τc 六条引用链扩展出 56 篇严格主题相关论文。  
> 精读：逐篇读取 DOI 元数据与原始摘要，共 19 篇；排除“预测未来地震”“灾害评估”“检测/相位关联”等偏题论文。  
> 工具边界：当前环境未暴露 anysearch / Playwright 工具，故用 OpenAlex 与 Crossref 双源完成检索；未把自动排名当作精读。

## 一、直接改变本项目设计的五条证据

### 1. MagNet 是冠军级路线的直接证据

**Mousavi & Beroza (2019/2020), GRL, DOI 10.1029/2019GL085976, 299 引**。

原文结论：端到端 CNN+RNN 从单台原始波形估计震级；网络“not sensitive to data normalization,
hence waveform amplitude information can be utilized”，能从训练数据直接学习距离与场地依赖函数；
单台预测误差均值近零、标准差约 0.2。这个数值与冠军 159.28 分对应的 MAE≈0.204 高度吻合。

**对本项目：** 当前 SeismicXM 在输入端逐道 peak-normalize，恰好违反这条设计。轮 22 主路线应复刻
“保留跨样本绝对振幅 + CNN 时域压缩 + RNN/attention 聚合”，而不是冻结分类 backbone 后 Ridge。

### 2. 大震低估不是普通过拟合，而是截窗饱和的物理上限

**Münchmeyer et al. / AIMag (2024), BSSA, DOI 10.1785/0120230171**：
2.4M 条 P 波、单台、无位置先验；114s 窗在 M2.3–7.6 可靠。窗口从 114s 缩到 1s 时，
高震级开始低估的拐点向更低 M 移动，低估程度加剧。

**Melgar et al. (2019), JGR, DOI 10.1029/2018JB017093, 99 引**：
14 万条 M4.5–9 波形证明 PGD–M 标度在由窗长决定的震级处发生饱和。

**Meier et al. (2018), GRL, DOI 10.1029/2018GL078991**：
俯冲带约 30s 可达残差 0.01±0.28，但 Mw≥7.5 仍系统低估 −0.70±0.30。

**对本项目：** T2 的 60s P 居前窗口必须做大震重采样、分位数回归/不确定性建模和饱和校准；
不能只靠普通 L1 在 STEAD 的小震长尾上训练。

### 3. 比赛数据条件已有近乎同构的成功范式

**Real-Time Detection & Magnitude using ViT (2022), JGR, DOI 10.1029/2021JB023657, 79 引**：
明确使用**非归一化 30s 窗（P 前1s+后29s）**做单台震级回归。

**Complex CNN (2021), IEEE GRSL, DOI 10.1109/LGRS.2021.3059422, 39 引**：
单台原始三分量 1min STEAD，同时预测震中距/深度/震级；震级 MAE=0.26。

**对本项目：** 官方 T2 多数为 60s/100Hz，正好与上述 1min 单台设置同构。应同时尝试：
时域 LogAmp-CNN+GRU，以及 STFT 复数/双通道 CNN；无需依赖 S-P。

### 4. 中国域迁移必须显式做，不能假设 STEAD 直接可用

**TLDCNN-M (2023), Frontiers in Physics, DOI 10.3389/fphy.2023.1070010**：
中国强震动数据上迁移学习优于无迁移 CNN；芦山 M6.1 在 P 后 3s 误差<0.5、10s<0.2。

**SVM transfer Sichuan–Yunnan (2022), BSSA, DOI 10.1785/0120210232**：
区域迁移能改善快速定级，说明震级映射包含区域/仪器响应差异。

**DiTing (2023), Earthquake Science, DOI 10.1016/j.eqs.2022.01.022, 78 引**：
273万三分量、78.7万中国区域事件、180s/50Hz，含 M0–7.7、距离、P/S、SNR，是最合适的中国域二阶段预训练集。

**对本项目：** STEAD→DiTing→R1/R2 固定留出是合理链路；R1/R2 可训练但必须事件/样本固定留出；
08 永远只读。

### 5. 物理特征混合与“禁用归一化”得到独立验证

**MEANet (2022), Geophysics, DOI 10.1190/GEO2022-0196.1**：
物理特征时间序列+attention，3s 单台误差标准差±0.25，优于单一 τc/Pd。

**E3WS (2023), JGR, DOI 10.1029/2023JB026575**：
融合时间、频谱、倒谱特征，从单台 3s P 波同时估计震级/距离/深度/方位。

**EEWMagNet (2023), AI in Geosciences, DOI 10.1016/J.AIIG.2023.03.001**：
中国 7s 三分量 DenseBlock+MHA；论文明确指出“epicentral distance is indispensable，
normalization has a negative effect on capturing accurate amplitude information”。

**对本项目：** 双分支合理：raw log-amplitude 波形分支 + 可解释的振幅/频谱/倒谱分支；
但禁止 per-trace peak normalization，禁止 log(npts) 等协议伪特征。

## 二、19 篇精读清单与项目结论

| # | 论文（年份） | DOI | 可执行结论 |
|---:|---|---|---|
| 1 | A Machine-Learning Approach for Earthquake Magnitude Estimation (2019) | 10.1029/2019GL085976 | MagNet CNN+RNN；保留振幅；隐式学习距离/场地；σ≈0.2 |
| 2 | AIMag: Rapid Single-Station Magnitudes on Global Scale (2024) | 10.1785/0120230171 | 2.4M P波；窗短导致饱和拐点下移；需大震校准 |
| 3 | End-to-End LSTM Single Station (2022) | 10.1109/LGRS.2022.3175108 | LSTM+工程特征解决大震少样本；M≥4相对误差4.01% |
| 4 | EQGraphNet (2024) | 10.1016/J.AIIG.2024.100089 | 深残差图卷积增强低SNR稳健性与跨环境泛化 |
| 5 | E3WS: 3s Single Station EEW (2023) | 10.1029/2023JB026575 | 时间/谱/倒谱集成；同时回归距离深度震级 |
| 6 | MEANet (2022) | 10.1190/GEO2022-0196.1 | 物理特征时间序列+attention；单台3s σ±0.25 |
| 7 | ViT Detection & Magnitude (2022) | 10.1029/2021JB023657 | 非归一化30s P窗与比赛高度同构 |
| 8 | Multi-feature + 3D corrections + boosting (2019) | 10.1093/GJI/GGZ416 | 距离/场地校正能到低不确定性；本赛缺元数据时作为上界 |
| 9 | SVM transfer Sichuan–Yunnan (2022) | 10.1785/0120210232 | 区域迁移是必要步骤 |
|10 | TLDCNN-M China (2023) | 10.3389/FPHY.2023.1070010 | 中国域迁移：3s误差<0.5，10s<0.2 |
|11 | AMAG attention (2025) | 10.1785/0220240289 | 可变窗≥1s、attention聚焦初动；需测不同窗泛化 |
|12 | PGD Saturation (2019) | 10.1029/2018JB017093 | 饱和阈值由窗长决定；大震需显式先验 |
|13 | How Fast for Subduction M? (2018) | 10.1029/2018GL078991 | 30s仍低估Mw≥7.5；短窗物理上限 |
|14 | DiTing (2023) | 10.1016/J.EQS.2022.01.022 | 中国域273万条；二阶段预训练首选 |
|15 | STEAD (2019) | 10.1109/ACCESS.2019.2947848 | 120万全球局地波形；主公开预训练集 |
|16 | Transformer M+Location (2021) | 10.1093/GJI/GGAB139 | 数据量增4倍可使平均误差减半；数据规模是首要杠杆 |
|17 | Network-based MagNet (2021) | 10.1785/0220200317 | 震级均衡增广；M3–5.9 σ0.21；大震样本不足是瓶颈 |
|18 | Complex CNN on STEAD (2021) | 10.1109/LGRS.2021.3059422 | 1min STFT复数CNN，震级MAE0.26 |
|19 | EEWMagNet China (2023) | 10.1016/J.AIIG.2023.03.001 | 距离重要；归一化损害幅值信息；与本项目诊断一致 |

## 三、下一轮实验设计（按优先级）

1. **A0：LogAmp-CNN（已排队）**  
   只吃非归一化 60s 波形；STEAD 按 `source_id` 分组；L1 损失；禁用 npts/S-P。

2. **A1：MagNet 同构 CNN+BiGRU**  
   将 A0 全局平均池化替换成 1–2 层 BiGRU/attention；理由是 MagNet 的直接 SOTA 证据。

3. **A2：多任务隐变量头**  
   训练时辅助预测 STEAD 的 log(distance)、depth；推理只输出 magnitude。距离虽然比赛不可见，
   但辅助监督可逼迫 backbone 学到路径效应；输入仍只有波形，不发生协议不一致。

4. **A3：大震均衡 + 分位数/异方差回归**  
   按事件震级分箱采样，避免 STEAD 小震占70%；输出 μ/σ 或 q10/q50/q90，减少 M≥6 系统低估。

5. **B：STFT 双分支**  
   raw log-amplitude 时域分支 + STFT 复数/功率谱分支 + 六维物理特征，验证是否能从谱形隐式估距。

6. **C：中国域二阶段迁移**  
   下载 DiTing（远程专用目录），STEAD→DiTing；随后仅按预注册固定划分使用 R1/R2，08永不参与。

## 四、准入门槛

- 公开集：事件级分组 STEAD MAE <0.26（达到 Complex CNN 水平），目标 <0.22（接近 MagNet）
- 域迁移：R1/R2 固定留出必须同时优于不含幅值的 SeismicXM baseline 与常数基线
- 大震：单独报告 M≥5、M≥6 MAE 与 bias；平均分上涨但大震低估不得上线
- 任何含 `log(npts)`、文件名、答案包统计或 08 反馈的模型直接判违规/伪高分
- 08 仅最终冻结只读报告一次，不据此回改方法
