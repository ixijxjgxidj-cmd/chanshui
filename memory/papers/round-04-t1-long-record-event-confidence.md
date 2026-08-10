# 研究轮次 04：T1 长记录事件级联合置信度过滤

- 日期：2026-08-11
- 状态：论文研究与历史审计完成；实验尚未运行
- 对应预注册：`memory/experiments/004-t1-long-record-event-confidence.md`
- 基准提交：`fb4345e4d3cfbeda55a8f8da74d719b1499220d1`
- 本轮问题：现有生产链路已经对 `>300s` 长记录的 P、S 拾取分别执行 20 秒同相位去重，但 7 个多事件超长文件仍贡献 T1 最大的集中损失。能否只在这一层之后，利用一个完整 P→S 事件的联合置信度删除低可信假事件，同时保持真实多事件、全部短记录和纯噪声安全性不退步？

## 1. 当前失败现象与分数证据

冻结基线为：

- `outputs/frozen_baseline/baseline_full_profile_prod_20260811.json`
- SHA-256：`2a55e4164db40a8eb87d6aa518fb040f11f7b2996788234f8fc1513bcfa3ac05`
- 生产长记录模型：前 5 个集成成员、`-1 dB` SNR 闸、20 秒 P/S 独立去重

长记录缓存为：

- `outputs/port_verify/_long_picks_cache.json`
- SHA-256：`b7e20333fbe97480017e8c8b5167be6f92a95c1969b5c7e91bb6e0319cc38699`
- 内容边界：仅包含 7 个长文件在 5 成员集成与 `-1 dB` SNR 闸之后、20 秒生产去重之前的 `(phase, relative_time, confidence)`；不含原始波形。

冻结逐文件结果表明，R2 的两个长文件和 08 的五个约 4000 秒文件是数量误差与总分损失的集中来源：

| 包 | 文件 | 真值 P/S | 生产预测 P/S | FP/FN | P/S 时差分 | 文件总分 | 文件级数量罚 |
|---|---|---:|---:|---:|---:|---:|---:|
| R2 | `T1.A.Q0001.mseed` | `35/34` | `64/68` | `76/13` | `26.822/28.350` | `25.172` | `30.0` |
| R2 | `T1.A.Q0002.mseed` | `53/52` | `64/67` | `50/24` | `41.344/37.239` | `68.083` | `10.5` |
| 08 | `T1.A.Q0001.mseed` | `47/51` | `47/58` | `37/30` | `28.000/39.344` | `65.844` | `1.5` |
| 08 | `T1.A.Q0002.mseed` | `41/48` | `63/72` | `60/14` | `34.000/40.194` | `53.194` | `21.0` |
| 08 | `T1.A.Q0003.mseed` | `39/50` | `46/55` | `36/24` | `27.956/36.706` | `60.661` | `4.0` |
| 08 | `T1.A.Q0004.mseed` | `36/34` | `61/73` | `71/7` | `30.989/31.961` | `32.450` | `30.5` |
| 08 | `T1.A.Q0005.mseed` | `47/50` | `57/69` | `50/21` | `32.844/42.878` | `63.222` | `12.5` |

按理想时差总分减当前文件总分计算：

- R2 全包 T1 总损失约 `343.722`，两个长文件损失约 `80.744`，占约 `23.5%`；两文件数量罚合计 `40.5`，FP/FN 为 `126/37`。
- 08 全包 T1 总损失约 `425.094`，五个长文件损失约 `167.628`，占约 `39.4%`；五文件包揽全部文件级数量罚 `69.5`，FP/FN 为 `254/96`。

对缓存复刻生产 20 秒去重后，7 文件预测数量为：

| 包 | 文件 | 20 秒后 P/S |
|---|---|---:|
| R2 | `T1.A.Q0001.mseed` | `64/68` |
| R2 | `T1.A.Q0002.mseed` | `64/67` |
| 08 | `T1.A.Q0001.mseed` | `47/58` |
| 08 | `T1.A.Q0002.mseed` | `63/72` |
| 08 | `T1.A.Q0003.mseed` | `46/55` |
| 08 | `T1.A.Q0004.mseed` | `61/73` |
| 08 | `T1.A.Q0005.mseed` | `57/69` |

这些数量与冻结生产逐文件预测完全一致，因此缓存可作为本轮离线反证入口；正式实验仍必须把这一一致性写成硬断言。

## 2. 前置结构诊断

### 2.1 只删除孤立相位不够

对生产 20 秒去重后的拾取做宽松 P→S 结构诊断，R2 两个长文件中：

- 与真值匹配的 P/S 对（MM）：`61`
- 假 P 与假 S 构成的完整假事件对（FF）：`49`
- 假 P/S 总数：`59/67`
- 假孤立相位仅：`5/15`

08 五个长文件中：

- MM：`142`
- FF：`73`
- 假 P/S 总数：`120/134`
- 假孤立相位仅：`24/54`

多数额外拾取并不是单独的孤立尖峰，而是能组成时间顺序合理的假 P/S 事件。因此，“必须同时有 P 和 S，否则删除”只能覆盖少量错误，不能解决主要数量罚。

### 2.2 联合置信度存在中等区分信号

7 个长文件、生产 20 秒去重后的只读诊断得到：

- 单相位 matched-over-false AUC：P confidence `0.7467`，S confidence `0.7386`。
- 宽松 60 秒 P→S 配对后，MM 与 FF 的事件分数 AUC：
  - `min(P_conf, S_conf)`：`0.7606`
  - `sqrt(P_conf × S_conf)`：`0.7616`
  - 算术均值：`0.7583`
- FF 几何均值中位数：`0.3959`
- MM 几何均值中位数：`0.5317`

这不是足以直接上线的证据：样本来自已见历史包，配对规则和阈值还没有预注册，08 也早已参与过 20 秒去重选择。但它足以支持一个很小、可否证、无需重新推理的实验：固定事件结构，只测试三个粗阈值，并用双向跨包验证拒绝分布特化。

### 2.3 继续扩大固定去重窗风险高

生产已经执行 20 秒同相位去重。剩余 false pick 到最近 matched 同相位拾取的距离分布为：

| 包 | 20–40 秒 | 40–60 秒 | >60 秒 |
|---|---:|---:|---:|
| R2 | 28 | 52 | 46 |
| 08 | 53 | 93 | 108 |

历史提交 `c82e75d` 已完整扫描固定长记录去重窗：30 秒在 7 文件上约 `+34.9`，45 秒开始合并真实事件并约 `-114`；生产因此选择更保守的 20 秒，约 `+26.7`。本轮不得原样重扫 `long_dedup_s`，也不得把 P/S 事件窗口伪装成新的同相位去重窗。

## 3. 历史 P/S 联合规则审计：为什么本轮不是重复

历史脚本：

- `outputs/port_verify/_reselect_exp.py`
- 候选缓存：`outputs/port_verify/_reselect_cands.json`
- 缓存 SHA-256：`5696f5ea2216deb8339db257243930d794a65438abb92dd53b465e80a5b34372`

该脚本只处理 `≤300s` 短文件，目标是在低阈值候选中重新选择唯一的 `1P+1S`：

- A：各相位独立取最高置信度；
- B：使用真值选最近候选的神谕上限；
- C：满足 `S>P` 的候选对中取联合置信度最高者；
- D：只在原选取违反物理或联合置信度显著更高时替换。

在不重新推理、只读取既有缓存并把失效的旧 R2 路径临时指向当前官方包后，结果为：

| 包 | 短文件数 | A 现行近似 | B 神谕上限 | C 物理+联合 | D 保守替换 |
|---|---:|---:|---:|---:|---:|
| R1 | 1000 | `1777.9` | `1818.4`（`+40.5`） | `1781.6`（`+3.7`） | `1781.6`（`+3.7`） |
| R2 | 913 | `1552.6` | `1586.0`（`+33.3`） | `1552.6`（`+0.0`） | `1552.6`（`+0.0`） |
| 08 | 779 | `1299.0` | `1344.8`（`+45.8`） | `1299.0`（`+0.1`） | `1299.0`（`+0.1`） |

短文件联合重选只有 R1 小幅上涨，R2 与 08 几乎为零，不能作为跨包生产方向。本轮与它有四个实质差异：

1. 只作用于 `>300s` 长记录；
2. 固定在现有 20 秒生产去重之后；
3. 不从多个候选中重选一个事件，而是尝试删除完整的低可信假事件；
4. 所有短文件必须逐位保持现行输出，不能借本轮重开短文件规则。

## 4. 检索与核验协议

### 4.1 AnySearch

本轮同时使用普通检索与学术垂直检索。主要检索式包括：

1. `seismic phase association PhaseLink GaMMA REAL PyOcto GENIE earthquake monitoring`
2. `seismic phase association earthquake picks confidence long continuous waveform`
3. `seismic phase association graph neural network earthquake event building`
4. `earthquake detection probability phase picking event confidence EQTransformer CRED continuous waveform`
5. `single station earthquake detection phase picking confidence continuous seismic multiple events`
6. `REAL a rapid earthquake association and location method paper DOI`
7. `GaMMA Bayesian Gaussian mixture model earthquake phase association paper DOI arXiv`
8. `Earthquake Phase Association with Graph Neural Networks GENIE paper DOI arXiv`
9. `QuakeFlow scalable machine-learning earthquake monitoring workflow cloud computing paper DOI arXiv`
10. `Which Picker Fits My Data quantitative evaluation deep learning seismic pickers arXiv DOI`
11. `DeepPhasePick method detecting picking seismic phases local earthquakes full paper`
12. 16 个候选标题、DOI 或 arXiv 编号的精确检索，用于核对作者、年份、正式页面和实验条件。

### 4.2 Playwright 实际打开并读取的页面

本轮有效核验 16 篇新的原始实验研究；不重复计入前三轮论文数量：

1. `https://ar5iv.labs.arxiv.org/html/1809.02880`
2. `https://par.nsf.gov/biblio/10129287`
3. `https://ar5iv.labs.arxiv.org/html/2109.09008`
4. `https://arxiv.org/html/2310.11157`
5. `https://ar5iv.labs.arxiv.org/html/2209.07086`
6. `https://ar5iv.labs.arxiv.org/html/2301.02597`
7. `https://seismica.library.mcgill.ca/article/view/1559`
8. `https://ar5iv.labs.arxiv.org/html/2208.14564`
9. `https://www.nature.com/articles/s43247-023-01188-4`
10. `https://ar5iv.labs.arxiv.org/html/1805.01075`
11. `https://arxiv.org/abs/1810.01965`
12. `https://www.nature.com/articles/s41467-020-17591-w`
13. `https://ar5iv.labs.arxiv.org/html/2110.13671`
14. `https://eartharxiv.org/repository/view/1752/`
15. `https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2022.848237/full`
16. `https://arxiv.org/html/2506.06939v1`

其中 REAL、CRED、DeepPhasePick 本轮可用页面主要提供正式元数据与摘要，以下明确标记“仅摘要”；其余页面实际读取了正文的结果、讨论或结论部分。本轮综述计数为 0，16 篇均为有方法和实验的原始研究或原始比较实验。

## 5. 多台站震相关联证据

### 1. PhaseLink: A Deep Learning Approach to Seismic Phase Association

- 作者/年份：Zachary E. Ross、Yisong Yue、Men-Andrin Meier、Egill Hauksson、Thomas H. Heaton；2019。
- 标识：DOI `10.1029/2018JB016674`；arXiv `1809.02880`。
- 方法：用堆叠双向循环网络学习跨台站到时序列是否来自同一事件；训练数据可由 1D 速度模型完全合成。
- 数据与实验：2016 Borrego Springs 序列含 1708 个目录事件、73,353 个人工相位；实验再加入等量 73,353 个均匀假拾取，并构造高事件率与日本合成场景。
- 关键结果：论文报告可关联起源时刻仅约 12 秒间隔的事件，并在受控序列上优于传统网格关联。
- 局限：输入核心是多个传感器的位置和跨台站走时模式；还需要区域速度模型生成训练序列。后续基准表明其在高噪声、高事件率和复杂俯冲场景可显著退化。
- 本项目关系：支持“完整事件结构能拒绝假拾取”，但比赛每次只给单台站，不能把 PhaseLink 的跨台站证据等同为单台站 P/S 配对证据。
- 决策：不实现模型；部分采用“联合结构优于单点阈值”的思想。

### 2. Rapid Earthquake Association and Location（REAL）

- 作者/年份：Miao Zhang、William L. Ellsworth、Gregory C. Beroza；2019。
- 标识：DOI `10.1785/0220190052`。
- 阅读状态：仅摘要与正式元数据页。
- 方法：在候选网格上首先最大化理论走时窗内的 P/S 拾取数量，再以走时残差细化位置；本质是“足够多跨台站一致拾取”构成事件。
- 关键实验结论：论文在合成和真实地震序列中验证了实时关联与定位能力；官方代码同时提供合成与真实数据流程。
- 局限：需要站点坐标、速度模型、多个站的 P/S 数量与走时残差；单台站无法形成其主要目标函数。
- 本项目关系：其“数量优先、残差其次”说明假事件删除应依赖完整事件证据，但不能移植其网格和最少台站阈值。
- 决策：拒绝直接实现；仅作为事件一致性证据。

### 3. Earthquake Phase Association Using a Bayesian Gaussian Mixture Model（GaMMA）

- 作者/年份：Weiqiang Zhu、Ian W. McBrearty、S. Mostafa Mousavi、William L. Ellsworth、Gregory C. Beroza；2022。
- 标识：DOI `10.1029/2021JB023249`；arXiv `2109.09008`。
- 方法：把每个地震视为一个混合分量，以跨台站到时双曲走时、振幅随距离衰减、P/S 类型和拾取质量共同做 EM 聚类，同时估计位置、起源时刻和震级。
- 数据与实验：合成实验含 6 个事件、40 个台站、178 个 P/S 拾取，加入 `0.5s` 到时误差、`0.3–3` 振幅扰动和 30% 随机假拾取；另在 2019 Ridgecrest 六天数据上验证。
- 关键结果：合成与 Ridgecrest 实验均能处理时空密集事件；相位质量和振幅可进入概率模型。
- 局限：需要站点坐标、速度模型、距离相关振幅和目标区域批次；复杂度约随事件数与拾取数乘积增长，长序列需切窗/DBSCAN。
- 本项目关系：直接支持“置信度只能作为事件结构中的一部分”，也说明只凭单台站置信度无法获得 GaMMA 的物理约束。
- 决策：不实现 GaMMA；采用几何联合分数作为最小代理，但设置严格跨包门槛。

### 4. PyOcto: A High-Throughput Seismic Phase Associator

- 作者/年份：Jannes Münchmeyer；2024。
- 标识：DOI `10.26443/seismica.v3i1.1130`；arXiv `2310.11157`。
- 方法：用四维时空分割和 0D/1D 速度模型快速搜索满足走时约束的事件与拾取集合。
- 数据与实验：两类合成场景覆盖 100–2000 个事件和噪声/真拾取比 `0.3/1/3`；真实实验为 2014 Iquique 密集序列。
- 关键结果：合成场景检测能力不低于 GaMMA/REAL，常见加速超过 50 倍；Iquique 每天关联约 `12–15s`，GaMMA 约 17 分钟、REAL 约 25–26 分钟，加速约 70–130 倍。
- 局限：速度来自强物理约束和多台站空间分割，不是单台站置信度阈值带来的。
- 本项目关系：若比赛提供网络级多站数据，PyOcto 是很强的候选；当前输入契约下不可用。
- 决策：拒绝本周实现；采用“后处理应远低于模型推理成本”的工程标准。

### 5. Earthquake Phase Association with Graph Neural Networks（GENIE）

- 作者/年份：Ian W. McBrearty、Gregory C. Beroza；2023。
- 标识：DOI `10.1785/0120220182`；arXiv `2209.07086`。
- 方法：在站点图、候选源空间与预计算走时场上使用图神经网络，同时预测时空源和离散相位归属。
- 数据与实验：合成测试包括约 1200 秒、22 个事件的未见序列；连续应用以 PhaseNet 拾取处理北加州逐日数据，并显式测试变化台网、噪声和重叠事件。
- 关键结果：模型能在连续时间中形成平滑源概率并处理可变台站覆盖；论文强调需同时判断事件数、位置、起源时刻和测量伪影。
- 局限：必须有台站图、源域、速度模型和区域训练；并非可直接套用的通用预训练关联器。
- 本项目关系：其最关键证据是“区分真实到时与测量伪影需要事件级上下文”，但当前只有同一站的一对 P/S 证据。
- 决策：不实现模型；部分采用事件级而非拾取级判断。

### 6. Neural Mixture Model Association of Seismic Phases（Neuma）

- 作者/年份：Zachary E. Ross、Weiqiang Zhu、Kamyar Azizzadenesheli；2023。
- 标识：arXiv `2301.02597`。
- 方法：在 GaMMA 类混合模型中加入 Eikonal 神经走时前向模型和显式噪声类别，把每个拾取分配到某个地震或噪声。
- 数据与实验：多个挑战性合成集与 2019 Ridgecrest 序列，对比 PhaseLink、GaMMA 和人工审阅目录。
- 关键结果：论文报告在 Ridgecrest 比最佳基线多检出约 3285 个事件（约 13.5%），同时改善震源深度偏差；显式噪声类可减少错误归属。
- 局限：仍需要多台站、3D 走时、振幅、位置与震级；其噪声概率不能由当前单台站 pick confidence 等价替代。
- 本项目关系：支持“宁可给噪声一个显式出口，也不要强行把所有拾取组成地震”。本轮因此保留所有无法可靠成对的 orphan，不强制归并，也不把它们自动删除。
- 决策：采用显式保守拒绝原则；拒绝模型实现。

### 7. Benchmarking Seismic Phase Associators: Insights from Synthetic Scenarios

- 作者/年份：Jorge Puente Huerta、Christian Sippl、Jannes Münchmeyer、Ian W. McBrearty；2025。
- 标识：DOI `10.26443/seismica.v4i2.1559`；arXiv `2501.03621`。
- 方法：在统一合成协议下比较 PhaseLink、REAL、GaMMA、GENIE、PyOcto，系统改变噪声、事件密度、台站密度、网外事件和地壳/俯冲场景。
- 关键结果：GENIE/PyOcto 在多数场景近乎完美，最困难高噪声高事件率下 F1 仍高于 0.8；PhaseLink 在复杂俯冲场景可接近失效，GaMMA 在高噪声/高密度下受精度与扩展性限制，REAL 在极端条件损失 recall。
- 局限：仍是有台站几何和走时模型的合成网络级评测；不能证明单台站规则可达到相同鲁棒性。
- 本项目关系：直接反对“某个关联器论文先进，所以换上就会涨分”；关联性能高度依赖噪声、事件率和场景。
- 决策：采用统一压力测试和最坏组准入思想；不引入完整关联器。

### 8. QuakeFlow: A Scalable Machine-Learning-Based Earthquake Monitoring Workflow with Cloud Computing

- 作者/年份：Weiqiang Zhu、Alvin Brian Hou、Robert Yang、Avoy Datta、S. Mostafa Mousavi、William L. Ellsworth、Gregory C. Beroza；2023。
- 标识：DOI `10.1093/gji/ggac355`；arXiv `2208.14564`。
- 方法：把波形下载、PhaseNet 拾取、GaMMA 关联、定位和目录生成封装成可扩展云工作流。
- 数据与实验：Puerto Rico 地震序列与 Hawaii 火山/深部地震，处理连续网络数据并生成增强目录。
- 关键结果：论文展示了大规模连续数据中端到端目录构建能力；价值主要来自完整工作流和横向扩展，而不是某个单一阈值。
- 局限：依赖多台站与云组件；当前比赛 API 的单请求单站、CPU 小服务器不需要这一系统复杂度。
- 本项目关系：支持保持“拾取—事件判断—输出”层次清晰；本轮只改最小后处理层，不碰推理和部署架构。
- 决策：采用模块化边界；拒绝云化重构。

### 9. An All-in-One Seismic Phase Picking, Location, and Association Network for Multi-Task Multi-Station Earthquake Monitoring（PLAN）

- 作者/年份：Xu Si、Xinming Wu、Zefeng Li、Shenghou Wang、Jun Zhu；2024。
- 标识：DOI `10.1038/s43247-023-01188-4`。
- 方法：用图神经网络统一波形特征、多站关联、位置和物理约束拾取，输入同时包含多站波形和站点位置。
- 数据与实验：在 Ridgecrest 与日本分别重训，和 PhaseNet、EQTransformer、Aggregated-GNN 做共同测试。
- 关键结果：正文报告 PLAN 的 P/S 残差分布更集中，并在日本数据上对拾取和位置表现出更明显优势。
- 局限：优势正来自多站上下文、站点几何和多任务重训；当前没有相同训练数据和输入。
- 本项目关系：说明联合任务可以互相约束，但也强化了本轮边界：不能把单站 P/S 几何均值包装成等价的“全事件关联”。
- 决策：延期到未来网络级输入或新训练阶段。

## 6. 单台站联合检测与置信度证据

### 10. Generalized Seismic Phase Detection with Deep Learning（GPD）

- 作者/年份：Zachary E. Ross、Men-Andrin Meier、Egill Hauksson、Thomas H. Heaton；2018。
- 标识：DOI `10.1785/0120180080`；arXiv `1805.01075`。
- 方法：把 4 秒三分量窗分类为 P、S 或噪声；连续应用时高重叠滑窗并在概率序列峰值处声明拾取。
- 数据与实验：273,882 个地震、各约 150 万 P/S 人工拾取；独立验证集 110 万条；连续测试为 2016 Bombay Beach 群震首日。
- 关键结果：论文明确展示 precision/recall 随概率阈值变化；连续应用采用 `0.98` 高阈值，强调阈值控制灵敏度/误报权衡。
- 局限：输出仍是相位类别，不是完整事件概率；滑窗推理开销大，且阈值依赖训练域。
- 本项目关系：支持“置信度可以做拒识”，也提醒阈值必须跨包验证；本轮不增加模型调用，只使用现有 confidence。
- 决策：采用阈值需跨域冻结的原则；不运行 GPD。

### 11. CRED: A Deep Residual Network of Convolutional and Recurrent Units for Earthquake Signal Detection

- 作者/年份：S. Mostafa Mousavi、Weiqiang Zhu、Yixiao Sheng、Gregory C. Beroza；2019。
- 标识：DOI `10.1038/s41598-019-45748-1`；arXiv `1810.01965`。
- 阅读状态：仅摘要与正式元数据页。
- 方法：在单台站三分量时频表示上结合 CNN、双向 LSTM 和残差结构，输出地震信号检测概率。
- 数据与实验：50 万条波形，地震/噪声各 25 万；论文报告测试 F-score `99.95`，并在 Arkansas 一个月连续数据中检出 700 多个、最低约 `ML -1.3` 的微震。
- 局限：需要一个独立、经过大量噪声训练的事件检测器；当前 DiTing PhaseNet 集成没有保存等价事件分支。
- 本项目关系：从原理上，独立 event detector 比 P/S 几何均值更可靠；但本周不能凭少量历史标签训练一个可信 CRED 替代品。
- 决策：延期；本轮联合置信度只作为廉价代理。

### 12. Earthquake Transformer—An Attentive Deep-Learning Model for Simultaneous Earthquake Detection and Phase Picking

- 作者/年份：S. Mostafa Mousavi、William L. Ellsworth、Weiqiang Zhu、Lindsay Y. Chuang、Gregory C. Beroza；2020。
- 标识：DOI `10.1038/s41467-020-17591-w`。
- 方法：共享编码器加三个解码分支，同时输出 earthquake detection、P 和 S 概率，用全局事件上下文约束局部相位。
- 数据与实验：全球 STEAD 约 100 万地震和 30 万噪声；测试超过 113,000 条 1 分钟波形；另在训练未包含的日本 Tottori 五周连续数据上使用 18 个台站。
- 关键结果：正文报告在共同测试集优于传统与若干深度模型；日本应用只用原研究不到三分之一台站仍定位约两倍事件。连续应用阈值示例为 detection `0.5`、P/S `0.3/0.3`。
- 局限：结果来自联合训练出的独立事件分支和大规模噪声数据，不等同于后处理相乘两个未经校准的 phase confidence。
- 本项目关系：这是本轮最直接的方法论支持：完整事件证据应联合 P/S；同时也是最重要的风险提示，当前代理必须通过严格跨包验证，不能声称等价于 EQTransformer。
- 决策：采用联合任务思想；不替换当前生产 picker。

### 13. Which Picker Fits My Data? A Quantitative Evaluation of Deep Learning Based Seismic Pickers

- 作者/年份：Jannes Münchmeyer、Jack Woollam、Andreas Rietbrock、Frederik Tilmann、Dietrich Lange 等；2022。
- 标识：DOI `10.1029/2021JB023499`；arXiv `2110.13671`。
- 方法：在 8 个数据集、事件检测/相位识别/到时拾取 3 个任务上统一比较 6 个深度模型和经典 picker，并专门做跨域实验。
- 关键结果：总体以 EQTransformer、GPD、PhaseNet 最强；区域间迁移通常只有温和退化，但区域与远震域之间不能安全互换。吞吐也差异显著，例如 GPD 的滑窗评测远慢于输出完整概率曲线的模型。
- 局限：模型阈值仍取决于数据和任务；论文不是长记录事件删除研究。
- 本项目关系：直接支持按数据包冻结阈值、跨包验证和不把单包最优阈值当通用常数。
- 决策：采用跨域协议和低成本偏好。

### 14. DeepPhasePick

- 作者/年份：Hugo Soto、Bernd Schurr；2021（本轮打开 2020 EarthArXiv 预印本页）。
- 标识：正式刊于 *Geophysical Journal International* 227(2):1268–1294；EarthArXiv 页面 `https://eartharxiv.org/repository/view/1752/`。
- 阅读状态：仅摘要与项目元数据页。
- 方法：两阶段单台站流程，先用三分量 CNN 检测 P/S，再分别用循环网络精确拾取，并通过 Monte Carlo dropout 输出不确定性。
- 数据与实验：北智利约 39,000 条检测记录、36,000 条拾取记录；在不同构造区测试检测和人工级到时精度。
- 局限：需要单独训练检测器和拾取器；过滤、采样率和站域差异可产生大量假阳性。
- 本项目关系：支持把“是否存在相位”和“精确到时”分层，也说明置信度/不确定性应由训练产生；当前只能做保守代理。
- 决策：延期模型替换；采用分层判断思想。

### 15. LEQNet: Light Earthquake Deep Neural Network for Earthquake Detection and Phase Picking

- 作者/年份：Jongseong Lim、Sunghun Jung、Chan JeGal、Gwanghoon Jung、Jung Ho Yoo、Jin Kyu Gahm、Giltae Song；2022。
- 标识：DOI `10.3389/feart.2022.848237`。
- 方法：用深度可分离卷积和递归结构压缩联合检测/P/S 拾取模型。
- 数据与实验：从 STEAD 采样 50,000 组地震与噪声，阈值固定为 detection `0.5`、P/S `0.3/0.3`，与 EQTransformer 等比较。
- 关键结果：检测/P/S F1 为 `0.99/0.98/0.97`，接近 EQTransformer 的 `1.00/0.99/0.98`；参数量比 EQTransformer 减少 `87.68%`，CNN FLOPs 减少 `93.38%`。
- 局限：同域 STEAD 结果不能证明比赛域；仍需要训练好的事件分支。
- 本项目关系：若未来需要低成本联合模型，LEQNet 比完整大型替换更符合服务器条件；当前无需为一个最小后处理实验引入新权重。
- 决策：延期；保留为 DiTing 数据到位后的候选。

### 16. Towards End-to-End Earthquake Monitoring Using a Multitask Deep Learning Model（PhaseNet+）

- 作者/年份：Weiqiang Zhu、Junhao Song、Haoyu Wang、Jannes Münchmeyer；2025。
- 标识：arXiv `2506.06939`。
- 方法：在 PhaseNet 上联合相位到时、初动极性、事件检测与起源时刻预测；单次扫描连续波形，并用预测起源时刻帮助关联。
- 数据与实验：北加州 CEED 约 325,000 个事件、110 万三分量波形与 P/S 对；北加州训练后评估 2023 年北/南加州，并应用于 2019 Ridgecrest。
- 关键结果：联合训练没有牺牲 PhaseNet 到时精度；Ridgecrest 中 PhaseNet+ 与 GaMMA 目录均比 SCSN 多 73–81% 事件，PhaseNet+ 数量少约 4% 但空间散布更小，论文据此认为 GaMMA 可能含更多假阳性。
- 局限：依赖百万级带事件起源时刻标签；论文也明确承认近同时事件会削弱单站起源时刻关联，需要站点位置或波形相似度补充。
- 本项目关系：直接证明“单台站联合事件分支”是合理方向，也说明在没有该分支时，简单 P/S 代理必须保守、不能处理近同时事件。FIFO 非交叉配对和 orphan 保留正是为此设定。
- 决策：延期模型训练；采用固定事件结构与严格失败条件。

## 7. 证据综合

16 篇论文共同给出四条稳定结论：

1. **真实的震相关联依赖比单相位置信度更强的物理信息。** PhaseLink、REAL、GaMMA、PyOcto、GENIE、Neuma、PLAN 都依赖多台站位置、走时、振幅或源空间；当前比赛单台站输入不能复制这些收益。
2. **单台站中，联合事件检测与 P/S 拾取通常优于彼此独立。** EQTransformer、CRED、DeepPhasePick、LEQNet、PhaseNet+ 都把事件/相位任务联合或分层建模；这支持检验完整 P/S 事件分数，而不是继续扩大同相位去重窗。
3. **阈值不可跨域想当然。** GPD 和统一 picker 基准都显示 precision/recall 与阈值、训练域、任务距离类型相关；因此本轮只能用粗网格、源包选型和双向跨包硬门槛。
4. **完整模型替换的数据和时间条件不具备。** 当前没有百万级事件/噪声标签、站点网络或速度模型；而离线 confidence 缓存已经存在，事件级后处理几乎没有推理成本，最适合作为最小反证。

## 8. 方法决策矩阵

评分 `1`（差/高风险）到 `5`（好/低风险）；数据现成、API 兼容和可回滚越高越好。

| 方向 | 预期分数收益 | 数据现成 | 本周可完成 | 单站 API 兼容 | 过拟合安全 | 推理成本 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| PhaseLink/GaMMA/PyOcto/GENIE 等完整关联器 | 5 | 1 | 1 | 1 | 4 | 2–4 | 拒绝当前输入条件 |
| 新训练 CRED/EQTransformer/PhaseNet+ 事件分支 | 5 | 1 | 1 | 5 | 2 | 2–4 | 等 DiTing/噪声标签 |
| 20 秒后 FIFO P/S 事件 + 几何联合置信度过滤 | 3 | 5 | 5 | 5 | 3 | 5 | 采用为最小反证 |
| 继续扩大固定 `long_dedup_s` | 2 | 5 | 5 | 5 | 1 | 5 | 历史已失败，禁止重复 |
| 短文件 P/S 联合重选 | 1 | 5 | 5 | 5 | 2 | 5 | 历史跨包近零，拒绝 |
| 删除全部 orphan 或强制每个 P/S 成对 | 2 | 5 | 5 | 5 | 1 | 5 | 假事件多为完整对，且漏检风险高 |

## 9. 本轮采用的最小机制

实验仅测试以下机制，不扩展成一般相位关联：

1. 固定既有 5 成员、`-1 dB` SNR 闸和 20 秒同相位去重；
2. 只处理 `duration > 300s`；短文件函数必须原样返回；
3. 按时间顺序用 FIFO、非交叉的一对一规则配 P→S，固定 `0.2s ≤ S-P ≤ 60s`；
4. orphan 永远保留，不因缺少另一相位被删除；
5. 事件分数固定为 `sqrt(P_conf × S_conf)`；
6. 只测试阈值 `0.35/0.40/0.45`，低于阈值时同时删除该 P/S 对；
7. R2、08 分别只用自身标签选阈值，再做 R2→08、08→R2 双向跨包验证；
8. 两包选出的较低阈值是唯一允许的共同保守候选，不能根据跨包结果换成另一个阈值；
9. 必须在四种数量罚口径、逐文件 FN/FP、P/S 时差分和短文件逐位一致性上同时通过。

完整冻结条件见预注册文件。特别说明：08 已参与历史 20 秒去重选择和本轮前置结构诊断，因此本轮不是盲测。双向源包选型只能降低继续过拟合风险，不能恢复真正独立终检的证据等级。

## 10. 明确拒绝的解释

- 若候选失败，只能否证“当前 5 成员 confidence + FIFO 单站 P/S 代理”这一小机制，不能否证事件级检测或多台站关联本身。
- 若某包上涨、另一包下跌，按生产纪律拒绝，不能用合计正收益覆盖方向不一致。
- 若分数上涨但新增任何真实相位漏检、降低 P/S 时差分或让某个长文件变差，仍拒绝；本轮目标是删除原本未匹配的假事件，不是用漏检换数量罚。
- 不根据结果增加 `0.30/0.50`、改变 60 秒窗、加入额外 min/max confidence、扩大同相位去重窗或改配对顺序。
- 不把本轮代理称为 PhaseLink、GaMMA、EQTransformer 或真正的 phase association。

## 11. 失败后的含义

若预注册实验失败，当前 evidence 将说明：现有 pick confidence 虽有中等 AUC，但不足以在单台站、跨历史包条件下安全识别完整假事件。下一步不再扫类似阈值，而应转向：

- 不依赖历史提分的生产鲁棒性与文档漂移；
- DiTing 原始数据到位后训练带显式 event/noise 分支的轻量多任务学生；
- 若未来获得多台站输入、站点坐标和速度模型，再评估 PyOcto/GaMMA/GENIE；
- 新的独立长记录包，用于重新判断 confidence 是否可校准。

