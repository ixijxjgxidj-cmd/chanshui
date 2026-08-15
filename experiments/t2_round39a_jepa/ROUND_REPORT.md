# T2 第 39A 轮：JEPA 自监督预训练的跨区域验证

> 日期：2026-08-15
> 状态：完成。仅作为公开数据上的候选研究证据，不变更任何生产模型或提交策略。

## 结论

在 5 组严格的 `训练区域 A → 标定区域 B → 评估区域 C` 协议中，JEPA 两种预训练池配置均同时超过 CNN 基线和同架构从零训练的 Transformer：

| 臂 | 评估域平均相关性 | 标定后平均分 | 标定后最差分 | 相对 CNN 的平均分变化 | 相对 CNN 的最差分变化 |
|---|---:|---:|---:|---:|---:|
| CNN（轮 38 架构） | 0.354 | 158.461 | 155.574 | — | — |
| Transformer，从零训练 | 0.336 | 158.299 | 155.561 | -0.162 | -0.013 |
| JEPA-A（仅 A 训练事件预训练） | 0.368 | 158.565 | 155.797 | +0.104 | +0.223 |
| JEPA-MR（多区域预训练） | **0.376** | 158.523 | **155.834** | +0.062 | +0.260 |

预注册式录取规则是「相对同架构从零训练和 CNN，标定后平均分严格更高、最差分不低」。`JEPA-A` 与 `JEPA-MR` 四项比较均通过；`scratch` 没有超过 CNN。因此，下一轮可保留 JEPA 表征，优先将 `JEPA-MR` 作为公开 INSTANCE 混合实验的候选；但尚不足以替换任何生产模型。

## 合规与数据隔离

本轮严格遵守项目最高优先级约束：`08-an.zip`、`08-exam.zip`、第 1 轮和第 2 轮比赛包及全部衍生产物均未被读取。

- 唯一数据输入是远程公开 STEAD 缓存：`/root/5.6+chanshui1/outputs/t2_cache_station27`。
- 缓存为 `60,466 × 3 × 1,000` 的 P−5 s 至 P+5 s、100 Hz 三分量窗口；标签范围只在实验内筛到 3.8–6.5。
- 运行时审计守卫对文件打开事件做白名单控制：只放行该缓存和本轮输出目录。
- 红-绿验证证实：公开 `y.npy` 可读；历史比赛波形 `t2data/T2.A.Q0001.mseed`、`outputs/r1_t2_meta.csv`、`outputs/c3_r1r2_devmix.json` 与另一未登记缓存均被 `ComplianceViolation` 阻断。
- 五组区域划分按台网互斥，且每个训练区域内部按 `source_id` 切分训练事件与域内留出事件。
- 对每一组，JEPA 预训练池排除 B、C 的全部记录，以及 A 的留出事件；因此标定域和评估域从未进入预训练、监督训练、早停或选择。

评分完全在公开数据上执行，使用与第 38 轮相同的官方 T2 hinge 形式：`200 × mean(max(0, 1 - |clip(p,0,9.9)-y|))`。缩放系数 `λ` 只在 B 选择，C 仅做一次评估。

## 方法

输入前端保持与轮 38 一致（RMS 归一化、对数幅度、SNR 和 8 段频谱统计），从而尽量隔离结构与自监督目标的影响。

- **CNN**：轮 38 的卷积成员，直接监督 L1 训练。
- **scratch**：`3 × 1,000` 波形经 stride 8 tokenizer 变成 125 个 token，由 4 层 Transformer 编码，直接监督 L1 训练。
- **JEPA-A**：与 scratch 相同的编码器。用 A 的训练事件做 10 epoch 掩码潜表示预测，再监督微调。
- **JEPA-MR**：同一 JEPA 目标，但预训练池为排除 B、C 和 A 留出事件后的多区域公开记录，最多 30,000 条。

JEPA 使用 EMA 目标编码器、4 个长度为 10 token 的遮蔽块、55% 可见上下文和 Smooth L1 潜表示预测；振幅增益扰动在预训练和监督阶段一致使用。所有臂在同一 AMD HIP 环境（PyTorch 2.9.0，HIP 6.3.26093）运行。

## 分组结果

下表是各臂在 C 上、使用 B 选出的 `λ` 得到的分数。它不是比赛分数，也不能用于对比赛测试集反向调参。

| A → B → C | CNN | scratch | JEPA-A | JEPA-MR |
|---|---:|---:|---:|---:|
| ALASKA → CALIF → GREECE | 160.927 | 160.499 | **160.951** | 160.604 |
| CALIF → GREECE → ALASKA | 155.594 | 155.901 | 155.972 | **156.012** |
| GREECE → CHILE → CALIF | 161.362 | 160.778 | 161.420 | **161.611** |
| CHILE → ALASKA → NZ | 155.574 | 155.561 | 155.797 | **155.834** |
| OTHER → NZ → CHILE | **158.846** | 158.755 | 158.688 | 158.552 |

JEPA 并非逐域统治：`OTHER → NZ → CHILE` 中 CNN 仍更高。正因如此，下一轮不应直接替换，而应通过独立 INSTANCE 域验证是否存在稳定增益。

## 可复现资产与验证证据

- 实验脚本：[train39a_jepa.py](train39a_jepa.py)
- 运行时守卫：[compliance_guard.py](compliance_guard.py)
- 红-绿自检：[test_compliance_guard.py](test_compliance_guard.py)
- 远程结果：`/root/5.6+chanshui1/outputs/t2_round39a_jepa/jepa_cross_region.json`
- 远程日志：`/root/5.6+chanshui1/round39a.log`
- 结果 JSON 明确记录缓存路径与 SHA-256、设备、HIP 版本、掩码训练轮数、预训练上限、完整区域协议和每组指标。

实际聚合录取判断：

```json
{
  "jepa_a_beats_scratch_mean_and_worst": true,
  "jepa_mr_beats_scratch_mean_and_worst": true,
  "jepa_a_beats_cnn_mean_and_worst": true,
  "jepa_mr_beats_cnn_mean_and_worst": true,
  "scratch_beats_cnn_mean_and_worst": false
}
```

## 文献精读与对下一轮的影响

检索使用远程代理访问 OpenAlex、Crossref 与 arXiv：共获取 720 条原始命中、637 条题名去重记录；从 30 篇定向清单中取得 28 篇「至少两源记录且摘要长度超过 200 字」的论文。以下列出本轮实际影响设计判断的 18 篇（多源以 OpenAlex / Crossref / arXiv 表示）。

| 论文 | 多源 | 精读结论 | 对本项目的可执行影响 |
|---|---|---|---|
| Assran et al., *I-JEPA* (2023) | OpenAlex、Crossref | 预测语义尺度较大的目标块，且上下文要足够分散，是潜表示预测有效的关键。 | 保留块遮蔽，下一轮扫描的是块尺度和上下文比例，而非改为逐点重构。 |
| Bardes et al., *V-JEPA* (2024) | OpenAlex、Crossref、arXiv | 无生成重构、无负样本的潜表示预测可以得到可迁移表征。 | 支持继续采用 EMA 目标编码器，而不引入可能放大波形细节噪声的重构损失。 |
| Baevski et al., *data2vec* (2022) | OpenAlex、Crossref、arXiv | 以被遮蔽视图预测完整输入的上下文化教师表征，跨模态有效。 | 下一轮增加「全序列教师目标」消融，对比当前仅目标 token。 |
| Nie et al., *PatchTST* (2022) | OpenAlex、Crossref、arXiv | patch token 可保存局部语义并降低注意力复杂度；掩码预训练跨数据集迁移有效。 | 当前 8 点 patch 合理；预注册比较 8/16/32 点，不能用比赛数据选择。 |
| Yue et al., *TS2Vec* (2022) | OpenAlex、Crossref、arXiv | 分层上下文对比表征可服务多种时序任务。 | 可作为 JEPA 的公开数据对照臂，而非直接假设 JEPA 必胜。 |
| He et al., *Masked Autoencoders Are Scalable Vision Learners* (2022) | OpenAlex、Crossref、arXiv | 高遮蔽率有利于高冗余信号的表征预训练。 | 在公开域上预注册遮蔽率 32%、48%、64% 三点，不用测试集筛选。 |
| Chen et al., *SeisLM* (2024) | OpenAlex、Crossref、arXiv | 大规模地震波形预训练可迁移到多个地震任务，但域和任务评估仍需明确隔离。 | INSTANCE 完成后，用公开跨区域评估验证多域波形预训练，而非只报域内精度。 |
| Mousavi et al., *Earthquake Transformer* (2020) | OpenAlex、Crossref | 注意力可联合编码长程时序信息，训练与测试域差异不能由模型复杂度自动消除。 | 用 Transformer 但必须保留 A→B→C 验证；本轮 scratch 逊色是重要反例。 |
| Woollam et al., *SeisBench* (2022) | OpenAlex、Crossref、arXiv | 统一数据接口和跨数据集基准是可比较地震机器学习的基础。 | INSTANCE 读取后先做格式、分量顺序、采样率与事件切分审计。 |
| Steinberg et al., *Which picker fits my data?* (2021) | OpenAlex、Crossref | 拾取器表现会因区域和数据条件显著变化。 | 不以单个区域胜利录取模型；继续报告平均与最差自然域。 |
| Mousavi et al., *STEAD* (2019) | OpenAlex、Crossref | 全球公开数据促进训练，但数据异质性和元数据质量直接影响结果。 | 固定缓存哈希、事件切分与过滤清单；不混入未审计来源。 |
| Michelini et al., *INSTANCE* (2021) | OpenAlex、Crossref | 意大利强震动数据集提供与 STEAD 明显不同的独立区域。 | 完成下载后作为独立外域，只用其官方 train/dev，绝不读官方 test。 |
| *Network-Based Earthquake Magnitude Determination via Deep Learning* (2021) | OpenAlex、Crossref | 震级估计受多台站/网络结构和观测条件影响。 | 本轮单台站相对尺度的不足不应被掩盖；下一轮记录并分析站点与网络层面的误差，但只在公开验证域。 |
| *Earthquake Magnitude Estimation Based on ML for EEW* (2021) | OpenAlex、Crossref | EEW 中震级估计需要面对有限窗口和快速可用性约束。 | 继续固定 P±5 s 窗口，不以事后更长窗口换取公开分数。 |
| Koh et al., *WILDS* (2021) | OpenAlex、Crossref、arXiv | 真实分布偏移需要按有意义的域/组评估，而不是随机切分。 | A→B→C 和最差域门槛继续作为录取条件。 |
| Kumar et al., *In Search of Lost Domain Generalization* (2022) | OpenAlex、Crossref、arXiv | 一些域泛化改进在严格比较下不稳定。 | JEPA 只在独立 INSTANCE 域再现后才进入候选清单。 |
| Ovadia et al., *Deep Ensembles* (2019) | OpenAlex、Crossref、arXiv | 独立初始化的深度集成可改善分布外稳健性与不确定性。 | 若 JEPA 在 INSTANCE 上通过，才试验公开域训练的 3–7 成员集成；成员选择不能接触比赛集。 |
| Komodakis & Zagoruyko, *Knowledge Distillation* (2022) | OpenAlex、Crossref、arXiv | 蒸馏成效对教师一致性和实现细节高度敏感。 | 不在测试包上扫温度/权重；仅在公开 train/dev 用预注册配方试验 JEPA 教师蒸馏。 |

原始检索和摘要快照位于远程：`/root/5.6+chanshui1/outputs/round39_lit/search_raw.json` 与 `readlist.json`。网页检索接口在本会话未暴露 AnySearch / Playwright MCP，因此使用了可复核的三公开学术索引 API；这是工具可用性限制，不应被描述成使用过 AnySearch。

## 下一轮：第 39B 轮预注册方案

前置条件是 INSTANCE 压缩包完整下载、解压、HDF5 随机抽检和缓存审计全部通过；在此之前不读取不完整 HDF5，也不以任何比赛包替代该独立域。

1. **数据白名单**：仅 `t2_cache_station27` 和审计通过后的 `t2_cache_instance31`；INSTANCE 只使用官方 `train` / `dev`，绝不读取官方 `test`。
2. **四条臂**：STEAD-only scratch、STEAD-only JEPA-MR、INSTANCE-only scratch、STEAD+INSTANCE JEPA-MR。参数量、窗口、增强、微调 epoch 固定。
3. **双向协议**：STEAD 训练 / INSTANCE dev 评估，以及 INSTANCE train 训练 / STEAD 自然域评估；训练域内再按事件划出独立 B 来选择 `λ`。
4. **录取门槛**：JEPA 相对相同数据的 scratch，平均得分严格提高、最差域不降低；跨数据源双向都要成立。只满足单向时登记为负面或不充分证据。
5. **不确定性与集成**：仅在上述门槛通过后，使用 3 个公开数据训练的独立随机种子成员测量均值/最差值与方差；不将任何比赛数据用于成员或权重选择。
6. **可解释误差审计**：按公开数据的事件、网络、台站和震级段报告误差；这些诊断不触碰 08 / R1 / R2。
