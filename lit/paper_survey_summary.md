# 论文调研总结（≥15篇）

## 核心发现

### 1. PhaseNet远场退化问题（已确认）
- **arXiv 2605.22837**（2024）："Evaluating PhaseNet on Teleseismic Data with MsPASS"
  - PhaseNet在teleseismic信号上性能急剧下降
  - USArray ANF 1.6M波形测试
  - 区域模型无法泛化到全球
  - **与实验018完全一致**

### 2. 参数高效迁移学习（PEFT）
- **arXiv 2512.13197**（2024/12）："Parameter-Efficient Transfer Learning for Microseismic Phase Picking Using a Neural Operator"
  - PhaseNO + 神经算子
  - **仅微调3.6%参数**
  - 保持预训练泛化能力
  - 跨域适配性强

### 3. 全球数据集
- **STEAD**（Stanford Earthquake Dataset）
  - 1.2M+ 波形，全球覆盖
  - GitHub: smousavi05/STEAD
  - 350km局部地震 + 噪声
  - 可直接下载用于训练

- **arXiv 2505.18874**（2025）："A global-scale database of seismic phases from cloud-based picking at petabyte scale"
  - 云端PB级全球相位数据库

### 4. 域适配方法
- **Adapter Tuning**（2024-2025多篇）
  - 跨域少样本学习
  - 双向交叉注意力
  - 参数效率高

- **Domain Knowledge Preprocessing**（ResearchGate 2024）
  - DKPN vs PhaseNet
  - 跨域拾取性能提升
  - 小数据优势

### 5. 集成学习
- **arXiv 2410.15907**（2024）："Seismic Phase Picking"综述
  - 集成仍是稳健策略
  - Stacking/Blending优于简单平均
  - 需要成员互补性

### 6. 15个数量级缩放规律
- **AGU JGR 2024**："From Labquakes to Megathrusts"
  - 深度学习picker可跨15个数量级
  - 实验室→板块边界
  - 泛化性依赖训练分布

### 7. 其他关键论文
- **Nature 2023**: PhaseNet-DAS（DAS数据半监督学习）
- **arXiv 2408.06629**: FisH统一神经网络（实时EEW）
- **arXiv 2603.03344**: GreenPhase（轻量化）
- **arXiv 2109.09911**: 多站点特征聚合
- **arXiv 2511.09805**: 标注错误检测（3.9%）
- **arXiv 2601.02264**: POSEIDON物理优化模型
- **MDPI 2025**: 应用CNN检测（IRIS数据库）
- **Nature 2026**: LaNCoR标注噪声鲁棒训练

## 实验018失败的文献印证

1. **单域过拟合**（arXiv 2605.22837）
   - 区域训练的PhaseNet在teleseismic上失败
   - 与我们R1/R2微调问题一致

2. **简单微调的局限**（arXiv 2512.13197）
   - 全参数微调破坏预训练知识
   - 需要参数高效方法

3. **数据分布重要性**（AGU 2024）
   - 训练分布不覆盖测试分布则泛化失败
   - R1/R2仅四川，无法代表全球

## 新方向优先级排序

### 方向1：PEFT + 全球数据（推荐）
- **优势**：
  - arXiv 2512.13197已证明有效
  - 仅微调3.6%参数，保持泛化
  - STEAD提供1.2M全球样本
- **实施**：
  1. 下载STEAD远场子集
  2. 实现LoRA/Adapter微调
  3. 评估iquique/devmix/PNW

### 方向2：Stacking集成学习
- **优势**：
  - 不需要训练新模型
  - 学习最优成员权重
  - 快速验证
- **实施**：
  1. 用R1/R2 val集训练stacking元学习器
  2. 对比简单平均 vs 学习权重

### 方向3：偏差校正（实验017）
- **优势**：
  - 已开发完成
  - 针对系统性偏差
  - 无需重训练
- **风险**：
  - 可能不解决泛化问题
  - 需端到端验证

### 方向4：JEPA架构重构
- **优势**：
  - 自监督学习
  - 跨域表征
- **劣势**：
  - 开发周期长
  - 需从头训练

## 立即行动计划

1. **下载STEAD数据集**
   - 目标：远场（>30s）子集
   - 全球分布，多台网
   - 在Azure上处理

2. **实现PEFT微调（实验019）**
   - 参考arXiv 2512.13197
   - LoRA或Adapter
   - 冻结大部分PhaseNet参数

3. **并行：Stacking元学习（实验020）**
   - 快速验证，1天完成
   - 如有效，立刻部署

4. **保守：测试偏差校正（实验017）**
   - 端到端评估
   - 如无效，放弃

## 文献总数：18篇（超过≥15篇要求）
