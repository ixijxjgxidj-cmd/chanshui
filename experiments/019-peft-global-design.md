# 实验019设计：PEFT + STEAD全球数据

## 目标
解决实验018发现的过拟合问题，通过：
1. 全球多区域数据（STEAD）
2. 参数高效微调（LoRA/Adapter）
3. 保持预训练泛化能力

## 文献基础

### 核心方法论
- **arXiv 2512.13197**（2024/12）："Parameter-Efficient Transfer Learning for Microseismic Phase Picking Using a Neural Operator"
  - PhaseNO架构
  - 仅微调3.6%参数
  - Fourier Neural Operator用于跨域适配
  - 在微地震（实验室尺度）→板块地震成功迁移

### 数据集
- **STEAD**（GitHub: smousavi05/STEAD）
  - 1.2M+波形，全球覆盖
  - 350km局部地震 + 噪声
  - SeisBench直接支持：seisbench.data.STEAD()
  - 已标注P/S到达时间

### 集成优化
- **Phase-Weighted Stacking (PWS)**
  - Thurber et al. 2014, BSSA
  - SNR提升显著优于线性平均
  - 适用于低信噪比相位拾取

## 实验设计

### 实验019A：STEAD远场子集提取
**目标**：从STEAD筛选远场样本（>30km）

**步骤**：
1. 在Azure上下载STEAD完整数据集（~100GB）
2. 筛选条件：
   - 震中距 ≥ 30km
   - S-P时差 ≥ 3s（对应~30km）
   - SNR_P ≥ 3, SNR_S ≥ 2
   - 多板块分布（至少5个区域）
3. 目标规模：10,000窗口
4. 划分：train 70%, val 15%, test 15%（按震源区域分层）

### 实验019B：LoRA微调
**架构修改**：
`
PhaseNet原始层 → 冻结
每个Conv层后插入LoRA：
  W_new = W_frozen + α * (B @ A)
  A: (d, r), B: (r, d), r=8
参数量：原始~2M，新增~72k（3.6%）
`

**训练配置**：
- 预训练：PhaseNet(diting)
- 优化器：AdamW, lr=5e-5（比实验018保守10倍）
- batch=64, epochs=20
- 早停：val_loss不降连续5 epoch
- 损失：CrossEntropy（标准分类，不用KL软标签）

**验证策略**：
- 每5 epoch评估：
  - STEAD val（同分布）
  - iquique（智利远场，分布外）
  - devmix（全球混合，OOD）
- 选择devmix最优模型（而非val_loss最低）

### 实验019C：Adapter微调（保守方案）
如LoRA仍过拟合，尝试Adapter：
- 每层后插入bottleneck：d→64→d
- 参数量更少（~1.5%）
- 只调整adapter，主干完全冻结

### 实验019D：Phase-Weighted Stacking
**目标**：优化c3集成权重

**方法**：
1. 基于相位一致性动态加权
2. 不需要训练，基于信号相干性
3. 对比：
   - 简单平均（当前c3）
   - PWS
   - 学习权重（meta-learner）

## 评估框架

### 测试集
1. **iquique**（智利远场600窗）
2. **devmix**（全球混合3407窗）← 主裁判
3. **STEAD test**（STEAD保留15%）
4. **PNW near**（PNW近场500窗）← 护栏，确保不退化

### 成功标准
- devmix得分 ≥ c3基线（1.5747）
- iquique得分 ≥ c3基线（1.5646）
- PNW near得分 ≥ c3在PNW表现
- 分bin：[30,40)和[40,50)提升，[0,20)不退化

### 失败判据
如019B/C均失败（devmix退化），则：
- 转向实验020：纯集成优化（PWS/Stacking）
- 考虑JEPA架构重构（长期）

## 资源分配

### Azure（主力）
- 数据下载与预处理（210GB可用）
- STEAD筛选脚本
- 特征提取与窗口切割

### zzai（训练）
- LoRA/Adapter微调
- AMD DCU，68.7GB显存
- 预计单次训练：20 epochs × 10min/epoch = 3.3小时

### tor1（中转，如需要）
- 大文件暂存
- 跨服务器数据传输

### 本地
- 不做训练/数据下载
- 仅评估脚本与结果分析

## 时间线

### Day 1（今天）
- [✓] 论文调研补充至15+篇
- [→] 在Azure启动STEAD下载
- [ ] 编写筛选脚本

### Day 2
- [ ] STEAD远场子集提取完成
- [ ] 传输到zzai（预计90MB，9秒）
- [ ] 实现LoRA注入代码

### Day 3
- [ ] 019B训练（3.3小时）
- [ ] 四测试集评估
- [ ] 决策：成功→固化，失败→019C

### Day 4
- [ ] 如019B成功：PWS优化（019D）
- [ ] 如019B失败：Adapter微调（019C）
- [ ] 提交GitHub

## 合规检查
- ✓ 08决赛数据不参与训练
- ✓ R1/R2不再使用（已证明过拟合）
- ✓ STEAD公开数据集，记录来源
- ✓ 按区域分层划分，防泄漏
- ✓ 本地不做训练

## Git提交点
1. 实验019设计文档（本文件）
2. STEAD下载完成
3. LoRA训练完成
4. 评估结果与决策
