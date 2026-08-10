# 本周服务器训练手册（2026-08-11，初赛周·CPU 方案）

> 目标：按已验证的 S-P 重叠度判据，从 SeisBench 公开数据集里筛出 CREW 之外的
> 高重叠数据集，复刻 crew1 成功配方（小池 ~18k 窗 + 真题锚点）CPU 微调出
> 候选第 6 成员。判据与配方出处：fork 提交 8a77846（crew_sp23，1.754→1.764）。

## 已知事实（决定方案形状，勿凭直觉推翻）

- 真题 S-P 时差：中位 21.31s，5%~95% [4.15, 67.05]s（区域震~远震）
- CREW 重叠 84.3% → 成功；CEED 重叠 29.8%（近场微震）→ 十连负
- "更多 CREW"已证否（48k 大池两侧分裂）；同域增广 augA/augB 也已证否
- 结论：本周唯一有先验支撑的新成员来源 = **未筛过的高重叠外部数据集**

## 服务器步骤（依次执行）

```bash
# 0) 环境（CPU torch 即可）
git clone https://github.com/ixijxjgxidj-cmd/dizheng-gpt5.6-sol.git dizheng && cd dizheng
pip install torch obspy seisbench scikit-learn h5py tqdm
# 上传本机产物：outputs/train/r2train.h5（33MB，真题锚点池）

# 1) 只下 metadata 筛数据集（不下波形，每个数据集几十MB~GB级）
python - <<'EOF'
import seisbench.data as sbd
for name in ("GEOFON","MLAAPDE","NEIC","LenDB","OBS","VolPick","CWA","Iquique"):
    try:
        getattr(sbd, name)(download_kwargs={"blocksize": 2**20}, missing_components="ignore", metadata_only=True)
    except TypeError:
        pass  # 老版本无 metadata_only：手动下 metadata.csv 或小 chunk
EOF
python scripts/scout_sp_distance.py --cache ~/.seisbench --datasets geofon,mlaapde,neic,lendb,obs,volpick,cwa,iquique

# 2) 选重叠度最高且 ≠CREW 的 1~2 个数据集，只下 2~3 个分片的波形

# 3) 转池 + 混真题锚点（复刻 crew1 比例：外部 ~18k 窗 + r2train 全量锚点）
python scripts/mix_pool.py --help   # 看 --upsample 用法（锚点上采样验稀释假设）

# 4) CPU 微调（PhaseNet 27万参数，18k 窗 CPU 数小时）
python scripts/finetune_phasenet.py --data <混合池.h5> --holdout outputs/train/r2train.h5 \
    --out outputs/ft_<数据集名> --epochs 30 --batch 16 --lr 1e-4
# 产物瘦身成 state_dict（fork 惯例 3.26→1.11MB）后拉回本机
```

## 回本机后的验收闸门（不在服务器上做）

1. A 规则快筛：新权重作第 6 成员，`outputs/port_verify/_eqt_exp.py` 同款
   A 规则框架，r2 基线 1552.6，不涨直接弃
2. 过筛后全流程三分布 + 逐文件 diff（基线 1.779/1.801/2.009），同向不劣才上线
3. 警惕捆绑效应：fork 发现过"第5成员与限额捆绑"，新成员要在完整生产配置下验

## 本周并行任务（本机，不占服务器）

- 第 6 区域成员筛选：41 个 USTC 权重里筛华南邻省（广东/海南/云南/湖南/贵州），
  纯推理 A 规则每个 ~3 分钟——历史"top5 持平"是三区域时代结论，当前配置未筛过
