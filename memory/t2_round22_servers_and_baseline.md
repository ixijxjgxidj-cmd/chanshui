# 轮 22 服务器台账与 R1 域内基线（2026-08-15）

## 一、四台机器实测能力

| 机器 | 角色 | GPU | 磁盘 | 关键事实 |
|---|---|---|---|---|
| zzai | T2 主训练 | HCU/ROCm（torch 2.9, hip 6.3） | 1.2T 可用 | STEAD 官方源单流约 4.6 MB/s；并行反而退化（4路合计3.4MB/s） |
| **趋动云 virtai** | **CUDA 并行路线** | **NVIDIA, torch 2.13+cu130** | `/gemini/code` 14P | 直连 SeisBench 源支持 Range；已装 obspy 1.5.0 / seisbench 0.12.3 / sklearn 1.7.2 |
| Azure | 数据中转/CPU | 无 | 207G | HF 3.4 MB/s；无 torch |
| tor1 | 隧道与暂存 | 无 | 6×100G | 单流 14.6 MB/s；`sing-box` 与用户项目未动 |

### 趋动云连接根因（此前判为不可用，已修正）
失败信息是 `sign_and_send_pubkey: no mutual signature supported`，
原因是网关只接受旧式 `ssh-rsa` 签名，而新 OpenSSH 默认禁用该算法，**不是密钥不对**。
可用连接方式：

```
ssh -i <id_rsa> -o IdentitiesOnly=yes -o PubkeyAcceptedAlgorithms=+ssh-rsa \
    -p 30022 'shitt@root@ssh-<hash>.<token>@direct.virtaicloud.com'
```

两个平台特性：`pip3` 不在 PATH（用 `python3 -m pip`）；容器盘仅 22G，
seisbench 缓存必须重定向到 `/gemini/code`（已写入 `env.sh` 的 `SEISBENCH_CACHE_ROOT`）。

## 二、R1 域内诚实基线（预注册 train=140，holdout 60 未读取）

去掉协议伪特征 `log(npts)` 后，6 维物理特征在 R1 训练集上的 5 折 CV：

| 模型 | MAE | score/200 |
|---|---:|---:|
| **常数（中位数）** | **0.3593** | **133.71** |
| GBM（6 物理特征） | 0.3849 | 129.85 |
| Ridge（6 物理特征） | 0.4004 | 124.50 |

特征相关性全部不显著：`logAmax_Z +0.064`、`logAmax_H +0.057`、`logRMS_Z +0.093`、
`logRMS_H +0.083`、`fc_Z -0.143`、`fc_H -0.123`（p 均 >0.09）。

**结论：常数基线比最好的物理模型高 3.87 分 —— R1 域内手工物理特征无技能。**
这与轮 22 的诊断链自洽：R1 振幅信息存在但缺距离项，而 T2 截窗内无 S 波可供估距。
因此 T2 的提分只能来自「深度模型从波形隐式学习距离/场地」，即 MagNet 族路线。

同时这条基线给出官方域的**可信参照线 133.71**：任何 STEAD 迁移模型若在 R1 上低于此值，
即使 STEAD 验证再好也不构成真实增益。

## 三、当前分工

- zzai：STEAD 88.7G/91.1G，完成后自动串行 A0→A1→A2→A3（单卡串行，无资源冲突）
- 趋动云：8 路并行下载 STEAD（metadata 已完成 402,560,190 字节），承担 CUDA 侧候选
- R1 数据与预注册划分已同步到趋动云并校验（201 文件，split sha 一致）
- 冻结评估器 `eval_t2_r1_frozen.py` 已部署，硬编码拒绝任何 `08`/`exam` 路径

## 四、合规

训练数据只用 STEAD 公开集与 R1 预注册 train；R1 holdout 60 条至今零读取；
08 全程未参与训练/调参/选择；本地未做任何训练与数据集下载（R1 中转 4.2MB 仅为服务器间搬运既有比赛数据）。
