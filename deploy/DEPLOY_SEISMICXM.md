# SeismicXM + 广西权重 部署清单（2026-08 模型升级批次）

> 状态：**2026-08-11 已部署到新服务器，服务端与公网四端点验收全部通过**。
> 当前仓库与 GitHub `main` 均为七成员发布版；本文档描述本次升级与验收口径；
> 通用部署和评测日流程见 `deploy/README.md` / `deploy/EVAL_DAY.md`。

## 当前生产部署验收（2026-08-11）

- Debian 13、4 vCPU、6.8 GiB RAM、CPU 推理；仓库路径和服务账号按目标机器配置，
  不写入公开仓库；systemd 单元 `phasepick-api` 开机自启。
- 服务以非 root 普通用户运行；七个 T1 成员、T2/T3 bundle 与
  SeismicXM encoder 共 10 个发布资产均已逐一校验哈希，启动日志没有回退。
- `/pick`、`/magnitude`、`/classify` 的真实样本以及纯噪声、50 Hz、多台站、
  4000 秒长记录、空文件和损坏文件边界测试均通过。
- 300 次连续真实 `/pick` 请求全部成功：均值 42.17 ms、P95 44.91 ms、最大
  56.35 ms；RSS 仅增长约 4 MiB，内存泄漏修复在 Linux 上复验通过。
- 服务已监听 `0.0.0.0:8000`，云平台 TCP 8000 入站规则已生效。从非服务器机器经
  `http://<公网地址>:8000` 复测：`/health`、`/pick`、`/magnitude`、`/classify`
  均为 HTTP 200，返回值与服务器本机一致；公网耗时分别约 0.16s、0.27s、0.39s、1.03s。
- 部署侧已无阻塞项。上线前剩余操作是把各任务 URL 登记到比赛平台；评测日 watchdog
  cron 尚未安装，因为当前脚本每次真实 `/pick` 探活都会写入生产 `captured/`，安装前需
  明确接受并过滤探针记录，或先为探活请求增加不采集标记。

## 本批次上线内容（本地已全部验收，随七成员发布提交）

| 组件 | 变化 | 验收成绩 |
|---|---|---|
| T1 拾取 | diting → **7成员概率集成（长记录门控前5）+ 短文件限额 + SNR闸 + 条件式强制成对 + 长记录去重**（2026-08-11 定稿） | 三套历史包冻结回放（答案已知，非盲测）：r1 **1.786294** / r2 **1.810140** / 08 **2.010084**（GEOFON m1/m3 为第6/7成员；长记录固定排除） |
| T2 震级 | joblib 特征树 → **SeismicXM deep1024+Ridge** | MAE 0.817→**0.621** |
| T3 分类 | joblib 特征树 → **SeismicXM TTA+余弦kNN(k=5)** | r2 两类 81.5%→**98.94%**；08 包答案实际只出现标签 1–4，**183/205=89.27%**（非盲测，不能证明第 5 类泛化） |

2026-08-11 官方群确认的规则情报：
- 计分规则与本仓 scorer.py 逐条一致；"每文件答案个数不定"确认多相位合法
- **今年数据"不限制加速度记录，以速度为主"**：实测积分转速度全负（原始加速度
  直接喂反而最好，逐窗归一化吃掉幅值差），生产**不做任何加速度特殊处理**
- 今年 T1 **含纯噪声条目** → 强制成对默认切条件式（--force-pair-mode conditional，
  另一相位阈值上有触发才补；纯噪声两相位皆无触发=空输出，完全免疫）

已证否勿再试（2026-08-10/11 实验记录）：TTA 极性翻转（r2 −0.4 且 2× 耗时）、
overlap 0.75（r2 −2.1）、overlap 0.9（r2 +6.5 但 08 −2.4 三分布不同向，5× 耗时）、
加速度积分转速度（08 两 HN 文件全变体负）、兜底 SNR 闸（真实兜底拾取 SNR
中位 0.03dB 与噪声不可分）。

## 需要新上传到容器的文件

- `weights/seismicxm/seismicxm.middle.pt`（207,709,060 bytes，不在 git；scp/rsync 上去；
  SHA-256 `671d02d677c25c3d075963889602299ec71f52c724470f2fa85bb28035fe1528`）
- `git pull` 带上：`weights/ustc_pickers/guangxi_sd.pt`、
  `weights/aug/{exam_aug6,crew_sp23}_r2train_sd.pt`、`weights/geofon/geofon_m{1,3}_last_sd.pt`（T1 集成第4-7成员，已在 git）、
  `weights/official_r1_to_r2/t{2,3}_seismicxm_r1r2.joblib`、
  `src/phasepicker/vendor/`（SeismicXM 模型定义）及全部代码改动

## 容器内步骤

```bash
pip install einops           # SeismicXM 唯一新依赖
git pull
# 权重就位检查
ls -la weights/seismicxm/seismicxm.middle.pt weights/ustc_pickers/*_sd.pt \
      weights/aug/*_sd.pt weights/official_r1_to_r2/t2_seismicxm_r1r2.joblib
# 重启前强制核对生产参数、全部跟踪资产和外置 encoder
python scripts/verify_release_manifest.py --require-external
# 启动（T1 生产配置 2026-08-11 定稿：7成员集成、长记录仅前5 + 短文件限额 + SNR闸 +
#       条件式强制成对 + 长记录去重(20s)，三分布 r1 1.786294 / r2 1.810140 / 08 2.010084。
#       cap/SNR/force-pair/long-dedup 均为 serve_api 默认值，显式写出防误改）
python scripts/serve_api.py --port 8000 \
  --weights "guangxi,jiangxi,shandong,weights/aug/exam_aug6_r2train_sd.pt,weights/aug/crew_sp23_r2train_sd.pt,weights/geofon/geofon_m1_last_sd.pt,weights/geofon/geofon_m3_last_sd.pt" \
  --cap-short-s 300 --cap-max-p 1 --cap-max-s 1 --long-snr-db -1.0 \
  --force-pair-short-s 300 --force-pair-mode conditional \
  --long-dedup-s 20 --ensemble-long-members 5
```

## 启动日志必须出现（缺一不可，出现"回退"字样=权重没就位）

```
拾取器: 概率集成 × 7 成员 ['guangxi', 'jiangxi', 'shandong', 'weights/aug/exam_aug6_r2train_sd.pt', 'weights/aug/crew_sp23_r2train_sd.pt', 'weights/geofon/geofon_m1_last_sd.pt', 'weights/geofon/geofon_m3_last_sd.pt']
T1 有效配置: cap<=300s (P<=1,S<=1); long_snr=-1dB; force_pair=conditional; long_dedup=20s; ensemble_long_members=5
震级估计器: seismicxm 已就绪
分类器: seismicxm 已就绪
```

## 外网验收（三端点，用 outputs/*/smoke/ 下的样本文件）

```bash
curl -F "file=@T3.A.Q0001.mseed" http://<外网地址>/pick       # 期望 P/S 各一个
curl -F "file=@T2.Q0001.mseed"   http://<外网地址>/magnitude  # 期望 M≈4.9（真值5.0）
curl -F "file=@T3.A.Q0001.mseed" http://<外网地址>/classify   # 期望 class=1
```

## 性能预期与注意

- SeismicXM 51.9M 模型 CPU 上 TTA 分类约 1.2 文件/秒（长波形 5 窗）；容器 GPU 会快一个量级。
  若确需应急降级，应显式选择 baseline；`deploy_api.sh` 默认拒绝缺 encoder 的
  SeismicXM 部署，只有 `ALLOW_MODEL_FALLBACK=1` 才允许缺失时继续。
- 决策记录：**评测日固定用广西权重，不做临场权重切换**——两个无标签选择信号
  （平均置信度 Spearman 0.08、共识一致度 0.53）均未通过去年数据验证，同源模型
  分差小于无监督代理的分辨率（2026-08-01 实验，见 memory/ustc-guangxi-t1-result）。
- 真题微调路线已取消（第一届数据=四川系，今年广西主办大概率换源，方向错误）。
- 容器重启后端口映射地址会变，报名平台登记地址需核对（见 memory/p0-deploy-status）。
