# SeismicXM + 广西权重 部署清单（2026-08 模型升级批次）

> 状态：**待执行**（用户暂缓部署）。本文档只描述"这次升级比上次部署多做什么"，
> 基线部署流程见 EVAL_DAY.md / 容器实况。

## 本批次上线内容（本地已全部验收，git 已提交）

| 组件 | 变化 | 验收成绩（去年第2轮盲测） |
|---|---|---|
| T1 拾取 | diting → **广西+江西+山东 概率集成** + 亚采样精细化 | 均分 1.669→**1.723**（r1 同验 1.732→1.744） |
| T2 震级 | joblib 特征树 → **SeismicXM deep1024+Ridge** | MAE 0.817→**0.621** |
| T3 分类 | joblib 特征树 → **SeismicXM TTA+余弦kNN(k=5)** | 81.5%→**98.9%** |

## 需要新上传到容器的文件

- `weights/seismicxm/seismicxm.middle.pt`（**208MB**，不在 git；scp/rsync 上去）
- `git pull` 带上：`weights/ustc_pickers/guangxi_sd.pt`、
  `weights/official_r1_to_r2/t{2,3}_seismicxm_r1r2.joblib`、
  `src/phasepicker/vendor/`（SeismicXM 模型定义）及全部代码改动

## 容器内步骤

```bash
pip install einops           # SeismicXM 唯一新依赖
git pull
# 权重就位检查（缺 seismicxm 会自动回退旧 baseline 不报错——要看启动日志！）
ls -la weights/seismicxm/seismicxm.middle.pt weights/ustc_pickers/*_sd.pt \
      weights/official_r1_to_r2/t2_seismicxm_r1r2.joblib
# 启动（T1 = 三区域概率集成，2026-08-02 定稿：两轮均超最优单模型）
python scripts/serve_api.py --port 8000 --weights "guangxi,jiangxi,shandong"
```

## 启动日志必须出现（缺一不可，出现"回退"字样=权重没就位）

```
拾取器: 概率集成 × 3 成员 ['guangxi', 'jiangxi', 'shandong']
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
  若评测吞吐紧张，可 `--cls-weights/--mag-weights` 临时切回 baseline（各自独立回退）。
- 决策记录：**评测日固定用广西权重，不做临场权重切换**——两个无标签选择信号
  （平均置信度 Spearman 0.08、共识一致度 0.53）均未通过去年数据验证，同源模型
  分差小于无监督代理的分辨率（2026-08-01 实验，见 memory/ustc-guangxi-t1-result）。
- 真题微调路线已取消（第一届数据=四川系，今年广西主办大概率换源，方向错误）。
- 容器重启后端口映射地址会变，报名平台登记地址需核对（见 memory/p0-deploy-status）。
