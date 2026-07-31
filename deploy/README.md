# API 部署（P0 保底上线）

## 一键部署（推荐：2~4 vCPU 按量付费云主机，Ubuntu/Debian/CentOS 均可）

```bash
git clone https://gitee.com/hulk-cheng/dizheng.git
cd dizheng
sudo bash deploy/deploy_api.sh
```

脚本做的事：装 venv 依赖（torch 优先 CPU 轮子）→ 从 `weights/*.tar.gz` 恢复
seisbench 权重缓存（**不走跨境下载**）→ 装 systemd 服务（开机自启+崩溃自拉起）→
健康检查 → 合成波形跑 `check_api.py` 官方格式自检。

完成后**必须手工做**的两件事：

1. 云控制台安全组放行 **TCP 8000**；
2. 从外网机器复测后，把 `http://<公网IP>:8000/pick` 登记到比赛平台。

## 为什么默认 `--pretrained diting`（2026-07-31 实测）

去年真题 A/B（`scripts/ab_compare.py`，Task1 官方计分，满分 2 分/文件）：

| 权重 | 第1轮前200条 | 第1轮全量1000 | 第2轮全量915 |
|---|---|---|---|
| stead | 1.496 | — | — |
| stead+Iquique全量微调 | 1.678 | — | — |
| **diting（默认）** | **1.899** | **1.629** | **1.500** |

diting（271万条中国 DiTing 数据训练）显著优于 stead 及其微调版，P2 微调从
diting 起步。注意 diting 模型原生 50Hz、窗长 60.02s——短于一窗的波形由
`picker._to_stream` 自动尾部边缘补齐（已修，否则官方 51.5s 文件全空报）。

## 常用运维

```bash
journalctl -u phasepick-api -f          # 看日志
systemctl restart phasepick-api         # 重启
curl http://127.0.0.1:8000/health       # 健康检查
```

换微调权重（P2 产出 best.pt 后）：

```bash
sudo WEIGHTS=/root/dizheng/weights/best.pt bash deploy/deploy_api.sh
```

（重跑脚本是幂等的：依赖已装则跳过，只重写 systemd 单元并重启。）

## Windows 本地调试注意

- 必须带 `PYTHONUTF8=1`（seisbench 读 diting 元数据 JSON 会被 GBK 默认编码坑崩）：
  `PYTHONUTF8=1 python scripts/serve_api.py --pretrained diting --device cpu`
- Linux 服务器默认 UTF-8，无此问题。
