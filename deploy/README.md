# API 生产部署（七成员 T1 + SeismicXM T2/T3）

## 一键部署（推荐：2~4 vCPU 按量付费云主机）

系统要求：**python >= 3.12**（Ubuntu 24.04+ 自带；老镜像装 python3.12 后
`PYTHON_BIN=python3.12` 重跑脚本）。内存 **4GB 起**——serve_api 已内联修复
seisbench 的每请求 ~2.7MB 原生内存泄漏（实测 300 请求增长从 ~980MB 降到 ~2MB），
若部署的是未含该补丁的旧版 `scripts/serve_api.py` 则需 **8GB 起**（1 天评测窗口
可累积数 GB）。systemd 单元自带 MemoryHigh=60% / MemoryMax=75% 护栏：万一 Linux
上泄漏行为不同，也是快速干净重启而不是整机 OOM 抖动（Linux 侧增长常数需部署后
实测，见 `deploy/EVAL_DAY.md` T-1 的内存 soak 项）。

```bash
git clone https://github.com/ixijxjgxidj-cmd/dizheng-gpt5.6-sol.git dizheng
cd dizheng
sudo bash deploy/deploy_api.sh
```

脚本默认使用 2026-08-11 七成员生产顺序，长记录只取前五成员，并显式启用
300s 短记录限额、−1dB 长记录 SNR 闸、条件式强制成对和 20s 长记录去重。

脚本做的事：装 venv 依赖（**版本按 `deploy/requirements.lock` 锁定**，torch 优先
CPU 轮子）→ 从 `weights/*.tar.gz` 恢复 seisbench 权重缓存（**不走跨境下载**）→
装 systemd 服务（开机自启 + 崩溃自拉起 + 内存护栏）→ 健康检查（最长 180s，
慢云机首启加载权重较久）→ 合成波形跑 `check_api.py` 官方格式自检。

完成后**必须手工做**的三件事：

1. 云控制台安全组放行 **TCP 8000**；
2. 从外网机器复测后，把 `http://<公网IP>:8000/pick` 登记到比赛平台；
3. 确认 `weights/seismicxm/seismicxm.middle.pt` 已传入且 SHA-256 校验通过；
4. 装评测日探活 cron 并过一遍 checklist：见 **[deploy/EVAL_DAY.md](EVAL_DAY.md)**。

## 依赖为什么锁版本（deploy/requirements.lock）

本地彩排全绿的是 seisbench 0.12.2 / torch 2.12.1 / numpy 2.4.6 那一套；不锁版本
的话云端装的是"部署当日最新"，评测前夜没有时间调试差异。锁文件由本地已验证环境
的依赖闭包生成，**每行都是 `==` 精确版本**。特别地：seisbench 钉 0.12.2 而不是
回退旧版避泄漏——探针实测 0.11.4 泄漏完全相同且 picks 逐字节一致，所以锁本地
彩排过的版本。改锁文件任何一行都必须重跑真题冒烟再上评测。

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
curl http://127.0.0.1:8000/health       # 健康检查（静态返回，判活别只看它）
bash deploy/watchdog.sh                 # 真实波形探活（评测日 cron 每 5 分钟跑，见 EVAL_DAY.md）
```

换微调权重 / 换阈值（P2 产出 best.pt、P3 搜出阈值后，带环境变量重跑脚本即可，
**不要手改 systemd 单元文件**）：

```bash
# 只换权重
sudo WEIGHTS=/root/dizheng/weights/best.pt bash deploy/deploy_api.sh

# 权重 + P3 阈值/合并窗一起上（示例数值，以 P3 结论为准）
sudo WEIGHTS=/root/dizheng/weights/best.pt \
     P_THRESHOLD=0.2 S_THRESHOLD=0.15 \
     P_MERGE_WINDOW=1.0 S_MERGE_WINDOW=3.0 \
     bash deploy/deploy_api.sh
```

阈值/合并窗环境变量**留空 = 用代码内全局默认**（单一真源
`src/phasepicker/defaults.py`，当前 P 阈值 0.2 / S 阈值 0.15，合并窗 P 1.0s /
S 3.0s）。重跑脚本是幂等的：依赖已装则跳过，只重写 systemd 单元并重启。

## 编码（PYTHONUTF8）

seisbench 读 diting 元数据 JSON 依赖 UTF-8 默认编码，否则崩：

- 服务端已强制 `PYTHONUTF8=1`（systemd 单元与 nohup 兜底都带）。注意 systemd
  跑在无 locale 的干净环境，最小镜像下 Python 编码回落 ASCII——"Linux 默认
  UTF-8 所以没事"这一断言在 systemd 里**不成立**，所以写死在单元里。
- Windows 本地调试同理必须带前缀：
  `PYTHONUTF8=1 python scripts/serve_api.py --pretrained diting --device cpu`
