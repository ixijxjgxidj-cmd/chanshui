# 地震震相拾取系统 — 参赛工程仓库

面向"中国（广西）—东盟人工智能+应急管理科技创新大赛 / 防震减灾专项赛道 · 自动震相拾取"的工程代码。

这个仓库存在的首要目的：**GPU 免费主机关机即清空，用 Gitee 仓库做持久化。每次开机 `git clone` 一条命令拉回全部环境。**

---

## ⚡ 2026 官方标准（第二届"震智杯"，以《比赛说明》PPT 为准）

**提交方式已从"上传 .an 文件"改为"公网 HTTP API"**：

- 评测方 `requests.post(url, files=files)` 上传 .mseed；
- 响应必须是 JSON：`{"台站名": {"P": ["2025-06-07T12:34:56.789000Z", ...], "S": [...]}}`
  （绝对 UTC 到时，微秒 6 位 + `Z` 后缀）；
- 评分：P ≤0.1s 得 1 分、0.1–1s 线性、>1s 不计分；S ≤0.2s 满分 / 0.2–2s 线性；
  数量误差 >5% 每超 1 个扣 0.5 分；单项最低 0 分。输入 100Hz、无位置信息、含纯噪声条目。

对应入口：

```bash
# 起官方标准 API（模型常驻 + 预热 + 多台站合批推理）
# 默认基座/阈值统一取自 src/phasepicker/defaults.py：diting + P=0.2 / S=0.15
#（去年真题 A/B：diting 1.899 > stead 1.496，见 deploy/README.md；勿再用 stead 起服务）
pip install fastapi uvicorn python-multipart requests
python scripts/serve_api.py --port 8000                                  # 保底（diting 零微调）
python scripts/serve_api.py --weights weights/<best>.pt --port 8000      # P2 微调权重（先过 ab_compare）

# 提交前自检：状态码 / JSON 结构 / ISO 到时格式 / 升序，全对才算过
python scripts/check_api.py --url http://<公网IP>:8000/pick --input <mseed目录或zip>
```

去年格式的 `.an` 工具链（run_official_task1/23、本地评分）完整保留，
用于在**去年真题包上离线验证模型分数**——这是唯一可靠的调参依据。
训练与提分排期见 `训练与提分计划.md`。

## 🚀 性能（极限优化，输出与优化前逐字段一致）

| 优化 | 位置 | 效果 |
|---|---|---|
| 嵌套 zip 归档缓存 | `io/official_waveforms.py` | 消除"每样本全量重解压内层 zip"，官方包读取加速数十~上百倍（`scripts/bench_io.py` 实测） |
| 跨文件合批推理 | `inference/picker.py::pick_batch` | 多波形拼一个 Stream 单次 classify，滑窗张量凑满 batch，GPU/CPU 不空转 |
| I/O 流水线 | `tasks/task1_runner.py::run_task1_samples_fast` | 读 mseed（线程池预取）与推理重叠；失败隔离语义不变 |
| 整批预测 | `tasks/baseline_models.py::predict_many` | T2/T3 树模型一次矩阵调用替代逐样本循环 |
| 运行时开关 | `--fp16 / --threads / --batch-size / --file-batch / --workers` | CUDA 半精度、CPU 满核、批大小可调；`--no-fast` 一键回退旧路径对照 |
| API 常驻+预热 | `scripts/serve_api.py` | 启动即加载权重并跑一条合成波形，正式请求零冷启动 |

```bash
# 去年真题端到端（默认启用全部优化；--no-fast 回退旧实现对照）
python scripts/run_official_task1.py --input round2.zip --output T1.an \
    --device auto --workers 8 --file-batch 16 --answer-package round2.zip

# 任何提速开关（--overlap 调小 / --fp16 / --compile / 升级 seisbench / 换权重）
# 启用前必须过 A/B 同分闸门：
python scripts/ab_compare.py --input round2.zip \
    --a "--device cuda" --b "--device cuda --fp16 --overlap 0.2" \
    --answer-package round2.zip
```

再快一档（按收益排序）：升级 `seisbench>=0.11`（annotate 后端已 C 重写，官方称 CPU≥20%、
GPU>50% 提速，setup.sh 已固定）→ `--overlap` 从 0.5 下调（窗口数近乎线性减少）→
`--fp16`（CUDA）→ `--compile`（torch 2.x）。每一步都用 ab_compare 验证同分后再启用。

## 🧠 训练捷径（Colab 一键，90GB 盘吃超大数据集）

基座直接用 `PhaseNet('diting')`（USTC 271 万条中国数据训好的 picker，一行加载）；
微调数据用 `scripts/chunked_fetch.py` 分块下载→抽 3001 点窗→删块循环取 CWA 等大集；
`notebooks/colab_bigdata_finetune.ipynb` 从零到 best.pt 全自动、断点续传。

---

## 一、每次开机三步复现（最重要）

免费 GPU 机器（FunHPC / Cloud Studio）关机会清空一切。开新机器后：

```bash
# 1) 拉回整个仓库（含代码 + 权重备份，总共几 MB，秒级）
cd /data/coding
git clone https://gitee.com/你的用户名/你的仓库名.git
cd 你的仓库名

# 2) 一键装环境 + 恢复权重 + GPU 自检（几分钟）
bash scripts/setup.sh

# 3) 开不断线会话，跑闭环验证
tmux new -s seis
python scripts/closed_loop.py
```

看到 `CLOSED LOOP OK` 就说明环境、GPU、模型、评分全部就位。

> **注意镜像选择**：进机器时务必选 `PyTorch 2.0.1 / Python 3.10 / CUDA 11.8` 的镜像。
> 这是 Tesla P4（老架构 sm_61）唯一稳定可用的组合，新版 torch 会报 `no kernel image`。

---

## 二、仓库里有什么

| 路径 | 作用 |
|---|---|
| `scripts/setup.sh` | 开机一键：设缓存目录、装 seisbench+obspy、从 `weights/` 恢复权重、GPU 自检 |
| `scripts/baseline_synth.py` | 合成三分量波形 → PhaseNet 推理，验证推理链路 |
| `scripts/closed_loop.py` | **读取→推理→评分**完整闭环（自包含，内嵌已测评分逻辑） |
| `scripts/run_local_scoring.py` | 本地评分 CLI，官方评分规则复刻 |
| `scripts/train.py` | 微调训练入口（官方数据到位后用） |
| `weights/phasenet_diting_weights.tar.gz` | **部署基座** diting 预训练权重备份（约 1MB，`deploy_api.sh` 离线恢复，免跨境重下）；另有 stead 包作应急 fallback。全部权重清单与取舍状态见 `weights/README.md` |
| `src/phasepicker/defaults.py` | 全局默认单一真源：基座 diting、阈值 P=0.2 / S=0.15、去重合并窗 |
| `src/phasepicker/` | 工程内核：mseed 读取、预处理、推理封装、去重、评分、训练脚手架、EEW 展示层 |
| `tests/` | 纯逻辑核心单元测试（评分/时间对齐/去重/训练/EEW） |
| `MASTER_PLAN.md` | 融合版方案总纲（策略 + 工程 + 决赛叙事） |
| `ROADMAP.md` | 优先级路线图，三个"胜负手" |

---

## 三、关键约束与设计原则

- **网络**：国内免费机跨境下载数据集极慢且关机不留。因此不下大数据集，
  用合成数据验证链路；预训练权重包（diting/stead，各约 1MB）备份进仓库。
- **持久化**：GPU 机器上一切关机即毁。唯一可信副本 = 本地 + Gitee 仓库。
  训练产物（checkpoint）需实时外送（见 `src/phasepicker/training/checkpoint.py`）。
- **时间对齐**：模型输出采样点下标，换算绝对到时的逻辑集中在
  `src/phasepicker/utils/timing.py`，有专门测试守护（最易"整盘皆输"处）。
- **评分驱动**：调参唯一客观依据是本地评分脚本（复刻官方规则），
  偏向高精确率（数量误差每超 1 个扣 0.5 分，宁漏勿多报）。

---

## 四、去年真题数据画像（2026-08-01 评审实测；压测/超时/调参预算按此估）

两轮真题包已在手，"每条 51.5s"的旧假设不成立：

1. **第1轮 1000 条**：单台站三分量，真值全部 1P+1S；时长 22~141s（150 条抽样
   min=22.3 / med=64.0 / max=140.6s，仅约 41% 短于 diting 一窗 61.02s 需尾部补齐）。
2. **第2轮 915 条**：时长 34.6s~**3600s**；含 3600s 大文件（单条约 70 倍计算量，
   并发压测与超时预算必须覆盖）；其中 2 条密集余震文件（35P+34S / 53P+52S），
   其余 913 条均为 1P+1S。
3. 两轮答案均**无纯噪声条目**——今年规则明示会有（应返回空表），
   噪声条目下的误报率只能用合成噪声波形抽查，去年真题测不出。

真实基线已在手（diting 零微调，满分 2 分/文件）：新默认阈值 P=0.2/S=0.15 下
第1轮 **1.717** / 第2轮 **1.654**（旧阈值 0.3 时为 1.629 / 1.500）。一切提分
（微调、后处理）从这里起步，排期见 `训练与提分计划.md`。
