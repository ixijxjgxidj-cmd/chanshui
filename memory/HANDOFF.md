# 交接摘要 · 第 9 轮（2026-08-13）

> 本文件为会话中断时的事实交接。所有数字均为本轮实测，脚本与产物路径已给出。

## 0. 用户本轮新增的关键信息（改变战略）

**本届由广西地震局举办；去年为四川地震局；去年全部测试数据仅供参考。**

推论（必须写入后续所有决策）：

- 第 1/2 轮（R1/R2）与 08 决赛包都是**四川域**数据，对新赛季属**域外**。
- 继续在 R1/R2 上追分存在**负迁移风险**。R1/R2 的 8 单元门槛应从「优化目标」
  降级为「不回归的合理性检查」，不能再作为唯一取舍依据。
- 生产集成的成员选择（尤其第 3 成员 shandong）是**用四川数据拟合出来的**，
  其正当性随赛区变更而失效，需要重新论证。

## 1. 本轮已完成

- **GitHub 推送成功**：`git push -u origin main` 到新仓库
  `https://github.com/ixijxjgxidj-cmd/chanshui.git` 已完成（`origin/main` 已存在）。
  网络曾短暂不可达，恢复后用
  `git -c http.postBuffer=524288000 -c http.version=HTTP/1.1 push -u origin main` 成功。
- **本地工具限制已确诊**：`apply_patch` 与 `rg` 均因 WindowsApps ACL 被拒
  （`Access is denied.`）。可用写文件方式 = PowerShell 单引号 here-string +
  `[System.IO.File]::WriteAllText/AppendAllText`，**分块写**（单次过大或命令过复杂会被 policy 拒绝）。
  搜索用 `Select-String` 替代 `rg`。
- **新增脚本**（已落盘，语法自检通过）：
  - `scripts/lit_search.py`：四通道检索（arxiv/crossref/openalex/s2）+ `--verify` 原文核验
    + `verify_doi_metadata()` Crossref 书目级核验。
  - `scripts/triage_round09.py`：主题打分与去重，产出候选精读清单。
  - `scripts/verify_round09.py`：对 25 篇策展清单逐条 HTTP 核验并留证。
- **检索产物**：`memory/papers/_raw/round09_q_*.json` 共 9 个查询轴，
  去重后 **900 条**，候选 **134 条**；`round09_scored.json` 为打分全量；
  `round09_verified.json` 为策展清单核验结果（**arXiv 全部 OK，出版商 DOI 页多被
  Cloudflare/JS 墙挡住，ratio=0 属抓取失败而非论文不存在**，需用 Crossref 元数据补证）。
  实际核验通过 14/25。
- **anysearch / Playwright MCP 在本会话不可用**（`list_mcp_resources` 返回空）。
  已改用可核验 HTTP 通道（arXiv API / Crossref / OpenAlex / Semantic Scholar +
  直接抓 abs 页），**必须在论文记录中如实标注通道，不得声称用了未调用的工具**。

## 2. 本轮最重要的三个实测结论

### 2.1 软标签 sigma 的真实口径（原假设需修正）

- SeisBench `ProbabilisticLabeller` 默认 `sigma=10`，单位是**采样点**。
  USTC pickers 原生 50Hz → 10 采样点 = **0.2s**。
- PhaseNet 参考实现：`label_width=30`，且 `sigma = label_width/5 = 6` 采样点
  → 100Hz 下 **0.06s**（比 SeisBench 默认窄约 3.3 倍）。
- 本仓 `scripts/finetune_phasenet.py::make_soft_label` 默认
  `sigma_p_s=0.2, sigma_s_s=0.3`（按秒定义）。50Hz 下 = 10 / 15 采样点。
- **结论修正**：P 的 0.2s **与 SeisBench 默认一致，不是异常**；
  真正异常的是 **S 用 0.3s，比 P 宽 1.5 倍**——SeisBench 与 PhaseNet 都**不按相位区分 sigma**。
  且 0.3s > S 满分容差 0.2s。
- `--sigma-s` 已是 CLI 参数（`nargs=2`，默认 `(0.2, 0.3)`），**收紧 sigma 无需改代码，属配置扫描**。

### 2.2 带符号残差诊断：**不存在系统性偏差**（新证据，此前从未做过）

脚本 `outputs/port_verify/_signed_residual.py` → `_signed_residual_r1r2.json`
（只读 R1/R2 冻结 g7 预测，脚本内含封存 08 路径硬拒绝）。

| 包 | 相位 | n | mean | median | frac_late | 满分率 | 最优常数平移 | 平移后满分率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | P | 936 | -0.0105 | -0.01 | 0.361 | 0.7724 | -0.01 | 0.7895 |
| r1 | S | 944 | +0.0176 | -0.01 | 0.442 | 0.7214 | +0.02 | 0.7288 |
| r2 | P | 918 | +0.0055 | -0.01 | 0.394 | 0.7560 | -0.01 | 0.7582 |
| r2 | S | 913 | +0.0200 | -0.02 | 0.436 | 0.6320 | -0.02 | 0.6484 |

**结论：常数平移标定最多只加 1.7 个百分点，这条路等于没有。误差是方差/抖动，不是偏置。**
这否证了「软标签不对称 → 系统偏晚」这一机制，同时**强化了「压方差 / 锐化峰」才是正确方向**。

### 2.3 收益天花板对比：**漏检路线的天花板大于残差路线**（修正历史结论）

脚本 `outputs/port_verify/_shrink_ceiling.py` → `_shrink_ceiling_r1r2.json`
（把已匹配残差按系数 k 收缩，模拟更锐的峰）。

| k | R1 Δ均分 | R2 Δ均分 |
|---:|---:|---:|
| 0.8 | +0.0282 | +0.0416 |
| 0.6 | +0.0540 | +0.0803 |
| 0.5 | +0.0659 | +0.0983 |
| 0.4 | +0.0764 | +0.1151 |
| 0.0（完美到时）| **+0.0937** | **+0.1461** |

未匹配真值（漏检）天花板：R1 P 64 + S 56 = 120 → **+0.120 均分**；
R2 P 83 + S 86 = 169 → **+0.185 均分**。

**结论：即使把所有已匹配拾取的到时做到完美，R1 只能 +0.094、R2 只能 +0.146；
而把所有漏检补上分别是 +0.120 / +0.185。历史交接摘要「主战场=时间残差」的说法不准确，
两条路量级相当且漏检略大。后续必须两条一起打，且不要再把残差当成唯一主战场。**

## 3. 必须由用户裁决的合规冲突（**尚未获得答复**）

`outputs/train/r2train.h5`（33,902,952 字节）实测：**1001 个窗，来自 915 个不同
`source_file`，恰好等于 R2 的 T1 文件数**；键名形如 `EXAM_T1_A_Q0001_mseed_ev0`。
`deploy/WEEK_TRAINING_RUNBOOK.md` 明确它是「R2 真题锚点池」，且被同时用作
`--holdout` 与混池训练输入。

生产第 4/5 成员 `weights/aug/exam_aug6_r2train_sd.pt`、
`weights/aug/crew_sp23_r2train_sd.pt`（各 1,114,514 字节）由此产生。

**这直接违反用户硬约束「测试数据集永远不参与模型微调与训练」**，
且它们还是**四川域**拟合物，在广西赛区双重失效。

**需要用户决定**：是否必须用纯公开数据重建这两个成员（会改变冻结分数），
或明确豁免既有冻结资产。**在得到答复前不要把它们写进任何新的发布清单。**

## 4. 失分归因（本轮实测，供后续复用）

R1（466 个有失分文件，总失分 230.044）：
`p_partial 16.7% / s_partial 23.8% / p_offtol_full 23.0% / s_offtol_full 23.0%
/ p_miss 6.3% / s_miss 2.8% / pen_miss 4.5% / pen_extra 0%`
→ 到时精度合计 **86.5%**；满分率 P **48.2%**、S **36.4%**；
已匹配残差 P 中位 0.10 / p90 0.39，S 中位 0.28 / p90 0.95。
`offtol` 96 个文件中 89 个是 1P/1S 对 1P/1S——**是精度失败，不是检测失败**。

R2（915 文件，总失分 375.172）：
- 短记录 ≤300s：913 文件，失分 287.867（76.7%）；
  `miss_p 66.0 + miss_s 79.0 = 145`、`tol_p 52.67 + tol_s 82.24 = 134.91`、`pen 7.96`。
- 长记录 >300s：仅 2 文件，失分 87.306（**23.3%**），其中**数量罚 63.0** 为主，
  `T1.A.Q0001` 单文件 `extra_p=43, extra_s=46, pen_eff=39.5`。**长记录是过检问题。**

## 5. 其它已核验事实

- USTC 省级权重与 guangxi 的平均相对 L2（111 个张量）：
  huanan **0.0167**、guizhou 0.0206、jiangxi 0.0247、hunan 0.0250、hainan 0.0261、
  sichuan 0.0769、guangdong 0.0830、yunnan 0.0852、**shandong 0.1750（最远）**。
  生产成员 3 是 shandong，其入选理由来自四川数据拟合 → 赛区变更后需重新论证；
  **huanan（华南，覆盖广西/广东/福建/海南）目前不在集成内，是明显的域先验缺口。**
- `r2train.h5` 的 S-P：n=838，中位 **16.36s**，p5 2.92 / p25 12.02 / p75 26.28 / p95 45.92 / max 52.3。
- 官方规则 PPT（已从 pptx 提取全文）：数据为「测震台网（速度计）观测数据」MSEED；
  **「一条数据震相个数不定，可能有多个 P 波、S 波震相（初动），也可能没有」**；
  P/S 分别计算误差；API 公网、初赛/复赛/决赛可变更；复赛自动算分；决赛取复赛前二十。
  主讲：姚志祥，中国地震局地球物理研究所 / 地震科学国际数据中心。**PPT 未写明数据所属区域。**
- **`cap_max_p=1 cap_max_s=1`（≤300s）是一个需要重新评估的风险**：
  依据是 R1+R2+08 短文件 2692/2692 恰好 1P+1S，实测增益却很小
  （r1 +0.015 / r2 +0.026 / 08 +0.026）；而官方规则明确允许多震相与零震相。
  **小收益、大尾部风险**，赛区变更后风险上升。
- 服务器：`/data` 50G，**可用 13G**（邻居 ETHZ 占 22G，仍可能增长）；
  根文件系统 30G 可用 9.4G；`/data/dizheng-sol` 187M，workspace/env.sh/venv/repo 完好；
  Tesla P4 8GB，利用率 0%。新数据集必须小体积/分片/流式，下载前先 `df -h /data`。
- 本地测试基线：`py -3.13 -m pytest tests -q` = **377 collected / 376 passed / 1 skipped**。
  注意 `python` 指向 LibreOffice 内置 Python，无 pytest，必须用 `py -3.13`。

## 6. 下一步（按顺序）

1. **向用户提出第 3 节的合规裁决**，同时报告第 2 节三个结论与赛区变更的战略含义。
2. 完成 `memory/papers/round-09-*.md`：≥15 篇（≥10 篇原始实验论文），
   用 `round09_verified.json` + Crossref 元数据补证，如实标注检索/核验通道。
   已核验通过且高度相关的关键篇目：
   `1803.03211`(PhaseNet)、`2606.15377`(Learning ... from Labels with Inaccuracies)、
   `rs-10439246/v1`(Label Imbalance)、`2410.15765`(SeisLM)、`2306.04753`(OBSTransformer)、
   `2302.08747`(DAS semi-supervised)、`10.3389/feart.2022.1032839`(CSESnet)、
   `10.3389/feart.2023.1306488`(local data customization)、
   `1910.06278`(DarkPose)、`1911.07524`(UDP)、`2012.15175`(Rethinking Heatmap Regression)、
   `2102.00650`(Soft labels bias-variance)、`1909.11723`(KD via LSR)。
3. 写 `memory/experiments/014-*.md` **预注册**，再训练。候选机制（按证据强弱排序）：
   - ① **S sigma 收紧到 0.2s（与 P 对齐、与评分容差对齐）** ——
     由 2.1 的口径证据 + 2.2 的「误差是方差不是偏置」共同支持；`--sigma-s 0.2 0.2`。
   - ② **把 huanan 纳入集成、重新论证 shandong** —— 纯域先验，不依赖四川数据拟合。
   - ③ 用纯公开数据 + `--split-mode event-hash` 重建第 4/5 成员（待用户裁决后执行）。
   - ④ 漏检路线（天花板更大）：阈值/召回侧机制，注意不能用 R1/R2 调阈值。
4. 训练一律在 `/data/dizheng-sol` 内，先 `. /data/dizheng-sol/env.sh`；产物写 `runs/`；
   不提交大 checkpoint、缓存、预测、私有审计 JSON。
5. **评估口径新规**：R1/R2 只作「不回归检查」并同时报四口径；
   真正的取舍依据应来自**公开数据的多区域留出集**（防泄漏按事件切分），
   以「跨区域最坏表现不退化」为准，而不是 R1/R2 均分最大化。

## 7. 禁止重复（沿用 `memory/failed-experiments.md`，本轮新增）

- **常数时间平移标定**（P 或 S）：已实测最多 +1.7pp，判定无效，不要再试。
- 已否证清单其余条目照旧：加速度积分、兜底 SNR 闸、极性 TTA、overlap 0.75/0.9、
  亚采样抛物线、SeisT-L 零样本、拾取级投票、model soup、T2 quantile/QuantReg/Huber/双模型、
  T3 中心化等类头、T1 长记录 event confidence、T1 最终列表/annotation gap mask、
  候选四口径重排、T1 三折 LOPO 基础蒸馏、包—记录层级等权 KD+hard。