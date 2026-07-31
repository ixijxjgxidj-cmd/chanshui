"""全局默认参数——单一真源（2026-08-01 全项目评审后设立）.

serve_api / run_official_task1 / ab_compare / finetune 验收评分统一从这里取默认，
不得在各脚本里各写各的。数值依据：去年两轮真题全量阈值扫描
（outputs/probe/*.pkl，独立复核通过），详见 outputs/评审报告_20260801.md。
"""

# 基座权重：去年真题 A/B 实测 diting(1.899) > stead+Iquique微调(1.678) > stead(1.496)
DEFAULT_PRETRAINED = "diting"

# 拾取阈值：两轮全量扫描最优（第1轮均分 1.629→1.717，第2轮 1.500→1.654）
DEFAULT_P_THRESHOLD = 0.2
DEFAULT_S_THRESHOLD = 0.15

# 去重合并窗（秒）：P3 网格结论（outputs/p3_grid/结论.md）。远小于去年真题
# 同相位最小真值间隔（P-P 10.49s / S-S 11.43s），只并分裂双检测、不会误并
# 两个真实震相；12 个(读法×取法×轮次)刻度全部改善后才采纳
DEFAULT_P_MERGE_WINDOW_S = 1.0
DEFAULT_S_MERGE_WINDOW_S = 3.0
