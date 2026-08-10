#!/usr/bin/env python3
"""按 S−P 时差（震中距代理）筛选与真题同震距范围的数据集。

===== 为什么用 S−P 选数据集 =====
2026-08-06 实测真题答案：短文件（913/915）真值 S−P **中位 21.31s、95% 67.05s、
最大 111.76s**，对应震中距约 170~800km ——**区域震到远震**。
只有那 2 个 3600s 长记录是近震（S−P 中位 5.67/6.53s）。

这解释了之前 CEED 微调为什么失败：加州 CEED 以**近震**为主，S−P 只有几秒，
与真题的区域震尺度差一个量级。模型学到的是"P 后几秒必有 S"的近震先验，
用到真题上就在错误的时间尺度上找 S。之前只说"域不匹配"太笼统，
真正错的是**震中距范围**——这是可量化、可筛选的判据。

本脚本只读每个候选数据集的 metadata.csv（几十 MB，不下波形），算：
  - S−P 时差分布（用 trace_{p,s}_arrival_sample / trace_dt_s 换算）
  - 与真题分布的重叠度（落在真题 5%~95% 区间 [4.15, 67.05]s 内的比例）
  - 台网/通道/采样率构成

只有重叠度高的才值得下波形训练。

===== 用法 =====
    python scripts/scout_sp_distance.py --cache /mnt/vol/sbcache \\
        --datasets geofon,crew,mlaapde,neic
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

# 真题短文件真值 S−P 分位（2026-08-06 实测 913 个文件）
EXAM_SP = {"min": 1.32, "p5": 4.15, "median": 21.31, "p95": 67.05, "max": 111.76}


def sp_stats(csv_path: str, max_rows: int = 200_000) -> dict:
    """从 metadata.csv 算 S−P 时差分布。返回 {} 表示无可用列。"""
    import numpy as np
    import pandas as pd

    # 只读需要的列，避免几百 MB csv 吃内存
    head = pd.read_csv(csv_path, nrows=5)
    cols = set(head.columns)

    def pick(*names):
        for n in names:
            if n in cols:
                return n
        return None

    c_p = pick("trace_p_arrival_sample", "trace_P_arrival_sample", "p_arrival_sample")
    c_s = pick("trace_s_arrival_sample", "trace_S_arrival_sample", "s_arrival_sample")
    c_dt = pick("trace_dt_s")
    c_sr = pick("trace_sampling_rate_hz", "trace_sampling_rate")
    if not (c_p and c_s):
        return {"error": "无 P/S 到时列"}

    use = [c for c in (c_p, c_s, c_dt, c_sr) if c]
    for extra in ("station_network_code", "station_channel", "station_channels",
                  "path_ep_distance_km", "source_distance_km"):
        if extra in cols:
            use.append(extra)
    df = pd.read_csv(csv_path, usecols=use, nrows=max_rows, low_memory=False)

    # 采样率 → 秒
    if c_dt and df[c_dt].notna().any():
        dt = pd.to_numeric(df[c_dt], errors="coerce")
    elif c_sr and df[c_sr].notna().any():
        dt = 1.0 / pd.to_numeric(df[c_sr], errors="coerce")
    else:
        dt = pd.Series([0.01] * len(df))  # 兜底 100Hz

    p = pd.to_numeric(df[c_p], errors="coerce")
    s = pd.to_numeric(df[c_s], errors="coerce")
    sp = (s - p) * dt
    sp = sp[(sp > 0) & (sp < 600)].to_numpy()  # 去掉缺失/异常
    if len(sp) < 50:
        return {"error": f"有效 S−P 样本仅 {len(sp)} 条"}

    q = np.percentile(sp, [5, 25, 50, 75, 95])
    inside = float(((sp >= EXAM_SP["p5"]) & (sp <= EXAM_SP["p95"])).mean())
    out = {
        "n_rows": len(df),
        "n_sp": len(sp),
        "p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4],
        "max": float(sp.max()),
        "overlap": inside,   # 落在真题 [4.15, 67.05]s 内的比例
    }
    # 附带台网/通道
    for k in ("station_network_code", "station_channel", "station_channels"):
        if k in df.columns:
            vc = df[k].astype(str).value_counts().head(4)
            out[k] = dict(vc)
    if "path_ep_distance_km" in df.columns:
        d = pd.to_numeric(df["path_ep_distance_km"], errors="coerce").dropna()
        if len(d):
            out["dist_km_median"] = float(d.median())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="SeisBench 缓存根")
    ap.add_argument("--datasets", default="",
                    help="逗号分隔的数据集目录名；空=扫缓存里全部")
    args = ap.parse_args()

    root = os.path.join(args.cache, "datasets")
    names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if not names:
        names = sorted(d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d)))

    print("真题短文件真值 S−P: 中位 %.2fs | 5%%~95%% [%.2f, %.2f]s | 最大 %.2fs"
          % (EXAM_SP["median"], EXAM_SP["p5"], EXAM_SP["p95"], EXAM_SP["max"]))
    print("=" * 78)
    rows = []
    for name in names:
        d = os.path.join(root, name)
        csvs = sorted(glob.glob(os.path.join(d, "metadata*.csv")))
        if not csvs:
            print(f"{name:<18} 无 metadata csv，跳过")
            continue
        # 取最大的 csv 抽样：小分块（如 CEED 早期年份）只有几十条有效 S−P，
        # 统计不出分布。max_rows 已封顶内存。
        csvs.sort(key=os.path.getsize, reverse=True)
        st = sp_stats(csvs[0])
        if "error" in st:
            print(f"{name:<18} {st['error']}")
            continue
        rows.append((name, st))
        print(f"{name:<18} S−P 中位 {st['median']:>6.2f}s  "
              f"5%~95% [{st['p5']:>5.2f}, {st['p95']:>6.2f}]s  "
              f"与真题重叠 {st['overlap']*100:>5.1f}%  (n={st['n_sp']})")
        for k in ("station_network_code", "station_channel", "station_channels"):
            if k in st:
                print(f"{'':<18}   {k}: {st[k]}")
        if "dist_km_median" in st:
            print(f"{'':<18}   震中距中位 {st['dist_km_median']:.0f} km")

    if rows:
        rows.sort(key=lambda r: -r[1]["overlap"])
        print("\n" + "=" * 78)
        print("按与真题震距重叠度排序（越高越值得训）:")
        for name, st in rows:
            print(f"  {st['overlap']*100:>5.1f}%  {name:<18} "
                  f"(中位 {st['median']:.1f}s vs 真题 21.3s)")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
