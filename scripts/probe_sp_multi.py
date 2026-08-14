#!/usr/bin/env python3
"""只下载 metadata.csv 的前若干 MB，统计各 SeisBench 数据源的 S-P 分布。

为什么只取前几 MB：metadata.csv 最大的 scedc 有 2.1 GB，txed 125 MB。我们只
需要知道"这个源有多少可用的远场双标注道"，抽样前 N 行足够定性。用 Range 请求
截断，比整块下载快两个数量级。

注意 CSV 被截断后最后一行可能不完整，pandas 会报错，所以按最后一个换行符裁掉。

用法: python probe_sp_multi.py --sources pnw,lendb,txed --mb 12
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request

import numpy as np
import pandas as pd

REMOTE = "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets"
# 池窗 3001 点 @50Hz = 60.02s，首尾各 250 点盲区 => P/S 同窗要求 S-P <= 50.02s
SP_MAX = 50.02
SP_MIN = 10.0


def head_csv(url: str, mb: int) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-%d" % (mb * 1024 * 1024)})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
    cut = raw.rfind(b"\n")
    if cut > 0:
        raw = raw[:cut]
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def pick(df: pd.DataFrame, names: list[str]):
    for n in names:
        if n in df.columns:
            return n
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--mb", type=int, default=12)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    res = {}
    for src in args.sources.split(","):
        src = src.strip()
        if not src:
            continue
        try:
            df = head_csv(REMOTE + "/" + src + "/metadata.csv", args.mb)
        except Exception as exc:
            print("%-12s ERROR %s" % (src, exc))
            res[src] = {"error": str(exc)}
            continue
        pcol = pick(df, ["trace_P_arrival_sample", "trace_p_arrival_sample",
                         "trace_Pg_arrival_sample", "trace_P1_arrival_sample"])
        scol = pick(df, ["trace_S_arrival_sample", "trace_s_arrival_sample",
                         "trace_Sg_arrival_sample", "trace_S1_arrival_sample"])
        srcol = pick(df, ["trace_sampling_rate_hz", "trace_dt_s"])
        if not pcol or not scol:
            print("%-12s 无 P/S 列; 列样例=%s" % (src, list(df.columns)[:12]))
            res[src] = {"rows": len(df), "cols": list(df.columns)[:40]}
            continue
        sr = 100.0
        if srcol == "trace_sampling_rate_hz":
            sr = float(pd.to_numeric(df[srcol], errors="coerce").median())
        elif srcol == "trace_dt_s":
            sr = 1.0 / float(pd.to_numeric(df[srcol], errors="coerce").median())
        p = pd.to_numeric(df[pcol], errors="coerce")
        s = pd.to_numeric(df[scol], errors="coerce")
        both = p.notna() & s.notna()
        sp = ((s - p) / sr)[both]
        sp = sp[sp > 0]
        usable = sp[(sp >= SP_MIN) & (sp <= SP_MAX)]
        d = dict(rows=len(df), sr=sr, both=int(both.sum()),
                 sp_median=float(sp.median()) if len(sp) else None,
                 far_10_50=int(len(usable)),
                 far_frac=float(len(usable) / max(len(sp), 1)),
                 over_50=int((sp > SP_MAX).sum()),
                 net_col=pick(df, ["station_network_code", "trace_network_code"]))
        nc = d["net_col"]
        if nc:
            d["nets"] = df[nc].astype(str).value_counts().head(6).to_dict()
        res[src] = d
        print("%-12s rows=%6d sr=%5.1f both=%6d sp_med=%7s far[10,50]=%6d (%.1f%%) over50=%5d"
              % (src, d["rows"], d["sr"], d["both"],
                 "%.1f" % d["sp_median"] if d["sp_median"] else "NA",
                 d["far_10_50"], 100 * d["far_frac"], d["over_50"]))
        if nc:
            print("             nets=%s" % d["nets"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("wrote " + args.out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())