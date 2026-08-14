#!/usr/bin/env python3
"""从一个池里按 S-P 分层抽样出固定大小的子集（真实拷贝，不用 ExternalLink）。

为什么要真实拷贝：裁判集要跨机器传输，ExternalLink 依赖源文件在原位，
传过去会变成断链。这里直接复制数据集内容。

为什么分层：裁判集只有几百窗，如果随机抽，深远场箱（S-P>40s）可能只剩十几条，
统计不稳。按分箱等额抽样让每个箱都有足够样本。

用法: python subset_pool.py --src a.hdf5 --out b.hdf5 --per-bin 120
"""
from __future__ import annotations

import argparse
import random
import sys

import h5py
import numpy as np

BINS_FAR = [(10.0, 15.0), (15.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 50.1)]
# 近场护栏集用细分箱：护栏要检查的是 S-P<10s 区间，用远场分箱会全落进一个箱
BINS_NEAR = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0)]
BINS = BINS_FAR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bin", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--near", action="store_true",
                    help="用近场分箱 [0,2,4,6,8,10)s，给近场护栏集用")
    args = ap.parse_args()

    global BINS
    BINS = BINS_NEAR if args.near else BINS_FAR
    rng = random.Random(args.seed)
    with h5py.File(args.src, "r") as f:
        g = f["data"]
        keys = list(g.keys())
        # 按 sp_s attr 分箱；若缺失则由 p/s 下标现算
        buckets: dict[tuple, list] = {b: [] for b in BINS}
        for k in keys:
            d = g[k]
            sp = d.attrs.get("sp_s")
            if sp is None:
                sr = float(d.attrs.get("sampling_rate", 50.0))
                sp = (float(d.attrs["s_sample_100hz"])
                      - float(d.attrs["p_sample_100hz"])) / sr
            sp = float(sp)
            for b in BINS:
                if b[0] <= sp < b[1]:
                    buckets[b].append(k)
                    break
        # 事件隔离：同一 source_event 只取一条，防同事件多台站相关样本膨胀
        chosen: list[str] = []
        for b in BINS:
            ks = buckets[b]
            rng.shuffle(ks)
            seen_ev = set()
            take = []
            for k in ks:
                ev = str(g[k].attrs.get("source_event", k))
                if ev in seen_ev:
                    continue
                seen_ev.add(ev)
                take.append(k)
                if len(take) >= args.per_bin:
                    break
            print("  bin [%g,%g): 候选 %d -> 取 %d（唯一事件）"
                  % (b[0], b[1], len(ks), len(take)))
            chosen += take
        with h5py.File(args.out, "w") as o:
            og = o.create_group("data")
            for k in chosen:
                src = g[k]
                d = og.create_dataset(k, data=src[:], compression="gzip",
                                      compression_opts=4)
                for a, v in src.attrs.items():
                    d.attrs[a] = v
        sp_all = []
        for k in chosen:
            d = g[k]
            sr = float(d.attrs.get("sampling_rate", 50.0))
            sp_all.append((float(d.attrs["s_sample_100hz"])
                           - float(d.attrs["p_sample_100hz"])) / sr)
    a = np.array(sp_all)
    print("[subset] %d 窗 -> %s" % (len(chosen), args.out))
    print("[subset] S-P median=%.1f p10=%.1f p90=%.1f" %
          (np.median(a), np.percentile(a, 10), np.percentile(a, 90)))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())