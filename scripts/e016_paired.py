#!/usr/bin/env python3
"""对同一留出集上的两组评估结果做配对显著性检验。

为什么必须配对：e016 各臂评的是**同一批窗**，窗间难度差异远大于臂间差异
（单窗分数 0~2 全域分布，std ~0.6）。非配对 t 检验会被窗难度方差淹没，
把真实效应判成不显著。配对后只看每窗的差值。

同时给出 bootstrap 置信区间——n 小（GEOFON dev 只有 187 窗）时 t 分布假设
不稳，bootstrap 更保守。

用法: python e016_paired.py A.json B.json [--metric total]
"""
from __future__ import annotations

import argparse
import json
import math
import random


def load(p: str):
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    ra, rb = A["rows"], B["rows"]
    if len(ra) != len(rb):
        print("行数不一致 %d vs %d，无法配对" % (len(ra), len(rb)))
        return 1
    # 行序由 load_hdf5_dataset 决定，同一 pool 同一版本脚本 => 顺序一致。
    # 用 sp_s 作一致性校验，避免静默错配。
    bad = sum(1 for x, y in zip(ra, rb) if abs(x["sp_s"] - y["sp_s"]) > 1e-6)
    if bad:
        print("警告: %d 行 sp_s 不匹配，配对可能错位" % bad)

    rng = random.Random(args.seed)
    n = len(ra)

    def report(name, va, vb, lower_is_better=False):
        d = [y - x for x, y in zip(va, vb)]
        mean = sum(d) / n
        var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
        se = math.sqrt(var / n) if n > 1 else 0.0
        t = mean / se if se > 0 else float("nan")
        boots = []
        for _ in range(args.boot):
            s = sum(d[rng.randrange(n)] for _ in range(n)) / n
            boots.append(s)
        boots.sort()
        lo = boots[int(0.025 * args.boot)]
        hi = boots[int(0.975 * args.boot)]
        sign = "退化" if ((mean > 0) == lower_is_better) else "改善"
        crosses = (lo <= 0.0 <= hi)
        verdict = "不显著(CI跨0)" if crosses else "显著"
        print("  %-12s A=%.4f B=%.4f  delta=%+.4f  95%%CI[%+.4f,%+.4f]  t=%+.2f  %s %s"
              % (name, sum(va) / n, sum(vb) / n, mean, lo, hi, t, sign, verdict))

    print("配对检验 n=%d" % n)
    print("A = %s" % (A.get("label") or args.a))
    print("B = %s" % (B.get("label") or args.b))
    report("mean_score", [r["total"] for r in ra], [r["total"] for r in rb])
    report("gross_rate",
           [1.0 if (r["gross_p"] or r["gross_s"]) else 0.0 for r in ra],
           [1.0 if (r["gross_p"] or r["gross_s"]) else 0.0 for r in rb],
           lower_is_better=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())