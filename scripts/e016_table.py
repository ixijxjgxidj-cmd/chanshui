#!/usr/bin/env python3
"""把 e016_eval.py 的多个 JSON 汇总成一张对照表。

为什么单独一个脚本：远端 heredoc/引号嵌套在 PowerShell -> ssh -> bash -> python
四层转义下极易出错（本轮踩过一次 f-string unmatched '['），落成文件最稳。

用法: python e016_table.py runs/e016_eval_*.json
"""
from __future__ import annotations

import json
import sys

BIN_ORDER = ["[0,10)", "[10,20)", "[20,30)", "[30,40)", "[40,50.1)"]


def main(paths: list[str]) -> int:
    docs = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                docs.append(json.load(f))
        except FileNotFoundError:
            print("MISSING " + p)
    if not docs:
        return 1
    w = max(len(d.get("label") or "?") for d in docs) + 1
    hdr = "label".ljust(w) + "  pool                     n     score   gross"
    print(hdr)
    print("-" * len(hdr))
    for d in docs:
        o = d["overall"]
        pool = d.get("pool", "?").split("/")[-1][:24]
        print((d.get("label") or "?").ljust(w)
              + "  " + pool.ljust(24)
              + " %5d  %.4f  %.4f" % (o["n"], o["mean_score"], o["gross_rate"]))
    for metric in ("mean_score", "gross_rate"):
        print()
        print("== per-bin " + metric + " ==")
        print("label".ljust(w) + "".join(b.rjust(12) for b in BIN_ORDER))
        for d in docs:
            cells = []
            for b in BIN_ORDER:
                v = d["per_bin"].get(b)
                cells.append(("-" if v is None else "%.4f" % v[metric]).rjust(12))
            print((d.get("label") or "?").ljust(w) + "".join(cells))
    print()
    print("== per-bin n ==")
    for d in docs[:1]:
        print("".join(b.rjust(12) for b in BIN_ORDER))
        print("".join(str(d["per_bin"].get(b, {}).get("n", "-")).rjust(12)
                      for b in BIN_ORDER))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))