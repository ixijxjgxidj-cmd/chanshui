#!/usr/bin/env python3
"""把 lit_search.py 的产物按与本轮问题的相关性排序，只留核验通过的记录。

排序依据：标题/摘要里命中的关键词权重之和。权重是手工设的——本轮的问题是
"训练池分布构成如何影响跨区域远场泛化"，所以 domain adaptation / distance /
generalization 这类词权重最高，纯方法词（transformer、attention）权重低。

用法: python lit_rank.py lit/r11_c.json --top 40 --out lit/r11_c_sorted.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys

WEIGHTS = {
    r"domain (adaptation|generaliz|shift|transfer)": 6.0,
    r"cross[- ](region|domain|dataset)": 6.0,
    r"out[- ]of[- ]distribution": 5.0,
    r"generaliz": 3.0,
    r"(epicentral|hypocentral|source[- ])distance": 5.0,
    r"teleseismic|regional distance|far[- ]field": 5.0,
    r"transfer learning|fine[- ]tun": 4.0,
    r"training (data|set|dataset) (composition|distribution|selection|mixing)": 6.0,
    r"curriculum|sample (weight|mixing|selection)|data (augmentation|mixing)": 4.0,
    r"knowledge distillation|distill": 4.0,
    r"self[- ]supervised|pretrain|masked (modeling|autoencoder)|JEPA": 4.0,
    r"ensemble": 2.5,
    r"phase pick": 3.0,
    r"(P|S)[- ]wave arrival|arrival[- ]time": 2.0,
    r"benchmark|seisbench": 2.5,
    r"uncertainty|calibrat": 2.0,
    r"false (positive|detection)|missed detection|recall": 2.0,
    r"deep learning|neural network": 0.5,
}
COMPILED = [(re.compile(k, re.I), v) for k, v in WEIGHTS.items()]


# 领域闸门：必须命中至少一个地震学词，否则丢弃。
# 为什么必需：纯关键词加权会把 "cross-dataset generalization" 的遥感/EEG/医学影像
# 论文排到最前（本轮实测前 20 里 14 篇与地震无关），它们对本问题零参考价值。
SEISMO = re.compile(
    r"seismi|seismo|earthquake|phase pick|micro[- ]?seismic|"
    r"tremor|aftershock|hypocent|epicent|teleseism|magnitude estimat|"
    r"waveform|station network|arrival[- ]time|P[- ]wave|S[- ]wave|"
    r"borehole|subduction|volcan|labquake|megathrust", re.I)


def in_domain(rec: dict) -> bool:
    text = " ".join(str(rec.get(k) or "") for k in ("title", "abstract", "venue"))
    return bool(SEISMO.search(text))


def score(rec: dict) -> tuple[float, list[str]]:
    text = " ".join(str(rec.get(k) or "") for k in ("title", "abstract", "venue"))
    s = 0.0
    hits = []
    for rx, w in COMPILED:
        if rx.search(text):
            s += w
            hits.append(rx.pattern[:28])
    y = rec.get("year")
    try:
        y = int(y)
        if y >= 2022:
            s += 1.5
        elif y >= 2020:
            s += 0.8
    except (TypeError, ValueError):
        pass
    return s, hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out", default="")
    ap.add_argument("--only-verified", action="store_true")
    ap.add_argument("--no-domain-gate", action="store_true",
                    help="关闭地震学领域闸门（默认开启，只留地震相关文献）")
    args = ap.parse_args()

    recs = []
    seen = set()
    for p in args.inputs:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        for r in d.get("records", []):
            key = (r.get("doi") or "").lower() or (r.get("title") or "").lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            if args.only_verified and not r.get("verified_ok"):
                continue
            if not args.no_domain_gate and not in_domain(r):
                continue
            recs.append(r)
    for r in recs:
        r["_score"], r["_hits"] = score(r)
    recs.sort(key=lambda r: -r["_score"])
    top = recs[: args.top]
    for i, r in enumerate(top, 1):
        print("%2d %5.1f  %-4s %-9s %s" % (
            i, r["_score"], str(r.get("year") or "?")[:4],
            (r.get("channel") or "")[:9],
            (r.get("title") or "")[:96]))
        ident = r.get("arxiv_id") or r.get("doi") or r.get("url") or ""
        if ident:
            print("            %s" % str(ident)[:96])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"n_in": len(recs), "records": top}, f,
                      ensure_ascii=False, indent=1)
        print("wrote %s (%d of %d)" % (args.out, len(top), len(recs)))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())