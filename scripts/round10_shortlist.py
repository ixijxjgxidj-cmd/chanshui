"""Round-10 shortlist + real verification.

Reads memory/papers/_raw/round10_q_*.json, scores every record against the
round-10 themes, dedupes across themes, then calls lit_search.verify_record on
the shortlist so that every cited paper carries a real HTTP fetch as evidence.

Touches no competition data. Read-only network + JSON output.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lit_search  # noqa: E402

THEMES = {
    "T2_MAG": (
        ["magnitude"],
        ["single station", "single-station", "deep learning", "neural", "regression",
         "estimation", "amplitude", "early warning", "transformer", "cnn"],
    ),
    "T3_CLS": (
        ["discrimination", "classification", "explosion", "blast", "source type",
         "event type"],
        ["earthquake", "seismic", "deep learning", "neural", "cnn", "quarry",
         "induced", "noise"],
    ),
    "FP_LONG": (
        ["false", "association", "detection", "continuous"],
        ["phase", "picker", "picking", "seismic", "associator", "swarm",
         "aftershock", "precision", "recall"],
    ),
    "FARFIELD": (
        ["distance", "teleseismic", "regional", "imbalance", "long tail",
         "long-tail", "resampling"],
        ["phase", "picking", "picker", "seismic", "generalization", "stratified",
         "regression", "deep"],
    ),
    "DATA_XFER": (
        ["transfer", "fine tuning", "fine-tuning", "benchmark", "dataset",
         "domain adaptation", "seisbench"],
        ["seismic", "phase", "picking", "region", "cross", "generaliz",
         "pretrain", "pre-train"],
    ),
}

# Records whose title matches none of these are off-topic noise for this round.
DOMAIN_GATE = [
    "seismic", "seismo", "earthquake", "phase pick", "phasenet", "microseismic",
    "aftershock", "tremor", "hypocenter", "epicent", "magnitude", "waveform",
    "explosion", "blast",
]


def score(rec_dict, theme):
    must, bonus = THEMES[theme]
    hay = ((rec_dict.get("title") or "") + " " + (rec_dict.get("abstract") or "")).lower()
    title = (rec_dict.get("title") or "").lower()
    if not any(g in hay for g in DOMAIN_GATE):
        return 0.0
    m = sum(1 for k in must if k in hay)
    if m == 0:
        return 0.0
    b = sum(1 for k in bonus if k in hay)
    s = 3.0 * m + 1.0 * b
    # prefer on-topic titles over incidental abstract mentions
    if any(k in title for k in must):
        s += 3.0
    yr = rec_dict.get("year") or 0
    if yr >= 2020:
        s += 1.5
    if yr >= 2023:
        s += 1.0
    cit = rec_dict.get("citations") or 0
    if cit >= 50:
        s += 1.0
    if cit >= 300:
        s += 1.0
    if rec_dict.get("arxiv_id"):
        s += 0.5  # verifiable via abs page
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description="round-10 shortlist + verify")
    ap.add_argument("--per-theme", type=int, default=5)
    ap.add_argument("--raw-glob", default="memory/papers/_raw/round10_q_*.json")
    ap.add_argument("--scored-out", default="memory/papers/_raw/round10_scored.json")
    ap.add_argument("--verified-out", default="memory/papers/_raw/round10_verified.json")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    raw = []
    for path in sorted(glob.glob(args.raw_glob)):
        blob = json.load(open(path, encoding="utf-8"))
        for r in blob["records"]:
            r = dict(r)
            r["_src"] = os.path.basename(path)
            raw.append(r)
    print("loaded %d records" % len(raw))

    # dedupe by doi / arxiv / title
    uniq = {}
    for r in raw:
        key = (r.get("doi") or "").lower() or ("arxiv:" + r["arxiv_id"] if r.get("arxiv_id") else (r.get("title") or "").lower()[:120])
        if not key:
            continue
        if key not in uniq or len(r.get("abstract") or "") > len(uniq[key].get("abstract") or ""):
            uniq[key] = r
    recs = list(uniq.values())
    print("unique %d" % len(recs))

    scored = []
    for r in recs:
        best_theme, best = None, 0.0
        per = {}
        for th in THEMES:
            s = score(r, th)
            per[th] = s
            if s > best:
                best, best_theme = s, th
        if best_theme:
            r2 = dict(r)
            r2["_theme"] = best_theme
            r2["_score"] = round(best, 2)
            r2["_scores"] = {k: round(v, 2) for k, v in per.items() if v > 0}
            scored.append(r2)
    scored.sort(key=lambda x: -x["_score"])
    json.dump({"n": len(scored), "records": scored}, open(args.scored_out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("scored %d -> %s" % (len(scored), args.scored_out))

    picks, per_theme_count = [], {k: 0 for k in THEMES}
    for r in scored:
        th = r["_theme"]
        if per_theme_count[th] >= args.per_theme:
            continue
        per_theme_count[th] += 1
        picks.append(r)
    print("shortlist %d  breakdown=%s" % (len(picks), per_theme_count))
    for p in picks:
        print("  [%s %.1f] %s (%s)" % (p["_theme"], p["_score"], (p.get("title") or "")[:96], p.get("year")))

    if args.no_verify:
        return 0

    out = []
    for i, p in enumerate(picks, 1):
        rec = lit_search.Record(
            channel=p.get("channel", ""), title=p.get("title", ""),
            authors=p.get("authors") or [], year=p.get("year"),
            venue=p.get("venue", ""), doi=p.get("doi", ""),
            arxiv_id=p.get("arxiv_id", ""), url=p.get("url", ""),
            abstract=p.get("abstract", ""), citations=p.get("citations"),
        )
        v = lit_search.verify_record(rec)
        d = asdict(rec)
        d["_theme"] = p["_theme"]
        d["_score"] = p["_score"]
        d["verification"] = v
        out.append(d)
        print("  verify %2d/%d ok=%s ratio=%s status=%s | %s" % (
            i, len(picks), v.get("ok"), v.get("title_token_match_ratio"),
            v.get("http_status"), (rec.title or "")[:70]))
    ok = sum(1 for d in out if d["verification"].get("ok"))
    json.dump({"n": len(out), "n_ok": ok, "records": out},
              open(args.verified_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("verified_ok=%d/%d -> %s" % (ok, len(out), args.verified_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())