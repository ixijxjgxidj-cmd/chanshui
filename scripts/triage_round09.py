"""对 round09 检索结果做主题打分与去重，输出候选精读清单。"""
import json
import glob
import re
import sys

TOPICS = {
    "label_width": [
        "gaussian", "label width", "soft label", "target function", "label shape",
        "sigma", "smoothing", "probability target", "labeling strategy",
    ],
    "pick_precision": [
        "arrival time", "onset", "picking accuracy", "residual", "time error",
        "sub-sample", "precision", "travel time", "mae", "absolute error",
    ],
    "domain_adapt": [
        "transfer learning", "domain adaptation", "fine-tun", "generalization",
        "cross-region", "out-of-distribution", "domain shift", "catastrophic forgetting",
    ],
    "distill": ["distillation", "teacher", "student", "ensemble", "knowledge transfer"],
    "ssl_jepa": [
        "self-supervised", "jepa", "joint embedding", "masked", "pretrain",
        "representation learning", "contrastive", "foundation model",
    ],
    "recall": [
        "detection", "recall", "missed", "low snr", "sensitivity", "false negative",
        "association", "multi-event",
    ],
}
SEISMIC = ["seismic", "earthquake", "phase pick", "phasenet", "microseismic", "seismol", "waveform"]


def score(rec):
    text = ((rec.get("title") or "") + " " + (rec.get("abstract") or "")).lower()
    hits = {}
    for topic, kws in TOPICS.items():
        n = sum(1 for k in kws if k in text)
        if n:
            hits[topic] = n
    seis = sum(1 for k in SEISMIC if k in text)
    total = sum(hits.values()) + 2 * min(seis, 3)
    return total, hits, seis


def main():
    recs = []
    for path in sorted(glob.glob("memory/papers/_raw/round09_q_*.json")):
        payload = json.load(open(path, encoding="utf-8"))
        for r in payload["records"]:
            r["_src"] = path.replace("\\", "/").split("/")[-1]
            recs.append(r)
    merged = {}
    for r in recs:
        key = (r.get("doi") or "").lower() or ("arxiv:" + (r.get("arxiv_id") or "")) or (r.get("title") or "").lower()[:110]
        if key in ("", "arxiv:"):
            key = (r.get("title") or "").lower()[:110]
        cur = merged.get(key)
        if cur is None:
            merged[key] = r
        else:
            if len(r.get("abstract") or "") > len(cur.get("abstract") or ""):
                cur["abstract"] = r["abstract"]
            if r["_src"] not in cur["_src"]:
                cur["_src"] = cur["_src"] + "," + r["_src"]
    out = []
    for r in merged.values():
        total, hits, seis = score(r)
        r["_score"] = total
        r["_topics"] = hits
        r["_seismic"] = seis
        out.append(r)
    out.sort(key=lambda r: (-r["_score"], -(r.get("citations") or 0)))
    print("unique_total", len(out))
    keep = [r for r in out if r["_seismic"] >= 1 and r["_score"] >= 6]
    print("candidates", len(keep))
    json.dump(out, open("memory/papers/_raw/round09_scored.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for i, r in enumerate(keep[:60]):
        yr = r.get("year")
        cit = r.get("citations")
        tps = ",".join(sorted(r["_topics"]))
        print("%2d s=%-3d cit=%-5s %s | %s | %s | %s" % (
            i + 1, r["_score"], cit, str(yr), (r.get("title") or "")[:95], tps, (r.get("doi") or r.get("arxiv_id") or "")[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main())