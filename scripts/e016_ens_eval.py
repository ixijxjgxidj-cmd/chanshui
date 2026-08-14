#!/usr/bin/env python3
"""在窗池上评估「概率集成」，用于判断单模型增益能否传导到 7 成员集成。

为什么需要：实验 016 证明 F1_half 单模型在跨区域远场上显著优于起点，但生产是
7 成员 annotation 平均。集成会把单成员的改善摊薄（甚至抵消），所以必须在集成
层面复验。现有 ensemble_prob_t1.py 只吃官方 mseed 包，而合规上 R1/R2 不能进
训练机，所以这里在公开裁判集（窗池）上做等价评估。

与生产同构的部分：annotation 逐点平均 → picks_from_annotations → 同阈值
（phasepicker.defaults）。不含长记录门控/去重（窗池是 60s 短窗，那些逻辑不触发）。

用法:
    python e016_ens_eval.py --members guangxi,huanan,jiangxi \
        --pool pool/judge_iquique600.hdf5 --out runs/ens_c3.json --label c3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

GROSS_P_S = 1.0
GROSS_S_S = 2.0
BINS = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 50.1)]


def _load_ft(repo_scripts: str):
    sys.path.insert(0, os.path.join(os.path.dirname(repo_scripts), "src"))
    sys.path.insert(0, repo_scripts)
    spec = importlib.util.spec_from_file_location(
        "ft", os.path.join(repo_scripts, "finetune_phasenet.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve(name: str, weights_dir: str) -> str | None:
    """成员名 -> 权重路径。'diting' = 纯预训练（None）。"""
    if name == "diting":
        return None
    if name.endswith(".pt") or os.sep in name or "/" in name:
        return name
    return os.path.join(weights_dir, name + "_sd.pt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True,
                    help="逗号分隔的成员：区域简名 / .pt 路径 / diting")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--repo-scripts", default="repo/scripts")
    ap.add_argument("--weights-dir", default="repo/weights/ustc_pickers")
    ap.add_argument("--pretrained", default="diting")
    ap.add_argument("--sr", type=float, default=50.0)
    ap.add_argument("--win", type=int, default=3001)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ft = _load_ft(args.repo_scripts)
    import torch
    import seisbench.models as sbm
    from phasepicker.defaults import DEFAULT_P_THRESHOLD, DEFAULT_S_THRESHOLD

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = [x.strip() for x in args.members.split(",") if x.strip()]
    models = []
    for n in names:
        m = sbm.PhaseNet.from_pretrained(args.pretrained).to(dev)
        wp = resolve(n, args.weights_dir)
        if wp:
            if not os.path.exists(wp):
                raise SystemExit("成员权重不存在: %s" % wp)
            ck = torch.load(wp, map_location=dev, weights_only=False)
            sd = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
            miss, unexp = m.load_state_dict(sd, strict=False)
            print("[member] %-46s missing=%d unexpected=%d" % (n, len(miss), len(unexp)))
        else:
            print("[member] %-46s (pretrained only)" % n)
        m.eval()
        models.append(m)
    print("[ens] %d 成员, 阈值 P=%.2f S=%.2f" %
          (len(models), DEFAULT_P_THRESHOLD, DEFAULT_S_THRESHOLD))

    items = ft.load_hdf5_dataset(args.pool, args.win, expect_sr=args.sr)
    if args.limit:
        items = items[: args.limit]
    print("[ens] %d windows from %s" % (len(items), args.pool))

    from obspy import Stream, Trace, UTCDateTime
    t0 = UTCDateTime(0)

    def to_stream(wave):
        st = Stream()
        for ch, nm in zip(wave, ["Z", "N", "E"]):
            tr = Trace(data=np.asarray(ch, dtype="float32"))
            tr.stats.sampling_rate = args.sr
            tr.stats.starttime = t0
            tr.stats.channel = "HH" + nm
            tr.stats.station = "XX"
            st.append(tr)
        return st

    rows = []
    for idx, it in enumerate(items):
        wave, p, s = it[0], it[1], it[2]
        if p < 0 or s < 0:
            continue
        st = to_stream(ft.normalize(wave))
        # annotation 逐点平均：与 ProbEnsemblePicker._classify_refined 同构
        anns = [m.annotate(st) for m in models]
        base = anns[0].copy()
        for tr in base:
            stack = [tr.data.astype(np.float64)]
            for other in anns[1:]:
                match = [t for t in other
                         if t.id == tr.id and t.stats.starttime == tr.stats.starttime
                         and len(t.data) == len(tr.data)]
                if len(match) != 1:
                    raise RuntimeError("集成 trace 对不齐: %s 命中 %d" % (tr.id, len(match)))
                stack.append(match[0].data.astype(np.float64))
            tr.data = np.mean(stack, axis=0)

        prefix = models[0].__class__.__name__
        pred = []
        for phase, th in (("P", DEFAULT_P_THRESHOLD), ("S", DEFAULT_S_THRESHOLD)):
            sel = base.select(channel="%s_%s" % (prefix, phase))
            for pk in models[0].picks_from_annotations(sel, th, phase):
                pt = getattr(pk, "peak_time", None)
                if pt is not None:
                    pred.append((phase, float(pt - t0)))

        truth = [("P", p / args.sr), ("S", s / args.sr)]
        rep = ft.score_file(pred, truth)
        pres = rep["pres"][0] if rep["pres"] else None
        sres = rep["sres"][0] if rep["sres"] else None
        rows.append(dict(sp_s=float((s - p) / args.sr), total=float(rep["total"]),
                         p_res=None if pres is None else float(pres),
                         s_res=None if sres is None else float(sres),
                         gross_p=bool(pres is None or pres > GROSS_P_S),
                         gross_s=bool(sres is None or sres > GROSS_S_S),
                         n_pred=len(pred)))
        if (idx + 1) % 100 == 0:
            print("  %d/%d" % (idx + 1, len(items)), flush=True)

    per_bin = {}
    for lo, hi in BINS:
        sub = [r for r in rows if lo <= r["sp_s"] < hi]
        if not sub:
            continue
        n = len(sub)
        pr = [r["p_res"] for r in sub if r["p_res"] is not None]
        sr_ = [r["s_res"] for r in sub if r["s_res"] is not None]
        per_bin["[%g,%g)" % (lo, hi)] = dict(
            n=n,
            mean_score=float(np.mean([r["total"] for r in sub])),
            gross_rate=float(sum(1 for r in sub if r["gross_p"] or r["gross_s"]) / n),
            gross_p_rate=float(sum(1 for r in sub if r["gross_p"]) / n),
            gross_s_rate=float(sum(1 for r in sub if r["gross_s"]) / n),
            p_res_median=float(np.median(pr)) if pr else None,
            s_res_median=float(np.median(sr_)) if sr_ else None,
            p_hit=float(np.mean([1.0 if (r["p_res"] is not None and r["p_res"] <= 0.1)
                                 else 0.0 for r in sub])),
            s_hit=float(np.mean([1.0 if (r["s_res"] is not None and r["s_res"] <= 0.2)
                                 else 0.0 for r in sub])),
        )
    overall = dict(
        n=len(rows),
        mean_score=float(np.mean([r["total"] for r in rows])) if rows else None,
        gross_rate=float(sum(1 for r in rows if r["gross_p"] or r["gross_s"])
                         / max(len(rows), 1)))
    out = dict(label=args.label or ",".join(names), members=names, pool=args.pool,
               sr=args.sr, overall=overall, per_bin=per_bin, rows=rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(dict(overall=overall, per_bin=per_bin), ensure_ascii=False, indent=1))
    print("[ens] wrote " + args.out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())