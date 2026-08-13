#!/usr/bin/env python3
"""实验 016 主指标评估器：按 S-P 分箱统计粗大错时率与官方分。

为什么需要单独一个评估器：finetune_phasenet.py 的 --holdout 只输出总均分，
而实验 016 的预注册主指标是**按 S-P 分箱**的粗大错时率（官方评分对 P>1s、
S>2s 直接判 0，这是失分的主要来源，且集中在远场窗）。总均分会把近场的
大样本量摊薄掉远场的改善，所以必须分箱看。

判定口径与 scripts/finetune_phasenet.py 完全一致（同一 _predict_window /
score_file / 阈值来源 phasepicker.defaults），保证与训练期 monitor 可比。

用法：
    python e016_eval.py --weights runs/e016_A0_near/best.pt \
        --pool pool/dev_mixed.hdf5 --out runs/e016_eval_A0.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_ft(repo_scripts: str):
    """把 finetune_phasenet.py 作为模块载入，复用它的打分口径。"""
    sys.path.insert(0, os.path.join(os.path.dirname(repo_scripts), "src"))
    sys.path.insert(0, repo_scripts)
    spec = importlib.util.spec_from_file_location(
        "ft", os.path.join(repo_scripts, "finetune_phasenet.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 分箱边界（秒）：与实验 015 诊断口径一致，近场 <10s 作为护栏分箱
BINS = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 50.1)]
GROSS_P_S = 1.0   # 官方：P 残差 >1.0s 得 0
GROSS_S_S = 2.0   # 官方：S 残差 >2.0s 得 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="checkpoint (.pt) 或 sb: 名")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-scripts", default="repo/scripts")
    ap.add_argument("--sr", type=float, default=50.0)
    ap.add_argument("--win", type=int, default=3001)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pretrained", default="diting")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ft = _load_ft(args.repo_scripts)
    import torch
    import seisbench.models as sbm

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = sbm.PhaseNet.from_pretrained(args.pretrained).to(dev)
    if args.weights and not args.weights.startswith("sb:"):
        ck = torch.load(args.weights, map_location=dev, weights_only=False)
        sd = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] {args.weights} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    items = ft.load_hdf5_dataset(args.pool, args.win, expect_sr=args.sr)
    if args.limit:
        items = items[: args.limit]
    print(f"[eval] {len(items)} windows from {args.pool}")

    per_bin: dict[str, dict] = {}
    rows = []
    for idx, it in enumerate(items):
        wave, p, s = it[0], it[1], it[2]
        if p < 0 or s < 0:
            continue
        sp_s = (s - p) / args.sr
        truth = [("P", p / args.sr), ("S", s / args.sr)]
        pred = ft._predict_window(model, ft.normalize(wave), args.sr)
        rep = ft.score_file(pred, truth)
        pres = rep["pres"][0] if rep["pres"] else None
        sres = rep["sres"][0] if rep["sres"] else None
        gross_p = (pres is None) or (pres > GROSS_P_S)
        gross_s = (sres is None) or (sres > GROSS_S_S)
        rows.append(dict(sp_s=float(sp_s), total=float(rep["total"]),
                         p_res=None if pres is None else float(pres),
                         s_res=None if sres is None else float(sres),
                         gross_p=bool(gross_p), gross_s=bool(gross_s),
                         n_pred=len(pred)))
        if (idx + 1) % 200 == 0:
            print(f"  {idx+1}/{len(items)}", flush=True)

    for lo, hi in BINS:
        sub = [r for r in rows if lo <= r["sp_s"] < hi]
        if not sub:
            continue
        n = len(sub)
        gp = sum(1 for r in sub if r["gross_p"])
        gs = sum(1 for r in sub if r["gross_s"])
        gross = sum(1 for r in sub if r["gross_p"] or r["gross_s"])
        pr = [r["p_res"] for r in sub if r["p_res"] is not None]
        sr_ = [r["s_res"] for r in sub if r["s_res"] is not None]
        per_bin[f"[{lo:g},{hi:g})"] = dict(
            n=n,
            mean_score=float(np.mean([r["total"] for r in sub])),
            gross_rate=float(gross / n),
            gross_p_rate=float(gp / n),
            gross_s_rate=float(gs / n),
            p_res_median=float(np.median(pr)) if pr else None,
            s_res_median=float(np.median(sr_)) if sr_ else None,
            p_hit=float(np.mean([1.0 if (r["p_res"] is not None and r["p_res"] <= 0.1) else 0.0
                                 for r in sub])),
            s_hit=float(np.mean([1.0 if (r["s_res"] is not None and r["s_res"] <= 0.2) else 0.0
                                 for r in sub])),
        )

    overall = dict(
        n=len(rows),
        mean_score=float(np.mean([r["total"] for r in rows])) if rows else None,
        gross_rate=float(sum(1 for r in rows if r["gross_p"] or r["gross_s"]) / max(len(rows), 1)),
    )
    out = dict(label=args.label or os.path.basename(args.weights),
               weights=args.weights, pool=args.pool, sr=args.sr,
               overall=overall, per_bin=per_bin, rows=rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(dict(overall=overall, per_bin=per_bin), ensure_ascii=False, indent=1))
    print(f"[eval] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())