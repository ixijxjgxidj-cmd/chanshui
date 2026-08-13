"""Experiment 016 probe: does any public pretrained picker beat our production
member in the far-field (large S-P) regime that produces our gross timing errors?

Selection set: SeisBench public CWA dataset (Taiwan, 2011 chunk), already cached
locally. Public data, evaluation only, no training, no download. The four sealed
competition packages are never touched by this script.

Metric is deliberately threshold-free: for each labeled window we take the argmax
of the P and S probability channels and report the residual against the catalog
arrival. That isolates "does the model know where the phase is" from "is its
threshold calibrated", which matters because candidate members ship with wildly
different default thresholds.

Scoring uses the official competition rule (P: <=0.1s full, >=1.0s zero;
S: <=0.2s full, >=2.0s zero) so the numbers are comparable to our other reports.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch

CWA_DIR = os.path.join(os.path.expanduser("~"), ".seisbench", "datasets", "cwa")
META = os.path.join(CWA_DIR, "metadata_2011.csv")
WAVE = os.path.join(CWA_DIR, "waveforms_2011.hdf5")

PHASE_RULE = {"P": (0.1, 1.0), "S": (0.2, 2.0)}


def phase_score(residual_s, phase):
    full, zero = PHASE_RULE[phase]
    r = abs(float(residual_s))
    if r <= full:
        return 1.0
    if r >= zero:
        return 0.0
    return (zero - r) / (zero - full)


def build_members(spec, weights_dir="weights/ustc_pickers"):
    """spec: name -> loaded PhaseNet. Accepts 'sb:<pretrained>' or a local .pt."""
    import seisbench.models as sbm

    out = {}
    for name in spec:
        if name.startswith("sb:"):
            out[name] = sbm.PhaseNet.from_pretrained(name[3:])
        else:
            path = name if (os.sep in name or name.endswith(".pt")) else os.path.join(weights_dir, name + "_sd.pt")
            m = sbm.PhaseNet(in_channels=3, classes=3, phases="NPS", sampling_rate=50, norm="std")
            sd = torch.load(path, map_location="cpu")
            m.load_state_dict(sd, strict=True)
            out[name] = m
        out[name].eval()
    return out


def phase_index(model, want):
    """Locate the probability channel for a phase, honoring each model's order."""
    phases = getattr(model, "labels", None) or getattr(model, "phases", "NPS")
    for i, ch in enumerate(list(phases)):
        if str(ch).upper().startswith(want):
            return i
    return None


def annotate_peaks(model, z_n_e, sr_in, want=("P", "S")):
    """Return {phase: peak_time_seconds} using argmax of the probability curve."""
    from obspy import Stream, Trace, UTCDateTime

    t0 = UTCDateTime(0)
    st = Stream()
    for ch, nm in zip(z_n_e, ["Z", "N", "E"]):
        tr = Trace(data=np.asarray(ch, dtype="float32"))
        tr.stats.sampling_rate = float(sr_in)
        tr.stats.starttime = t0
        tr.stats.channel = "HH" + nm
        tr.stats.station = "CWA"
        st.append(tr)
    ann = model.annotate(st)
    res = {}
    for ph in want:
        cand = [tr for tr in ann if str(tr.stats.channel).upper().endswith(ph)]
        if not cand:
            continue
        tr = cand[0]
        if tr.stats.npts == 0:
            continue
        k = int(np.argmax(tr.data))
        res[ph] = float(tr.stats.starttime - t0) + k / float(tr.stats.sampling_rate)
        res[ph + "_peak"] = float(np.max(tr.data))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="e016 CWA far-field member probe")
    ap.add_argument("--members", default="guangxi,weights/pub/pub_ethzcrew_a_sd.pt,sb:neic,sb:pisdl,sb:diting")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--win-s", type=float, default=60.0, help="window length seconds (match competition records)")
    ap.add_argument("--pre-p-s", type=float, default=10.0, help="seconds of pre-P context")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/e016_cwa_farfield.json")
    args = ap.parse_args(argv)

    df = pd.read_csv(META, low_memory=False)
    df = df.dropna(subset=["trace_p_arrival_sample", "trace_s_arrival_sample"]).copy()
    df["sp_s"] = (df.trace_s_arrival_sample - df.trace_p_arrival_sample) / df.trace_sampling_rate_hz
    df = df[df.sp_s > 0.5]
    # 50 Hz members (diting family) need 3001 samples = 60.02 s of input, so a
    # trace shorter than the window silently yields an empty annotation and the
    # member drops out of the comparison. Require enough duration up front so
    # every member is scored on exactly the same windows.
    df["dur_s"] = df.trace_npts / df.trace_sampling_rate_hz
    df["p_off_s"] = df.trace_p_arrival_sample / df.trace_sampling_rate_hz
    need = args.win_s + 1.0
    df = df[df.dur_s >= need]
    # window must be placeable so that both P and S fall inside it
    df = df[(df.p_off_s + df.sp_s + 2.0) <= (df.p_off_s - args.pre_p_s).clip(lower=0) + args.win_s]
    rng = np.random.RandomState(args.seed)

    # Stratify so the far-field regime is actually represented, instead of being
    # swamped by the near-field majority the way our training pool was.
    bins = [(0, 5), (5, 10), (10, 20), (20, 60)]
    per = max(1, args.limit // len(bins))
    picks_idx = []
    for lo, hi in bins:
        sub = df[(df.sp_s >= lo) & (df.sp_s < hi)]
        if len(sub) == 0:
            continue
        take = min(per, len(sub))
        picks_idx.extend(rng.choice(sub.index.values, size=take, replace=False).tolist())
    sel = df.loc[picks_idx].copy()
    print("selected %d windows; per-bin: %s" % (
        len(sel), {("%d-%ds" % b): int(((sel.sp_s >= b[0]) & (sel.sp_s < b[1])).sum()) for b in bins}))

    members = build_members([m.strip() for m in args.members.split(",") if m.strip()])
    print("members:", list(members))
    for nm, m in members.items():
        print("   %-42s sr=%s phases=%s" % (nm, m.sampling_rate, getattr(m, "labels", getattr(m, "phases", "?"))))

    rows = []
    with h5py.File(WAVE, "r") as f:
        grp = f["data"]
        for i, (_, r) in enumerate(sel.iterrows(), 1):
            key = r.trace_name
            if key not in grp:
                continue
            arr = np.asarray(grp[key][()], dtype=np.float64)
            sr = float(r.trace_sampling_rate_hz)
            p_smp = float(r.trace_p_arrival_sample)
            s_smp = float(r.trace_s_arrival_sample)
            n = arr.shape[-1]
            start = int(max(0, p_smp - args.pre_p_s * sr))
            length = int(args.win_s * sr)
            if start + length > n:
                start = max(0, n - length)
            seg = arr[:, start:start + length]
            if seg.shape[-1] < int(args.win_s * sr) - 1:
                continue  # keep every member on an identical-length window
            p_true = (p_smp - start) / sr
            s_true = (s_smp - start) / sr
            if s_true > seg.shape[-1] / sr:
                continue  # S beyond the window; nothing to score fairly
            rec = {"trace": key, "sp_s": float(r.sp_s), "sr": sr,
                   "dist_km": float(r.path_ep_distance_km) if pd.notna(r.path_ep_distance_km) else None,
                   "mag": float(r.source_magnitude) if pd.notna(r.source_magnitude) else None}
            for nm, m in members.items():
                try:
                    pk = annotate_peaks(m, seg, sr)
                except Exception as exc:
                    rec[nm] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:80])}
                    continue
                d = {}
                if "P" in pk:
                    d["p_res"] = pk["P"] - p_true
                    d["p_score"] = phase_score(d["p_res"], "P")
                    d["p_peak"] = pk.get("P_peak")
                if "S" in pk:
                    d["s_res"] = pk["S"] - s_true
                    d["s_score"] = phase_score(d["s_res"], "S")
                    d["s_peak"] = pk.get("S_peak")
                rec[nm] = d
            rows.append(rec)
            if i % 25 == 0:
                print("  %d/%d" % (i, len(sel)), flush=True)

    # ---- aggregate ----
    report = {"n": len(rows), "members": {}, "config": vars(args)}
    for nm in members:
        agg = {}
        for label, lo, hi in [("all", 0, 1e9), ("near_0_10s", 0, 10), ("far_10_20s", 10, 20), ("far_20plus", 20, 1e9)]:
            sub = [r for r in rows if lo <= r["sp_s"] < hi and isinstance(r.get(nm), dict) and "p_res" in r[nm]]
            if not sub:
                continue
            ps = [r[nm]["p_score"] for r in sub]
            ss = [r[nm]["s_score"] for r in sub if "s_score" in r[nm]]
            pr = [abs(r[nm]["p_res"]) for r in sub]
            sr_ = [abs(r[nm]["s_res"]) for r in sub if "s_res" in r[nm]]
            gross_p = sum(1 for x in pr if x > 1.0)
            gross_s = sum(1 for x in sr_ if x > 2.0)
            agg[label] = {
                "n": len(sub),
                "mean_score": float(np.mean(ps) + np.mean(ss)) if ss else float(np.mean(ps)),
                "p_score": float(np.mean(ps)), "s_score": float(np.mean(ss)) if ss else None,
                "p_res_med": float(np.median(pr)), "s_res_med": float(np.median(sr_)) if sr_ else None,
                "gross_p": gross_p, "gross_s": gross_s,
                "gross_rate": float((gross_p + gross_s) / (len(pr) + len(sr_))) if (pr or sr_) else None,
            }
        report["members"][nm] = agg

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwrote", args.out)
    hdr = "%-44s %8s %8s %8s %8s %8s" % ("member/bin", "n", "mean", "p_med", "s_med", "gross%")
    for nm, agg in report["members"].items():
        print("\n" + hdr)
        for label, a in agg.items():
            print("%-44s %8d %8.4f %8.3f %8s %8s" % (
                nm[:30] + "/" + label, a["n"], a["mean_score"], a["p_res_med"],
                ("%.3f" % a["s_res_med"]) if a["s_res_med"] is not None else "-",
                ("%.2f" % (100 * a["gross_rate"])) if a["gross_rate"] is not None else "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())