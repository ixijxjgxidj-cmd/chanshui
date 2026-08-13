import io, sys, time, urllib.request
import numpy as np, pandas as pd

R = "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets"

def load_csv(url):
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=600) as r:
        raw = r.read()
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    print(f"  {url.split('/')[-2]}/{url.split('/')[-1]}: {len(raw)}B rows={len(df)} in {time.time()-t0:.1f}s")
    return df

def sp_report(name, md):
    pcols = [c for c in md.columns if c.startswith("trace_P") and c.endswith("_arrival_sample")]
    pcols += [c for c in md.columns if c == "trace_p_arrival_sample"]
    scols = [c for c in md.columns if c.startswith("trace_S") and c.endswith("_arrival_sample")]
    scols += [c for c in md.columns if c == "trace_s_arrival_sample"]
    srcol = "trace_sampling_rate_hz" if "trace_sampling_rate_hz" in md.columns else None
    P = md[pcols].astype("float64").bfill(axis=1).iloc[:, 0] if pcols else None
    S = md[scols].astype("float64").bfill(axis=1).iloc[:, 0] if scols else None
    if P is None or S is None:
        print(f"{name}: no arrival cols P={pcols} S={scols}"); return
    sr = md[srcol].astype("float64") if srcol else pd.Series(100.0, index=md.index)
    sp = (S - P) / sr
    sp = sp[(sp > 0) & np.isfinite(sp)]
    print(f"{name}: rows={len(md)} with_PS={len(sp)} ({100*len(sp)/max(len(md),1):.1f}%)")
    if len(sp):
        q = np.percentile(sp, [10,25,50,75,90,99])
        print(f"  S-P s: p10={q[0]:.1f} p25={q[1]:.1f} median={q[2]:.1f} p75={q[3]:.1f} p90={q[4]:.1f} p99={q[5]:.1f} max={sp.max():.1f}")
        for th in (10, 15, 20, 30, 40):
            n = int((sp >= th).sum())
            print(f"  >= {th:>2}s : {n:>7} ({100*n/len(sp):.2f}%)")
    if "source_magnitude" in md.columns:
        m = pd.to_numeric(md["source_magnitude"], errors="coerce").dropna()
        print(f"  magnitude n={len(m)} min={m.min():.2f} median={m.median():.2f} max={m.max():.2f}")
    for c in ("path_ep_distance_km","source_distance_km","path_travel_time_P_s"):
        if c in md.columns:
            v = pd.to_numeric(md[c], errors="coerce").dropna()
            if len(v):
                print(f"  {c}: n={len(v)} median={v.median():.1f} p90={np.percentile(v,90):.1f} max={v.max():.1f}")

for name, url in [("CREW-000", f"{R}/crew/metadata000.csv"),
                  ("GEOFON",   f"{R}/geofon/metadata.csv")]:
    try:
        md = load_csv(url)
        sp_report(name, md)
    except Exception as e:
        print(name, "FAIL", type(e).__name__, e)
    print()