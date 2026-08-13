import io, time, urllib.request
import numpy as np, pandas as pd

R = "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets"
CHUNKS = ["000","001","002","003","004"]
WIN_S = 3001/50.0          # 60.02 s  (pool sampling rate 50 Hz)
MARGIN_S = 250/50.0        # 5.0 s blind margin each side
BOTH_MAX = (3001-500)/50.0 # 50.02 s : max S-P that still fits both picks

rows_all = []
for c in CHUNKS:
    url = f"{R}/crew/metadata{c}.csv"
    try:
        t0=time.time()
        with urllib.request.urlopen(url, timeout=600) as r:
            raw = r.read()
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
        df["__chunk"]=c
        rows_all.append(df)
        print(f"chunk {c}: rows={len(df)} bytes={len(raw)} in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"chunk {c}: FAIL {type(e).__name__} {e}")
if not rows_all:
    raise SystemExit(1)
md = pd.concat(rows_all, ignore_index=True)
print("TOTAL rows", len(md))
print("cols with arrival:", [c for c in md.columns if "arrival_sample" in c])
sr = pd.to_numeric(md.get("trace_sampling_rate_hz", pd.Series(100.0,index=md.index)), errors="coerce").fillna(100.0)
P = pd.to_numeric(md["trace_P_arrival_sample"], errors="coerce")
S = pd.to_numeric(md["trace_S_arrival_sample"], errors="coerce")
npts = pd.to_numeric(md.get("trace_npts", pd.Series(np.nan,index=md.index)), errors="coerce")
sp = (S-P)/sr
ok = np.isfinite(sp) & (sp>0)
print(f"\nwith P+S: {int(ok.sum())}")
spv = sp[ok]
print("sampling rates:", sr.value_counts().to_dict())
if npts.notna().any():
    dur = (npts/sr)[ok]
    print(f"duration s: median={dur.median():.1f} p10={np.percentile(dur,10):.1f} max={dur.max():.1f}")
both = spv[spv <= BOTH_MAX]
print(f"\nS-P <= {BOTH_MAX:.1f}s (both picks fit 60.02s window): {len(both)} = {100*len(both)/len(spv):.1f}%")
for lo,hi in [(0,10),(10,15),(15,20),(20,25),(25,30),(30,35),(35,40),(40,45),(45,50.02)]:
    n = int(((spv>=lo)&(spv<hi)).sum())
    print(f"  [{lo:>4.0f},{hi:>5.1f})s : {n:>6}")
print(f"  >  {BOTH_MAX:.1f}s (only one pick fits) : {int((spv>BOTH_MAX).sum())}")
if "source_magnitude" in md.columns:
    m = pd.to_numeric(md["source_magnitude"],errors="coerce")[ok]
    print(f"\nmagnitude: median={m.median():.2f} p10={np.percentile(m.dropna(),10):.2f} max={m.max():.2f}")
for c in ("path_ep_distance_km","source_depth_km","station_network_code"):
    if c in md.columns:
        v = md[c][ok]
        vn = pd.to_numeric(v, errors="coerce").dropna()
        if len(vn):
            print(f"{c}: median={vn.median():.1f} p90={np.percentile(vn,90):.1f} max={vn.max():.1f}")
        else:
            print(f"{c}: top={v.value_counts().head(5).to_dict()}")
ev = [c for c in ("source_id","source_event_id","event_id","source_origin_time") if c in md.columns]
print("event id cols:", ev)
if ev:
    print("unique events:", md.loc[ok, ev[0]].nunique())