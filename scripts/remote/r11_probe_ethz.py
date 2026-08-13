import sys, time, io, urllib.request
sys.path.insert(0, "/root/5.6+chanshui1/repo/scripts")
from sb_http import HttpRangeFile
import h5py, numpy as np, pandas as pd

R = "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets"
md_url = f"{R}/ethz/metadata.csv"
t0 = time.time()
with urllib.request.urlopen(md_url, timeout=300) as r:
    raw = r.read()
print("metadata bytes", len(raw), "in", round(time.time()-t0,1), "s")
md = pd.read_csv(io.BytesIO(raw), low_memory=False)
print("rows", len(md), "cols", len(md.columns))
pcols = [c for c in md.columns if "arrival_sample" in c]
print("arrival cols:", pcols)
print("trace_name sample:", md["trace_name"].head(3).tolist())
print("sampling rate col:", [c for c in md.columns if "sampling" in c or c=="trace_dt_s"])
for c in ("trace_sampling_rate_hz","trace_dt_s"):
    if c in md.columns:
        print(c, md[c].describe().to_dict())
# S-P distribution using primary + variants
def firstval(row, pref):
    for c in pcols:
        if c.startswith(pref):
            v = row.get(c)
            if pd.notna(v):
                return float(v)
    return None
sub = md.head(20000)
sp = []
for _, row in sub.iterrows():
    p = firstval(row, "trace_P") or firstval(row, "trace_p")
    s = firstval(row, "trace_S") or firstval(row, "trace_s")
    sr = row.get("trace_sampling_rate_hz", 100.0)
    if p and s and sr and s > p:
        sp.append((s-p)/float(sr))
sp = np.array(sp)
print("S-P n", len(sp), "median", round(float(np.median(sp)),2),
      "p90", round(float(np.percentile(sp,90)),2), "max", round(float(sp.max()),2))
print("frac >=10s", round(float((sp>=10).mean()),4), " >=20s", round(float((sp>=20).mean()),4))
# now fetch one waveform by trace_name via range reads
fo = HttpRangeFile(f"{R}/ethz/waveforms.hdf5", block=1<<20, max_blocks=256)
with h5py.File(fo, "r", driver="fileobj") as f:
    g = f["data"]
    tn = md["trace_name"].iloc[0]
    print("try trace_name", tn)
    if "$" in str(tn):
        base, _, loc = str(tn).partition("$")
        print(" bucket form -> base", base, "loc", loc)
        d = g[base]
        print(" bucket shape", d.shape, d.dtype)
    else:
        d = g[str(tn)]
        arr = d[:]
        print(" direct shape", arr.shape, arr.dtype, "nan", int(np.isnan(arr).sum()))
print("stats", fo.stats)