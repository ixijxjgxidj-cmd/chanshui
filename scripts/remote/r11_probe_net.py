import os, time, urllib.request, concurrent.futures as cf

R = "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets"
PATHS = ["ethz/waveforms.hdf5","ethz/metadata.csv","geofon/waveforms.hdf5",
         "geofon/metadata.csv","crew/metadata000.csv","crew/waveforms000.hdf5",
         "lendb/waveforms.hdf5","iquique/waveforms.hdf5"]

def head(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, dict(r.headers)

def rng(url, lo, hi):
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    return r.status, len(data), dict(r.headers).get("Content-Range"), time.time()-t0

print("=== HEAD ===")
for p in PATHS:
    try:
        st, h = head(f"{R}/{p}")
        print(f"{p} http={st} len={h.get('Content-Length')} ranges={h.get('Accept-Ranges')}")
    except Exception as e:
        print(f"{p} HEAD-FAIL {type(e).__name__} {e}")

print("=== RANGE probe (first 1KB) ===")
for p in ["ethz/waveforms.hdf5","crew/waveforms000.hdf5"]:
    try:
        st, n, cr, dt = rng(f"{R}/{p}", 0, 1023)
        print(f"{p} http={st} got={n}B content_range={cr} in {dt:.2f}s")
    except Exception as e:
        print(f"{p} RANGE-FAIL {type(e).__name__} {e}")

url = f"{R}/ethz/waveforms.hdf5"
print("=== single stream 20MB ===")
try:
    st, n, cr, dt = rng(url, 0, 20*1024*1024-1)
    print(f"single got={n/2**20:.1f}MB in {dt:.1f}s = {n/2**20/dt:.2f} MB/s")
except Exception as e:
    print("single FAIL", e)

print("=== 6 parallel x 10MB ===")
def job(i):
    lo = i*10*1024*1024; hi = lo + 10*1024*1024 - 1
    return rng(url, lo, hi)[1]
t0 = time.time()
try:
    with cf.ThreadPoolExecutor(6) as ex:
        tot = sum(ex.map(job, range(6)))
    dt = time.time()-t0
    print(f"parallel got={tot/2**20:.1f}MB in {dt:.1f}s = {tot/2**20/dt:.2f} MB/s")
except Exception as e:
    print("parallel FAIL", e)