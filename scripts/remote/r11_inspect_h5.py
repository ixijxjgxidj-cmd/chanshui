import sys, time
sys.path.insert(0, "/root/5.6+chanshui1/repo/scripts")
from sb_http import HttpRangeFile
import h5py

R = "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets"
url = f"{R}/ethz/waveforms.hdf5"
t0 = time.time()
fo = HttpRangeFile(url, block=1 << 20, max_blocks=256)
print("size_gb", round(fo.size / 2**30, 2))
with h5py.File(fo, "r", driver="fileobj") as f:
    print("root keys", list(f.keys()))
    for k in list(f.keys()):
        obj = f[k]
        print(" ", k, type(obj).__name__, dict(obj.attrs) if obj.attrs else "")
        if isinstance(obj, h5py.Group):
            sub = list(obj.keys())[:5]
            print("   first children:", sub, "n=", len(obj))
            for s in sub[:2]:
                d = obj[s]
                print("    ", s, getattr(d, "shape", None), getattr(d, "dtype", None),
                      dict(d.attrs) if d.attrs else "")
print("elapsed", round(time.time() - t0, 1), "stats", fo.stats)