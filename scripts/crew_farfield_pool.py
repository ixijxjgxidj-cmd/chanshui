#!/usr/bin/env python3
"""CREW/GEOFON 远场窗池构建器（数据前置站专用，不依赖 seisbench）。

为什么单独写一个：
1. 训练机（zzai 容器）出网只有 ~1.6 MB/s，CREW 五块原始波形约 25 GB，直接在训练机
   下载要 4h+；而数据前置站（tor1，多伦多）到 DESY 有 9.4 MB/s 单流 / 20 MB/s 四并发。
2. 池格式与 scripts/prepare_pool.py 完全一致（group "data"，attrs
   p_sample_100hz / s_sample_100hz / sampling_rate），所以 finetune_phasenet.py 直接可读。
   注意 attr 名里的 "100hz" 是历史名，真实语义是"池自身采样率下的窗内下标"。
3. tor1 只有 2 GB 内存、1 vCPU，所以逐 bucket 流式处理，不整块载入。

窗口策略与 prepare_pool._crop 对齐：50 Hz、3001 点（60.02 s）、首尾 250 点盲区，
只收 P 与 S 能同窗的样本（S-P <= 50.02 s）。

用法：
    python crew_farfield_pool.py --dataset crew --chunks 000,001 \
        --out-dir /root/5.6+chanshui1/pool --sp-min-s 10 --sp-max-s 50
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import time
import urllib.request

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

REMOTE_ROOT = "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets"
MARGIN = 250          # 首尾盲区（池采样率下的点）
WIN = 3001            # 池窗长（点）
POOL_SR = 50.0        # 池采样率，对齐 diting/USTC PhaseNet 原生


def stable_hash01(text: str) -> float:
    d = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(d[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def fetch_csv(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=900) as r:
        raw = r.read()
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def download(url: str, dest: str, retries: int = 5) -> str:
    """带断点续传的整块下载（bucket 随机读在 1.6MB/s 下不划算，整块更快）。"""
    if os.path.exists(dest):
        return dest
    tmp = dest + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"),
                               timeout=120) as r:
        total = int(r.headers["Content-Length"])
    for attempt in range(retries):
        if have >= total:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=900) as r, open(tmp, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    if have % (64 << 20) < (1 << 20):
                        mb = have / 2 ** 20
                        print(f"    {mb:.0f}/{total/2**20:.0f} MB "
                              f"({100*have/total:.1f}%) {mb/max(time.time()-t0,1e-9):.1f} MB/s",
                              flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"    retry {attempt+1}: {type(exc).__name__} {exc}", flush=True)
            time.sleep(5)
    if have < total:
        raise IOError(f"incomplete download {have}/{total} for {url}")
    os.replace(tmp, dest)
    return dest


def parse_trace_name(name: str):
    """'bucket1$0,:3,:30001' -> ('bucket1', 0, 3, 30001)；非 bucket 形式返回 (name, None…)"""
    if "$" not in name:
        return name, None, None, None
    base, _, loc = name.partition("$")
    parts = loc.split(",")
    idx = int(parts[0])
    dims = []
    for p in parts[1:]:
        p = p.strip()
        if p.startswith(":"):
            dims.append(int(p[1:]))
        else:
            dims.append(int(p))
    ncomp = dims[0] if dims else 3
    npts = dims[1] if len(dims) > 1 else None
    return base, idx, ncomp, npts


def first_arrival(row, phase: str) -> float:
    """按 直达 > 区域(g) > 折射(n) 的优先级取到时。"""
    for col in (f"trace_{phase}_arrival_sample",
                f"trace_{phase.lower()}_arrival_sample",
                f"trace_{phase}g_arrival_sample",
                f"trace_{phase}n_arrival_sample"):
        if col in row.index:
            v = pd.to_numeric(row[col], errors="coerce")
            if np.isfinite(v) and v >= 0:
                return float(v)
    return float("nan")


def crop(wave: np.ndarray, p: int, s: int, rng) -> tuple | None:
    n = wave.shape[1]
    if n < WIN:
        out = np.zeros((wave.shape[0], WIN), dtype="float32")
        out[:, :n] = wave
        return out, p, s
    start = None
    if p >= 0 and s >= 0 and (s - p) <= WIN - 2 * MARGIN:
        lo = max(0, s - WIN + MARGIN)
        hi = min(p - MARGIN, n - WIN)
        if lo <= hi:
            start = int(rng.randint(lo, hi + 1))
    if start is None:
        return None
    w = wave[:, start:start + WIN]
    pp, ss = p - start, s - start
    if not (0 <= pp < WIN and 0 <= ss < WIN):
        return None
    return w, pp, ss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="crew")
    ap.add_argument("--chunks", default="000")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--sp-min-s", type=float, default=10.0)
    ap.add_argument("--sp-max-s", type=float, default=50.0)
    ap.add_argument("--dev-frac", type=float, default=0.08)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-raw", action="store_true",
                    help="处理完不删除下载的 waveforms 分块（默认删，省磁盘）")
    args = ap.parse_args()

    ds = args.dataset.lower()
    cache = args.cache_dir or os.path.join(args.out_dir, "raw")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    # 空 --chunks 表示单文件数据集（metadata.csv / waveforms.hdf5），保留一个空 suffix
    chunks = [c.strip() for c in args.chunks.split(",")] if args.chunks.strip() else [""]

    tag = f"{ds}_sp{int(args.sp_min_s)}_{int(args.sp_max_s)}" + ("_" + "_".join(c for c in chunks if c) if any(chunks) else "")
    paths = {k: os.path.join(args.out_dir, f"{tag}_{k}.hdf5") for k in ("train", "dev")}
    fh = {k: h5py.File(v + ".part", "w") for k, v in paths.items()}
    grp = {k: f.create_group("data") for k, f in fh.items()}
    stat = {"train": 0, "dev": 0}
    skip = {"no_ps": 0, "sp_range": 0, "crop": 0, "read": 0, "nan": 0}
    sp_hist: list[float] = []

    for c in chunks:
        # 非分块数据集（ETHZ/GEOFON 等）用 --chunks "" 表示单文件 metadata.csv
        suffix = c
        md_url = f"{REMOTE_ROOT}/{ds}/metadata{suffix}.csv"
        wf_url = f"{REMOTE_ROOT}/{ds}/waveforms{suffix}.hdf5"
        print(f"[chunk {c}] metadata {md_url}", flush=True)
        md = fetch_csv(md_url)
        sr = pd.to_numeric(md.get("trace_sampling_rate_hz",
                                  pd.Series(100.0, index=md.index)),
                           errors="coerce").fillna(100.0)
        P = md.apply(lambda r: first_arrival(r, "P"), axis=1)
        S = md.apply(lambda r: first_arrival(r, "S"), axis=1)
        sp_s = (S - P) / sr
        keep = np.isfinite(sp_s) & (sp_s >= args.sp_min_s) & (sp_s <= args.sp_max_s)
        sel = md.index[keep]
        print(f"[chunk {c}] rows={len(md)} in S-P[{args.sp_min_s},{args.sp_max_s}] = {len(sel)}",
              flush=True)
        if args.limit:
            sel = sel[: args.limit]
        if len(sel) == 0:
            continue

        local = os.path.join(cache, f"{ds}_waveforms{suffix}.hdf5")
        print(f"[chunk {c}] download {wf_url}", flush=True)
        download(wf_url, local)
        print(f"[chunk {c}] downloaded {os.path.getsize(local)/2**20:.0f} MB", flush=True)

        ev_col = next((x for x in ("source_id", "source_event_id", "event_id",
                                   "source_origin_time") if x in md.columns), None)
        with h5py.File(local, "r") as f:
            data = f["data"]
            # 按 bucket 分组，避免反复打开同一大数组
            by_bucket: dict[str, list] = {}
            for i in sel:
                base, idx, ncomp, npts = parse_trace_name(str(md.at[i, "trace_name"]))
                by_bucket.setdefault(base, []).append((i, idx, ncomp, npts))
            for base, items in by_bucket.items():
                if base not in data:
                    skip["read"] += len(items)
                    continue
                dset = data[base]
                for (i, idx, ncomp, npts) in items:
                    try:
                        if idx is None:
                            raw = dset[:]
                        else:
                            raw = dset[idx, :3, :npts] if npts else dset[idx, :3, :]
                    except Exception:  # noqa: BLE001
                        skip["read"] += 1
                        continue
                    raw = np.asarray(raw, dtype="float64")
                    if raw.ndim != 2 or raw.shape[0] < 3:
                        skip["read"] += 1
                        continue
                    if not np.isfinite(raw).all():
                        raw = np.nan_to_num(raw, copy=False)
                    sr_in = float(sr.iloc[i])
                    p_in, s_in = float(P.iloc[i]), float(S.iloc[i])
                    if sr_in != POOL_SR:
                        from fractions import Fraction
                        fr = Fraction(POOL_SR / sr_in).limit_denominator(1000)
                        raw = resample_poly(raw, fr.numerator, fr.denominator, axis=-1)
                        scale = POOL_SR / sr_in
                    else:
                        scale = 1.0
                    wave = raw[:3].astype("float32")
                    p = int(round(p_in * scale))
                    s = int(round(s_in * scale))
                    got = crop(wave, p, s, rng)
                    if got is None:
                        skip["crop"] += 1
                        continue
                    w, pp, ss = got
                    if not np.isfinite(w).all():
                        skip["nan"] += 1
                        continue
                    ev = str(md.at[i, ev_col]) if ev_col else f"row{i}"
                    bucket = "dev" if stable_hash01(ev) < args.dev_frac else "train"
                    name = f"{ds}{suffix}_{i}"
                    if name in grp[bucket]:
                        continue
                    d = grp[bucket].create_dataset(name, data=w, compression="gzip",
                                                   compression_opts=4)
                    d.attrs["p_sample_100hz"] = pp
                    d.attrs["s_sample_100hz"] = ss
                    d.attrs["sampling_rate"] = POOL_SR
                    d.attrs["sp_s"] = float((ss - pp) / POOL_SR)
                    d.attrs["source_event"] = ev
                    stat[bucket] += 1
                    sp_hist.append(float((ss - pp) / POOL_SR))
                print(f"    bucket {base}: cum train={stat['train']} dev={stat['dev']}",
                      flush=True)
        if not args.keep_raw:
            os.remove(local)
            print(f"[chunk {c}] removed raw to save disk", flush=True)

    for k, f in fh.items():
        f.close()
        os.replace(paths[k] + ".part", paths[k])
    arr = np.array(sp_hist) if sp_hist else np.zeros(0)
    print(f"\n[pool] train={stat['train']} dev={stat['dev']} skip={skip}")
    if len(arr):
        print(f"[pool] S-P s: median={np.median(arr):.1f} p10={np.percentile(arr,10):.1f} "
              f"p90={np.percentile(arr,90):.1f} min={arr.min():.1f} max={arr.max():.1f}")
    for k, v in paths.items():
        if os.path.exists(v):
            print(f"[pool] {v} {os.path.getsize(v)/2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())