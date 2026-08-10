#!/usr/bin/env python3
"""SeisBench 数据集 → 微调窗口池（finetune_phasenet.py 直接可读的格式）。

===== 为什么需要这一层 =====
下载来的是 SeisBench 原生格式（metadata.csv + waveforms.hdf5 分桶存储），而
finetune_phasenet.py 吃的是"group data 下每条一个窗口 + attrs 标注到时"的池格式。
中间有四个**静默出错**的坑，全部在这里拦死：

1. **采样率**：PhaseNet(diting/广西) 原生 50Hz、窗长 3001 点(60.02s)。CEED 是
   100Hz、AQ2009 是 125Hz、MLAAPDE 是 40Hz。训练脚本只校验不重采样，错位不报错
   只会学错时间尺度 → 这里统一重采样到 --sr 并同步换算到时下标。
2. **分量序**：CEED 是 ENZ，PhaseNet 预训练要 ZNE（实测 component_order='ZNE'
   会真重排为 [2,1,0]，已验证）。靠 seisbench 的 component_order 参数做，不手撸。
3. **到时下标基准**：池格式的 attr 名 p_sample_100hz 是历史名，实际语义是
   "**池自身采样率下的窗内下标**"，且 attrs['sampling_rate'] 会被训练脚本校验。
4. **噪声窗**：p=s=-1 表示纯噪声，训练侧生成全零软标签当负样本（已核对
   make_soft_label：p_sample>=0 才生成高斯）。噪声集必须走这条路，不能丢。

===== 用法 =====
    # CEED 某几块 → 训练池 + dev 池（dev 用官方 split，做提升闸）
    python scripts/prepare_pool.py --dataset CEED --chunks nc1987,nc1988 \\
        --cache /mnt/vol1/sbcache --out-dir /mnt/vol4/pool --sr 50

    # 噪声集（无到时）
    python scripts/prepare_pool.py --dataset InstanceNoise \\
        --cache /mnt/vol1/sbcache --out-dir /mnt/vol4/pool --noise-only
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_MARGIN = 250  # diting 首尾盲区（点），窗口裁剪时避开


def _resample(wave: np.ndarray, sr_in: float, sr_out: float) -> np.ndarray:
    """带抗混叠的重采样。sr 相同则原样返回。"""
    if abs(sr_in - sr_out) < 1e-6:
        return wave
    from fractions import Fraction

    from scipy.signal import resample_poly

    fr = Fraction(sr_out / sr_in).limit_denominator(1000)
    return resample_poly(wave, fr.numerator, fr.denominator, axis=-1
                         ).astype("float32")


def _crop(wave: np.ndarray, p: int, s: int, win: int,
          rng: np.random.RandomState) -> tuple[np.ndarray, int, int] | None:
    """裁到 win 点，尽量 P/S 同窗且避开首尾盲区。返回 None 表示这条不可用。

    与 finetune_phasenet._fit_window 同策略（那边是兜底，这里提前做掉，避免
    池里存 12000 点全长白占 4 倍磁盘）。
    """
    n = wave.shape[1]
    if n < win:
        out = np.zeros((wave.shape[0], win), dtype="float32")
        out[:, :n] = wave
        return out, p, s
    start = None
    if p >= 0 and s >= 0 and (s - p) <= win - 2 * _MARGIN:
        lo = max(0, s - win + _MARGIN)
        hi = min(p - _MARGIN, n - win)
        if lo <= hi:
            start = int(rng.randint(lo, hi + 1))
    if start is None:
        center = p if p >= 0 else (s if s >= 0 else n // 2)
        if center < 0:  # 纯噪声：随机取一窗
            start = int(rng.randint(0, n - win + 1))
        else:
            off = int(rng.randint(int(win * 0.15), int(win * 0.55) + 1))
            start = int(np.clip(center - off, 0, n - win))
    w = wave[:, start:start + win]
    np_, ns_ = p - start, s - start
    if np_ < 0 or np_ >= win:
        np_ = -1
    if ns_ < 0 or ns_ >= win:
        ns_ = -1
    if p >= 0 and np_ < 0 and (s < 0 or ns_ < 0):
        return None  # 有标注却全被裁掉 = 假噪声窗，宁可丢弃
    return w, np_, ns_


def _pick_sample(row, key: str) -> int:
    """取到时样点，缺失/NaN → -1。

    列名大小写与相位变体都要兼容——各数据集命名不统一，这是**静默出错**的位置：
    - SeisBench 多数集: trace_p_arrival_sample / trace_s_arrival_sample
    - CREW:            trace_P_arrival_sample（大写！）+ Pg/Pn、Sg/Sn 变体
    若只按小写精确匹配，CREW 每行都会拿到 -1 → 整池退化成噪声窗，
    训练照跑不报错，最后白训一轮才发现（2026-08-06 踩到，已加此兜底）。

    变体优先级：直达相位 (P/S) > 区域相位 (Pg/Sg) > 折射相位 (Pn/Sn)。
    区域震常只标 Pg/Pn 而无 P，丢掉它们等于丢掉大半标注。
    """
    phase = "P" if "_p_" in key.lower() else "S"
    cands = [key]                                  # 原始（小写惯例）
    cands.append(key.replace(f"_{phase.lower()}_", f"_{phase}_"))  # 大写主相位
    for suf in ("g", "n"):                         # Pg/Pn、Sg/Sn
        cands.append(f"trace_{phase}{suf}_arrival_sample")
        cands.append(f"trace_{phase.lower()}{suf}_arrival_sample")
    # 最后兜底：忽略大小写扫一遍现有键
    lower_map = {str(k).lower(): k for k in getattr(row, "index", row.keys())}

    def _val(name):
        if name in row:
            return row.get(name)
        real = lower_map.get(name.lower())
        return row.get(real) if real is not None else None

    for name in cands:
        v = _val(name)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f) and f >= 0:
            return int(round(f))
    return -1


def _row_sr(row, default: float) -> float:
    for k in ("trace_sampling_rate_hz", "trace_sampling_rate"):
        v = row.get(k, None)
        try:
            f = float(v)
            if np.isfinite(f) and f > 0:
                return f
        except (TypeError, ValueError):
            pass
    dt = row.get("trace_dt_s", None)
    try:
        f = float(dt)
        if np.isfinite(f) and f > 0:
            return 1.0 / f
    except (TypeError, ValueError):
        pass
    return default


def convert(args) -> int:
    # 必须在 import seisbench 之前设好缓存根：seisbench 在 import 时就读这个值。
    # 顺序写反过一次，后果是无视本地已下好的分块、重新去 DESY 下一遍。
    os.environ["SEISBENCH_CACHE_ROOT"] = args.cache
    import h5py
    import seisbench
    import seisbench.data as sbd
    from pathlib import Path
    seisbench.cache_root = Path(args.cache)  # 双保险
    if str(seisbench.cache_root) != args.cache:
        print(f"[池] !! cache_root 未生效: {seisbench.cache_root}")

    cls = getattr(sbd, args.dataset, None)
    if cls is None:
        raise SystemExit(f"seisbench.data 里没有 {args.dataset}")
    kw = {"lazyload": False, "cache": None, "component_order": "ZNE"}
    if args.chunks:
        kw["chunks"] = [c.strip() for c in args.chunks.split(",") if c.strip()]
    print(f"[池] 打开 {args.dataset} chunks={kw.get('chunks','(全部)')}")
    ds = cls(**kw)
    md = ds.metadata
    print(f"[池] {len(ds)} 条; 列 {len(md.columns)}")

    rng = np.random.RandomState(args.seed)
    # 官方 split 列可用则据此分流；否则全进 train，dev 由 --dev-frac 随机切
    has_split = "split" in md.columns
    if has_split:
        print("[池] 使用数据集自带 split:",
              md["split"].value_counts().to_dict())

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or (args.chunks.replace(",", "_") if args.chunks
                       else args.dataset.lower())
    paths = {
        "train": os.path.join(args.out_dir, f"{args.dataset.lower()}_{tag}_train.hdf5"),
        "dev": os.path.join(args.out_dir, f"{args.dataset.lower()}_{tag}_dev.hdf5"),
    }
    if all(os.path.exists(p) for p in paths.values()) and not args.force:
        print(f"[池] 已存在且未加 --force，跳过: {tag}")
        return 0

    # 先写临时名，全部成功后再 rename —— 训练侧在扫池，绝不能看到半成品
    tmp = {k: v + ".part" for k, v in paths.items()}
    fh = {k: h5py.File(v, "w") for k, v in tmp.items()}
    grp = {k: f.create_group("data") for k, f in fh.items()}
    stat = {"train": 0, "dev": 0}
    skipped = {"no_pick": 0, "crop_fail": 0, "read_fail": 0, "short": 0}
    t0 = time.time()
    n = len(ds)
    for i in range(n):
        row = md.iloc[i]
        try:
            wave = ds.get_waveforms(i)
        except Exception:  # noqa: BLE001 - 单条坏数据不该中断整块
            skipped["read_fail"] += 1
            continue
        if wave is None or wave.ndim != 2 or wave.shape[0] < 3:
            skipped["read_fail"] += 1
            continue
        wave = np.nan_to_num(wave[:3].astype("float32"), copy=False)
        sr_in = _row_sr(row, args.assume_sr)
        p = -1 if args.noise_only else _pick_sample(row, "trace_p_arrival_sample")
        s = -1 if args.noise_only else _pick_sample(row, "trace_s_arrival_sample")
        if not args.noise_only and p < 0 and s < 0:
            skipped["no_pick"] += 1
            continue
        wave = _resample(wave, sr_in, args.sr)
        scale = args.sr / sr_in
        if p >= 0:
            p = int(round(p * scale))
        if s >= 0:
            s = int(round(s * scale))
        if wave.shape[1] < args.min_len:
            skipped["short"] += 1
            continue
        got = _crop(wave, p, s, args.win, rng)
        if got is None:
            skipped["crop_fail"] += 1
            continue
        w, pp, ss = got

        if has_split:
            sp = str(row.get("split", "train")).lower()
            bucket = "dev" if sp in ("dev", "test", "val") else "train"
        else:
            bucket = "dev" if rng.rand() < args.dev_frac else "train"
        d = grp[bucket].create_dataset(
            f"{tag}_{i}", data=w, compression="gzip", compression_opts=1)
        # attr 名是历史名，语义 = 本池采样率下的窗内下标（见文件头第 3 条）
        d.attrs["p_sample_100hz"] = pp
        d.attrs["s_sample_100hz"] = ss
        d.attrs["sampling_rate"] = float(args.sr)
        d.attrs["src"] = f"{args.dataset}:{tag}:{i}"
        stat[bucket] += 1
        if args.limit and (stat["train"] + stat["dev"]) >= args.limit:
            break
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n} 已写 train={stat['train']} dev={stat['dev']} "
                  f"({(i+1)/max(el,1e-6):.0f} 条/s)", flush=True)

    for f in fh.values():
        f.close()
    for k, src in tmp.items():
        os.replace(src, paths[k])
    el = time.time() - t0
    print(f"[池] 完成 {tag}: train={stat['train']} dev={stat['dev']} "
          f"用时 {el:.1f}s")
    print(f"[池] 丢弃统计 {skipped}")
    for k, v in paths.items():
        if os.path.exists(v):
            print(f"  {k}: {v} ({os.path.getsize(v)/1e6:.1f} MB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="seisbench.data 类名，如 CEED / InstanceNoise / ETHZ")
    ap.add_argument("--chunks", default="",
                    help="逗号分隔的 chunk 名（CEED 用，如 nc1987,nc1988）")
    ap.add_argument("--cache", required=True, help="SeisBench 缓存根目录")
    ap.add_argument("--out-dir", required=True, help="池输出目录")
    ap.add_argument("--tag", default="", help="产物文件名标签，默认取 chunks")
    ap.add_argument("--sr", type=float, default=50.0,
                    help="目标采样率；必须与训练 --sr 一致（diting 系=50）")
    ap.add_argument("--win", type=int, default=3001, help="窗长（点）")
    ap.add_argument("--assume-sr", type=float, default=100.0,
                    help="元数据缺采样率时的假定值")
    ap.add_argument("--min-len", type=int, default=1000,
                    help="重采样后短于此点数的丢弃")
    ap.add_argument("--noise-only", action="store_true",
                    help="纯噪声集：忽略到时列，全部标为 p=s=-1")
    ap.add_argument("--dev-frac", type=float, default=0.05,
                    help="数据集无 split 列时的 dev 随机比例")
    ap.add_argument("--limit", type=int, default=0, help="最多写多少条(0=全部)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="已存在也重建")
    args = ap.parse_args()
    return convert(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
