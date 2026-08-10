#!/usr/bin/env python3
"""按指定比例混合多个训练池分片（信号 + 噪声负样本）。

为什么不用 merge_pool.py：那个把目录里所有 *_train.hdf5 无差别并进来。这里
需要定向控制——例如"真题信号全要 + 噪声按 3:1 配比 + 已证伪的 CEED 一律排除"。

噪声配比为什么重要：噪声窗（p=s=-1）在训练侧生成全零软标签当负样本，是压
假阳性的唯一机理。但配比过高会把模型推向"什么都不报"，反而丢召回；过低则
没效果。所以做成参数，可扫。

用 h5py.ExternalLink 零拷贝引用源分片（源文件必须保持原位）。

用法:
    python scripts/mix_pool.py --out mixed.hdf5 \\
        --signal exam_r2_train_50hz.hdf5 --noise instancenoise_noise12k_train.hdf5 \\
        --noise-ratio 3.0
"""
from __future__ import annotations

import argparse
import os
import random
import sys


def keys_of(path: str) -> list[str]:
    import h5py

    with h5py.File(path, "r") as f:
        if "data" not in f:
            return []
        return list(f["data"].keys())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--signal", nargs="+", required=True,
                    help="信号池分片（全量取用）")
    ap.add_argument("--noise", nargs="*", default=[],
                    help="噪声池分片（按 --noise-ratio 抽样）")
    ap.add_argument("--noise-ratio", type=float, default=1.0,
                    help="噪声窗数 / 信号窗数。0=不加噪声")
    ap.add_argument("--signal-limit", type=int, default=0,
                    help="信号窗上限（0=全要）")
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--upsample", nargs="*", default=[],
                    metavar="PATH:N",
                    help="上采样池：PATH:N 表示该池的窗重复 N 次写入。"
                         "用于在大池里维持稀有域(如真题域)的占比——"
                         "2026-08-06 观察: 真题锚点占比从 3.1%(小池) 掉到 "
                         "1.2%(大池) 时提升消失，此参数用于验证该假设")
    args = ap.parse_args()

    import h5py

    rng = random.Random(args.seed)

    sig: list[tuple[str, str]] = []
    for p in args.signal:
        ks = keys_of(p)
        sig += [(p, k) for k in ks]
        print(f"  信号 {os.path.basename(p)}: {len(ks)} 窗")
    if args.signal_limit and len(sig) > args.signal_limit:
        rng.shuffle(sig)
        sig = sig[: args.signal_limit]
        print(f"  信号截断到 {len(sig)} 窗")

    noise: list[tuple[str, str]] = []
    for p in args.noise:
        ks = keys_of(p)
        noise += [(p, k) for k in ks]
        print(f"  噪声 {os.path.basename(p)}: {len(ks)} 窗")
    want = int(len(sig) * args.noise_ratio)
    if noise and want < len(noise):
        rng.shuffle(noise)
        noise = noise[:want]
    print(f"  噪声抽样 {len(noise)} 窗（目标比 {args.noise_ratio}）")

    # 上采样池：在 signal-limit 截断之后加入，所以不会被截掉——
    # 这正是它存在的意义（保证稀有域在大池里仍占足够比例）。
    ups: list[tuple[str, str, int]] = []
    for spec in args.upsample:
        path, _, n_str = spec.rpartition(":")
        if not path or not n_str.isdigit():
            raise SystemExit(f"--upsample 需要 PATH:N 形式，收到 {spec!r}")
        reps = int(n_str)
        ks = keys_of(path)
        for rep in range(reps):
            ups += [(path, k, rep) for k in ks]
        print(f"  上采样 {os.path.basename(path)}: {len(ks)} 窗 × {reps} 次 "
              f"= {len(ks) * reps} 窗")

    tmp = args.out + ".part"
    n = 0
    with h5py.File(tmp, "w") as f:
        g = f.create_group("data")
        for tag, items in (("SIG", sig), ("NOI", noise)):
            for src, k in items:
                base = os.path.splitext(os.path.basename(src))[0][:24]
                name = f"{tag}_{base}__{k}"
                if name in g:
                    continue
                g[name] = h5py.ExternalLink(src, f"/data/{k}")
                n += 1
        # 上采样条目名字带副本号，否则同名会被跳过、上采样静默失效
        for src, k, rep in ups:
            base = os.path.splitext(os.path.basename(src))[0][:24]
            name = f"UPS{rep}_{base}__{k}"
            if name in g:
                continue
            g[name] = h5py.ExternalLink(src, f"/data/{k}")
            n += 1
    os.replace(tmp, args.out)
    ratio = len(noise) / max(len(sig), 1)
    msg = (f"[混合] 共 {n} 窗 = 信号 {len(sig)} + 噪声 {len(noise)} "
           f"(实际比 {ratio:.2f})")
    if ups:
        msg += f" + 上采样 {len(ups)} (占比 {100.0 * len(ups) / max(n, 1):.1f}%)"
    print(msg + f" -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
