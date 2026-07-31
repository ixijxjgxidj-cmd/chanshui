#!/usr/bin/env python3
"""归档读取微基准：量化 read_source_bytes 缓存 vs 无缓存的差距——纯标准库.

合成一个"第2轮风格"的嵌套包（外层 zip → 内层 data zip → N 条 mseed），
分别用无缓存路径与缓存路径把全部条目读一遍，打印耗时与加速比。
mseed 用随机字节模拟（压缩率低，接近真实波形数据的解压代价）。

用法：
    python scripts/bench_io.py                     # 默认 200 条 × 256KB
    python scripts/bench_io.py --n 500 --kb 330    # 模拟更大的官方包
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import time
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.io.official_waveforms import (  # noqa: E402
    _read_source_bytes_nocache,
    clear_archive_cache,
    read_source_bytes,
)


def build_package(tmp: str, n: int, kb: int) -> tuple[str, list[str]]:
    rnd = os.urandom(kb * 1024)  # 随机字节：压缩不掉，逼近真实波形解压代价
    inner = io.BytesIO()
    names = []
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(n):
            name = f"exam/T1-Q/T1.A.Q{i:04d}.mseed"
            names.append(name)
            zf.writestr(name, rnd)
    outer = os.path.join(tmp, "round2_bench.zip")
    with zipfile.ZipFile(outer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("exam-data.zip", inner.getvalue())
    return outer, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="mseed 条目数")
    ap.add_argument("--kb", type=int, default=256, help="每条 mseed 大小 KB")
    ap.add_argument("--nocache-cap", type=int, default=30,
                    help="无缓存路径太慢，只测前 K 条后线性外推")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        outer, names = build_package(tmp, args.n, args.kb)
        inner_mb = os.path.getsize(outer) / 1e6
        print(f"合成包：{args.n} 条 × {args.kb}KB，外层 zip {inner_mb:.1f}MB")

        srcs = [f"{outer}!exam-data.zip!{n}" for n in names]

        cap = min(args.nocache_cap, len(srcs))
        t0 = time.perf_counter()
        for s in srcs[:cap]:
            _read_source_bytes_nocache(s)
        t_no = (time.perf_counter() - t0) / cap * len(srcs)

        clear_archive_cache()
        t0 = time.perf_counter()
        for s in srcs:
            read_source_bytes(s)
        t_yes = time.perf_counter() - t0
        clear_archive_cache()

        print(f"无缓存（每条全量重解压内层 zip，前 {cap} 条实测线性外推）："
              f"{t_no:.2f}s  ≈ {len(srcs)/max(t_no,1e-9):.1f} 条/秒")
        print(f"缓存后（内层 zip 只解压一次 + O(1) 条目索引）："
              f"{t_yes:.2f}s  ≈ {len(srcs)/max(t_yes,1e-9):.1f} 条/秒")
        print(f"加速比：×{t_no/max(t_yes,1e-9):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
