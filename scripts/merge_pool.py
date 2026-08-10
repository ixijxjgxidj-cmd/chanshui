#!/usr/bin/env python3
"""把池目录下的分片合并成一份 train / 一份 dev（训练脚本吃单文件）。

为什么用软链而不是拷贝内容：分片会越来越多（CEED 65 块），每轮训练都全量
复制一遍磁盘和时间都浪费。h5py 的 ExternalLink 可以让一个 hdf5 引用另一个
文件里的数据集，读起来跟本地一样，且零拷贝。

注意：ExternalLink 要求被引用文件在读取时仍在原路径——池目录是常驻的，成立。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys


def build(out: str, shards: list[str], kind: str) -> int:
    import h5py

    tmp = out + ".part"
    n = 0
    with h5py.File(tmp, "w") as f:
        g = f.create_group("data")
        for sp in shards:
            base = os.path.splitext(os.path.basename(sp))[0]
            try:
                with h5py.File(sp, "r") as src:
                    if "data" not in src:
                        continue
                    keys = list(src["data"].keys())
            except Exception as exc:  # noqa: BLE001 - 坏分片跳过, 不拖垮整轮
                print(f"  !! 跳过坏分片 {base}: {type(exc).__name__}")
                continue
            for k in keys:
                # 外部链接：/data/<分片>_<窗口名> -> 分片文件里的 /data/<窗口名>
                g[f"{base}__{k}"] = h5py.ExternalLink(sp, f"/data/{k}")
                n += 1
    os.replace(tmp, out)
    print(f"[合并] {kind}: {n} 窗口 ← {len(shards)} 分片 → {out}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--out", required=True, help="合并后的 train hdf5")
    ap.add_argument("--dev-out", required=True, help="合并后的 dev hdf5")
    args = ap.parse_args()

    def pick(suffix: str) -> list[str]:
        got = sorted(glob.glob(os.path.join(args.pool_dir, f"*_{suffix}.hdf5")))
        # 排除自己的产物，否则会自引用套娃
        bad = {os.path.abspath(args.out), os.path.abspath(args.dev_out)}
        return [p for p in got if os.path.abspath(p) not in bad
                and not os.path.basename(p).startswith("_merged")]

    train = pick("train")
    dev = pick("dev")
    if not train:
        print("[合并] 池里没有 *_train.hdf5，什么都没做")
        return 1
    n_tr = build(args.out, train, "train")
    n_dev = build(args.dev_out, dev, "dev") if dev else 0
    if n_dev == 0:
        print("[合并] !! dev 池为空 —— 训练将只能用合成分守门，"
              "提升闸失效。检查数据集是否带 split 列。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
