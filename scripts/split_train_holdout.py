#!/usr/bin/env python3
"""把 chunked_fetch 产出的训练池按【事件】切成 train / holdout 两个 HDF5.

按事件（而不是按窗口）切分是防泄漏的关键：同一地震在多台站/多窗口出现，
若随机按窗口切，holdout 里会混进训练见过的事件，分数虚高、上线即露馅。
散列切分（md5(event_id)）保证：跨机器/重跑结果一致；追加新数据后旧样本去向不变。

用法：
    python scripts/split_train_holdout.py --src pool.hdf5 \\
        --train train.hdf5 --holdout holdout.hdf5 --holdout-frac 0.1
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunked_fetch import stable_hash01  # noqa: E402  复用同一散列，口径一致


def split_keys(keys_events: list, holdout_frac: float) -> tuple:
    """(key, event_id) 列表 → (train_keys, holdout_keys)，按事件散列切分。"""
    train, hold = [], []
    for key, ev in keys_events:
        (hold if stable_hash01(str(ev)) < holdout_frac else train).append(key)
    return train, hold


def main() -> int:
    ap = argparse.ArgumentParser(description="按事件切分训练池")
    ap.add_argument("--src", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    args = ap.parse_args()

    import h5py

    with h5py.File(args.src, "r") as src:
        grp = src["data"]
        keys_events = [(k, grp[k].attrs.get("event_id", k)) for k in grp.keys()]
        train_keys, hold_keys = split_keys(keys_events, args.holdout_frac)
        print(f"总窗 {len(keys_events)} → train {len(train_keys)} / holdout {len(hold_keys)}")

        for path, keys in ((args.train, train_keys), (args.holdout, hold_keys)):
            if os.path.exists(path):
                os.remove(path)
            with h5py.File(path, "w") as dst:
                g = dst.require_group("data")
                for k in keys:
                    d = grp[k]
                    nd = g.create_dataset(k, data=d[()], compression="gzip", compression_opts=4)
                    for ak, av in d.attrs.items():
                        nd.attrs[ak] = av
            print(f"  写出 {path}（{len(keys)} 窗，{os.path.getsize(path)/1e9:.2f}GB）")

    # 事件不相交断言（防泄漏的最后一道闸）
    with h5py.File(args.train, "r") as t, h5py.File(args.holdout, "r") as h:
        ev_t = {t["data"][k].attrs.get("event_id", k) for k in t["data"].keys()}
        ev_h = {h["data"][k].attrs.get("event_id", k) for k in h["data"].keys()}
        inter = ev_t & ev_h
        if inter:
            print(f"[严重] train/holdout 事件交集非空（{len(inter)} 个）！", file=sys.stderr)
            return 1
    print("切分完成：train/holdout 事件零交集 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
