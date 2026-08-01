#!/usr/bin/env python3
"""把官方真题包(zip)转成微调训练用的 HDF5 窗口池.

===== 为什么存在 =====
平台迟迟未发样例数据, 而手里最接近今年分布的中国域标注数据就是去年两轮真题
(1915 文件, 数千 P/S 标注)。跨轮协议: 只拿第 1 轮训练、第 2 轮整轮当从未见过的
闸门——跨轮涨分 = 能泛化到"下一次出题"的最强证据。本脚本负责 zip → HDF5,
输出与 chunked_fetch.py 完全同构(group 'data', attrs sampling_rate /
p_sample_100hz / s_sample_100hz——历史名, 实为 --sr 采样率下的窗内下标),
finetune_phasenet.py --streaming 可直接吃。

===== 多事件配对 =====
第 2 轮单文件可含多组 P/S(最多 53 P)。配对规则: P 升序遍历, 每个 P 取
(本 P, 下一个 P) 区间内的第一个 S 为同事件 S; 配不上的 P 单独成窗(s=-1),
剩余孤儿 S 同理(p=-1)。每对/孤儿相位各切一窗(联合可行区间随机采样,
复用 chunked_fetch.cut_window, margin=diting 盲区 250 点)。

===== 用法 =====
    python scripts/exam_to_hdf5.py --package <官方zip> --out exam_r1_50hz.hdf5
    # 冒烟: --limit 30; 全量约 1-3 分钟(第2轮长文件多)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from phasepicker.io.official_exam import scan_exam_input  # noqa: E402
from phasepicker.io.official_waveforms import read_package_answers  # noqa: E402
from phasepicker.types import ExamTask  # noqa: E402


def pair_events(p_times, s_times):
    """P/S 相对秒列表 → [(p 或 None, s 或 None)] 事件配对(见模块 docstring 规则)。"""
    ps = sorted(float(t) for t in p_times)
    ss = sorted(float(t) for t in s_times)
    used_s = [False] * len(ss)
    pairs = []
    for i, p in enumerate(ps):
        nxt = ps[i + 1] if i + 1 < len(ps) else float("inf")
        mate = None
        for j, s in enumerate(ss):
            if not used_s[j] and p < s < nxt:
                mate = j
                break
        if mate is not None:
            used_s[mate] = True
            pairs.append((p, ss[mate]))
        else:
            pairs.append((p, None))
    for j, s in enumerate(ss):
        if not used_s[j]:
            pairs.append((None, s))
    return pairs


def resample_to(wave: np.ndarray, sr_in: float, sr_out: float) -> np.ndarray:
    """FIR 抗混叠有理重采样(100→50 即 1/2 抽取), 与推理侧 seisbench 重采样同量级。"""
    if abs(sr_in - sr_out) < 1e-6:
        return wave.astype("float32", copy=False)
    from scipy.signal import resample_poly
    from fractions import Fraction

    frac = Fraction(int(round(sr_out * 100)), int(round(sr_in * 100)))
    out = resample_poly(wave.astype("float64"), frac.numerator, frac.denominator, axis=1)
    return out.astype("float32")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="官方真题包 → 微调 HDF5 窗口池")
    ap.add_argument("--package", required=True, help="官方 zip(第1或第2轮)")
    ap.add_argument("--out", required=True, help="输出 hdf5 路径")
    ap.add_argument("--sr", type=float, default=50.0, help="目标采样率(diting=50)")
    ap.add_argument("--win", type=int, default=3001, help="窗长采样点(diting=3001)")
    ap.add_argument("--limit", type=int, default=None, help="只转前 N 个文件(冒烟)")
    ap.add_argument("--seed", type=int, default=20260801)
    args = ap.parse_args(argv)

    import h5py
    from chunked_fetch import cut_window
    from run_official_task1 import _make_load_waveforms_fn

    rng = np.random.RandomState(args.seed)
    samples = [s for s in scan_exam_input(args.package) if s.task == ExamTask.T1]
    if args.limit:
        samples = samples[: args.limit]
    answers = read_package_answers(args.package, ExamTask.T1)
    load_fn = _make_load_waveforms_fn()

    n_win = n_pair = n_p_only = n_s_only = n_skip = 0
    with h5py.File(args.out, "w") as f:
        grp = f.create_group("data")
        for sm in samples:
            ans = answers.get(sm.file_id)
            if ans is None:
                n_skip += 1
                continue
            wfs = load_fn(sm).waveforms if hasattr(load_fn(sm), "waveforms") else load_fn(sm)
            if not wfs:
                n_skip += 1
                continue
            wf = wfs[0]  # 去年真题实测均为单台站
            wave50 = resample_to(np.asarray(wf.data), float(wf.sampling_rate), args.sr)
            for k, (p_s, s_s) in enumerate(pair_events(ans.p_times_s, ans.s_times_s)):
                p_idx = int(round(p_s * args.sr)) if p_s is not None else -1
                s_idx = int(round(s_s * args.sr)) if s_s is not None else -1
                w, p2, s2 = cut_window(wave50, p_idx, s_idx, args.win, rng)
                if w is None or (p2 < 0 and s2 < 0):
                    n_skip += 1
                    continue
                key = f"EXAM_{sm.file_id.replace('.', '_')}_ev{k}"
                d = grp.create_dataset(key, data=np.ascontiguousarray(w, dtype="float32"),
                                       compression="gzip", compression_opts=1)
                d.attrs["sampling_rate"] = float(args.sr)
                d.attrs["p_sample_100hz"] = int(p2)
                d.attrs["s_sample_100hz"] = int(s2)
                d.attrs["source_file"] = sm.file_id
                n_win += 1
                if p2 >= 0 and s2 >= 0:
                    n_pair += 1
                elif p2 >= 0:
                    n_p_only += 1
                else:
                    n_s_only += 1
    print(f"完成: {args.out}")
    print(f"  文件 {len(samples)} -> 窗 {n_win} (P+S {n_pair} / 仅P {n_p_only} / 仅S {n_s_only}), 跳过 {n_skip}")
    print(f"  采样率 {args.sr}Hz, 窗长 {args.win} 点; 键前缀 EXAM_, source_file 属性可溯源")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
