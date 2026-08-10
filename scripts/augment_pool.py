#!/usr/bin/env python3
"""真题域内数据增强：不引入任何外域分布，只放大已有的 582 窗。

为什么必须走这条路（2026-08-06）：
- 域内数据上限已确认 = 582 窗（train 侧 548 文件对应 582 个 P，提取率 100%）
- 公开数据集十连负（CEED/INSTANCE 噪声/真题域微调），统一失效模式是多报激增，
  数量罚 -0.5/个 吃掉召回收益 —— 根因是外域分布把模型推离中国域
- 增强的关键性质：**新样本仍在真题域内**，不带任何加州/意大利分布

四种增强，每种都保物理一致性：
1. time_shift  窗内平移到时（±N 秒）—— 教会模型到时可以在窗内任意位置，
                不改波形本身
2. amp_scale   整体幅度缩放（×0.3~×3）—— 模拟不同震级/震中距的幅度差，
                PhaseNet 输入本就逐窗归一化, 主要是抗归一化残差
3. polarity    三分量同时取反 —— P 初动极性对拾取不应敏感（拾取≠极性判定）
4. add_noise   叠加**真题自身静默段**噪声到指定 SNR —— 直接对着诊断出的
                133 个多报, 且噪声本域, 不像 INSTANCE 意大利噪声那样带外域特征

刻意不做的两类（会破坏物理一致性）：
- 时间伸缩(time stretch)：改变 P-S 时差 = 伪造不同震中距，但波形形态不跟着变
- 频域滤波：改变仪器响应特征，而仪器类型正是本届赛题的关键变量（含强震仪）

用法:
    python scripts/augment_pool.py --in exam_r2_train_50hz.hdf5 \\
        --out aug.hdf5 --factor 6 --ops time_shift,amp_scale,polarity,add_noise
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def load_windows(path: str) -> list[tuple[str, np.ndarray, int, int, float]]:
    """读出 (名, 波形(3,N), p下标, s下标, 采样率)。"""
    import h5py

    out = []
    with h5py.File(path, "r") as f:
        for k in f["data"]:
            d = f["data"][k]
            a = d.attrs
            out.append((
                k, np.asarray(d[:], dtype=np.float32),
                int(a.get("p_sample_100hz", -1)),
                int(a.get("s_sample_100hz", -1)),
                float(a.get("sampling_rate", 50.0)),
            ))
    return out


def noise_bank(wins, sr: float, margin_s: float = 3.0) -> list[np.ndarray]:
    """从真题窗的事件前静默段收集噪声片段（P 到时前 margin 秒以外）。"""
    bank = []
    for _, w, p, _s, _ in wins:
        if p < 0:
            bank.append(w)          # 纯噪声窗整段可用
            continue
        end = int(p - margin_s * sr)
        if end > int(sr * 2):       # 至少 2 秒才有意义
            bank.append(w[:, :end])
    return bank


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def aug_time_shift(w, p, s, sr, rng):
    """窗内平移：把整段循环移位，到时下标同步跟着走。"""
    n = w.shape[1]
    # 移位量控制在 ±8 秒，且保证 p/s 不被移出窗
    lim = int(8.0 * sr)
    lo, hi = -lim, lim
    for idx in (p, s):
        if idx >= 0:
            lo = max(lo, -(idx - int(0.5 * sr)))
            hi = min(hi, (n - 1 - idx) - int(0.5 * sr))
    if hi <= lo:
        return None
    sh = int(rng.integers(lo, hi + 1))
    if sh == 0:
        return None
    out = np.zeros_like(w)
    if sh > 0:
        out[:, sh:] = w[:, :n - sh]
        out[:, :sh] = w[:, :sh][:, ::-1] * 0.1   # 边缘用镜像衰减填，避免硬阶跃
    else:
        k = -sh
        out[:, :n - k] = w[:, k:]
        out[:, n - k:] = w[:, n - k:][:, ::-1] * 0.1
    np_ = p + sh if p >= 0 else -1
    ns = s + sh if s >= 0 else -1
    return out, np_, ns


def aug_amp(w, p, s, sr, rng):
    g = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
    return w * g, p, s


def aug_polarity(w, p, s, sr, rng):
    return -w, p, s


def make_add_noise(bank):
    def aug_add_noise(w, p, s, sr, rng):
        if not bank:
            return None
        seg = bank[int(rng.integers(0, len(bank)))]
        n = w.shape[1]
        if seg.shape[1] < n:
            reps = int(np.ceil(n / seg.shape[1]))
            seg = np.tile(seg, (1, reps))
        st = int(rng.integers(0, seg.shape[1] - n + 1))
        nz = seg[:, st:st + n].astype(np.float32)
        # 按目标 SNR 定噪声增益（SNR 越低越难，覆盖弱信号场景）
        snr_db = float(rng.uniform(3.0, 18.0))
        want = rms(w) / (10.0 ** (snr_db / 20.0))
        nz = nz * (want / rms(nz))
        return w + nz, p, s
    return aug_add_noise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=int, default=6,
                    help="每条原窗生成多少条增强窗")
    ap.add_argument("--ops", default="time_shift,amp_scale,polarity,add_noise")
    ap.add_argument("--keep-original", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()

    import h5py

    rng = np.random.default_rng(args.seed)
    wins = load_windows(args.inp)
    print(f"原池 {len(wins)} 窗")
    sr = wins[0][4] if wins else 50.0
    bank = noise_bank(wins, sr)
    print(f"噪声库 {len(bank)} 段（来自真题自身静默段）")

    registry = {
        "time_shift": aug_time_shift,
        "amp_scale": aug_amp,
        "polarity": aug_polarity,
        "add_noise": make_add_noise(bank),
    }
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    for o in ops:
        if o not in registry:
            raise SystemExit(f"未知增强 {o}，可选 {list(registry)}")
    print(f"启用增强: {ops}")

    tmp = args.out + ".part"
    stats = {o: 0 for o in ops}
    n_out = 0
    with h5py.File(tmp, "w") as f:
        g = f.create_group("data")

        def put(name, w, p, s):
            nonlocal n_out
            d = g.create_dataset(name, data=w.astype(np.float32),
                                 compression="gzip", compression_opts=1)
            d.attrs["p_sample_100hz"] = int(p)
            d.attrs["s_sample_100hz"] = int(s)
            d.attrs["sampling_rate"] = float(sr)
            n_out += 1

        for k, w, p, s, _ in wins:
            if args.keep_original:
                put(f"orig__{k}", w, p, s)
            for j in range(args.factor):
                # 每条增强窗随机叠 1~2 种算子（组合更多样，但别叠太狠）
                pick = list(rng.choice(ops, size=int(rng.integers(1, 3)),
                                       replace=False))
                cw, cp, cs = w, p, s
                applied = []
                for o in pick:
                    r = registry[o](cw, cp, cs, sr, rng)
                    if r is None:
                        continue
                    cw, cp, cs = r
                    applied.append(o)
                    stats[o] += 1
                if not applied:
                    continue
                put(f"aug{j}_{'+'.join(applied)}__{k}", cw, cp, cs)

    os.replace(tmp, args.out)
    print(f"[增强] 输出 {n_out} 窗 -> {args.out} "
          f"({os.path.getsize(args.out)/1e6:.1f} MB)")
    print(f"[增强] 各算子生效次数: {stats}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
