"""T2 输入通道统一适配：所有来源最终转换为 E,N,Z。"""
from __future__ import annotations

import numpy as np


def to_enz_from_obspy(stream):
    """从 ObsPy Stream 读取三分量并返回 `(E,N,Z), sampling_rate`。"""
    channels = {tr.stats.channel[-1].upper(): tr for tr in stream}
    if not all(c in channels for c in ("E", "N", "Z")):
        raise ValueError("需要 E/N/Z 三分量")
    rates = [float(channels[c].stats.sampling_rate) for c in ("E", "N", "Z")]
    if max(rates) - min(rates) > 1e-6:
        raise ValueError(f"三分量采样率不一致: {rates}")
    arrays = [np.asarray(channels[c].data, dtype=np.float64) for c in ("E", "N", "Z")]
    length = min(len(x) for x in arrays)
    if length == 0:
        raise ValueError("空波形")
    return np.stack([x[:length] for x in arrays], axis=0), rates[0]


def to_enz_from_stead(waveform, component_order="ENZ"):
    """把 SeisBench 数组按显式 component_order 转为 E,N,Z。"""
    w = np.asarray(waveform)
    order = str(component_order).upper()
    if w.ndim != 2 or w.shape[0] != len(order):
        raise ValueError(f"期望 (C,N) 且 C={len(order)}，实际 {w.shape}")
    if set(order) != {"E", "N", "Z"} or len(order) != 3:
        raise ValueError(f"component_order 必须恰含 E/N/Z: {order}")
    return np.stack([w[order.index(c)] for c in ("E", "N", "Z")], axis=0)