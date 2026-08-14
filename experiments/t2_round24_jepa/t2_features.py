"""T2 震级估计的可迁移物理特征与官方评分代理。

刻意不包含记录长度/采样点数。长度由出题截窗协议决定，并非地震物理量；
它在 R1 与 08 上分别呈相反相关，作为特征会制造不可迁移的高分假象。
"""
from __future__ import annotations

import numpy as np

PHYS_FEATURE_NAMES = (
    "logAmax_Z", "logAmax_H", "logRMS_Z", "logRMS_H", "fc_Z", "fc_H",
)


def score200(pred, true) -> float:
    """返回 200 样本口径的截断绝对误差分。"""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape or pred.size == 0:
        raise ValueError("pred/true 必须同形且非空")
    return float(200.0 * np.maximum(0.0, 1.0 - np.abs(pred - true)).mean())


def _spectral_centroid(x: np.ndarray, sr: float) -> float:
    scale = float(np.max(np.abs(x)))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    x = x / scale
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freq = np.fft.rfftfreq(len(x), 1.0 / sr)
    band = (freq > 0.2) & (freq < min(40.0, 0.48 * sr))
    if not np.any(band):
        return float("nan")
    total = float(power[band].sum())
    if total <= 0:
        return float("nan")
    return float((freq[band] * power[band]).sum() / total)


def extract_features(waveform, sampling_rate: float):
    """从 E,N,Z 三分量数组提取 6 个长度无关特征。"""
    w = np.asarray(waveform, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != 3 or w.shape[1] < 512:
        return None
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        return None
    w = np.nan_to_num(w - w.mean(axis=1, keepdims=True))
    e, n, z = w
    h = np.sqrt(e * e + n * n)
    values = np.asarray((
        np.max(np.abs(z)), np.max(h), np.sqrt(np.mean(z * z)),
        np.sqrt(np.mean((e * e + n * n) / 2.0)),
    ), dtype=np.float64)
    if values.min() <= 0 or not np.isfinite(values).all():
        return None
    return np.asarray((
        np.log10(values[0]), np.log10(values[1]), np.log10(values[2]),
        np.log10(values[3]), _spectral_centroid(z, sampling_rate),
        _spectral_centroid(h, sampling_rate),
    ), dtype=np.float64)