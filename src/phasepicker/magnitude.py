"""震级估计器（Magnitude estimators）——供 /magnitude API 与离线评测复用.

===== 背景 =====
报名页有独立的"震级 API"字段（第一届主题就含震级预测），但今年的输出格式
官方未公布。本模块只负责"波形(+拾取) → 震级数值"，与 HTTP 层解耦：
serve_api.py 用可配 formatter 把数值组装成 JSON，格式一旦公布只改外层。

===== 两个实现 =====
1. ``BaselineJoblibMagnitude``：复用去年 T2 资产
   （weights/official_r1_to_r2/t2_magnitude_baseline.joblib，
   ExtraTrees + 60 维波形特征，r1 训练 / r2 留出 MAE=0.817，见 manifest.json）。
   **局限（诚实边界）**：训练样本是"单台站单事件 60s 短窗"，特征含时长/
   峰值位置等窗级统计——对今年的多事件长连续记录，它只能给**整文件一个 M**，
   无法按事件拆分；且长记录的时长特征已偏离训练分布，数值仅供保底。
2. ``PSDeltaMagnitude``：可解释的占位公式。用拾取的 S-P 时差做震中距代理
   （地壳平均速度下 R≈8×Δt km），配合事件窗内最大幅值做量规回归：
       M = a·log10(A) + b·log10(Δt_SP) + c
   无 S-P 配对时退化为纯幅值回归 M = a₂·log10(A) + c₂；连幅值都取不到时
   回落常数。系数全部可配，默认值由 scripts/run_magnitude_eval.py 在去年
   第 1 轮真题上拟合（r2 留出集上与 baseline 对比见该脚本输出）。
   **诚实边界**：赛题波形无台站位置、无仪器响应标定，幅值是原始 counts；
   这是"方法可解释 + 支持按事件分组"的占位，不是科学定标的绝对震级。

===== 契约 =====
输入 ``MagnitudeInput``（waveforms + 与之下标对齐的 picks_per_wf），
输出与 waveforms 下标对齐的 ``List[List[float]]``：每台站 0..n 个震级
（多事件各一个；噪声台站空表）。异常向上抛，由 API 层统一降级。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .types import CHANNEL_ORDER, PhaseType, Pick, Waveform


# =========================================================================
# 输入/输出数据结构
# =========================================================================
@dataclass
class MagnitudeInput:
    """一次震级估计请求（一个 mseed 文件解析后的全部台站）。

    Attributes:
        waveforms: mseed_reader.load_waveforms 产出的多台站波形。
        picks_per_wf: 与 waveforms **下标对齐**的拾取结果；不需要拾取的
            估计器（baseline）可收到空表。
        stream: 可选的原始 ObsPy Stream。离线评测走 read_mseed_stream 时
            传入，可与去年训练/评测管线的特征取数逐字节一致；不传时
            baseline 用 waveforms 重建伪 Stream（官方规整数据上两者等价，
            有 gap 的文件可能有微小差异）。
    """

    waveforms: List[Waveform]
    picks_per_wf: List[List[Pick]] = field(default_factory=list)
    stream: Any = None


@dataclass
class EventGroup:
    """一个"事件"的 P/S 到时组（epoch 秒）。至少有其一。"""

    p_time: Optional[float] = None
    s_time: Optional[float] = None

    @property
    def sp_delta(self) -> Optional[float]:
        if self.p_time is None or self.s_time is None:
            return None
        dt = self.s_time - self.p_time
        return dt if dt > 0 else None


def group_picks_into_events(picks: Sequence[Pick]) -> List[EventGroup]:
    """把单台站的 P/S 拾取分组为事件：每个 P 开一个事件，配它到下一个 P
    之间的第一个 S；落在首个 P 之前/无 P 可配的 S 各自成孤儿事件。

    这是长连续记录"多事件各出一个震级"的最简确定性分组：依据是同一事件
    S 必然晚于 P、而相邻事件的 P-P 间隔远大于本事件 S-P（去年真题最小
    P-P 10.49s）。不做震相关联网络那种全局最优——占位实现以可解释优先。
    """
    p_times = sorted(float(p.time_utc) for p in picks if p.phase == PhaseType.P)
    s_times = sorted(float(p.time_utc) for p in picks if p.phase == PhaseType.S)

    events: List[EventGroup] = []
    used = [False] * len(s_times)
    for i, pt in enumerate(p_times):
        window_end = p_times[i + 1] if i + 1 < len(p_times) else math.inf
        matched: Optional[float] = None
        for j, st in enumerate(s_times):
            if used[j] or st <= pt:
                continue
            if st >= window_end:
                break
            matched = st
            used[j] = True
            break
        events.append(EventGroup(p_time=pt, s_time=matched))
    for j, st in enumerate(s_times):
        if not used[j]:
            events.append(EventGroup(p_time=None, s_time=st))
    events.sort(key=lambda e: e.p_time if e.p_time is not None else e.s_time)
    return events


# =========================================================================
# 抽象契约
# =========================================================================
class MagnitudeEstimator(ABC):
    """震级估计器抽象：吃 MagnitudeInput，吐与台站对齐的震级列表。"""

    name: str = "abstract"
    #: 是否需要 API 层先跑震相拾取（baseline 不需要，可省一次推理）。
    needs_picks: bool = False

    @abstractmethod
    def estimate(self, inp: MagnitudeInput) -> List[List[float]]:
        """返回与 ``inp.waveforms`` 下标对齐的震级列表（每台站 0..n 个）。"""
        raise NotImplementedError


# =========================================================================
# 实现 1：去年 T2 joblib 基线
# =========================================================================
def _waveforms_to_pseudo_stream(waveforms: Sequence[Waveform]) -> List[Tuple[str, float, np.ndarray]]:
    """把已解析的 Waveform 列表还原成 extract_waveform_features 可吃的
    ``[(channel, sampling_rate, data), ...]`` 伪 Stream。

    多台站时全部铺平——与原始 Stream 喂给 stream_to_components 的行为一致
    （每分量保留样本最多的一段）。"""
    out: List[Tuple[str, float, np.ndarray]] = []
    for wf in waveforms:
        data = np.asarray(wf.data)
        for row, comp in enumerate(CHANNEL_ORDER):
            out.append((comp, float(wf.sampling_rate), data[row]))
    return out


class BaselineJoblibMagnitude(MagnitudeEstimator):
    """去年 T2 基线（ExtraTrees + 60 维特征）的在线封装。

    **局限**：模型按"整文件单事件"训练，这里对整文件出**一个** M，
    并把它赋给文件内每个台站（单元素数组）；多事件长记录无法按事件拆分。
    r2 留出集 MAE=0.817（manifest.json），优于常数基线 0.884。
    """

    name = "baseline"
    needs_picks = False

    def __init__(self, model_path: str):
        from .tasks.baseline_models import load_bundle

        self._bundle = load_bundle(model_path, expected_task="T2")

    def magnitude_for_stream(self, stream) -> float:
        """单文件 Stream（或伪 Stream 三元组列表）→ 一个震级。

        与 scripts/run_official_task23.py 的离线路径同源：同一特征函数、
        同一模型包、同一 [0, 9.9] 外推裁剪。"""
        from .tasks.waveform_features import extract_waveform_features

        magnitude = float(self._bundle.predict_one(extract_waveform_features(stream)))
        return min(9.9, max(0.0, magnitude))

    def estimate(self, inp: MagnitudeInput) -> List[List[float]]:
        if not inp.waveforms:
            return []
        stream = inp.stream if inp.stream is not None else _waveforms_to_pseudo_stream(inp.waveforms)
        magnitude = self.magnitude_for_stream(stream)
        return [[magnitude] for _ in inp.waveforms]


# =========================================================================
# 实现 2：S-P 时差 + 幅值回归占位
# =========================================================================
@dataclass
class PSDeltaCoefficients:
    """PSDeltaMagnitude 的全部可配系数。

    默认值由 ``scripts/run_magnitude_eval.py --fit`` 在去年第 1 轮真题
    （200 条 T2，diting 默认拾取管线）最小二乘拟合得出；r2 留出集 MAE
    见该脚本输出。重新拟合后把打印的系数粘回这里即可。
    """

    # M = a·log10(A_max) + b·log10(Δt_SP) + c（有 S-P 配对的事件）
    a: float = 0.7290
    b: float = 0.6858
    c: float = -1.7229
    # M = amp_a·log10(A_max) + amp_c（无 S-P 配对：孤 P / 孤 S）
    amp_a: float = 0.7541
    amp_c: float = -1.3745
    # 连幅值都无效时的兜底常数（r1 训练集震级均值）
    fallback_m: float = 4.464


class PSDeltaMagnitude(MagnitudeEstimator):
    """S-P 时差做距离代理 + 事件窗峰值幅值的可解释占位公式。

    每台站按 ``group_picks_into_events`` 分组，**每个事件出一个 M**——
    这是它相对 baseline 的唯一结构优势（多事件长记录友好）。
    幅值取事件窗（P 前 1s 到 S 后 coda_s，无 S 则 P 后 fallback_window_s）
    内三分量去均值后的最大绝对值，原始 counts 无标定，诚实边界见模块头。
    """

    name = "psdelta"
    needs_picks = True

    def __init__(
        self,
        coef: Optional[PSDeltaCoefficients] = None,
        coda_s: float = 10.0,
        fallback_window_s: float = 10.0,
        pre_p_s: float = 1.0,
        clip_range: Tuple[float, float] = (0.0, 9.9),
    ):
        self.coef = coef or PSDeltaCoefficients()
        self.coda_s = float(coda_s)
        self.fallback_window_s = float(fallback_window_s)
        self.pre_p_s = float(pre_p_s)
        self.clip_range = clip_range

    # ---- 特征取数（拟合脚本与在线推理共用，保证训练/推理同分布） ----
    def event_features(self, wf: Waveform, event: EventGroup) -> Tuple[Optional[float], Optional[float]]:
        """返回 (log10 幅值, log10 S-P 秒差)；取不到的一侧为 None。"""
        log_dt: Optional[float] = None
        dt = event.sp_delta
        if dt is not None and dt > 1e-3:
            log_dt = math.log10(dt)

        anchor = event.p_time if event.p_time is not None else event.s_time
        if anchor is None:
            return None, log_dt
        start_s = anchor - self.pre_p_s
        if event.s_time is not None:
            end_s = event.s_time + self.coda_s
        else:
            end_s = anchor + self.fallback_window_s

        data = np.asarray(wf.data, dtype=np.float64)
        sr = float(wf.sampling_rate)
        n = data.shape[-1]
        i0 = int(max(0, math.floor((start_s - wf.starttime_utc) * sr)))
        i1 = int(min(n, math.ceil((end_s - wf.starttime_utc) * sr)))
        if i1 <= i0:
            return None, log_dt
        seg = data[:, i0:i1]
        seg = seg - seg.mean(axis=1, keepdims=True)
        amp = float(np.max(np.abs(seg)))
        if not np.isfinite(amp) or amp <= 0.0:
            return None, log_dt
        return math.log10(amp), log_dt

    def magnitude_for_event(self, wf: Waveform, event: EventGroup) -> float:
        log_a, log_dt = self.event_features(wf, event)
        k = self.coef
        if log_a is None:
            m = k.fallback_m
        elif log_dt is None:
            m = k.amp_a * log_a + k.amp_c
        else:
            m = k.a * log_a + k.b * log_dt + k.c
        lo, hi = self.clip_range
        return min(hi, max(lo, float(m)))

    def estimate(self, inp: MagnitudeInput) -> List[List[float]]:
        picks_per_wf = inp.picks_per_wf or [[] for _ in inp.waveforms]
        out: List[List[float]] = []
        for wf, picks in zip(inp.waveforms, picks_per_wf):
            events = group_picks_into_events(picks)
            out.append([self.magnitude_for_event(wf, ev) for ev in events])
        return out


# =========================================================================
# 工厂：serve_api CLI --mag-model 的接线点
# =========================================================================
def build_estimator(kind: str, model_path: Optional[str] = None) -> Optional[MagnitudeEstimator]:
    """按名字构建估计器；``off``/空 返回 None（API 层据此 501）。"""
    kind = (kind or "off").strip().lower()
    if kind in {"off", "none", ""}:
        return None
    if kind == "baseline":
        if not model_path:
            raise ValueError("baseline 震级模型需要 --mag-weights 指向 t2_magnitude_baseline.joblib")
        return BaselineJoblibMagnitude(model_path)
    if kind == "psdelta":
        return PSDeltaMagnitude()
    raise ValueError(f"未知震级模型：{kind}（可选 baseline / psdelta / off）")
