"""mseed 读取 + 校验 + 容错（Data ingestion）.

⚠️ 依赖 ObsPy，在本沙箱无法运行；请在你的机器上 `pip install obspy` 后运行。
本模块所有逻辑已按 ObsPy 官方 API 编写，函数签名与内部数据结构（types.py）
严格对齐，配套的时间对齐核心（utils/timing.py）已在纯 numpy 环境通过单元测试。

设计目标（对应赛题"数据处理模块"）：
1. 读取三分量 mseed，做基本校验：分量完整性、采样率、时间连续性。
2. 对异常数据（缺分量、非标准采样率、超短/超长波形、多台站混合）明确容错：
   缺分量以零填充降级继续推理（保留台站），其余不可恢复异常以"跳过该台站 +
   结构化告警"收场，绝不让进程崩溃。
3. 输出统一的 Waveform 列表（每台站一个），通道顺序固定 [Z, N, E]。

容错哲学：**宁可跳过一个台站，也不让整个请求挂掉。** 每个可预见的异常都
被捕获并记录为 IngestWarning，调用方（API 层）据此决定是否降级返回空结果。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..types import Waveform, CHANNEL_ORDER

logger = logging.getLogger(__name__)

# ObsPy 延迟导入：让本模块在无 obspy 环境也能被 import（便于测试其它模块）。
try:
    from obspy import read, Stream, Trace, UTCDateTime  # type: ignore

    _OBSPY_AVAILABLE = True
except Exception:  # pragma: no cover - 环境相关
    _OBSPY_AVAILABLE = False


# 允许的采样率白名单（Hz）。非白名单会被重采样到 TARGET_RATE。
# 100Hz 是 SeisBench PhaseNet/EQTransformer 的原生采样率。
TARGET_RATE = 100.0
# 波形时长的合理边界（秒）。超短无法给模型足够上下文；超长仅告警不截断——
# 长波形由 seisbench annotate 原生滑窗直接处理（serve_api 链路不调用
# preprocess；实测第2轮 3600s 文件可正常滑窗出结果）。上界同时是"超时防护"
# 的第一道闸——限制输入长度，比指望中途 kill 正在跑的推理更可靠。
MIN_DURATION_S = 5.0
MAX_DURATION_S = 3600.0


@dataclass
class IngestWarning:
    """一条结构化告警，用于赛后复盘与 API 层降级决策。"""

    station: str
    reason: str
    detail: str = ""


@dataclass
class IngestResult:
    """读取结果：成功的波形 + 所有告警。"""

    waveforms: List[Waveform] = field(default_factory=list)
    warnings: List[IngestWarning] = field(default_factory=list)

    def add_warning(self, station: str, reason: str, detail: str = "") -> None:
        w = IngestWarning(station=station, reason=reason, detail=detail)
        self.warnings.append(w)
        logger.warning("摄入告警 [%s] %s: %s", station, reason, detail)


def _channel_key(channel_code: str) -> Optional[str]:
    """从 SEED 通道代码（如 'BHZ'/'HHN'/'EHE'）提取方向分量 Z/N/E。

    SEED 通道代码约定：末位是方向（Z/N/E，或 1/2/3 等价于 N/E）。
    这里做一个稳健映射，兼容常见的数字分量命名。
    """
    if not channel_code:
        return None
    last = channel_code[-1].upper()
    mapping = {"Z": "Z", "N": "N", "E": "E", "1": "N", "2": "E", "3": "Z"}
    return mapping.get(last)


def read_mseed_bytes(raw: bytes) -> "Stream":
    """从二进制字节读取 mseed 为 ObsPy Stream。API 层接收上传流后调用。

    Raises:
        RuntimeError: ObsPy 不可用。
        ValueError: 无法解析为有效 mseed。
    """
    if not _OBSPY_AVAILABLE:
        raise RuntimeError("ObsPy 未安装；请在部署环境 `pip install obspy`。")
    import io

    try:
        return read(io.BytesIO(raw), format="MSEED")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"mseed 解析失败：{exc}") from exc


def group_by_station(stream: "Stream") -> dict:
    """把 Stream 中的 Trace 按台站 NET.STA 分组，支持多台站混合文件。"""
    groups: dict = {}
    for tr in stream:
        key = f"{tr.stats.network}.{tr.stats.station}"
        groups.setdefault(key, []).append(tr)
    return groups


def _collect_gaps(traces: List["Trace"]) -> List[Tuple[float, float]]:
    """merge 前记录同一分量多段之间的缺口区间（绝对 epoch 秒）。

    缺口判定：按起始时间排序后，若后段起点比"已覆盖范围末尾"晚超过 1.5 个
    采样间隔，则 (已覆盖末尾, 后段起点) 是一个缺口。1.5 倍容差吸收半个采样点
    的时间戳抖动；重叠段（后段起点 <= 已覆盖末尾）不是缺口，交给 merge 插值。
    用"运行最大末尾"而非相邻两段比较，正确处理一段完全包含在另一段内的情形。

    这些区间随后被 merge(fill_value=0) 零填充——零填充区跑出的 pick 是伪造的
    （seisbench issue #273），必须记录下来供推理层否决。
    """
    gaps: List[Tuple[float, float]] = []
    if len(traces) < 2:
        return gaps
    trs = sorted(traces, key=lambda t: t.stats.starttime)
    covered_end: Optional[float] = None
    for tr in trs:
        start = float(tr.stats.starttime.timestamp)
        end = float(tr.stats.endtime.timestamp)
        delta = 1.0 / float(tr.stats.sampling_rate) if tr.stats.sampling_rate else 0.0
        if covered_end is not None and start - covered_end > 1.5 * delta:
            gaps.append((covered_end, start))
        covered_end = end if covered_end is None else max(covered_end, end)
    return gaps


def _union_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """把多分量收集来的缺口区间做并集合并（重叠/相接的并成一段），升序输出。

    任一分量有缺口，该时间段的三分量输入就已残缺，pick 均不可信，
    所以 Waveform 级别只需要一份并集，无须区分来自哪个分量。
    """
    if not intervals:
        return []
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _merge_gappy(traces: List["Trace"], station: str, result: IngestResult) -> Optional["Stream"]:
    """合并同一分量可能存在的多段（gap/overlap）。

    时间不连续（gap）是地震数据常态。用 ObsPy 的 merge 填充：
    - method=1：重叠段取插值；缺口用 fill_value 填 0（后续预处理会去均值，
      填 0 不会引入直流偏置到模型可感知的程度，且保持采样点与时间的严格对应）。
    保持时间连续性是"采样点↔绝对时间"换算成立的前提，这一步至关重要。
    缺口区间由调用方在 merge 前用 _collect_gaps 记录（本函数只管填充）。
    """
    try:
        st = Stream(traces=traces)
        st.merge(method=1, fill_value=0, interpolation_samples=0)
        return st
    except Exception as exc:  # noqa: BLE001
        result.add_warning(station, "merge_failed", str(exc))
        return None


def build_waveform(
    traces: List["Trace"],
    station: str,
    result: IngestResult,
) -> Optional[Waveform]:
    """把某台站的一组 Trace 组装成统一的三分量 Waveform，带完整校验。

    返回 None 表示该台站不可用（原因已记入 result.warnings），调用方应跳过。
    """
    # 1) 按分量归类
    by_comp: dict = {}
    for tr in traces:
        comp = _channel_key(tr.stats.channel)
        if comp is None:
            result.add_warning(station, "unknown_channel", tr.stats.channel)
            continue
        by_comp.setdefault(comp, []).append(tr)

    # 2) 分量完整性检查：只要还有 ≥1 个可用分量就继续（缺失行稍后零填充），
    #    完全无分量才丢台站——单个死道不该让整站 P/S 全部弃权（评审 U4/U20：
    #    分量置零的退化输入 PhaseNet 仍能给出接近满分的 P/S）。
    if not by_comp:
        result.add_warning(station, "missing_component", "无任何可识别分量")
        return None

    # 3) 每个在场分量合并多段 + 采样率一致性；合并失败/合并后为空的分量
    #    降级视同缺失（零填充），而不是丢整站。合并前先记录缺口区间——
    #    merge 会把缺口零填充抹平，之后就再也看不出哪里是假数据了。
    merged: dict = {}
    rates = set()
    gap_intervals: List[Tuple[float, float]] = []
    for comp in CHANNEL_ORDER:
        if comp not in by_comp:
            continue
        comp_gaps = _collect_gaps(by_comp[comp])
        st = _merge_gappy(by_comp[comp], station, result)
        if st is None or len(st) == 0:
            result.add_warning(station, "empty_after_merge", comp)
            continue
        gap_intervals.extend(comp_gaps)
        if comp_gaps:
            result.add_warning(
                station,
                "gap_zero_filled",
                f"{comp} 分量 {len(comp_gaps)} 个缺口已零填充并记录: "
                + ", ".join(f"[{a:.2f}, {b:.2f}]" for a, b in comp_gaps),
            )
        tr = st[0]
        merged[comp] = tr
        rates.add(round(float(tr.stats.sampling_rate), 6))

    if not merged:
        result.add_warning(station, "missing_component", "全部分量合并失败")
        return None
    zero_fill = [c for c in CHANNEL_ORDER if c not in merged]
    if zero_fill:
        result.add_warning(
            station,
            "missing_component_zero_filled",
            f"缺失分量 {zero_fill} 以零填充降级推理；可用 {list(merged)}",
        )

    if len(rates) > 1:
        result.add_warning(station, "inconsistent_sampling_rate", f"{rates}")
        return None
    sampling_rate = float(next(iter(rates)))
    if sampling_rate <= 0:
        result.add_warning(station, "nonpositive_sampling_rate", str(sampling_rate))
        return None

    # 4) 在场分量对齐到公共时间窗（取交集），保证各行严格同长且同起点
    starts = [merged[c].stats.starttime for c in merged]
    ends = [merged[c].stats.endtime for c in merged]
    common_start = max(starts)
    common_end = min(ends)
    if common_end <= common_start:
        result.add_warning(station, "no_time_overlap", "分量时间窗无交集")
        return None

    arrays: dict = {}
    for c in merged:
        tr = merged[c].copy()
        try:
            tr.trim(common_start, common_end, pad=False)
        except Exception as exc:  # noqa: BLE001
            result.add_warning(station, "trim_failed", f"{c}: {exc}")
            return None
        arrays[c] = np.asarray(tr.data, dtype=np.float32)

    # 5) 长度对齐（trim 后可能差 1 个采样点，取最短，保证矩阵规整）；
    #    缺失分量补同长零行——采样率与起始时间不变，到时换算不受影响
    n = min(a.shape[0] for a in arrays.values())
    if n <= 0:
        result.add_warning(station, "empty_data", "trim 后无采样点")
        return None
    data = np.stack(
        [
            arrays[c][:n] if c in arrays else np.zeros(n, dtype=np.float32)
            for c in CHANNEL_ORDER
        ],
        axis=0,
    )  # (3, n) = [Z, N, E]

    # 6) 时长边界校验（超短拒绝；超长仅告警，annotate 原生滑窗可处理）
    duration = n / sampling_rate
    if duration < MIN_DURATION_S:
        result.add_warning(
            station, "too_short", f"{duration:.2f}s < {MIN_DURATION_S}s"
        )
        return None
    if duration > MAX_DURATION_S:
        result.add_warning(
            station, "too_long", f"{duration:.2f}s > {MAX_DURATION_S}s（annotate 滑窗处理）"
        )

    starttime_utc = float(common_start.timestamp)  # UTCDateTime -> epoch 秒
    return Waveform(
        data=data,
        sampling_rate=sampling_rate,
        starttime_utc=starttime_utc,
        station=station,
        gaps=_union_intervals(gap_intervals),
    )


def load_waveforms(raw: bytes) -> IngestResult:
    """顶层入口：原始 mseed 字节 → 校验过的多台站 Waveform 列表。

    绝不抛出未捕获异常（除非 ObsPy 缺失，那是部署问题应尽早暴露）。
    任何数据层面的问题都以 IngestWarning 形式返回。
    """
    result = IngestResult()
    try:
        stream = read_mseed_bytes(raw)
    except ValueError as exc:
        result.add_warning("<file>", "parse_error", str(exc))
        return result

    groups = group_by_station(stream)
    if not groups:
        result.add_warning("<file>", "empty_stream", "文件中无任何 Trace")
        return result

    for station, traces in groups.items():
        try:
            wf = build_waveform(traces, station, result)
            if wf is not None:
                result.waveforms.append(wf)
        except Exception as exc:  # noqa: BLE001 —— 最后一道防线，绝不让单台站拖垮整体
            result.add_warning(station, "unexpected_error", repr(exc))
    return result
