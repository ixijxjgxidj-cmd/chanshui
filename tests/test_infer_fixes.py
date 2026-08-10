"""推理修复与当前生产后处理的单元测试.

覆盖:
- U3 末端护栏: pick()/pick_batch() 丢弃越过真实数据末尾+0.5s 的 pick,
  batch 路径按 per-wf 各自末尾判定
- U4/U20 缺分量降级: build_waveform 对 ≥1 分量的台站零填充缺失行继续,
  完全无分量才丢; 有降级告警记录
- U17 置信度带出: Task1Result 新增并行置信度列表, 不破坏旧构造/相等性/写出格式
- 合并窗管道: PickerConfig 的 p/s_merge_window_s → DedupConfig, None 落到 defaults.py
- 生产后处理: 条件式强制成对、长记录 SNR 闸、20s 去重、七成员长记录门控

两种运行方式:
    pytest tests/test_infer_fixes.py
    PYTHONUTF8=1 python -m pytest tests/test_infer_fixes.py -q
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker import defaults  # noqa: E402
from phasepicker.types import PhaseType, Pick, Task1Result, Waveform  # noqa: E402
from phasepicker.inference.picker import (  # noqa: E402
    END_GUARD_TOLERANCE_S,
    PickerConfig,
    ProbEnsemblePicker,
    SeisBenchPicker,
)
from phasepicker.postprocess.dedup import DEFAULT_MERGE_WINDOW_S, DedupConfig  # noqa: E402
from phasepicker.tasks.task1_runner import (  # noqa: E402
    _merge_task1_results,
    picks_to_task1_result,
)
from phasepicker.io.submission_writer import format_task1_line  # noqa: E402


# ---------------------------------------------------------------------------
# 假模型: 只提供 pick()/pick_batch() 路径所需的最小接口, 不做真实推理
# ---------------------------------------------------------------------------
class _FakeSbPick:
    """SeisBench classify 输出 pick 的最小替身。"""

    def __init__(self, phase, ts, value=0.5, trace_id=""):
        self.phase = phase
        self.peak_time = SimpleNamespace(timestamp=ts)
        self.peak_value = value
        self.trace_id = trace_id


class _FakeModel:
    in_samples = 3001
    sampling_rate = 100.0

    def __init__(self, picks):
        self._picks = picks

    def classify(self, stream, **kwargs):
        return SimpleNamespace(picks=list(self._picks))

    def parameters(self):
        return iter([])


def _mk_wf(n=3000, sr=100.0, start=1000.0, station="XB.TST"):
    return Waveform(
        data=np.zeros((3, n), dtype=np.float32),
        sampling_rate=sr,
        starttime_utc=start,
        station=station,
    )


def _mk_picker(fake_picks, cfg=None, dedup_cfg=None):
    pytest.importorskip("torch")
    pytest.importorskip("obspy")
    return SeisBenchPicker(_FakeModel(fake_picks), cfg or PickerConfig(), dedup_cfg)


# ---------------------------------------------------------------------------
# U3 末端护栏
# ---------------------------------------------------------------------------
def test_pick_drops_beyond_data_end():
    wf = _mk_wf(n=3000, sr=100.0, start=1000.0)  # 真实末尾 = 1030.0
    end = wf.starttime_utc + wf.n_samples / wf.sampling_rate
    picker = _mk_picker(
        [
            _FakeSbPick("P", 1010.0, 0.9),
            _FakeSbPick("P", end + 0.4, 0.8),  # 容差内, 保留
            _FakeSbPick("S", end + 0.6, 0.7),  # 超容差, 丢弃
        ]
    )
    picks = picker.pick(wf)
    times = sorted(p.time_utc for p in picks)
    assert times == [1010.0, pytest.approx(end + 0.4)]
    assert all(p.phase == PhaseType.P for p in picks)
    assert END_GUARD_TOLERANCE_S == 0.5


def test_pick_batch_guard_is_per_waveform():
    # wf0 真实末尾 1030.0, wf1 真实末尾 2060.0 —— 护栏必须各用各的
    wf0 = _mk_wf(n=3000, start=1000.0, station="XB.AAA")
    wf1 = _mk_wf(n=6000, start=2000.0, station="XB.BBB")
    picker = _mk_picker(
        [
            _FakeSbPick("P", 1005.0, 0.9, trace_id="XB.B00000."),
            _FakeSbPick("P", 1031.0, 0.8, trace_id="XB.B00000."),  # wf0 超末尾+0.5, 丢
            _FakeSbPick("P", 2059.9, 0.9, trace_id="XB.B00001."),
            _FakeSbPick("S", 2060.6, 0.7, trace_id="XB.B00001."),  # wf1 超末尾+0.5, 丢
        ]
    )
    per_wf = picker.pick_batch([wf0, wf1])
    assert [p.time_utc for p in per_wf[0]] == [1005.0]
    assert [p.time_utc for p in per_wf[1]] == [pytest.approx(2059.9)]
    assert per_wf[0][0].station == "XB.AAA"
    assert per_wf[1][0].station == "XB.BBB"


# ---------------------------------------------------------------------------
# 合并窗配置管道: PickerConfig → DedupConfig → 生效
# ---------------------------------------------------------------------------
def test_merge_window_defaults_from_defaults_py():
    picker = _mk_picker([])
    assert picker._dedup_cfg.p_merge_window_s == defaults.DEFAULT_P_MERGE_WINDOW_S
    assert picker._dedup_cfg.s_merge_window_s == defaults.DEFAULT_S_MERGE_WINDOW_S
    # dedup 模块自己的默认表也必须来自同一真源
    assert DEFAULT_MERGE_WINDOW_S[PhaseType.P] == defaults.DEFAULT_P_MERGE_WINDOW_S
    assert DEFAULT_MERGE_WINDOW_S[PhaseType.S] == defaults.DEFAULT_S_MERGE_WINDOW_S


def test_merge_window_from_picker_config():
    cfg = PickerConfig(p_merge_window_s=2.0, s_merge_window_s=3.0)
    picker = _mk_picker([], cfg=cfg)
    assert picker._dedup_cfg.p_merge_window_s == 2.0
    assert picker._dedup_cfg.s_merge_window_s == 3.0


def test_explicit_dedup_cfg_wins_over_picker_config():
    cfg = PickerConfig(p_merge_window_s=2.0, s_merge_window_s=3.0)
    explicit = DedupConfig(p_merge_window_s=0.7, s_merge_window_s=0.9)
    picker = _mk_picker([], cfg=cfg, dedup_cfg=explicit)
    assert picker._dedup_cfg is explicit


def test_merge_window_actually_applied_in_pick():
    wf = _mk_wf(n=6000, start=1000.0)
    fake = [
        _FakeSbPick("P", 1010.0, 0.9),
        _FakeSbPick("P", 1011.5, 0.4),  # 相距 1.5s，明显大于默认 P 窗 1.0s
    ]
    # 默认窗 1.0s: 间距 1.5s 不合并，两个都保留
    assert len(_mk_picker(fake).pick(wf)) == 2
    # 配置 2s 窗: 合并为置信度最高的那个
    merged = _mk_picker(fake, cfg=PickerConfig(p_merge_window_s=2.0)).pick(wf)
    assert [(p.time_utc, p.confidence) for p in merged] == [(1010.0, 0.9)]


# ---------------------------------------------------------------------------
# 2026-08-11 生产后处理
# ---------------------------------------------------------------------------
class _FakeAnnotations:
    def __len__(self):
        return 1

    def select(self, **_kwargs):
        return self


def test_conditional_force_pair_keeps_pure_noise_empty():
    """两相位正常阈值都无触发时，不得在纯噪声上强行补 P/S。"""
    cfg = PickerConfig(
        force_pair_max_duration_s=300.0,
        force_pair_floor=0.03,
        force_pair_conditional=True,
    )
    picker = _mk_picker([], cfg=cfg)
    trace = SimpleNamespace(
        stats=SimpleNamespace(station="B00000", starttime=0.0, endtime=60.0)
    )
    got = picker._fallback_lowth_picks(
        "P",
        {"P": _FakeAnnotations(), "S": _FakeAnnotations()},
        {"P": [], "S": []},
        [trace],
    )
    assert got == []


def test_long_record_snr_gate_drops_falling_energy_pick():
    sr = 10.0
    data = np.ones((3, 4000), dtype=np.float32)  # 400s，超过生产阈值
    # 100s: 到时后能量下降 20dB，应丢弃。
    data[:, 980:1000] = 10.0
    data[:, 1000:1020] = 1.0
    # 200s: 到时后能量上升 20dB，应保留。
    data[:, 1980:2000] = 1.0
    data[:, 2000:2020] = 10.0
    wf = Waveform(data=data, sampling_rate=sr, starttime_utc=0.0, station="XB.TST")
    bad = _pick(PhaseType.P, 100.0, 0.9)
    good = _pick(PhaseType.S, 200.0, 0.8)
    cfg = PickerConfig(
        long_snr_threshold_db=-1.0,
        long_snr_min_duration_s=300.0,
        long_snr_pre_s=2.0,
        long_snr_post_s=2.0,
    )
    kept = _mk_picker([], cfg=cfg)._filter_long_snr(wf, [bad, good])
    assert kept == [good]


def test_long_record_dedup_uses_20_second_window_only_above_300s():
    cfg = PickerConfig(
        long_dedup_p_window_s=20.0,
        long_dedup_s_window_s=20.0,
        long_dedup_min_duration_s=300.0,
    )
    picker = _mk_picker([], cfg=cfg)
    picks = [
        _pick(PhaseType.P, 100.0, 0.4),
        _pick(PhaseType.P, 110.0, 0.9),
        _pick(PhaseType.P, 150.0, 0.8),
    ]
    long_wf = _mk_wf(n=4001, sr=10.0, start=0.0)
    got = picker._long_dedup(long_wf, picks)
    assert [(p.time_utc, p.confidence) for p in got] == [(110.0, 0.9), (150.0, 0.8)]

    boundary_wf = _mk_wf(n=3000, sr=10.0, start=0.0)  # 恰好 300s，不启用
    assert picker._long_dedup(boundary_wf, picks) is picks


class _FakeProbModel:
    in_samples = 3001
    sampling_rate = 100.0

    def __init__(self, value, record=False):
        self.value = float(value)
        self.means = [] if record else None

    def annotate(self, stream, **_kwargs):
        obspy = pytest.importorskip("obspy")
        out = obspy.Stream()
        for src in stream:
            for phase in ("P", "S"):
                tr = obspy.Trace(data=np.full(len(src.data), self.value, dtype=np.float32))
                tr.stats.network = src.stats.network
                tr.stats.station = src.stats.station
                tr.stats.location = src.stats.location
                tr.stats.channel = f"{self.__class__.__name__}_{phase}"
                tr.stats.starttime = src.stats.starttime
                tr.stats.sampling_rate = src.stats.sampling_rate
                out += tr
        return out

    def picks_from_annotations(self, annotations, _threshold, phase):
        if self.means is not None:
            self.means.append((phase, float(np.mean(annotations[0].data))))
        return []

    def parameters(self):
        return iter([])


def _ensemble_input(duration_s):
    obspy = pytest.importorskip("obspy")
    sr = 10.0
    tr = obspy.Trace(data=np.zeros(int(duration_s * sr) + 1, dtype=np.float32))
    tr.stats.network = "XB"
    tr.stats.station = "TST"
    tr.stats.location = ""
    tr.stats.channel = "HHZ"
    tr.stats.starttime = obspy.UTCDateTime(0)
    tr.stats.sampling_rate = sr
    return obspy.Stream([tr])


def test_seven_member_ensemble_uses_only_first_five_for_long_records():
    pytest.importorskip("seisbench")
    host = _FakeProbModel(1, record=True)
    picker = ProbEnsemblePicker(
        host,
        PickerConfig(ensemble_long_top_n=5, ensemble_long_max_duration_s=300.0),
    )
    picker._members = [host] + [_FakeProbModel(i) for i in range(2, 8)]

    picker._classify_refined(_ensemble_input(100.0))
    short_means = [value for _phase, value in host.means[-2:]]
    assert short_means == pytest.approx([4.0, 4.0])  # mean(1..7)

    picker._classify_refined(_ensemble_input(400.0))
    long_means = [value for _phase, value in host.means[-2:]]
    assert long_means == pytest.approx([3.0, 3.0])  # mean(1..5)


# ---------------------------------------------------------------------------
# U4/U20 缺分量降级 (依赖 obspy)
# ---------------------------------------------------------------------------
def _mk_trace(comp, n=1000, sr=100.0, start=0.0):
    obspy = pytest.importorskip("obspy")
    tr = obspy.Trace(data=np.arange(n, dtype=np.float32) + 1.0)
    tr.stats.sampling_rate = sr
    tr.stats.starttime = obspy.UTCDateTime(start)
    tr.stats.channel = f"HH{comp}"
    tr.stats.station = "TST"
    tr.stats.network = "XB"
    return tr


def _build(traces):
    from phasepicker.io.mseed_reader import IngestResult, build_waveform

    result = IngestResult()
    wf = build_waveform(traces, "XB.TST", result)
    return wf, result


def test_z_only_station_kept_with_zero_fill():
    wf, result = _build([_mk_trace("Z")])
    assert wf is not None, "改后: 只有 Z 分量的台站必须保留"
    assert wf.station == "XB.TST"
    assert wf.data.shape == (3, 1000)
    assert np.all(wf.data[0] != 0.0)          # Z 行是真实数据
    assert not np.any(wf.data[1])             # N 行零填充
    assert not np.any(wf.data[2])             # E 行零填充
    assert wf.sampling_rate == 100.0 and wf.starttime_utc == 0.0
    reasons = [w.reason for w in result.warnings]
    assert "missing_component_zero_filled" in reasons, "必须留一条降级记录"


def test_two_component_station_kept_with_zero_fill():
    wf, result = _build([_mk_trace("Z"), _mk_trace("N")])
    assert wf is not None
    assert np.all(wf.data[0] != 0.0) and np.all(wf.data[1] != 0.0)
    assert not np.any(wf.data[2])
    assert any(w.reason == "missing_component_zero_filled" for w in result.warnings)


def test_no_component_station_dropped():
    # 通道码无法识别 → 无任何可用分量 → 丢台站（与旧行为一致）
    tr = _mk_trace("Z")
    tr.stats.channel = "HHQ"
    wf, result = _build([tr])
    assert wf is None
    assert any(w.reason == "missing_component" for w in result.warnings)


def test_full_three_component_unchanged():
    wf, result = _build([_mk_trace("Z"), _mk_trace("N"), _mk_trace("E")])
    assert wf is not None
    assert wf.data.shape == (3, 1000)
    assert np.all(wf.data != 0.0)
    assert not any(
        w.reason.startswith("missing_component") for w in result.warnings
    ), "三分量齐全不该出现任何降级告警"


def test_short_waveform_default_rejected_but_padding_model_can_opt_in():
    from phasepicker.io.mseed_reader import IngestResult, build_waveform

    traces = [_mk_trace(comp, n=200, sr=100.0) for comp in ("Z", "N", "E")]

    default_result = IngestResult()
    assert build_waveform(traces, "XB.TST", default_result) is None
    assert any(w.reason == "too_short" for w in default_result.warnings)

    padding_result = IngestResult()
    wf = build_waveform(
        traces, "XB.TST", padding_result, min_duration_s=0.0
    )
    assert wf is not None
    assert wf.duration == pytest.approx(2.0)
    assert not any(w.reason == "too_short" for w in padding_result.warnings)


def test_degraded_waveform_flows_through_picker():
    # 端到端衔接: 零填充降级出的 Waveform 能被 pick() 正常消化并出 picks
    wf, _ = _build([_mk_trace("Z", n=3000)])
    picks = _mk_picker([_FakeSbPick("P", 10.0, 0.9)]).pick(wf)
    assert [p.time_utc for p in picks] == [10.0]


# ---------------------------------------------------------------------------
# U17 置信度带出
# ---------------------------------------------------------------------------
def _pick(phase, t, conf, station="XB.TST"):
    return Pick(phase=phase, time_utc=t, confidence=conf, station=station)


def test_confidences_parallel_and_sorted():
    start = 1000.0
    picks = [
        _pick(PhaseType.P, start + 1.5, 0.9),
        _pick(PhaseType.S, start + 3.25, 0.8),
        _pick(PhaseType.P, start + 0.5, 0.7),
    ]
    r = picks_to_task1_result("f.mseed", picks, start)
    assert r.p_times_s == [0.5, 1.5]
    assert r.p_confidences == [0.7, 0.9], "置信度必须跟着到时一起排序"
    assert r.s_times_s == [3.25]
    assert r.s_confidences == [0.8]


def test_negative_rel_drops_confidence_too():
    start = 1000.0
    picks = [
        _pick(PhaseType.P, start - 1.0, 0.99),
        _pick(PhaseType.P, start + 2.0, 0.6),
    ]
    r = picks_to_task1_result("f.mseed", picks, start)
    assert r.p_times_s == [2.0]
    assert r.p_confidences == [0.6]


def test_merge_keeps_confidences_aligned():
    a = Task1Result("f", [1.0, 5.0], [7.0], p_confidences=[0.9, 0.5], s_confidences=[0.8])
    b = Task1Result("f", [3.0], [6.0], p_confidences=[0.7], s_confidences=[0.4])
    merged = _merge_task1_results("f", [a, b])
    assert merged.p_times_s == [1.0, 3.0, 5.0]
    assert merged.p_confidences == [0.9, 0.7, 0.5]
    assert merged.s_times_s == [6.0, 7.0]
    assert merged.s_confidences == [0.4, 0.8]


def test_merge_pads_missing_confidences():
    # 旧式构造(无置信度)的 part 与新式混合: 缺的按 0.0 对齐, 并行列表必须等长
    old_style = Task1Result("f", [2.0], [8.0])
    new_style = Task1Result("f", [1.0], [], p_confidences=[0.9])
    merged = _merge_task1_results("f", [old_style, new_style])
    assert merged.p_times_s == [1.0, 2.0]
    assert merged.p_confidences == [0.9, 0.0]
    assert len(merged.s_times_s) == len(merged.s_confidences) == 1


def test_task1result_backward_compat():
    # 旧式位置参数构造不变
    r = Task1Result("f.mseed", [1.0], [2.0])
    assert r.p_confidences == [] and r.s_confidences == []
    # 相等性不受置信度影响 (compare=False): 旧代码的比较语义零变化
    r2 = Task1Result("f.mseed", [1.0], [2.0], p_confidences=[0.9], s_confidences=[0.8])
    assert r == r2
    # T1.an 行格式一个字符都不能变
    assert format_task1_line(r2) == "f.mseed : P : 1.00 : S : 2.00"
    assert format_task1_line(r2) == format_task1_line(r)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
