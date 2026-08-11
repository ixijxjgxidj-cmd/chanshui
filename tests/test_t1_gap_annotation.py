"""Structural and decision tests for the preregistered round-06 audit."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiment_t1_gap_annotation.py"
)
_SPEC = importlib.util.spec_from_file_location("experiment_t1_gap_annotation", _SCRIPT)
assert _SPEC and _SPEC.loader
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


def _trace(
    channel: str,
    data,
    *,
    start: float = 1000.0,
    sampling_rate: float = 10.0,
) -> Trace:
    trace = Trace(data=np.asarray(data))
    trace.stats.starttime = UTCDateTime(start)
    trace.stats.sampling_rate = sampling_rate
    trace.stats.network = "XB"
    trace.stats.station = "TEST"
    trace.stats.location = "00"
    trace.stats.channel = channel
    return trace


def _annotations(*, n: int = 40, sampling_rate: float = 10.0) -> Stream:
    return Stream(
        [
            _trace(
                "PhaseNet_P",
                np.linspace(0.1, 0.9, n, dtype=np.float32),
                sampling_rate=sampling_rate,
            ),
            _trace(
                "PhaseNet_S",
                np.linspace(0.9, 0.1, n, dtype=np.float64),
                sampling_rate=sampling_rate,
            ),
            _trace(
                "PhaseNet_N",
                np.full(n, 0.25, dtype=np.float32),
                sampling_rate=sampling_rate,
            ),
            _trace(
                "BHZ",
                np.arange(n, dtype=np.int16),
                sampling_rate=sampling_rate,
            ),
        ]
    )


def _waveform(*, duration: float = 40.0, sampling_rate: float = 10.0):
    n = int(duration * sampling_rate)
    return exp.Waveform(
        data=np.zeros((3, n), dtype=np.float32),
        sampling_rate=sampling_rate,
        starttime_utc=1000.0,
        station="XB.TEST",
        gaps=[],
    )


def _pick(phase, time_utc: float, confidence: float = 0.5):
    return exp.Pick(
        phase=phase,
        time_utc=float(time_utc),
        confidence=float(confidence),
        station="XB.TEST",
    )


def test_guard_grid_and_keys_are_frozen():
    assert exp.ACTIVE_GUARDS_S == (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
    assert exp.guard_key(None) == "OFF"
    assert [exp.guard_key(value) for value in exp.ACTIVE_GUARDS_S] == [
        "0.0",
        "0.5",
        "1.0",
        "2.0",
        "5.0",
        "10.0",
    ]


def test_no_gap_and_no_legal_gap_return_exact_stream_object():
    annotations = _annotations()
    invalid_only = [
        (1001.0, 1001.0),
        (1002.0, 1001.0),
        (math.nan, 1004.0),
        ("bad", 1005.0),
        (1006.0,),
    ]
    assert exp.mask_phase_annotations(annotations, [], 0.0) is annotations
    assert exp.mask_phase_annotations(annotations, invalid_only, 10.0) is annotations
    with pytest.raises(ValueError):
        exp.mask_phase_annotations(annotations, [], -0.1)
    with pytest.raises(ValueError):
        exp.mask_phase_annotations(annotations, [], math.inf)


def test_guard_expansion_uses_closed_sample_boundaries():
    values = np.arange(1, 21, dtype=np.float32)
    annotations = Stream([_trace("PhaseNet_P", values, sampling_rate=10.0)])
    candidate = exp.mask_phase_annotations(
        annotations,
        [(1001.0, 1001.2)],
        0.2,
    )
    assert candidate is not annotations
    assert np.array_equal(candidate[0].data[:8], values[:8])
    assert np.all(candidate[0].data[8:15] == 0.0)
    assert np.array_equal(candidate[0].data[15:], values[15:])
    assert np.array_equal(annotations[0].data, values)


def test_mask_changes_only_phase_channels_and_preserves_structure_and_raw_stream():
    annotations = _annotations()
    raw_hash = exp.annotation_hash(annotations)
    raw_metadata = [exp._trace_metadata(trace) for trace in annotations]
    raw_arrays = [np.array(trace.data, copy=True) for trace in annotations]

    candidate = exp.mask_phase_annotations(
        annotations,
        [(1001.0, 1001.2)],
        0.0,
    )

    assert candidate is not annotations
    assert exp.validate_mask_structure(
        annotations, candidate, [(1001.0, 1001.2)], 0.0
    )["pass"]
    assert exp.annotation_hash(annotations) == raw_hash
    assert [exp._trace_metadata(trace) for trace in candidate] == raw_metadata
    assert [trace.id for trace in candidate] == [trace.id for trace in annotations]
    assert [trace.data.dtype for trace in candidate] == [
        trace.data.dtype for trace in annotations
    ]
    for before, trace in zip(raw_arrays, annotations):
        assert np.array_equal(trace.data, before)
    assert np.all(candidate.select(channel="PhaseNet_P")[0].data[10:13] == 0.0)
    assert np.all(candidate.select(channel="PhaseNet_S")[0].data[10:13] == 0.0)
    assert np.array_equal(candidate.select(channel="PhaseNet_N")[0].data, raw_arrays[2])
    assert np.array_equal(candidate.select(channel="BHZ")[0].data, raw_arrays[3])


def _probability_streams() -> tuple[Stream, Stream]:
    reference = Stream(
        [
            _trace("PhaseNet_P", np.zeros(40, dtype=np.float64), sampling_rate=1.0),
            _trace("PhaseNet_S", np.zeros(40, dtype=np.float64), sampling_rate=1.0),
        ]
    )
    return reference, reference.copy()


def test_probability_remote_partition_and_strict_one_e_minus_six_tolerance():
    reference, gapped = _probability_streams()
    # The physical gap is [1010, 1012], so the frozen remote region starts at
    # 1023.  A large local difference and an exactly-on-tolerance remote delta
    # must not be counted.
    gapped.select(channel="PhaseNet_P")[0].data[5] = 0.9
    gapped.select(channel="PhaseNet_P")[0].data[23] = exp.PROBABILITY_TOLERANCE
    result = exp.probability_difference(
        reference,
        gapped,
        _waveform(),
        [(1010.0, 1012.0)],
        {"P": 0.5, "S": 0.5},
    )
    assert result["totals"]["remote_samples"] == 34
    assert result["totals"]["remote_count_gt_1e6"] == 0
    assert result["remote_probability_pass"]

    gapped.select(channel="PhaseNet_P")[0].data[24] = (
        exp.PROBABILITY_TOLERANCE * 1.01
    )
    changed = exp.probability_difference(
        reference,
        gapped,
        _waveform(),
        [(1010.0, 1012.0)],
        {"P": 0.5, "S": 0.5},
    )
    assert changed["totals"]["remote_count_gt_1e6"] == 1
    assert not changed["remote_probability_pass"]


def test_probability_counts_normal_threshold_and_floor_crossings_separately():
    reference, gapped = _probability_streams()
    reference.select(channel="PhaseNet_P")[0].data[30] = 0.49
    gapped.select(channel="PhaseNet_P")[0].data[30] = 0.51
    reference.select(channel="PhaseNet_S")[0].data[31] = 0.02
    gapped.select(channel="PhaseNet_S")[0].data[31] = 0.04
    result = exp.probability_difference(
        reference,
        gapped,
        _waveform(),
        [(1010.0, 1012.0)],
        {"P": 0.5, "S": 0.5},
    )
    assert result["totals"]["remote_normal_threshold_crossing_count"] == 1
    assert result["totals"]["remote_floor_crossing_count"] == 1
    assert not result["remote_probability_pass"]


def test_pick_change_record_separates_remote_induced_and_lost_picks():
    reference = [
        _pick(exp.PhaseType.P, 1005.0),
        _pick(exp.PhaseType.S, 1025.0),
    ]
    other = [
        _pick(exp.PhaseType.P, 1018.0),
        _pick(exp.PhaseType.S, 1040.0),
    ]
    result = exp.pick_change_record(reference, other, [(1020.0, 1022.0)])
    assert result["induced_count"] == 2
    assert result["lost_count"] == 2
    assert result["remote_induced_count"] == 1
    assert result["remote_lost_count"] == 1
    assert [item["remote"] for item in result["induced"]] == [False, True]
    assert [item["remote"] for item in result["lost"]] == [True, False]


def test_collateral_count_ignores_reference_pick_inside_physical_gap():
    reference = [
        _pick(exp.PhaseType.P, 1005.0),
        _pick(exp.PhaseType.S, 1021.0),
    ]
    raw_gapped = list(reference)
    candidate = [reference[1]]
    assert (
        exp.collateral_changed_count(
            reference,
            raw_gapped,
            candidate,
            [(1020.0, 1022.0)],
        )
        == 1
    )


def test_pick_replay_uses_frozen_time_and_confidence_tolerances():
    reference = [_pick(exp.PhaseType.P, 1000.0, 0.5)]
    accepted = [
        _pick(
            exp.PhaseType.P,
            1000.0 + exp.REPLAY_TIME_TOLERANCE_S * 0.99,
            0.5 + exp.REPLAY_CONFIDENCE_TOLERANCE * 0.99,
        )
    ]
    rejected = [
        _pick(
            exp.PhaseType.P,
            1000.0 + exp.REPLAY_TIME_TOLERANCE_S * 1.01,
            0.5 + exp.REPLAY_CONFIDENCE_TOLERANCE * 1.01,
        )
    ]
    assert exp._pick_replay_record(reference, accepted)["pass"]
    assert not exp._pick_replay_record(reference, rejected)["pass"]


def _change_record(*, induced: int = 0, lost: int = 0):
    return {
        "pass": induced == 0 and lost == 0,
        "reference_count": 1,
        "other_count": 1 + induced - lost,
        "matched_count": 1 - lost,
        "induced_count": induced,
        "lost_count": lost,
        "remote_induced_count": 0,
        "remote_lost_count": 0,
        "induced": [],
        "lost": [],
        "matching": {},
    }


def _aggregate_sample(
    guards,
    *,
    eligible_keys=(),
    no_gap_pass: bool = True,
):
    guard_records = {}
    for guard in guards:
        key = exp.guard_key(guard)
        eligible = key in eligible_keys
        guard_records[key] = {
            "eligible": eligible,
            "failure_reasons": [] if eligible else ["residual_induced_new"],
            "final_change": _change_record(induced=0 if eligible else 1),
            "collateral_changed_count": 0,
        }
    return {
        "no_gap_identity": {
            exp.guard_key(guard): {"pass": no_gap_pass} for guard in guards
        },
        "variants": [
            {
                "injection_identity_pass": True,
                "probability": {
                    "alignment_pass": True,
                    "remote_probability_pass": True,
                    "totals": {
                        "remote_samples": 10,
                        "remote_count_gt_1e6": 0,
                        "remote_normal_threshold_crossing_count": 0,
                        "remote_floor_crossing_count": 0,
                        "remote_max_abs_delta": 0.0,
                        "remote_mean_abs_delta": 0.0,
                    },
                },
                "raw_final_change": _change_record(),
                "remote_peak_pass": True,
                "remote_final_pick_pass": True,
                "stage_a_pass": True,
                "guards": guard_records,
            }
        ],
    }


def test_guard_aggregate_selects_smallest_eligible_guard():
    record = _aggregate_sample(
        exp.ACTIVE_GUARDS_S,
        eligible_keys=("1.0", "2.0", "5.0", "10.0"),
    )
    result = exp.aggregate_development([record], exp.ACTIVE_GUARDS_S)
    assert result["stage_a_pass"]
    assert result["no_gap_identity_pass"]
    assert result["selected_guard_s"] == 1.0
    assert list(result["guards"]) == ["0.0", "0.5", "1.0", "2.0", "5.0", "10.0"]


def test_holdout_aggregate_accepts_only_the_frozen_single_guard():
    record = _aggregate_sample((2.0,), eligible_keys=("2.0",))
    result = exp.aggregate_development([record], (2.0,))
    assert list(result["guards"]) == ["2.0"]
    assert result["selected_guard_s"] == 2.0


def test_aggregate_rejects_empty_inputs_and_duplicate_guard_keys():
    with pytest.raises(ValueError):
        exp.aggregate_development([], (1.0,))
    record = _aggregate_sample((1.0,), eligible_keys=("1.0",))
    with pytest.raises(ValueError):
        exp.aggregate_development([record], (1.0, 1.0))


def test_runtime_no_gap_audit_checks_object_identity_and_final_pick_replay(monkeypatch):
    annotations = _annotations()
    observed = []

    def fake_final(_picker, _waveform, candidate):
        observed.append(candidate)
        return []

    monkeypatch.setattr(exp, "final_picks_from_annotations", fake_final)
    result = exp.audit_no_gap_annotations(
        picker=object(),
        waveform=_waveform(),
        annotations=annotations,
        reference_final=[],
        guard_s=5.0,
    )
    assert observed == [annotations]
    assert result["same_stream_object"]
    assert result["annotation_unchanged"]
    assert result["final_pick_replay"]["pass"]
    assert result["pass"]


def test_analyze_guard_uses_zero_filled_waveform_for_peak_and_final_processing(
    monkeypatch,
):
    sample_waveform = _waveform()
    injected_waveform = _waveform()
    injected_waveform.data[:, :] = 7.0
    sample = SimpleNamespace(waveform=sample_waveform)
    injection = SimpleNamespace(
        waveform=injected_waveform,
        gaps_utc=((1010.0, 1012.0),),
    )
    candidate = object()
    observed = []

    monkeypatch.setattr(exp, "annotation_hash", lambda _value: "stable")
    monkeypatch.setattr(
        exp,
        "mask_phase_annotations",
        lambda _raw, _gaps, _guard: candidate,
    )
    monkeypatch.setattr(
        exp,
        "validate_mask_structure",
        lambda _raw, _candidate, _gaps, _guard: {
            "pass": True,
            "failure_reasons": [],
        },
    )

    def fake_threshold(_picker, waveform, _annotations, _thresholds):
        observed.append(("threshold", waveform))
        return {"P": [], "S": []}

    def fake_final(_picker, waveform, _annotations):
        observed.append(("final", waveform))
        return []

    monkeypatch.setattr(exp, "threshold_peak_sets", fake_threshold)
    monkeypatch.setattr(exp, "final_picks_from_annotations", fake_final)
    monkeypatch.setattr(exp, "pick_change_record", lambda *_args: _change_record())
    monkeypatch.setattr(exp, "collateral_changed_count", lambda *_args: 0)

    result = exp.analyze_guard(
        picker=SimpleNamespace(
            _cfg=SimpleNamespace(
                p_threshold=0.3,
                s_threshold=0.3,
                force_pair_floor=0.03,
            )
        ),
        sample=sample,
        injection=injection,
        raw_annotations=object(),
        reference_final=[],
        raw_final=[],
        stage_a_pass=True,
        guard_s=1.0,
    )
    assert result["eligible"]
    assert [kind for kind, _ in observed] == ["threshold", "threshold", "final"]
    assert all(waveform is injected_waveform for _, waveform in observed)


def test_benchmark_returns_frozen_shape_with_small_test_workload(monkeypatch):
    monkeypatch.setattr(exp, "PERFORMANCE_SAMPLES", 5_000)
    monkeypatch.setattr(exp, "PERFORMANCE_GAPS", 2)
    monkeypatch.setattr(exp, "PERFORMANCE_RUNS", 3)
    monkeypatch.setattr(exp, "PERFORMANCE_MAX_P95_MS", 1_000.0)
    result = exp.benchmark_mask()
    assert result["pass"]
    assert result["runs"] == 3
    assert result["samples_per_phase"] == 5_000
    assert result["gap_count"] == 2
    assert result["structure"]["pass"]
    assert result["p95_ms"] >= 0.0
