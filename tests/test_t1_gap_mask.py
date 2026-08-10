"""Preregistered structural and decision tests for round-05 gap masking."""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experiment_t1_gap_mask.py"
_SPEC = importlib.util.spec_from_file_location("experiment_t1_gap_mask", _SCRIPT)
assert _SPEC and _SPEC.loader
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


def _pick(phase, time_s, confidence=0.5, station="X"):
    return exp.Pick(
        phase=phase,
        time_utc=float(time_s),
        confidence=float(confidence),
        station=station,
    )


def _waveform(duration=60.0, sampling_rate=10.0):
    n = int(duration * sampling_rate)
    data = np.arange(3 * n, dtype=np.float32).reshape(3, n)
    return exp.Waveform(
        data=data,
        sampling_rate=sampling_rate,
        starttime_utc=1000.0,
        station="X.STA",
        gaps=[],
    )


def test_candidate_grid_is_exact_and_off_is_not_numeric():
    assert exp.candidate_grid() == [None, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    assert exp.margin_key(None) == "OFF"
    assert exp.margin_key(0.0) == "0.0"
    assert exp.margin_key(10.0) == "10.0"


def test_normalize_gaps_drops_invalid_expands_and_merges_touching():
    gaps = [
        (10.0, 12.0),
        (12.5, 14.0),
        (30.0, 30.0),
        (40.0, 39.0),
        (math.nan, 50.0),
        (100.0,),
        ("bad", 2.0),
    ]
    assert exp.normalize_gaps(gaps, 0.25) == [(9.75, 14.25)]
    with pytest.raises(ValueError):
        exp.normalize_gaps([], -0.1)
    with pytest.raises(ValueError):
        exp.normalize_gaps([], math.inf)


def test_empty_gap_path_returns_exact_input_list_object():
    picks = [_pick(exp.PhaseType.P, 1.0)]
    for margin in exp.ACTIVE_MARGINS_S:
        assert exp.mask_gap_picks(picks, [], margin) is picks


def test_mask_uses_closed_boundaries_and_preserves_objects_order_and_fields():
    before = [
        _pick(exp.PhaseType.P, 8.999),
        _pick(exp.PhaseType.P, 9.0),
        _pick(exp.PhaseType.S, 10.0),
        _pick(exp.PhaseType.P, 11.0),
        _pick(exp.PhaseType.S, 11.501),
        _pick(exp.PhaseType.P, math.nan),
    ]
    tokens = [exp.pick_token(pick) for pick in before]
    out = exp.mask_gap_picks(before, [(10.0, 10.0), (10.0, 10.5)], 1.0)
    assert out == [before[0], before[4], before[5]]
    assert all(actual is expected for actual, expected in zip(out, [before[0], before[4], before[5]]))
    assert [exp.pick_token(pick) for pick in before] == tokens
    assert exp.is_stable_identity_subsequence(out, before)


def test_mask_handles_unsorted_picks_without_reordering():
    picks = [
        _pick(exp.PhaseType.P, 20.0),
        _pick(exp.PhaseType.S, 5.0),
        _pick(exp.PhaseType.P, 11.0),
        _pick(exp.PhaseType.S, 30.0),
    ]
    out = exp.mask_gap_picks(picks, [(10.0, 12.0)], 0.0)
    assert out == [picks[0], picks[1], picks[3]]
    assert exp.is_stable_identity_subsequence(out, picks)


def test_matching_is_phase_specific_tolerance_bounded_and_deterministic():
    reference = [
        _pick(exp.PhaseType.P, 10.0),
        _pick(exp.PhaseType.P, 10.15),
        _pick(exp.PhaseType.S, 20.0),
    ]
    other = [
        _pick(exp.PhaseType.P, 10.08),
        _pick(exp.PhaseType.P, 10.16),
        _pick(exp.PhaseType.S, 20.19),
        _pick(exp.PhaseType.S, 30.0),
    ]
    result = exp.match_pick_lists(reference, other)
    assert result["matches"] == [
        (0, 0, pytest.approx(0.08)),
        (1, 1, pytest.approx(0.01)),
        (2, 2, pytest.approx(0.19)),
    ]
    assert result["induced_indices"] == [3]
    assert result["lost_indices"] == []
    assert not result["pass"]


def test_distance_to_gap_is_zero_inside_and_infinite_without_valid_gap():
    gaps = [(10.0, 12.0), (20.0, 21.0)]
    assert exp.distance_to_gaps(11.0, gaps) == 0.0
    assert exp.distance_to_gaps(12.0, gaps) == 0.0
    assert exp.distance_to_gaps(16.0, gaps) == 4.0
    assert math.isinf(exp.distance_to_gaps(math.nan, gaps))
    assert math.isinf(exp.distance_to_gaps(1.0, []))


def test_candidate_analysis_accepts_local_induced_pick_when_mask_removes_it():
    reference = [
        _pick(exp.PhaseType.P, 10.0),
        _pick(exp.PhaseType.S, 50.0),
    ]
    raw = [reference[0], _pick(exp.PhaseType.P, 20.0), reference[1]]
    result = exp.analyze_candidate(reference, raw, [(19.0, 21.0)], 0.0)
    assert result["induced_new_indices"] == [1]
    assert result["residual_induced_new_indices"] == []
    assert result["remote_induced_new_indices"] == []
    assert result["collateral_deleted"] == []
    assert result["pass"]


def test_candidate_analysis_rejects_remote_effect_even_with_wide_margin():
    reference = [_pick(exp.PhaseType.P, 10.0)]
    raw = [reference[0], _pick(exp.PhaseType.S, 40.0)]
    result = exp.analyze_candidate(reference, raw, [(19.0, 21.0)], 10.0)
    assert result["remote_induced_new_indices"] == [1]
    assert not result["pass"]


def test_candidate_analysis_rejects_collateral_deletion_outside_physical_gap():
    reference = [_pick(exp.PhaseType.P, 12.0)]
    raw = [_pick(exp.PhaseType.P, 12.0)]
    result = exp.analyze_candidate(reference, raw, [(10.0, 11.0)], 1.0)
    assert result["stable_outside_physical_gap"] == [[0, 0]]
    assert result["collateral_deleted"] == [[0, 0]]
    assert not result["pass"]


def test_variant_specs_are_exact_and_single_component_is_hash_determined():
    rows = exp.variant_specs(sample_key="sample-A", duration_s=60.0, anchor_s=20.0)
    assert [row[0] for row in rows] == [
        "mid-0p5-all",
        "mid-2-all",
        "mid-10-all",
        "anchor-center-2-all",
        "anchor-edge-2-all",
        "double-2-10-all",
        "anchor-center-2-one",
    ]
    assert all(row[1] == (0, 1, 2) for row in rows[:6])
    expected = exp.hashlib.sha256(b"sample-A").digest()[0] % 3
    assert rows[-1][1] == (expected,)
    assert rows[4][2] == ((17.5, 19.5),)


def test_injection_zeroes_only_target_region_and_is_deterministic():
    wf = _waveform()
    first = exp.inject_gaps(
        wf,
        variant_id="one",
        components=(1,),
        relative_intervals_s=((10.0, 12.0),),
    )
    second = exp.inject_gaps(
        wf,
        variant_id="one",
        components=(1,),
        relative_intervals_s=((10.0, 12.0),),
    )
    assert first.identity_pass
    assert first.array_sha256 == second.array_sha256
    assert first.gaps_utc == ((1010.0, 1012.0),)
    assert np.array_equal(first.waveform.data[0], wf.data[0])
    assert np.array_equal(first.waveform.data[2], wf.data[2])
    assert np.all(first.waveform.data[1, 100:120] == 0.0)
    assert np.array_equal(first.waveform.data[1, :100], wf.data[1, :100])


def test_select_anchor_prefers_earliest_in_guard_and_falls_back_to_midpoint():
    wf = _waveform(duration=60.0)
    picks = [
        _pick(exp.PhaseType.P, 1005.0),
        _pick(exp.PhaseType.S, 1030.0),
        _pick(exp.PhaseType.P, 1020.0),
    ]
    assert exp.select_anchor(picks, wf) == 20.0
    assert exp.select_anchor([], wf) == 30.0


def test_long_crop_is_exact_360_seconds_and_keeps_time_alignment():
    wf = _waveform(duration=1000.0, sampling_rate=10.0)
    picks = [_pick(exp.PhaseType.P, 1600.0)]
    cropped, record = exp.crop_long_waveform(wf, picks)
    assert cropped.duration == pytest.approx(360.0)
    assert record["full_anchor_relative_s"] == pytest.approx(600.0)
    assert record["crop_start_relative_s"] == pytest.approx(420.0)
    assert cropped.starttime_utc == pytest.approx(1420.0)
    assert np.array_equal(cropped.data, wf.data[:, 4200:7800])


def test_noise_samples_are_repeatable_and_independent_across_components():
    first = exp.make_noise_samples()
    second = exp.make_noise_samples()
    assert [item.key for item in first] == ["noise-short", "noise-long"]
    assert [exp.sha256_array(item.waveform.data) for item in first] == [
        exp.sha256_array(item.waveform.data) for item in second
    ]
    assert not np.array_equal(first[0].waveform.data[0], first[0].waveform.data[1])


def _analysis_record(*, fail=False):
    return {
        "subset_and_order_pass": True,
        "object_fields_unchanged_pass": True,
        "candidate_inside_expanded_gap_indices": [],
        "residual_induced_new_indices": [0] if fail else [],
        "remote_induced_new_indices": [],
        "remote_lost_reference_indices": [],
        "collateral_deleted": [],
    }


def _development_record(fail_margin_keys=()):
    margins = {}
    for margin in exp.ACTIVE_MARGINS_S:
        key = exp.margin_key(margin)
        margins[key] = {
            "single": _analysis_record(fail=key in fail_margin_keys),
            "batch": _analysis_record(fail=key in fail_margin_keys),
            "single_batch_candidate": {"pass": True},
        }
    return {
        "reference": {
            "no_gap_identity": {
                exp.margin_key(margin): True for margin in exp.ACTIVE_MARGINS_S
            },
            "single_batch": {"pass": True},
            "batch_repeat_exact": True,
        },
        "variants": {
            "v": {
                "input_identity_pass": True,
                "raw_single_batch": {"pass": True},
                "batch_repeat_exact": True,
                "margins": margins,
            }
        },
    }


def test_margin_selection_chooses_smallest_eligible_and_never_backfills_grid():
    records = {
        "sample": _development_record(fail_margin_keys=("0.0", "0.5"))
    }
    assessments, selected = exp.assess_development_margins(records)
    assert not assessments["0.0"]["eligible"]
    assert not assessments["0.5"]["eligible"]
    assert assessments["1.0"]["eligible"]
    assert selected == 1.0
    assert list(assessments) == ["0.0", "0.5", "1.0", "2.0", "5.0", "10.0"]


def test_no_eligible_margin_selects_off():
    records = {
        "sample": _development_record(
            fail_margin_keys=tuple(exp.margin_key(x) for x in exp.ACTIVE_MARGINS_S)
        )
    }
    assessments, selected = exp.assess_development_margins(records)
    assert selected is None
    assert all(not record["eligible"] for record in assessments.values())


def test_holdout_json_keeps_only_frozen_active_margin():
    records = {"sample": _development_record()}
    stripped = exp._strip_unselected_holdout_margins(copy.deepcopy(records), 2.0)
    assert list(stripped["sample"]["variants"]["v"]["margins"]) == ["2.0"]


def test_json_safe_replaces_nonfinite_numbers_recursively():
    value = {"a": math.inf, "b": [math.nan, {"c": -math.inf}], "d": 1.0}
    assert exp._json_safe(value) == {
        "a": None,
        "b": [None, {"c": None}],
        "d": 1.0,
    }


def test_large_filter_remains_a_stable_subsequence_without_quadratic_semantics():
    picks = [_pick(exp.PhaseType.P, idx * 0.1) for idx in range(10_000)]
    gaps = [(idx * 10.0 + 1.0, idx * 10.0 + 1.5) for idx in range(100)]
    out = exp.mask_gap_picks(picks, gaps, 1.0)
    assert exp.is_stable_identity_subsequence(out, picks)
    assert len(out) < len(picks)
