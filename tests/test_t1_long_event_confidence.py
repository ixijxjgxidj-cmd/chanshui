"""T1 长记录事件级置信度实验的预注册边界与决策测试。"""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiment_t1_long_event_confidence.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "experiment_t1_long_event_confidence", _SCRIPT
)
assert _SPEC and _SPEC.loader
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


def _pick(phase, time_s, confidence, *, station="X"):
    return exp.Pick(
        phase=phase,
        time_utc=time_s,
        confidence=confidence,
        station=station,
    )


def _package_evaluation(*, threshold=0.35, gain=0.02, fp_drop=2):
    baseline_modes = {
        "merged_file_floor0": {"total_score": 10.0},
        "per_phase_floor0": {"total_score": 10.0},
    }
    candidate_modes = {
        "merged_file_floor0": {"total_score": 10.0},
        "per_phase_floor0": {"total_score": 10.0},
    }
    baseline = {
        "p_time_score": 5.0,
        "s_time_score": 5.0,
        "phase_errors": {
            "P": {"false_positives": 3, "false_negatives": 1},
            "S": {"false_positives": 3, "false_negatives": 1},
        },
        "modes": baseline_modes,
    }
    candidate = {
        "p_time_score": 5.0,
        "s_time_score": 5.0,
        "phase_errors": {
            "P": {"false_positives": 2, "false_negatives": 1},
            "S": {"false_positives": 2, "false_negatives": 1},
        },
        "modes": candidate_modes,
    }
    mode_delta = gain * 100.0
    return {
        "threshold": threshold,
        "active_file_ids": ["long.mseed"],
        "per_file": {
            "long.mseed": {
                "baseline": baseline,
                "candidate": candidate,
                "output_is_baseline_subset": True,
                "filter": {"deleted_events": 1},
            }
        },
        "package_modes": {
            mode: {"delta": mode_delta, "normalized_gain": gain}
            for mode in exp.PENALTY_MODES
        },
        "worst_normalized_gain": gain,
        "errors": {
            "baseline": {"false_positives": 6, "false_negatives": 2},
            "candidate": {
                "false_positives": 6 - fp_drop,
                "false_negatives": 2,
            },
        },
        "deleted_events": 1,
    }


def test_candidate_grid_is_exact_and_rejects_unregistered_thresholds():
    assert exp.candidate_grid() == [None, 0.35, 0.40, 0.45]
    with pytest.raises(ValueError):
        exp.filter_deduplicated_events([], 0.30)
    with pytest.raises(ValueError):
        exp.filter_deduplicated_events([], 0.50)


def test_fifo_pairing_is_non_crossing_deterministic_and_ignores_confidence():
    picks = [
        _pick(exp.PhaseType.S, 0.10, 0.99),
        _pick(exp.PhaseType.P, 0.00, 0.01),
        _pick(exp.PhaseType.P, 1.00, 0.99),
        _pick(exp.PhaseType.S, 0.30, 0.01),
        _pick(exp.PhaseType.S, 1.30, 0.99),
    ]
    first = exp.fifo_pair_events(picks)
    second = exp.fifo_pair_events(picks)
    assert first == second
    assert first.pairs == (
        exp.EventPair(p_index=1, s_index=3),
        exp.EventPair(p_index=2, s_index=4),
    )
    assert first.orphan_s_indices == (0,)
    assert first.orphan_p_indices == ()


def test_orphans_are_always_retained():
    picks = [
        _pick(exp.PhaseType.P, 0.0, 0.01),
        _pick(exp.PhaseType.S, 100.0, 0.01),
        _pick(exp.PhaseType.P, 200.0, 0.01),
    ]
    outcome = exp.filter_deduplicated_events(picks, 0.45)
    assert outcome.picks == picks
    assert outcome.deleted_events == 0
    assert outcome.pairing.pairs == ()
    assert outcome.pairing.orphan_p_indices == (0, 2)
    assert outcome.pairing.orphan_s_indices == (1,)


def test_low_complete_pair_is_deleted_and_high_pair_is_retained():
    picks = [
        _pick(exp.PhaseType.P, 0.0, 0.20),
        _pick(exp.PhaseType.S, 1.0, 0.20),
        _pick(exp.PhaseType.P, 100.0, 0.90),
        _pick(exp.PhaseType.S, 101.0, 0.90),
    ]
    outcome = exp.filter_deduplicated_events(picks, 0.35)
    assert outcome.picks == picks[2:]
    assert outcome.dropped_indices == (0, 1)
    assert outcome.deleted_events == 1
    assert [record["keep"] for record in outcome.pair_decisions] == [False, True]


def test_threshold_comparison_is_strict_less_than():
    picks = [
        _pick(exp.PhaseType.P, 0.0, 0.40),
        _pick(exp.PhaseType.S, 1.0, 0.40),
    ]
    outcome = exp.filter_deduplicated_events(picks, 0.40)
    assert outcome.picks == picks
    assert outcome.pair_decisions[0]["event_confidence"] == pytest.approx(0.40)
    assert outcome.pair_decisions[0]["decision"] == "keep_at_or_above_threshold"


def test_nonfinite_confidence_pairs_are_kept_and_json_safe():
    picks = [
        _pick(exp.PhaseType.P, 0.0, math.nan),
        _pick(exp.PhaseType.S, 1.0, math.inf),
    ]
    outcome = exp.filter_deduplicated_events(picks, 0.45)
    decision = outcome.pair_decisions[0]
    assert outcome.picks == picks
    assert decision["keep"]
    assert decision["nonfinite_confidence_keep"]
    assert decision["p_confidence"] is None
    assert decision["s_confidence"] is None
    assert decision["event_confidence"] is None


def test_duration_at_or_below_300_is_object_and_order_identical():
    picks = [
        _pick(exp.PhaseType.S, 2.0, 0.2),
        _pick(exp.PhaseType.P, 1.0, 0.2),
        _pick(exp.PhaseType.P, 1.01, 0.9),
    ]
    for duration in (0.0, 299.999999, 300.0):
        outcome = exp.apply_event_confidence_filter(
            picks, duration_s=duration, threshold=0.45
        )
        assert outcome.path == "short_unchanged"
        assert len(outcome.picks) == len(picks)
        assert all(before is after for before, after in zip(picks, outcome.picks))
        assert outcome.picks == picks


def test_only_300_000001_enters_long_path_and_deduplicates():
    picks = [
        _pick(exp.PhaseType.P, 1.0, 0.2),
        _pick(exp.PhaseType.P, 1.01, 0.9),
    ]
    outcome = exp.apply_event_confidence_filter(
        picks, duration_s=300.000001, threshold=None
    )
    assert outcome.path == "long_dedup_then_event_filter"
    assert len(outcome.picks) == 1
    assert outcome.picks[0] is picks[1]


def test_candidate_output_is_always_a_baseline_subset():
    raw = [
        _pick(exp.PhaseType.P, 0.0, 0.2),
        _pick(exp.PhaseType.P, 0.01, 0.8),
        _pick(exp.PhaseType.S, 1.0, 0.2),
        _pick(exp.PhaseType.S, 1.01, 0.8),
        _pick(exp.PhaseType.P, 100.0, 0.2),
        _pick(exp.PhaseType.S, 101.0, 0.2),
    ]
    for threshold in exp.candidate_grid():
        outcome = exp.apply_event_confidence_filter(
            raw, duration_s=4000.0, threshold=threshold
        )
        baseline_ids = {id(pick) for pick in outcome.baseline_picks}
        assert all(id(pick) in baseline_ids for pick in outcome.picks)


def test_scoring_reproduces_two_decimal_submission_boundary():
    picks = [
        _pick(exp.PhaseType.P, 1.2349, 0.8),
        _pick(exp.PhaseType.S, 2.3451, 0.8),
    ]
    assert exp._prediction_pairs(picks) == [("P", 1.23), ("S", 2.35)]


def test_filter_result_is_deterministic():
    raw = [
        _pick(exp.PhaseType.S, 101.0, 0.2),
        _pick(exp.PhaseType.P, 100.0, 0.2),
        _pick(exp.PhaseType.P, 0.0, 0.9),
        _pick(exp.PhaseType.S, 1.0, 0.9),
    ]
    first = exp.apply_event_confidence_filter(
        raw, duration_s=4000.0, threshold=0.35
    )
    second = exp.apply_event_confidence_filter(
        raw, duration_s=4000.0, threshold=0.35
    )
    assert first.to_record() == second.to_record()
    assert [exp._pick_token(pick) for pick in first.picks] == [
        exp._pick_token(pick) for pick in second.picks
    ]


def test_source_eligibility_requires_every_conjunct():
    good = _package_evaluation()
    assert exp.source_eligibility(good)["eligible"]

    mutations = []
    fn_bad = copy.deepcopy(good)
    fn_bad["per_file"]["long.mseed"]["candidate"]["phase_errors"]["P"][
        "false_negatives"
    ] += 1
    mutations.append(fn_bad)

    time_bad = copy.deepcopy(good)
    time_bad["per_file"]["long.mseed"]["candidate"]["p_time_score"] -= 2e-9
    mutations.append(time_bad)

    file_score_bad = copy.deepcopy(good)
    file_score_bad["per_file"]["long.mseed"]["candidate"]["modes"][
        "merged_file_floor0"
    ]["total_score"] -= 1e-6
    mutations.append(file_score_bad)

    package_bad = copy.deepcopy(good)
    package_bad["package_modes"]["merged_exam"]["delta"] = 0.0
    mutations.append(package_bad)

    fp_bad = copy.deepcopy(good)
    fp_bad["errors"]["candidate"]["false_positives"] = 6
    mutations.append(fp_bad)

    gain_bad = copy.deepcopy(good)
    gain_bad["worst_normalized_gain"] = 0.009
    mutations.append(gain_bad)

    subset_bad = copy.deepcopy(good)
    subset_bad["per_file"]["long.mseed"]["output_is_baseline_subset"] = False
    mutations.append(subset_bad)

    for bad in mutations:
        result = exp.source_eligibility(bad)
        assert not result["eligible"]
        assert result["reasons"]


def test_source_selection_uses_lower_tau_within_tie_tolerance():
    evaluations = {
        "0.35": _package_evaluation(threshold=0.35, gain=0.02000),
        "0.40": _package_evaluation(threshold=0.40, gain=0.02005),
    }
    selection = exp.select_source_threshold(evaluations)
    assert selection["selected_tau"] == 0.35

    evaluations["0.40"] = _package_evaluation(
        threshold=0.40, gain=0.02020
    )
    selection = exp.select_source_threshold(evaluations)
    assert selection["selected_tau"] == 0.40

    evaluations["0.35"] = _package_evaluation(
        threshold=0.35, gain=0.02000
    )
    evaluations["0.40"] = _package_evaluation(
        threshold=0.40, gain=0.02005
    )
    evaluations["0.45"] = _package_evaluation(
        threshold=0.45, gain=0.02014
    )
    selection = exp.select_source_threshold(evaluations)
    # 0.40 距离全局最大 0.00009，应按冻结 tie 规则压过 0.45。
    assert selection["selected_tau"] == 0.40


def test_lofo_off_or_two_grid_step_span_is_unstable():
    stable = exp.assess_selection_stability(
        0.35, {"a": 0.35, "b": 0.40}
    )
    assert stable["pass"]
    assert stable["span"] == pytest.approx(0.05)

    off = exp.assess_selection_stability(0.35, {"a": None})
    assert not off["pass"]

    crossed = exp.assess_selection_stability(
        0.35, {"a": 0.45}
    )
    assert not crossed["pass"]
    assert crossed["span"] == pytest.approx(0.10)


def test_bidirectional_failure_fails_development_and_common_is_minimum():
    selection_r2 = {"selected_tau": 0.35}
    selection_08 = {"selected_tau": 0.40}
    stability = {"pass": True, "reasons": []}
    safety = {"pass": True, "reasons": []}
    r2_evaluations = {
        "0.35": _package_evaluation(threshold=0.35),
        "0.40": _package_evaluation(threshold=0.40),
    }
    final_evaluations = {
        "0.35": _package_evaluation(threshold=0.35),
        "0.40": _package_evaluation(threshold=0.40),
    }
    bad_reverse = copy.deepcopy(r2_evaluations["0.40"])
    bad_reverse["package_modes"]["merged_exam"]["delta"] = 0.0
    r2_evaluations["0.40"] = bad_reverse

    decision = exp.development_decision(
        round2_selection=selection_r2,
        final08_selection=selection_08,
        round2_stability=stability,
        final08_stability=stability,
        round2_evaluations=r2_evaluations,
        final08_evaluations=final_evaluations,
        safety=safety,
    )
    assert not decision["bidirectional_pass"]
    assert not decision["development_pass"]
    assert decision["common_candidate"]["tau_common"] == 0.35
    assert any("08→R2" in reason for reason in decision["failure_reasons"])

    r2_evaluations["0.40"] = _package_evaluation(threshold=0.40)
    decision = exp.development_decision(
        round2_selection=selection_r2,
        final08_selection=selection_08,
        round2_stability=stability,
        final08_stability=stability,
        round2_evaluations=r2_evaluations,
        final08_evaluations=final_evaluations,
        safety=safety,
    )
    assert decision["bidirectional_pass"]
    assert decision["common_candidate"]["tau_common"] == min(0.35, 0.40)
    assert decision["development_pass"]


def test_baseline_reproduction_failure_never_runs_active_thresholds():
    called = []

    def evaluator(threshold):
        called.append(threshold)
        return {"threshold": threshold}

    assert exp.evaluate_active_candidates_after_reproduction(False, evaluator) == {}
    assert called == []

    result = exp.evaluate_active_candidates_after_reproduction(True, evaluator)
    assert list(result) == ["0.35", "0.40", "0.45"]
    assert called == [0.35, 0.40, 0.45]
