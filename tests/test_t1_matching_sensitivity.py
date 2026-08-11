from __future__ import annotations

import pytest

from scripts.audit_t1_matching_sensitivity import (
    match_phases_exact,
    match_score,
    parse_case,
)
from phasepicker.scoring.scorer import match_phases


def test_exact_matcher_detects_greedy_counterexample():
    pred_times = [0.0, 0.1]
    true_times = [0.1, 0.2]

    greedy = match_phases(pred_times, true_times, "P")
    exact = match_phases_exact(pred_times, true_times, "P", "max_time_score")

    assert match_score(greedy.matched, "P") == pytest.approx(17.0 / 9.0)
    assert match_score(exact.matched, "P") == pytest.approx(2.0)


def test_exact_matcher_leaves_out_of_window_items_unmatched():
    result = match_phases_exact([0.0], [1.0], "P", "max_time_score")

    assert result.matched == []
    assert result.unmatched_pred == [0]
    assert result.unmatched_true == [0]


def test_exact_matcher_empty_inputs_are_reported():
    result = match_phases_exact([], [0.1, 0.2], "S")

    assert result.matched == []
    assert result.unmatched_pred == []
    assert result.unmatched_true == [0, 1]


def test_parse_case_requires_answer_and_prediction_paths():
    with pytest.raises(ValueError):
        parse_case("round1=answers.zip")
