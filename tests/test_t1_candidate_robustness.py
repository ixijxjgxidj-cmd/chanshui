from __future__ import annotations

import pytest

from scripts.audit_t1_candidate_robustness import (
    PENALTY_MODES,
    compare_candidate,
    parse_case,
    read_t1_answers,
    score_dataset,
)


def _write_t1(path, p_time: float, s_time: float) -> None:
    path.write_text(
        f"T1.A.Q0001.mseed : P : {p_time:.2f} : S : {s_time:.2f}\n",
        encoding="utf-8",
    )


def test_parse_case_requires_g7_baseline(tmp_path):
    answer = tmp_path / "answer.an"
    pred = tmp_path / "pred.an"

    with pytest.raises(ValueError, match="g7"):
        parse_case(f"round1={answer}::cond={pred}")


def test_parse_case_collects_named_predictions(tmp_path):
    answer = tmp_path / "answer.an"
    g7 = tmp_path / "g7.an"
    cond = tmp_path / "cond.an"

    spec = parse_case(f"round1={answer}::g7={g7}::cond={cond}")

    assert spec.name == "round1"
    assert spec.answer_path == answer.resolve()
    assert list(spec.predictions) == ["g7", "cond"]


def test_improved_candidate_passes_all_modes(tmp_path):
    answer_path = tmp_path / "answer.an"
    baseline_path = tmp_path / "g7.an"
    candidate_path = tmp_path / "candidate.an"
    _write_t1(answer_path, 10.0, 20.0)
    _write_t1(baseline_path, 10.5, 21.0)
    _write_t1(candidate_path, 10.0, 20.0)

    answers = read_t1_answers(answer_path)
    baseline = score_dataset(read_t1_answers(baseline_path), answers)
    candidate = score_dataset(read_t1_answers(candidate_path), answers)
    result = compare_candidate("candidate", candidate, baseline)

    assert result["coverage"]["complete"] is True
    assert result["all_modes_non_decreasing"] is True
    assert result["has_strict_improvement"] is True
    assert result["robust_pass"] is True
    assert all(result["package_modes"][mode]["delta"] > 0 for mode in PENALTY_MODES)
    assert result["errors"]["false_negative_delta"] == 0


def test_incomplete_candidate_fails_robust_gate(tmp_path):
    answer_path = tmp_path / "answer.an"
    baseline_path = tmp_path / "g7.an"
    candidate_path = tmp_path / "candidate.an"
    answer_path.write_text(
        "T1.A.Q0001.mseed : P : 10.00 : S : 20.00\n"
        "T1.A.Q0002.mseed : P : 30.00 : S : 40.00\n",
        encoding="utf-8",
    )
    baseline_path.write_text(answer_path.read_text(encoding="utf-8"), encoding="utf-8")
    _write_t1(candidate_path, 10.0, 20.0)

    answers = read_t1_answers(answer_path)
    baseline = score_dataset(read_t1_answers(baseline_path), answers)
    candidate = score_dataset(read_t1_answers(candidate_path), answers)
    result = compare_candidate("candidate", candidate, baseline)

    assert result["coverage"]["missing"] == 1
    assert result["coverage"]["complete"] is False
    assert result["robust_pass"] is False
