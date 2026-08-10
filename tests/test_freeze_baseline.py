from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phasepicker.types import Task1Result, Task2Result, Task3Result


def _load_module():
    path = ROOT / "scripts" / "freeze_baseline.py"
    spec = importlib.util.spec_from_file_location("freeze_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fb = _load_module()


def test_parse_dataset_same_and_split_packages(tmp_path):
    same = fb.parse_dataset(f"r1={tmp_path / 'r1.zip'}")
    assert same.exam_path == same.answer_path

    split = fb.parse_dataset(
        f"f08={tmp_path / 'exam.zip'}::{tmp_path / 'answers.zip'}"
    )
    assert split.name == "f08"
    assert split.exam_path.endswith("exam.zip")
    assert split.answer_path.endswith("answers.zip")


def test_t1_frozen_reports_all_modes_and_strict_missing():
    truth = {
        "a.mseed": Task1Result("a.mseed", [1.0], [2.0]),
        "b.mseed": Task1Result("b.mseed", [1.0], [2.0]),
    }
    pred = {
        "a.mseed": Task1Result("a.mseed", [1.0], [2.0]),
        "extra.mseed": Task1Result("extra.mseed", [1.0], [2.0]),
    }
    result = fb.evaluate_t1_frozen(pred, truth)
    assert set(result["penalty_modes"]) == set(fb.PENALTY_MODES)
    assert result["coverage"]["missing"] == ["b.mseed"]
    assert result["coverage"]["extra"] == ["extra.mseed"]
    assert result["penalty_modes"]["merged_file_floor0"]["n_files"] == 2
    assert result["per_file"]["b.mseed"]["false_negatives"] == 2


def test_t2_frozen_reports_bias_and_mae():
    truth = {
        "a": Task2Result("a", 4.0),
        "b": Task2Result("b", 5.0),
    }
    pred = {
        "a": Task2Result("a", 4.5),
        "b": Task2Result("b", 4.5),
    }
    result = fb.evaluate_t2_frozen(pred, truth)
    assert result["mae"] == 0.5
    assert result["mean_signed_error"] == 0.0
    assert result["suspicious_prediction_equals_answers"] is False


def test_t3_frozen_per_class_recall_and_label_coverage():
    truth = {
        "a": Task3Result("a", 1),
        "b": Task3Result("b", 1),
        "c": Task3Result("c", 2),
    }
    pred = {
        "a": Task3Result("a", 1),
        "b": Task3Result("b", 2),
        "c": Task3Result("c", 2),
    }
    result = fb.evaluate_t3_frozen(pred, truth)
    assert result["accuracy"] == 2 / 3
    assert result["per_class_recall"] == {"1": 0.5, "2": 1.0}
    assert result["labels_seen_in_truth"] == [1, 2]
    assert result["suspicious_prediction_equals_answers"] is False


def test_perfect_full_coverage_is_flagged_as_possible_answer_leakage():
    t2 = {"a": Task2Result("a", 4.0)}
    t3 = {"a": Task3Result("a", 1)}
    assert fb.evaluate_t2_frozen(t2, t2)["suspicious_prediction_equals_answers"] is True
    assert fb.evaluate_t3_frozen(t3, t3)["suspicious_prediction_equals_answers"] is True


def test_gap_intervals_use_station_relative_left_and_right_boundaries():
    class FakeUTC(float):
        @property
        def timestamp(self):
            return float(self)

    def trace(start, end):
        return SimpleNamespace(
            stats=SimpleNamespace(
                channel="BHZ",
                starttime=FakeUTC(start),
                endtime=FakeUTC(end),
                sampling_rate=100.0,
            )
        )

    gaps, overlaps = fb._station_gaps([trace(10.0, 11.0), trace(12.0, 13.0)], 10.0)
    assert overlaps == 0
    assert gaps == [
        {
            "channel": "BHZ",
            "start_relative_s": 1.0,
            "end_relative_s": 2.0,
            "duration_s": 1.0,
        }
    ]
