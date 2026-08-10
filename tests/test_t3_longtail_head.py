"""冻结 SeismicXM 特征分类头的轻量单元测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experiment_t3_longtail_head.py"
_SPEC = importlib.util.spec_from_file_location("experiment_t3_longtail_head", _SCRIPT)
assert _SPEC and _SPEC.loader
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


def _toy(seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = np.eye(5, 8) * 4.0
    xs = []
    ys = []
    for label, center in enumerate(centers, 1):
        xs.append(center + rng.normal(0.0, 0.15, size=(12, 8)))
        ys.extend([label] * 12)
    return np.vstack(xs), np.asarray(ys)


def test_candidate_grid_is_fixed_and_unique():
    grid = exp.candidate_grid()
    assert len(grid) == 26
    assert len({config.name for config in grid}) == len(grid)
    assert grid[0].name == "ncm_raw"


def test_all_heads_predict_five_class_integer_labels():
    x, y = _toy()
    for config in exp.candidate_grid():
        head = exp.FittedHead(config, seed=123).fit(x, y)
        pred, margin, support = head.predict_with_diagnostics(x[:7])
        assert pred.shape == margin.shape == support.shape == (7,)
        assert set(pred).issubset({1, 2, 3, 4, 5})
        assert np.all(np.isfinite(margin)) and np.all(margin >= 0)
        assert np.all(np.isfinite(support))


def test_gate_can_reproduce_baseline_and_candidate_extremes():
    baseline = np.asarray([1, 1, 2, 2])
    candidate = np.asarray([1, 2, 2, 3])
    margin = np.asarray([0.1, 0.2, 0.3, 0.4])
    support = np.asarray([0.5, 0.6, 0.7, 0.8])

    pred, coverage = exp.apply_gate(
        baseline,
        candidate,
        margin,
        support,
        {"margin_threshold": float("inf"), "support_threshold": float("inf")},
    )
    assert np.array_equal(pred, baseline) and coverage == 0.0

    pred, coverage = exp.apply_gate(
        baseline,
        candidate,
        margin,
        support,
        {"margin_threshold": -float("inf"), "support_threshold": -float("inf")},
    )
    assert np.array_equal(pred, candidate) and coverage == 1.0


def test_metrics_balance_only_classes_present_in_truth():
    metrics = exp.metric_record(
        np.asarray([1, 1, 2, 2]),
        np.asarray([1, 4, 2, 4]),
    )
    assert metrics["balanced_accuracy_present_classes"] == 0.5
    assert metrics["recall"] == {
        "1": 0.5,
        "2": 0.5,
        "3": 0.0,
        "4": 0.0,
        "5": 0.0,
    }


def test_small_nested_selection_runs_without_target_leakage():
    from sklearn.model_selection import StratifiedKFold

    x, y = _toy(4)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
    splits = list(cv.split(np.zeros(len(y)), y))
    config, metrics, top, baseline = exp.choose_config(x, y, splits, seed=7)
    assert config in exp.candidate_grid()
    assert metrics["accuracy"] > 0.9
    assert baseline["accuracy"] > 0.9
    assert len(top) == 5
