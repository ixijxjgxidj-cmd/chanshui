"""T2 源包 OOF residual 校准实验的防泄漏与数值测试。"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiment_t2_residual_calibration.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "experiment_t2_residual_calibration", _SCRIPT
)
assert _SPEC and _SPEC.loader
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


def _selection(config, *, baseline_mae=0.30, selected_mae=0.29):
    return {
        "selected_config": exp.config_record(config),
        "baseline_metrics": {"mae": baseline_mae},
        "selected_metrics": {"mae": selected_mae},
        "config_metrics": {
            candidate.name: {
                "mae_delta_candidate_minus_baseline": -0.01,
            }
            for candidate in exp.candidate_grid()
        },
    }


def _cross(*, baseline_mae, candidate_mae, baseline_bias, candidate_bias):
    return {
        "baseline_metrics": {
            "mae": baseline_mae,
            "signed_bias_prediction_minus_truth": baseline_bias,
        },
        "candidate_metrics": {
            "mae": candidate_mae,
            "signed_bias_prediction_minus_truth": candidate_bias,
        },
        "mae_gain_over_baseline": baseline_mae - candidate_mae,
        "absolute_signed_bias_gain": abs(baseline_bias) - abs(candidate_bias),
    }


def test_candidate_grid_is_fixed_unique_and_in_tie_order():
    grid = exp.candidate_grid()
    assert len(grid) == 12
    assert len({config.name for config in grid}) == 12
    assert grid[0] == exp.CandidateConfig(0, 40, 0.25)
    assert grid[-1] == exp.CandidateConfig(16, 20, 0.50)
    assert sorted(grid, key=exp.complexity_key) == grid


def test_cross_fitted_predictions_cover_once_and_exclude_validation_labels():
    x = np.linspace(-1.0, 1.0, 30)[:, None]
    x = np.hstack([x, x**2, np.sin(x * 3.0)])
    y = 5.0 + 0.7 * x[:, 0]
    y[4] += 3.0
    splits = exp.make_kfold_splits(len(y), seed=7)
    oof = exp.cross_fitted_base_predictions(x, y, splits)
    in_sample = exp.clip_prediction(exp.make_base_model().fit(x, y).predict(x))

    assert oof.shape == y.shape
    assert np.all(np.isfinite(oof))
    # 异常点标签没有进入预测它自己的折，和全量训练内预测应有明显差异。
    assert abs(oof[4] - in_sample[4]) > 0.05


def test_calibration_geometry_and_gate_have_expected_shapes():
    rng = np.random.default_rng(3)
    x_source = rng.normal(size=(60, 6))
    base_oof = 5.0 + 0.2 * x_source[:, 0]
    residual = 0.1 * x_source[:, 0] + rng.normal(0.0, 0.02, size=60)
    x_query = rng.normal(size=(11, 6))
    base_query = 5.0 + 0.2 * x_query[:, 0]

    geometry = exp.CalibrationGeometry(8, seed=4).fit(x_source, base_oof)
    distance = geometry.query_distances(x_query, base_query)
    diagnostics = geometry.local_residual_diagnostics(
        distance, residual, k=20
    )

    assert geometry.effective_pca_dim_ == 6
    assert distance.shape == (11, 60)
    assert diagnostics["local_residual"].shape == (11,)
    assert diagnostics["mean_distance"].shape == (11,)
    assert diagnostics["gate"].shape == (11,)
    assert np.isfinite(diagnostics["gate_threshold"])
    assert 0 <= diagnostics["effective_k"] <= 20


def test_prediction_only_local_residual_can_correct_regression_to_mean():
    source_prediction = np.linspace(4.0, 6.0, 80)
    source_residual = 0.20 * (source_prediction - 5.0)
    x_source = np.zeros((80, 2), dtype=np.float64)
    geometry = exp.CalibrationGeometry(0, seed=5).fit(
        x_source, source_prediction
    )

    base_target = np.asarray([5.70, 5.80, 5.90])
    truth = base_target + np.asarray([0.08, 0.09, 0.10])
    distance = geometry.query_distances(
        np.zeros((3, 2), dtype=np.float64), base_target
    )
    diagnostics = geometry.local_residual_diagnostics(
        distance, source_residual, k=20
    )
    candidate, correction = exp.apply_local_correction(
        base_target,
        diagnostics["local_residual"],
        np.ones(3, dtype=bool),
        shrinkage=0.50,
    )

    assert np.all(correction > 0)
    assert exp.metric_record(truth, candidate)["mae"] < exp.metric_record(
        truth, base_target
    )["mae"]


def test_source_fit_interface_has_no_target_label_parameter():
    parameters = inspect.signature(exp.fit_source_and_predict).parameters
    assert set(parameters) == {
        "x_source",
        "y_source",
        "x_target",
        "config",
        "seed",
    }
    assert "y_target" not in parameters


def test_development_decision_requires_both_directions_and_stability():
    config1 = exp.CandidateConfig(8, 40, 0.25)
    config2 = exp.CandidateConfig(16, 20, 0.25)
    selection1 = _selection(config1)
    selection2 = _selection(config2)
    good = _cross(
        baseline_mae=0.62,
        candidate_mae=0.60,
        baseline_bias=-0.58,
        candidate_bias=-0.50,
    )
    weak = _cross(
        baseline_mae=0.66,
        candidate_mae=0.655,
        baseline_bias=0.54,
        candidate_bias=0.52,
    )

    passed, decision = exp.development_decision(
        selection1, selection2, good, weak
    )
    assert not passed
    assert any("r2_to_r1" in reason for reason in decision["failure_reasons"])

    good_reverse = _cross(
        baseline_mae=0.66,
        candidate_mae=0.64,
        baseline_bias=0.54,
        candidate_bias=0.48,
    )
    passed, decision = exp.development_decision(
        selection1, selection2, good, good_reverse
    )
    assert passed
    assert decision["configuration_stable"]


def test_run_does_not_open_terminal_cache_when_development_fails(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(8)
    feature_path = tmp_path / "dev.npz"
    np.savez(
        feature_path,
        X1t2=rng.normal(size=(10, 4)),
        y1t2=rng.normal(4.5, 0.1, size=10),
        X2t2=rng.normal(size=(10, 4)),
        y2t2=rng.normal(5.2, 0.1, size=10),
    )

    failed_selection = {
        "selected_config": None,
        "baseline_metrics": {"mae": 0.2},
        "selected_metrics": {"mae": 0.2},
        "config_metrics": {},
    }

    monkeypatch.setattr(
        exp,
        "nested_source_selection",
        lambda *args, **kwargs: failed_selection,
    )

    def fake_fit(x_source, y_source, x_target, config, *, seed):
        prediction = np.full(len(x_target), float(np.mean(y_source)))
        return {
            "config": None,
            "baseline_prediction": prediction,
            "candidate_prediction": prediction.copy(),
            "local_residual": np.zeros(len(x_target)),
            "correction": np.zeros(len(x_target)),
            "mean_distance": np.full(len(x_target), np.nan),
            "gate": np.zeros(len(x_target), dtype=bool),
            "gate_threshold": None,
            "source_oof_prediction": None,
            "source_oof_metrics": None,
        }

    monkeypatch.setattr(exp, "fit_source_and_predict", fake_fit)
    missing_terminal = tmp_path / "must_not_be_opened.npz"
    result = exp.run(
        feature_path,
        terminal_features=missing_terminal,
        outer_repeats=1,
        expected_feature_sha256=None,
        expected_terminal_sha256=None,
    )

    assert not result["development_pass"]
    assert result["terminal_08"] == {
        "allowed": False,
        "evaluated": False,
        "feature_cache": None,
        "feature_cache_sha256": None,
        "result": None,
    }
    assert not missing_terminal.exists()


def test_small_nested_source_selection_runs_all_fixed_candidates():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(50, 20))
    y = 5.0 + 0.25 * x[:, 0] - 0.15 * x[:, 1] + rng.normal(
        0.0, 0.08, size=50
    )
    result = exp.nested_source_selection(
        x, y, seed=12, outer_repeats=1
    )

    assert len(result["config_metrics"]) == 12
    assert len(result["fold_records"]) == 5
    assert result["n_oof_predictions"] == 50
    assert np.isfinite(result["baseline_metrics"]["mae"])
