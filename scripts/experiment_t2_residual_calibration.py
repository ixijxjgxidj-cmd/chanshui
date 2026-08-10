#!/usr/bin/env python3
"""T2 源包 OOF residual 的低维局部校准实验。

协议已预注册在：
    memory/experiments/003-t2-cross-package-residual-calibration.md

该脚本只读取冻结的 SeismicXM 特征，不调用编码器。配置只能由源包嵌套
OOF 选择；R1/R2 双向开发门槛失败时，即使传入 ``--terminal-features``
也不会打开 08 缓存。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEV_FEATURE_SHA256 = (
    "ce213a04472713bb9f76667f62d9132ae0fa08539a9945058534759b746ea574"
)
TERMINAL_FEATURE_SHA256 = (
    "357b851dce6352648b0fadf169715ca6e47f0c170c078b4d5241fae97607d8d4"
)
BASELINE_COMMIT = "acad92d"
BASELINE_ALPHA = 30.0
PREDICTION_MIN = 0.0
PREDICTION_MAX = 9.9
RESIDUAL_CAP = 0.20
DISTANCE_EPSILON = 0.05
GATE_QUANTILE = 0.90
SOURCE_SELECTION_MIN_GAIN = 0.005
SELECTION_TIE_TOLERANCE = 0.001
DEVELOPMENT_DIRECTION_MIN_GAIN = 0.010
DEVELOPMENT_MEAN_MIN_GAIN = 0.015
TERMINAL_BASELINE_MAE = 0.523197455
TERMINAL_BASELINE_SIGNED_BIAS = 0.099781905
TERMINAL_REQUIRED_MAE = TERMINAL_BASELINE_MAE - 0.010


@dataclass(frozen=True)
class CandidateConfig:
    """固定的 residual 校准配置。"""

    pca_dim: int
    k: int
    shrinkage: float

    @property
    def name(self) -> str:
        shrink = int(round(self.shrinkage * 100))
        return f"pca{self.pca_dim}_k{self.k}_s{shrink:02d}"


def candidate_grid() -> list[CandidateConfig]:
    """返回预注册的 12 个候选，顺序也是复杂度并列顺序。"""

    return [
        CandidateConfig(pca_dim=pca_dim, k=k, shrinkage=shrinkage)
        for pca_dim in (0, 8, 16)
        for k in (40, 20)
        for shrinkage in (0.25, 0.50)
    ]


def config_record(config: CandidateConfig | None) -> dict | None:
    if config is None:
        return None
    return {"name": config.name, **asdict(config)}


def complexity_key(config: CandidateConfig) -> tuple[int, int, int, str]:
    return (
        {0: 0, 8: 1, 16: 2}[config.pca_dim],
        {40: 0, 20: 1}[config.k],
        {0.25: 0, 0.50: 1}[config.shrinkage],
        config.name,
    )


def make_base_model():
    """构造冻结的 T2 Ridge 基线。"""

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge(alpha=BASELINE_ALPHA))


def clip_prediction(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), PREDICTION_MIN, PREDICTION_MAX)


def metric_record(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError(f"truth/prediction shape 不一致：{truth.shape} vs {pred.shape}")
    error = pred - truth
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "signed_bias_prediction_minus_truth": float(np.mean(error)),
        "max_absolute_error": float(np.max(np.abs(error))),
        "prediction_mean": float(np.mean(pred)),
        "truth_mean": float(np.mean(truth)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or len(x) < 2:
        return None
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    if np.std(rx) <= 1e-15 or np.std(ry) <= 1e-15:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_kfold_splits(
    n_samples: int,
    *,
    seed: int,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold

    if n_samples < 2:
        raise ValueError("至少需要 2 个样本生成 OOF 预测")
    folds = min(int(n_splits), int(n_samples))
    cv = KFold(n_splits=folds, shuffle=True, random_state=int(seed))
    placeholder = np.zeros(n_samples, dtype=np.float64)
    return [(train, valid) for train, valid in cv.split(placeholder)]


def cross_fitted_base_predictions(
    x: np.ndarray,
    y: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """对每个样本只用未包含其标签的折模型产生预测。"""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) != len(y):
        raise ValueError("x/y 长度不一致")
    prediction = np.full(len(y), np.nan, dtype=np.float64)
    coverage = np.zeros(len(y), dtype=np.int64)
    for train_index, valid_index in splits:
        train_index = np.asarray(train_index, dtype=np.int64)
        valid_index = np.asarray(valid_index, dtype=np.int64)
        if np.intersect1d(train_index, valid_index).size:
            raise ValueError("交叉拟合 split 的 train/valid 有重叠")
        model = make_base_model().fit(x[train_index], y[train_index])
        prediction[valid_index] = clip_prediction(model.predict(x[valid_index]))
        coverage[valid_index] += 1
    if not np.all(coverage == 1):
        raise ValueError(f"OOF coverage 必须恰为 1，实际范围 {coverage.min()}–{coverage.max()}")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("OOF 预测含非有限值")
    return prediction


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


class CalibrationGeometry:
    """只由源包拟合的低维余弦邻域几何。"""

    def __init__(self, pca_dim: int, *, seed: int):
        self.pca_dim = int(pca_dim)
        self.seed = int(seed)

    def fit(self, x_source: np.ndarray, base_oof_prediction: np.ndarray):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        x_source = np.asarray(x_source, dtype=np.float64)
        base_oof_prediction = np.asarray(
            base_oof_prediction, dtype=np.float64
        ).reshape(-1)
        if len(x_source) != len(base_oof_prediction):
            raise ValueError("源特征与 OOF 预测长度不一致")

        self.feature_scaler_ = None
        self.pca_ = None
        self.effective_pca_dim_ = min(
            self.pca_dim,
            max(0, len(x_source) - 1),
            x_source.shape[1],
        )
        parts = [base_oof_prediction[:, None]]
        if self.effective_pca_dim_:
            self.feature_scaler_ = StandardScaler().fit(x_source)
            source_scaled = self.feature_scaler_.transform(x_source)
            self.pca_ = PCA(
                n_components=self.effective_pca_dim_,
                whiten=True,
                svd_solver="randomized",
                random_state=self.seed,
            ).fit(source_scaled)
            parts.append(self.pca_.transform(source_scaled))

        learned = np.column_stack(parts)
        self.calibration_scaler_ = StandardScaler().fit(learned)
        standardized = self.calibration_scaler_.transform(learned)
        homogeneous = np.column_stack(
            [standardized, np.ones(len(standardized), dtype=np.float64)]
        )
        self.source_unit_ = _unit_rows(homogeneous)
        self.source_distances_ = np.clip(
            1.0 - self.source_unit_ @ self.source_unit_.T,
            0.0,
            2.0,
        )
        return self

    def query_distances(
        self,
        x_query: np.ndarray,
        base_prediction: np.ndarray,
    ) -> np.ndarray:
        x_query = np.asarray(x_query, dtype=np.float64)
        base_prediction = np.asarray(base_prediction, dtype=np.float64).reshape(-1)
        if len(x_query) != len(base_prediction):
            raise ValueError("查询特征与基线预测长度不一致")
        parts = [base_prediction[:, None]]
        if self.effective_pca_dim_:
            query_scaled = self.feature_scaler_.transform(x_query)
            parts.append(self.pca_.transform(query_scaled))
        learned = np.column_stack(parts)
        standardized = self.calibration_scaler_.transform(learned)
        homogeneous = np.column_stack(
            [standardized, np.ones(len(standardized), dtype=np.float64)]
        )
        query_unit = _unit_rows(homogeneous)
        return np.clip(1.0 - query_unit @ self.source_unit_.T, 0.0, 2.0)

    def local_residual_diagnostics(
        self,
        query_distances: np.ndarray,
        source_residual: np.ndarray,
        *,
        k: int,
    ) -> dict[str, np.ndarray | float | int]:
        distances = np.asarray(query_distances, dtype=np.float64)
        residual = np.asarray(source_residual, dtype=np.float64).reshape(-1)
        if distances.ndim != 2 or distances.shape[1] != len(residual):
            raise ValueError("查询距离矩阵与 residual 库不匹配")

        effective_k = min(int(k), len(residual))
        neighbor_index = np.argpartition(
            distances, kth=effective_k - 1, axis=1
        )[:, :effective_k]
        neighbor_distance = np.take_along_axis(distances, neighbor_index, axis=1)
        neighbor_residual = residual[neighbor_index]
        weight = 1.0 / (neighbor_distance + DISTANCE_EPSILON)
        local_residual = np.sum(weight * neighbor_residual, axis=1) / np.sum(
            weight, axis=1
        )
        mean_distance = np.mean(neighbor_distance, axis=1)

        source_loo = self.source_distances_.copy()
        np.fill_diagonal(source_loo, np.inf)
        source_k = min(int(k), max(1, len(residual) - 1))
        source_neighbor = np.partition(source_loo, kth=source_k - 1, axis=1)[
            :, :source_k
        ]
        source_mean_distance = np.mean(source_neighbor, axis=1)
        threshold = float(np.quantile(source_mean_distance, GATE_QUANTILE))
        gate = mean_distance <= threshold
        return {
            "local_residual": local_residual,
            "mean_distance": mean_distance,
            "gate": gate,
            "gate_threshold": threshold,
            "effective_k": effective_k,
        }


def apply_local_correction(
    base_prediction: np.ndarray,
    local_residual: np.ndarray,
    gate: np.ndarray,
    *,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_prediction, dtype=np.float64).reshape(-1)
    local = np.asarray(local_residual, dtype=np.float64).reshape(-1)
    gate = np.asarray(gate, dtype=bool).reshape(-1)
    if base.shape != local.shape or base.shape != gate.shape:
        raise ValueError("base/local/gate shape 不一致")
    correction = (
        np.clip(local, -RESIDUAL_CAP, RESIDUAL_CAP)
        * float(shrinkage)
        * gate.astype(np.float64)
    )
    return clip_prediction(base + correction), correction


def _selection_from_metrics(
    baseline_metrics: dict[str, float],
    config_metrics: dict[str, dict[str, float]],
) -> tuple[CandidateConfig | None, str]:
    configs = candidate_grid()
    best_mae = min(config_metrics[config.name]["mae"] for config in configs)
    tied = [
        config
        for config in configs
        if config_metrics[config.name]["mae"] <= best_mae + SELECTION_TIE_TOLERANCE
    ]
    selected = min(tied, key=complexity_key)
    gain = baseline_metrics["mae"] - config_metrics[selected.name]["mae"]
    if gain + 1e-12 < SOURCE_SELECTION_MIN_GAIN:
        return None, (
            f"最优激活配置源包 OOF 改善 {gain:.6f}，低于 "
            f"{SOURCE_SELECTION_MIN_GAIN:.3f}，回退 Ridge"
        )
    return selected, f"源包 OOF 改善 {gain:.6f}，选择 {selected.name}"


def nested_source_selection(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    outer_repeats: int = 5,
) -> dict:
    """只用单个源包完成重复嵌套 OOF 配置选择。"""

    from sklearn.model_selection import KFold

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    configs = candidate_grid()
    truth_all: list[np.ndarray] = []
    base_all: list[np.ndarray] = []
    prediction_all: dict[str, list[np.ndarray]] = {
        config.name: [] for config in configs
    }
    gate_all: dict[str, list[np.ndarray]] = {config.name: [] for config in configs}
    distance_all: dict[str, list[np.ndarray]] = {
        config.name: [] for config in configs
    }
    fold_records: list[dict] = []

    for repeat in range(int(outer_repeats)):
        outer = KFold(
            n_splits=min(5, len(y)),
            shuffle=True,
            random_state=seed + repeat,
        )
        for fold, (train_index, valid_index) in enumerate(
            outer.split(np.zeros(len(y), dtype=np.float64))
        ):
            x_train, y_train = x[train_index], y[train_index]
            x_valid, y_valid = x[valid_index], y[valid_index]
            inner_splits = make_kfold_splits(
                len(y_train), seed=seed + repeat * 100 + fold + 1
            )
            train_oof = cross_fitted_base_predictions(
                x_train, y_train, inner_splits
            )
            train_residual = y_train - train_oof
            base_model = make_base_model().fit(x_train, y_train)
            base_valid = clip_prediction(base_model.predict(x_valid))

            fold_candidate: dict[str, list[float]] = {}
            fold_gate: dict[str, list[bool]] = {}
            fold_distance: dict[str, list[float]] = {}
            for pca_dim in (0, 8, 16):
                geometry = CalibrationGeometry(
                    pca_dim, seed=seed + repeat * 1000 + fold
                ).fit(x_train, train_oof)
                query_distances = geometry.query_distances(x_valid, base_valid)
                diagnostics_by_k = {
                    k: geometry.local_residual_diagnostics(
                        query_distances, train_residual, k=k
                    )
                    for k in (40, 20)
                }
                for config in [c for c in configs if c.pca_dim == pca_dim]:
                    diagnostics = diagnostics_by_k[config.k]
                    prediction, _ = apply_local_correction(
                        base_valid,
                        diagnostics["local_residual"],
                        diagnostics["gate"],
                        shrinkage=config.shrinkage,
                    )
                    prediction_all[config.name].append(prediction)
                    gate_all[config.name].append(
                        np.asarray(diagnostics["gate"], dtype=bool)
                    )
                    distance_all[config.name].append(
                        np.asarray(diagnostics["mean_distance"], dtype=np.float64)
                    )
                    fold_candidate[config.name] = prediction.tolist()
                    fold_gate[config.name] = np.asarray(
                        diagnostics["gate"], dtype=bool
                    ).tolist()
                    fold_distance[config.name] = np.asarray(
                        diagnostics["mean_distance"], dtype=np.float64
                    ).tolist()

            truth_all.append(y_valid)
            base_all.append(base_valid)
            fold_records.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "valid_indices": valid_index.tolist(),
                    "truth": y_valid.tolist(),
                    "baseline_prediction": base_valid.tolist(),
                    "candidate_prediction": fold_candidate,
                    "candidate_gate": fold_gate,
                    "candidate_mean_distance": fold_distance,
                }
            )

    truth_flat = np.concatenate(truth_all)
    base_flat = np.concatenate(base_all)
    baseline_metrics = metric_record(truth_flat, base_flat)
    config_metrics: dict[str, dict[str, float]] = {}
    for config in configs:
        prediction = np.concatenate(prediction_all[config.name])
        gate = np.concatenate(gate_all[config.name])
        distance = np.concatenate(distance_all[config.name])
        metrics = metric_record(truth_flat, prediction)
        metrics.update(
            {
                "mae_delta_candidate_minus_baseline": (
                    metrics["mae"] - baseline_metrics["mae"]
                ),
                "gate_coverage": float(np.mean(gate)),
                "distance_absolute_error_spearman": spearman_correlation(
                    distance, np.abs(prediction - truth_flat)
                ),
            }
        )
        config_metrics[config.name] = metrics

    selected, selection_reason = _selection_from_metrics(
        baseline_metrics, config_metrics
    )
    if selected is None:
        selected_prediction = base_flat
        selected_gate = np.zeros(len(base_flat), dtype=bool)
        selected_distance = np.full(len(base_flat), np.nan, dtype=np.float64)
    else:
        selected_prediction = np.concatenate(prediction_all[selected.name])
        selected_gate = np.concatenate(gate_all[selected.name])
        selected_distance = np.concatenate(distance_all[selected.name])
    selected_metrics = metric_record(truth_flat, selected_prediction)
    selected_metrics.update(
        {
            "mae_gain_over_baseline": (
                baseline_metrics["mae"] - selected_metrics["mae"]
            ),
            "gate_coverage": float(np.mean(selected_gate)),
            "distance_absolute_error_spearman": (
                None
                if selected is None
                else spearman_correlation(
                    selected_distance,
                    np.abs(selected_prediction - truth_flat),
                )
            ),
        }
    )
    return {
        "outer_repeats": int(outer_repeats),
        "outer_folds": 5,
        "n_oof_predictions": int(len(truth_flat)),
        "baseline_metrics": baseline_metrics,
        "config_metrics": config_metrics,
        "selected_config": config_record(selected),
        "selection_reason": selection_reason,
        "selected_metrics": selected_metrics,
        "fold_records": fold_records,
    }


def config_from_record(record: dict | None) -> CandidateConfig | None:
    if record is None:
        return None
    return CandidateConfig(
        pca_dim=int(record["pca_dim"]),
        k=int(record["k"]),
        shrinkage=float(record["shrinkage"]),
    )


def fit_source_and_predict(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    config: CandidateConfig | None,
    *,
    seed: int,
) -> dict[str, np.ndarray | float | int | dict | None]:
    """只用源特征/标签拟合；接口刻意没有目标标签参数。"""

    x_source = np.asarray(x_source, dtype=np.float64)
    y_source = np.asarray(y_source, dtype=np.float64).reshape(-1)
    x_target = np.asarray(x_target, dtype=np.float64)
    base_model = make_base_model().fit(x_source, y_source)
    base_target = clip_prediction(base_model.predict(x_target))

    if config is None:
        return {
            "config": None,
            "baseline_prediction": base_target,
            "candidate_prediction": base_target.copy(),
            "local_residual": np.zeros(len(base_target), dtype=np.float64),
            "correction": np.zeros(len(base_target), dtype=np.float64),
            "mean_distance": np.full(len(base_target), np.nan, dtype=np.float64),
            "gate": np.zeros(len(base_target), dtype=bool),
            "gate_threshold": None,
            "source_oof_prediction": None,
            "source_oof_metrics": None,
        }

    source_splits = make_kfold_splits(len(y_source), seed=seed)
    source_oof = cross_fitted_base_predictions(
        x_source, y_source, source_splits
    )
    source_residual = y_source - source_oof
    geometry = CalibrationGeometry(config.pca_dim, seed=seed).fit(
        x_source, source_oof
    )
    query_distances = geometry.query_distances(x_target, base_target)
    diagnostics = geometry.local_residual_diagnostics(
        query_distances, source_residual, k=config.k
    )
    candidate_target, correction = apply_local_correction(
        base_target,
        diagnostics["local_residual"],
        diagnostics["gate"],
        shrinkage=config.shrinkage,
    )
    return {
        "config": config_record(config),
        "baseline_prediction": base_target,
        "candidate_prediction": candidate_target,
        "local_residual": np.asarray(
            diagnostics["local_residual"], dtype=np.float64
        ),
        "correction": correction,
        "mean_distance": np.asarray(
            diagnostics["mean_distance"], dtype=np.float64
        ),
        "gate": np.asarray(diagnostics["gate"], dtype=bool),
        "gate_threshold": float(diagnostics["gate_threshold"]),
        "source_oof_prediction": source_oof,
        "source_oof_metrics": metric_record(y_source, source_oof),
    }


def evaluate_target_prediction(
    fitted: dict,
    y_target: np.ndarray,
) -> dict:
    y_target = np.asarray(y_target, dtype=np.float64).reshape(-1)
    baseline = np.asarray(fitted["baseline_prediction"], dtype=np.float64)
    candidate = np.asarray(fitted["candidate_prediction"], dtype=np.float64)
    local = np.asarray(fitted["local_residual"], dtype=np.float64)
    correction = np.asarray(fitted["correction"], dtype=np.float64)
    distance = np.asarray(fitted["mean_distance"], dtype=np.float64)
    gate = np.asarray(fitted["gate"], dtype=bool)
    baseline_metrics = metric_record(y_target, baseline)
    candidate_metrics = metric_record(y_target, candidate)
    finite_distance = np.isfinite(distance)
    correlation = (
        spearman_correlation(
            distance[finite_distance],
            np.abs(candidate[finite_distance] - y_target[finite_distance]),
        )
        if np.any(finite_distance)
        else None
    )
    coverage_metrics = {}
    for name, mask in (("gate_on", gate), ("gate_off", ~gate)):
        coverage_metrics[name] = (
            metric_record(y_target[mask], candidate[mask]) if np.any(mask) else None
        )
    per_sample = [
        {
            "index": int(index),
            "truth": float(y_target[index]),
            "baseline_prediction": float(baseline[index]),
            "candidate_prediction": float(candidate[index]),
            "baseline_error": float(baseline[index] - y_target[index]),
            "candidate_error": float(candidate[index] - y_target[index]),
            "local_residual": float(local[index]),
            "correction": float(correction[index]),
            "mean_distance": (
                None if not np.isfinite(distance[index]) else float(distance[index])
            ),
            "gate": bool(gate[index]),
        }
        for index in range(len(y_target))
    ]
    return {
        "config": fitted["config"],
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "mae_gain_over_baseline": (
            baseline_metrics["mae"] - candidate_metrics["mae"]
        ),
        "absolute_signed_bias_gain": (
            abs(baseline_metrics["signed_bias_prediction_minus_truth"])
            - abs(candidate_metrics["signed_bias_prediction_minus_truth"])
        ),
        "gate_coverage": float(np.mean(gate)),
        "gate_threshold": fitted["gate_threshold"],
        "coverage_metrics": coverage_metrics,
        "distance_absolute_error_spearman": correlation,
        "source_oof_metrics": fitted["source_oof_metrics"],
        "per_sample": per_sample,
    }


def configuration_stability(
    first: CandidateConfig | None,
    second: CandidateConfig | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if first is None or second is None:
        reasons.append("至少一个源包未达到 0.005 OOF 改善并回退 Ridge")
        return False, reasons
    if first.shrinkage != second.shrinkage:
        reasons.append("两个源包选择的 shrinkage 不同")
    rank = {0: 0, 8: 1, 16: 2}
    if abs(rank[first.pca_dim] - rank[second.pca_dim]) > 1:
        reasons.append("两个源包 PCA 维数相差两档")
    return not reasons, reasons


def development_decision(
    first_selection: dict,
    second_selection: dict,
    first_to_second: dict,
    second_to_first: dict,
) -> tuple[bool, dict]:
    first_config = config_from_record(first_selection["selected_config"])
    second_config = config_from_record(second_selection["selected_config"])
    stable, stability_reasons = configuration_stability(
        first_config, second_config
    )
    reasons = list(stability_reasons)

    directional = {
        "r1_to_r2": float(first_to_second["mae_gain_over_baseline"]),
        "r2_to_r1": float(second_to_first["mae_gain_over_baseline"]),
    }
    for direction, gain in directional.items():
        if gain + 1e-12 < DEVELOPMENT_DIRECTION_MIN_GAIN:
            reasons.append(
                f"{direction} MAE 改善 {gain:.6f} 低于 "
                f"{DEVELOPMENT_DIRECTION_MIN_GAIN:.3f}"
            )
    average_gain = float(np.mean(list(directional.values())))
    if average_gain + 1e-12 < DEVELOPMENT_MEAN_MIN_GAIN:
        reasons.append(
            f"双向平均 MAE 改善 {average_gain:.6f} 低于 "
            f"{DEVELOPMENT_MEAN_MIN_GAIN:.3f}"
        )
    for direction, result in (
        ("r1_to_r2", first_to_second),
        ("r2_to_r1", second_to_first),
    ):
        if result["absolute_signed_bias_gain"] <= 0.0:
            reasons.append(f"{direction} 的绝对 signed bias 未下降")
        if result["candidate_metrics"]["mae"] > result["baseline_metrics"]["mae"]:
            reasons.append(f"{direction} MAE 恶化")
    for package, selection in (
        ("r1", first_selection),
        ("r2", second_selection),
    ):
        oof_degradation = (
            selection["selected_metrics"]["mae"]
            - selection["baseline_metrics"]["mae"]
        )
        if oof_degradation > 0.010 + 1e-12:
            reasons.append(f"{package} 源包 OOF MAE 恶化超过 0.010")
    return not reasons, {
        "configuration_stable": stable,
        "directional_mae_gain": directional,
        "average_mae_gain": average_gain,
        "failure_reasons": reasons,
    }


def choose_consensus_config(
    first_selection: dict,
    second_selection: dict,
) -> tuple[CandidateConfig, dict[str, float]]:
    scores: dict[str, float] = {}
    for config in candidate_grid():
        first_delta = first_selection["config_metrics"][config.name][
            "mae_delta_candidate_minus_baseline"
        ]
        second_delta = second_selection["config_metrics"][config.name][
            "mae_delta_candidate_minus_baseline"
        ]
        scores[config.name] = float((first_delta + second_delta) / 2.0)
    best = min(scores.values())
    tied = [
        config
        for config in candidate_grid()
        if scores[config.name] <= best + SELECTION_TIE_TOLERANCE
    ]
    return min(tied, key=complexity_key), scores


def _load_development_cache(path: Path) -> tuple[np.ndarray, ...]:
    with np.load(path, allow_pickle=False) as cache:
        required = {"X1t2", "y1t2", "X2t2", "y2t2"}
        missing = sorted(required - set(cache.files))
        if missing:
            raise KeyError(f"开发特征缓存缺键：{missing}")
        return tuple(
            np.asarray(cache[key], dtype=np.float64)
            for key in ("X1t2", "y1t2", "X2t2", "y2t2")
        )


def _load_terminal_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        required = {"X", "y"}
        missing = sorted(required - set(cache.files))
        if missing:
            raise KeyError(f"08 特征缓存缺键：{missing}")
        return (
            np.asarray(cache["X"], dtype=np.float64),
            np.asarray(cache["y"], dtype=np.float64),
        )


def run(
    features: Path,
    *,
    terminal_features: Path | None = None,
    seed: int = 20260811,
    outer_repeats: int = 5,
    expected_feature_sha256: str | None = DEV_FEATURE_SHA256,
    expected_terminal_sha256: str | None = TERMINAL_FEATURE_SHA256,
) -> dict:
    started = time.perf_counter()
    feature_hash = sha256_file(features)
    if expected_feature_sha256 and feature_hash != expected_feature_sha256:
        raise ValueError(
            f"开发缓存 SHA-256 不匹配：{feature_hash} != {expected_feature_sha256}"
        )
    x1, y1, x2, y2 = _load_development_cache(features)
    y1 = y1.reshape(-1)
    y2 = y2.reshape(-1)

    first_selection = nested_source_selection(
        x1, y1, seed=seed, outer_repeats=outer_repeats
    )
    second_selection = nested_source_selection(
        x2, y2, seed=seed + 10000, outer_repeats=outer_repeats
    )
    first_config = config_from_record(first_selection["selected_config"])
    second_config = config_from_record(second_selection["selected_config"])

    first_fitted = fit_source_and_predict(
        x1, y1, x2, first_config, seed=seed + 20000
    )
    second_fitted = fit_source_and_predict(
        x2, y2, x1, second_config, seed=seed + 30000
    )
    first_to_second = evaluate_target_prediction(first_fitted, y2)
    second_to_first = evaluate_target_prediction(second_fitted, y1)
    development_pass, decision = development_decision(
        first_selection,
        second_selection,
        first_to_second,
        second_to_first,
    )

    terminal: dict = {
        "allowed": development_pass,
        "evaluated": False,
        "feature_cache": None,
        "feature_cache_sha256": None,
        "result": None,
    }
    if development_pass and terminal_features is not None:
        # 关键防泄漏边界：只有 development_pass 后才计算终检哈希并打开缓存。
        terminal_hash = sha256_file(terminal_features)
        if expected_terminal_sha256 and terminal_hash != expected_terminal_sha256:
            raise ValueError(
                "08 缓存 SHA-256 不匹配："
                f"{terminal_hash} != {expected_terminal_sha256}"
            )
        x_terminal, y_terminal = _load_terminal_cache(terminal_features)
        consensus, consensus_scores = choose_consensus_config(
            first_selection, second_selection
        )
        x_combined = np.vstack([x1, x2])
        y_combined = np.concatenate([y1, y2])
        terminal_fitted = fit_source_and_predict(
            x_combined,
            y_combined,
            x_terminal,
            consensus,
            seed=seed + 40000,
        )
        terminal_result = evaluate_target_prediction(
            terminal_fitted, y_terminal
        )
        terminal_reasons: list[str] = []
        if terminal_result["candidate_metrics"]["mae"] > TERMINAL_REQUIRED_MAE + 1e-12:
            terminal_reasons.append(
                f"08 MAE 高于 {TERMINAL_REQUIRED_MAE:.9f}"
            )
        terminal_bias = abs(
            terminal_result["candidate_metrics"][
                "signed_bias_prediction_minus_truth"
            ]
        )
        if terminal_bias > TERMINAL_BASELINE_SIGNED_BIAS + 1e-12:
            terminal_reasons.append(
                "08 绝对 signed bias 高于冻结基线 0.099781905"
            )
        terminal_result.update(
            {
                "consensus_config": config_record(consensus),
                "consensus_source_oof_mean_deltas": consensus_scores,
                "terminal_pass": not terminal_reasons,
                "terminal_failure_reasons": terminal_reasons,
                "required_mae": TERMINAL_REQUIRED_MAE,
                "maximum_absolute_signed_bias": TERMINAL_BASELINE_SIGNED_BIAS,
            }
        )
        terminal = {
            "allowed": True,
            "evaluated": True,
            "feature_cache": str(terminal_features),
            "feature_cache_sha256": terminal_hash,
            "result": terminal_result,
        }

    import sklearn

    return {
        "experiment_id": "t2-cross-package-residual-calibration-20260811",
        "baseline_commit": BASELINE_COMMIT,
        "seed": int(seed),
        "outer_repeats": int(outer_repeats),
        "development_feature_cache": str(features),
        "development_feature_cache_sha256": feature_hash,
        "shapes": {
            "X1t2": list(x1.shape),
            "y1t2": list(y1.shape),
            "X2t2": list(x2.shape),
            "y2t2": list(y2.shape),
        },
        "label_summary": {
            "round1": {
                "mean": float(np.mean(y1)),
                "std": float(np.std(y1)),
                "min": float(np.min(y1)),
                "max": float(np.max(y1)),
            },
            "round2": {
                "mean": float(np.mean(y2)),
                "std": float(np.std(y2)),
                "min": float(np.min(y2)),
                "max": float(np.max(y2)),
            },
        },
        "grid": [config_record(config) for config in candidate_grid()],
        "round1_source_selection": first_selection,
        "round2_source_selection": second_selection,
        "r1_to_r2": first_to_second,
        "r2_to_r1": second_to_first,
        "development_pass": development_pass,
        "development_decision": decision,
        "terminal_08": terminal,
        "runtime_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        default="outputs/seismicxm_t3/features_tta.npz",
    )
    parser.add_argument(
        "--terminal-features",
        default=None,
        help="仅 development_pass 后才会打开的冻结 08 T2 特征缓存",
    )
    parser.add_argument(
        "--output",
        default="outputs/experiments/round03_t2_residual_calibration.json",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--outer-repeats", type=int, default=5)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    result = run(
        Path(args.features),
        terminal_features=(
            None if args.terminal_features is None else Path(args.terminal_features)
        ),
        seed=args.seed,
        outer_repeats=args.outer_repeats,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output": str(output),
        "round1_selected": result["round1_source_selection"]["selected_config"],
        "round2_selected": result["round2_source_selection"]["selected_config"],
        "round1_oof": {
            "baseline": result["round1_source_selection"]["baseline_metrics"]["mae"],
            "selected": result["round1_source_selection"]["selected_metrics"]["mae"],
        },
        "round2_oof": {
            "baseline": result["round2_source_selection"]["baseline_metrics"]["mae"],
            "selected": result["round2_source_selection"]["selected_metrics"]["mae"],
        },
        "r1_to_r2": {
            "baseline_mae": result["r1_to_r2"]["baseline_metrics"]["mae"],
            "candidate_mae": result["r1_to_r2"]["candidate_metrics"]["mae"],
            "gain": result["r1_to_r2"]["mae_gain_over_baseline"],
        },
        "r2_to_r1": {
            "baseline_mae": result["r2_to_r1"]["baseline_metrics"]["mae"],
            "candidate_mae": result["r2_to_r1"]["candidate_metrics"]["mae"],
            "gain": result["r2_to_r1"]["mae_gain_over_baseline"],
        },
        "development_pass": result["development_pass"],
        "failure_reasons": result["development_decision"]["failure_reasons"],
        "terminal_08": {
            "allowed": result["terminal_08"]["allowed"],
            "evaluated": result["terminal_08"]["evaluated"],
        },
        "runtime_seconds": result["runtime_seconds"],
    }
    if result["terminal_08"]["evaluated"]:
        terminal = result["terminal_08"]["result"]
        summary["terminal_08"].update(
            {
                "candidate_mae": terminal["candidate_metrics"]["mae"],
                "terminal_pass": terminal["terminal_pass"],
                "failure_reasons": terminal["terminal_failure_reasons"],
            }
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
