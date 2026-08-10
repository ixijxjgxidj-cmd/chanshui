#!/usr/bin/env python3
"""预注册实验 002：冻结 SeismicXM 表征上的 T3 长尾分类头。

只读取 ``features_tta.npz``，不触碰生产权重。开发协议：

1. 第 1 轮做 5 折 × 5 重复外层验证；每个外层训练折内用 4 折选择头与门控。
2. 比较生产 ``L2 + cosine kNN(k=5)``、中心化 NCM、多原型、每类 top-m
   局部余弦分数及二者收缩组合。
3. 第 2 轮只作跨包准入检查，不参与配置排序。
4. 只有 JSON 中 ``development_pass=true``，才允许按预注册提取 08 特征终检。

运行：

    python scripts/experiment_t3_longtail_head.py \
      --features outputs/seismicxm_t3/features_tta.npz \
      --output outputs/experiments/round02_t3_longtail_head.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


LABELS = np.asarray([1, 2, 3, 4, 5], dtype=np.int64)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-12)


@dataclass(frozen=True)
class HeadConfig:
    kind: str
    center: bool = False
    top_m: int = 0
    alpha: float = 0.0
    n_prototypes: int = 0

    @property
    def name(self) -> str:
        bits = [self.kind, "center" if self.center else "raw"]
        if self.top_m:
            bits.append(f"m{self.top_m}")
        if self.alpha:
            bits.append(f"a{self.alpha:g}")
        if self.n_prototypes:
            bits.append(f"p{self.n_prototypes}")
        return "_".join(bits)


def candidate_grid() -> list[HeadConfig]:
    """固定顺序也是完全并列时的低复杂度 tie-break。"""
    grid: list[HeadConfig] = []
    for center in (False, True):
        grid.append(HeadConfig("ncm", center=center))
    for center in (False, True):
        for top_m in (1, 3, 5, 10):
            grid.append(HeadConfig("topm", center=center, top_m=top_m))
    for center in (False, True):
        for top_m in (3, 5):
            for alpha in (0.25, 0.5, 0.75):
                grid.append(
                    HeadConfig("hybrid", center=center, top_m=top_m, alpha=alpha)
                )
    for center in (False, True):
        for n_prototypes in (2, 3):
            grid.append(
                HeadConfig(
                    "multiproto", center=center, n_prototypes=n_prototypes
                )
            )
    return grid


class FittedHead:
    def __init__(self, config: HeadConfig, seed: int = 0):
        self.config = config
        self.seed = int(seed)
        self.mean_: np.ndarray | None = None
        self.class_vectors_: list[np.ndarray] = []
        self.centroids_: np.ndarray | None = None
        self.prototypes_: list[np.ndarray] = []

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.config.center:
            if self.mean_ is None:
                raise RuntimeError("centered head 尚未 fit")
            x = x - self.mean_
        return _l2_normalize(x)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FittedHead":
        from sklearn.cluster import KMeans

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.mean_ = x.mean(axis=0, keepdims=True) if self.config.center else None
        z = self._transform(x)
        self.class_vectors_ = [z[y == label] for label in LABELS]
        if any(v.size == 0 for v in self.class_vectors_):
            missing = [int(c) for c, v in zip(LABELS, self.class_vectors_) if not v.size]
            raise ValueError(f"训练折缺少类别：{missing}")

        centroids = [v.mean(axis=0) for v in self.class_vectors_]
        self.centroids_ = _l2_normalize(np.stack(centroids))

        self.prototypes_ = []
        if self.config.kind == "multiproto":
            for class_idx, vectors in enumerate(self.class_vectors_):
                k = min(self.config.n_prototypes, len(vectors))
                model = KMeans(
                    n_clusters=k,
                    n_init=10,
                    random_state=self.seed + class_idx,
                )
                model.fit(vectors)
                self.prototypes_.append(_l2_normalize(model.cluster_centers_))
        return self

    def score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 class score、top1 margin、最近同预测类训练支持。"""
        if self.centroids_ is None:
            raise RuntimeError("head 尚未 fit")
        q = self._transform(x)
        n = len(q)
        local_scores = np.empty((n, len(LABELS)), dtype=np.float64)
        support_scores = np.empty_like(local_scores)
        for j, vectors in enumerate(self.class_vectors_):
            sims = q @ vectors.T
            support_scores[:, j] = sims.max(axis=1)
            m = min(max(1, self.config.top_m), sims.shape[1])
            top = np.partition(sims, sims.shape[1] - m, axis=1)[:, -m:]
            local_scores[:, j] = top.mean(axis=1)

        centroid_scores = q @ self.centroids_.T
        if self.config.kind == "ncm":
            scores = centroid_scores
        elif self.config.kind == "topm":
            scores = local_scores
        elif self.config.kind == "hybrid":
            scores = self.config.alpha * centroid_scores + (1.0 - self.config.alpha) * local_scores
        elif self.config.kind == "multiproto":
            scores = np.column_stack(
                [(q @ prototypes.T).max(axis=1) for prototypes in self.prototypes_]
            )
        else:  # pragma: no cover - 配置网格固定
            raise ValueError(f"未知 head kind：{self.config.kind}")

        order = np.argsort(scores, axis=1)
        pred_idx = order[:, -1]
        margin = scores[np.arange(n), order[:, -1]] - scores[np.arange(n), order[:, -2]]
        support = support_scores[np.arange(n), pred_idx]
        return scores, margin, support

    def predict_with_diagnostics(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, margin, support = self.score(x)
        pred = LABELS[np.argmax(scores, axis=1)]
        return pred.astype(np.int64), margin, support

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_with_diagnostics(x)[0]


def fit_baseline(x: np.ndarray, y: np.ndarray):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    return make_pipeline(
        Normalizer(), KNeighborsClassifier(5, metric="cosine")
    ).fit(x, y)


def metric_record(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    recalls = recall_score(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    present_labels = np.unique(y_true)
    present_recalls = recall_score(
        y_true,
        y_pred,
        labels=present_labels,
        average=None,
        zero_division=0,
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy_present_classes": float(np.mean(present_recalls)),
        "recall": {str(int(k)): float(v) for k, v in zip(LABELS, recalls)},
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "correct": int(np.sum(y_true == y_pred)),
    }


def evaluate_config_on_splits(
    x: np.ndarray,
    y: np.ndarray,
    config: HeadConfig,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    seed: int,
    with_diagnostics: bool = False,
) -> dict:
    truths: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        head = FittedHead(config, seed=seed + 1009 * fold_id).fit(
            x[train_idx], y[train_idx]
        )
        pred, margin, support = head.predict_with_diagnostics(x[test_idx])
        truths.append(y[test_idx])
        preds.append(pred)
        margins.append(margin)
        supports.append(support)
        indices.append(np.asarray(test_idx, dtype=np.int64))
    out = metric_record(np.concatenate(truths), np.concatenate(preds))
    if with_diagnostics:
        out["y_true"] = np.concatenate(truths)
        out["y_pred"] = np.concatenate(preds)
        out["margin"] = np.concatenate(margins)
        out["support"] = np.concatenate(supports)
        out["indices"] = np.concatenate(indices)
    return out


def evaluate_baseline_on_splits(
    x: np.ndarray,
    y: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    with_predictions: bool = False,
) -> dict:
    truths: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for train_idx, test_idx in splits:
        pred = fit_baseline(x[train_idx], y[train_idx]).predict(x[test_idx])
        truths.append(y[test_idx])
        preds.append(np.asarray(pred, dtype=np.int64))
        indices.append(np.asarray(test_idx, dtype=np.int64))
    out = metric_record(np.concatenate(truths), np.concatenate(preds))
    if with_predictions:
        out["y_true"] = np.concatenate(truths)
        out["y_pred"] = np.concatenate(preds)
        out["indices"] = np.concatenate(indices)
    return out


def _json_metrics(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in {"y_true", "y_pred", "margin", "support", "indices"}}


def choose_config(
    x: np.ndarray,
    y: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[HeadConfig, dict, list[dict], dict]:
    baseline = evaluate_baseline_on_splits(x, y, splits)
    rows: list[tuple[int, HeadConfig, dict]] = []
    for idx, config in enumerate(candidate_grid()):
        metrics = evaluate_config_on_splits(x, y, config, splits, seed=seed)
        rows.append((idx, config, metrics))

    min_accuracy = baseline["accuracy"] - 0.01
    eligible = [row for row in rows if row[2]["accuracy"] >= min_accuracy]
    pool = eligible or rows
    best = max(
        pool,
        key=lambda row: (
            row[2]["balanced_accuracy_present_classes"],
            row[2]["accuracy"],
            -row[0],
        ),
    )
    ranked = sorted(
        rows,
        key=lambda row: (
            row[2]["balanced_accuracy_present_classes"],
            row[2]["accuracy"],
            -row[0],
        ),
        reverse=True,
    )
    top = [
        {"config": asdict(cfg), "name": cfg.name, "metrics": _json_metrics(metrics)}
        for _, cfg, metrics in ranked[:5]
    ]
    return best[1], best[2], top, baseline


def choose_gate(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    candidate_pred: np.ndarray,
    margin: np.ndarray,
    support: np.ndarray,
) -> dict:
    """在训练折 OOF 预测上选择安全门控；完全不用外层测试标签。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    baseline_pred = np.asarray(baseline_pred, dtype=np.int64)
    candidate_pred = np.asarray(candidate_pred, dtype=np.int64)
    baseline_metrics = metric_record(y_true, baseline_pred)
    min_accuracy = baseline_metrics["accuracy"] - 0.01

    quantiles = (0.0, 0.2, 0.4, 0.6, 0.8)
    margin_thresholds = [-math.inf] + [float(np.quantile(margin, q)) for q in quantiles] + [math.inf]
    support_thresholds = [-math.inf] + [float(np.quantile(support, q)) for q in quantiles] + [math.inf]

    rows = []
    for mt in sorted(set(margin_thresholds)):
        for st in sorted(set(support_thresholds)):
            use_candidate = (margin >= mt) & (support >= st)
            pred = np.where(use_candidate, candidate_pred, baseline_pred)
            metrics = metric_record(y_true, pred)
            coverage = float(np.mean(use_candidate))
            rows.append((mt, st, coverage, metrics))
    eligible = [row for row in rows if row[3]["accuracy"] >= min_accuracy]
    pool = eligible or rows
    best = max(
        pool,
        key=lambda row: (
            row[3]["balanced_accuracy_present_classes"],
            row[3]["accuracy"],
            -row[2],  # 完全并列时少切换，保守优先
            row[0],
            row[1],
        ),
    )
    return {
        "margin_threshold": float(best[0]),
        "support_threshold": float(best[1]),
        "coverage": float(best[2]),
        "metrics": best[3],
        "baseline_metrics": baseline_metrics,
    }


def apply_gate(
    baseline_pred: np.ndarray,
    candidate_pred: np.ndarray,
    margin: np.ndarray,
    support: np.ndarray,
    gate: dict,
) -> tuple[np.ndarray, float]:
    use_candidate = (
        (margin >= gate["margin_threshold"])
        & (support >= gate["support_threshold"])
    )
    pred = np.where(use_candidate, candidate_pred, baseline_pred)
    return pred.astype(np.int64), float(np.mean(use_candidate))


def _make_inner_splits(y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    dummy = np.zeros(len(y), dtype=np.uint8)
    return [(tr, te) for tr, te in cv.split(dummy, y)]


def nested_outer_evaluation(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    repeats: int,
) -> dict:
    from sklearn.model_selection import RepeatedStratifiedKFold

    outer = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=repeats, random_state=seed
    )
    dummy = np.zeros(len(y), dtype=np.uint8)
    outer_splits = [(tr, te) for tr, te in outer.split(dummy, y)]

    all_truth: list[np.ndarray] = []
    all_baseline: list[np.ndarray] = []
    all_candidate: list[np.ndarray] = []
    all_gated: list[np.ndarray] = []
    fold_rows: list[dict] = []
    selected_names: Counter[str] = Counter()

    for fold_id, (train_idx, test_idx) in enumerate(outer_splits):
        x_train, y_train = x[train_idx], y[train_idx]
        inner_splits = _make_inner_splits(y_train, seed + 7919 * fold_id)
        config, inner_metrics, top, inner_baseline = choose_config(
            x_train, y_train, inner_splits, seed + 104729 * fold_id
        )
        selected_names[config.name] += 1

        candidate_oof = evaluate_config_on_splits(
            x_train,
            y_train,
            config,
            inner_splits,
            seed=seed + 15485863 * fold_id,
            with_diagnostics=True,
        )
        baseline_oof = evaluate_baseline_on_splits(
            x_train, y_train, inner_splits, with_predictions=True
        )
        if not np.array_equal(candidate_oof["indices"], baseline_oof["indices"]):
            raise RuntimeError("candidate/baseline OOF 索引顺序不一致")
        gate = choose_gate(
            candidate_oof["y_true"],
            baseline_oof["y_pred"],
            candidate_oof["y_pred"],
            candidate_oof["margin"],
            candidate_oof["support"],
        )

        baseline_model = fit_baseline(x_train, y_train)
        baseline_pred = np.asarray(
            baseline_model.predict(x[test_idx]), dtype=np.int64
        )
        head = FittedHead(config, seed=seed + fold_id).fit(x_train, y_train)
        candidate_pred, margin, support = head.predict_with_diagnostics(x[test_idx])
        gated_pred, coverage = apply_gate(
            baseline_pred, candidate_pred, margin, support, gate
        )

        all_truth.append(y[test_idx])
        all_baseline.append(baseline_pred)
        all_candidate.append(candidate_pred)
        all_gated.append(gated_pred)
        fold_rows.append(
            {
                "fold": fold_id,
                "selected_config": asdict(config),
                "selected_name": config.name,
                "inner_baseline": _json_metrics(inner_baseline),
                "inner_candidate": _json_metrics(inner_metrics),
                "inner_top5": top,
                "gate": {
                    "margin_threshold": gate["margin_threshold"],
                    "support_threshold": gate["support_threshold"],
                    "inner_coverage": gate["coverage"],
                    "inner_metrics": gate["metrics"],
                },
                "outer_baseline": metric_record(y[test_idx], baseline_pred),
                "outer_candidate": metric_record(y[test_idx], candidate_pred),
                "outer_gated": metric_record(y[test_idx], gated_pred),
                "outer_gate_coverage": coverage,
            }
        )

    truth = np.concatenate(all_truth)
    return {
        "folds": len(outer_splits),
        "selected_config_counts": dict(selected_names.most_common()),
        "baseline": metric_record(truth, np.concatenate(all_baseline)),
        "candidate": metric_record(truth, np.concatenate(all_candidate)),
        "gated": metric_record(truth, np.concatenate(all_gated)),
        "mean_outer_gate_coverage": float(
            np.mean([row["outer_gate_coverage"] for row in fold_rows])
        ),
        "fold_records": fold_rows,
    }


def fit_final_r1_and_check_r2(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    seed: int,
) -> dict:
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    dummy = np.zeros(len(y1), dtype=np.uint8)
    splits = [(tr, te) for tr, te in cv.split(dummy, y1)]
    config, cv_metrics, top, cv_baseline = choose_config(x1, y1, splits, seed)
    candidate_oof = evaluate_config_on_splits(
        x1, y1, config, splits, seed=seed + 1, with_diagnostics=True
    )
    baseline_oof = evaluate_baseline_on_splits(
        x1, y1, splits, with_predictions=True
    )
    gate = choose_gate(
        candidate_oof["y_true"],
        baseline_oof["y_pred"],
        candidate_oof["y_pred"],
        candidate_oof["margin"],
        candidate_oof["support"],
    )

    baseline_model = fit_baseline(x1, y1)
    baseline_pred = np.asarray(baseline_model.predict(x2), dtype=np.int64)
    head = FittedHead(config, seed=seed).fit(x1, y1)
    candidate_pred, margin, support = head.predict_with_diagnostics(x2)
    gated_pred, coverage = apply_gate(
        baseline_pred, candidate_pred, margin, support, gate
    )
    return {
        "selected_config": asdict(config),
        "selected_name": config.name,
        "r1_cv_baseline": _json_metrics(cv_baseline),
        "r1_cv_candidate": _json_metrics(cv_metrics),
        "r1_cv_top5": top,
        "gate": {
            "margin_threshold": gate["margin_threshold"],
            "support_threshold": gate["support_threshold"],
            "r1_oof_coverage": gate["coverage"],
            "r1_oof_metrics": gate["metrics"],
        },
        "r1_to_r2_baseline": metric_record(y2, baseline_pred),
        "r1_to_r2_candidate": metric_record(y2, candidate_pred),
        "r1_to_r2_gated": metric_record(y2, gated_pred),
        "r2_gate_coverage": coverage,
    }


def choose_development_variant(outer: dict) -> str:
    candidates = [("candidate", outer["candidate"]), ("gated", outer["gated"])]
    return max(
        candidates,
        key=lambda item: (
            item[1]["balanced_accuracy_present_classes"],
            item[1]["accuracy"],
            item[0] == "gated",
        ),
    )[0]


def development_decision(outer: dict, cross: dict) -> tuple[bool, str, list[str]]:
    variant = choose_development_variant(outer)
    metrics = outer[variant]
    cross_metrics = cross[f"r1_to_r2_{variant}"]
    baseline = outer["baseline"]
    reasons: list[str] = []

    if metrics["balanced_accuracy_present_classes"] < baseline["balanced_accuracy_present_classes"] + 0.020 - 1e-12:
        reasons.append("outer balanced accuracy 未提升至少 +0.020")
    if metrics["accuracy"] < 0.8980 - 1e-12:
        reasons.append("outer accuracy 低于 0.8980")
    for label in ("2", "3"):
        if metrics["recall"][label] + 1e-12 < baseline["recall"][label]:
            reasons.append(f"类别 {label} recall 低于生产基线")
    if min(metrics["recall"].values()) < 0.67 - 1e-12:
        reasons.append("五类最差 recall 低于 0.67")
    if cross_metrics["correct"] < 186:
        reasons.append("第1轮训练→第2轮正确数低于 186/189")
    return not reasons, variant, reasons


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(features: Path, seed: int = 20260811, outer_repeats: int = 5) -> dict:
    started = time.perf_counter()
    z = np.load(features)
    required = {"X1t3", "y1t3", "X2t3", "y2t3"}
    missing = sorted(required - set(z.files))
    if missing:
        raise KeyError(f"特征缓存缺键：{missing}")
    x1 = np.asarray(z["X1t3"], dtype=np.float64)
    y1 = np.asarray(z["y1t3"], dtype=np.int64)
    x2 = np.asarray(z["X2t3"], dtype=np.float64)
    y2 = np.asarray(z["y2t3"], dtype=np.int64)

    outer = nested_outer_evaluation(x1, y1, seed=seed, repeats=outer_repeats)
    cross = fit_final_r1_and_check_r2(x1, y1, x2, y2, seed=seed)
    passed, variant, reasons = development_decision(outer, cross)

    import sklearn

    return {
        "experiment_id": "t3-long-tail-domain-generalization-20260811",
        "baseline_commit": "c08bda58bad856fa1bd24f8975385696dbad11bc",
        "seed": seed,
        "outer_repeats": outer_repeats,
        "feature_cache": str(features),
        "feature_cache_sha256": sha256_file(features),
        "shapes": {
            "X1t3": list(x1.shape),
            "y1t3": list(y1.shape),
            "X2t3": list(x2.shape),
            "y2t3": list(y2.shape),
        },
        "class_counts": {
            "round1": {str(int(k)): int(v) for k, v in zip(*np.unique(y1, return_counts=True))},
            "round2": {str(int(k)): int(v) for k, v in zip(*np.unique(y2, return_counts=True))},
        },
        "grid": [asdict(config) for config in candidate_grid()],
        "outer_nested_cv": outer,
        "final_r1_selection_and_r2_check": cross,
        "development_variant": variant,
        "development_pass": passed,
        "failure_reasons": reasons,
        "terminal_08_allowed": passed,
        "runtime_seconds": time.perf_counter() - started,
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
        "--features", default="outputs/seismicxm_t3/features_tta.npz"
    )
    parser.add_argument(
        "--output",
        default="outputs/experiments/round02_t3_longtail_head.json",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--outer-repeats", type=int, default=5)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    output = Path(args.output)
    result = run(
        Path(args.features), seed=args.seed, outer_repeats=args.outer_repeats
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "output": str(output),
        "baseline": result["outer_nested_cv"]["baseline"],
        "candidate": result["outer_nested_cv"]["candidate"],
        "gated": result["outer_nested_cv"]["gated"],
        "selected_counts": result["outer_nested_cv"]["selected_config_counts"],
        "r1_to_r2": {
            key: result["final_r1_selection_and_r2_check"][key]
            for key in (
                "r1_to_r2_baseline",
                "r1_to_r2_candidate",
                "r1_to_r2_gated",
            )
        },
        "development_variant": result["development_variant"],
        "development_pass": result["development_pass"],
        "failure_reasons": result["failure_reasons"],
        "runtime_seconds": result["runtime_seconds"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
