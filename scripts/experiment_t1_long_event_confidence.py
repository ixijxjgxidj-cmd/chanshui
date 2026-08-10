#!/usr/bin/env python3
"""T1 长记录 FIFO P/S 事件级联合置信度过滤实验。

实验协议已冻结在：
    memory/experiments/004-t1-long-record-event-confidence.md

本脚本只读取冻结的长记录拾取缓存、冻结基线和官方答案，不调用模型。任何
active 阈值评分都必须先通过 OFF 路径的逐文件与完整包复现硬闸。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from phasepicker.io.official_waveforms import read_package_answers
from phasepicker.postprocess.dedup import deduplicate
from phasepicker.scoring.scorer import (
    PENALTY_MODES,
    ScoreReport,
    exam_total_score,
    match_phases,
    score_file,
)
from phasepicker.types import ExamTask, PhaseType, Pick, Task1Result


EXPERIMENT_ID = "t1-long-record-event-confidence-20260811"
BASELINE_COMMIT = "fb4345e4d3cfbeda55a8f8da74d719b1499220d1"
BASELINE_SHA256 = "2a55e4164db40a8eb87d6aa518fb040f11f7b2996788234f8fc1513bcfa3ac05"
CACHE_SHA256 = "b7e20333fbe97480017e8c8b5167be6f92a95c1969b5c7e91bb6e0319cc38699"
SEED = 20260811

LONG_DEDUP_S = 20.0
LONG_DURATION_SENTINEL_S = 4000.0
SHORT_MAX_DURATION_S = 300.0
SP_MIN_S = 0.2
SP_MAX_S = 60.0
ACTIVE_THRESHOLDS = (0.35, 0.40, 0.45)
FLOAT_TOLERANCE = 1e-9
SOURCE_MIN_NORMALIZED_GAIN = 0.01
SELECTION_TIE_TOLERANCE = 1e-4
SELECTION_MAX_SPAN = 0.05
POSTPROCESS_P95_LIMIT_MS = 10.0
SUBMISSION_NDIGITS = 2


@dataclass(frozen=True)
class PackageSpec:
    key: str
    label: str
    cache_prefix: str
    long_file_ids: tuple[str, ...]
    expected_counts: Mapping[str, tuple[int, int]]
    expected_long_truth_phase_count: int


PACKAGE_SPECS = {
    "round2": PackageSpec(
        key="round2",
        label="R2",
        cache_prefix="r2:",
        long_file_ids=("T1.A.Q0001.mseed", "T1.A.Q0002.mseed"),
        expected_counts={
            "T1.A.Q0001.mseed": (64, 68),
            "T1.A.Q0002.mseed": (64, 67),
        },
        expected_long_truth_phase_count=174,
    ),
    "final08": PackageSpec(
        key="final08",
        label="08",
        cache_prefix="f08:",
        long_file_ids=(
            "T1.A.Q0001.mseed",
            "T1.A.Q0002.mseed",
            "T1.A.Q0003.mseed",
            "T1.A.Q0004.mseed",
            "T1.A.Q0005.mseed",
        ),
        expected_counts={
            "T1.A.Q0001.mseed": (47, 58),
            "T1.A.Q0002.mseed": (63, 72),
            "T1.A.Q0003.mseed": (46, 55),
            "T1.A.Q0004.mseed": (61, 73),
            "T1.A.Q0005.mseed": (57, 69),
        },
        expected_long_truth_phase_count=443,
    ),
}


@dataclass(frozen=True)
class EventPair:
    p_index: int
    s_index: int


@dataclass(frozen=True)
class PairingResult:
    pairs: tuple[EventPair, ...]
    orphan_p_indices: tuple[int, ...]
    orphan_s_indices: tuple[int, ...]


@dataclass
class FilterOutcome:
    picks: list[Pick]
    baseline_picks: list[Pick]
    path: str
    threshold: float | None
    pairing: PairingResult
    pair_decisions: list[dict]
    dropped_indices: tuple[int, ...]

    @property
    def deleted_events(self) -> int:
        return len(self.dropped_indices) // 2

    def to_record(self, *, include_pair_details: bool = True) -> dict:
        record = {
            "path": self.path,
            "threshold": self.threshold,
            "n_input_after_scope": len(self.baseline_picks),
            "n_output": len(self.picks),
            "n_pairs": len(self.pairing.pairs),
            "n_orphan_p": len(self.pairing.orphan_p_indices),
            "n_orphan_s": len(self.pairing.orphan_s_indices),
            "orphan_p_indices": list(self.pairing.orphan_p_indices),
            "orphan_s_indices": list(self.pairing.orphan_s_indices),
            "deleted_events": self.deleted_events,
            "deleted_picks": len(self.dropped_indices),
            "dropped_indices": list(self.dropped_indices),
        }
        record["pair_decisions"] = self.pair_decisions if include_pair_details else []
        return record


def candidate_grid() -> list[float | None]:
    """返回唯一允许的预注册状态，顺序固定。"""

    return [None, *ACTIVE_THRESHOLDS]


def threshold_key(threshold: float | None) -> str:
    return "OFF" if threshold is None else f"{threshold:.2f}"


def _validate_threshold(threshold: float | None) -> None:
    if threshold not in candidate_grid():
        raise ValueError(
            f"阈值 {threshold!r} 不在冻结网格 {candidate_grid()} 中"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _pick_token(pick: Pick) -> tuple:
    return (
        pick.phase.value,
        float(pick.time_utc),
        float(pick.confidence),
        pick.station,
        pick.sample_index,
    )


def dedup_cached_picks(picks: Sequence[Pick]) -> list[Pick]:
    """严格复刻生产长记录 20 秒同相位去重。"""

    return deduplicate(
        picks,
        merge_window_s={
            PhaseType.P: LONG_DEDUP_S,
            PhaseType.S: LONG_DEDUP_S,
        },
    )


def fifo_pair_events(picks: Sequence[Pick]) -> PairingResult:
    """按预注册规则构建确定性、非交叉的 FIFO P→S 事件对。

    非有限到时无法安全参与比较，因此按对应相位作为 orphan 保留。相同时间由
    原始下标打破并列，不使用 confidence 决定配对。
    """

    p_items: list[tuple[float, int]] = []
    s_items: list[tuple[float, int]] = []
    orphan_p: list[int] = []
    orphan_s: list[int] = []
    for index, pick in enumerate(picks):
        target = p_items if pick.phase == PhaseType.P else s_items
        orphan_target = orphan_p if pick.phase == PhaseType.P else orphan_s
        if not math.isfinite(float(pick.time_utc)):
            orphan_target.append(index)
        else:
            target.append((float(pick.time_utc), index))

    p_items.sort(key=lambda item: (item[0], item[1]))
    s_items.sort(key=lambda item: (item[0], item[1]))

    pairs: list[EventPair] = []
    p_pos = 0
    s_pos = 0
    while p_pos < len(p_items) and s_pos < len(s_items):
        p_time, p_index = p_items[p_pos]
        while s_pos < len(s_items) and s_items[s_pos][0] < p_time + SP_MIN_S:
            orphan_s.append(s_items[s_pos][1])
            s_pos += 1
        if s_pos >= len(s_items):
            break

        s_time, s_index = s_items[s_pos]
        if s_time <= p_time + SP_MAX_S:
            pairs.append(EventPair(p_index=p_index, s_index=s_index))
            p_pos += 1
            s_pos += 1
        else:
            orphan_p.append(p_index)
            p_pos += 1

    orphan_p.extend(index for _, index in p_items[p_pos:])
    orphan_s.extend(index for _, index in s_items[s_pos:])
    return PairingResult(
        pairs=tuple(pairs),
        orphan_p_indices=tuple(sorted(orphan_p)),
        orphan_s_indices=tuple(sorted(orphan_s)),
    )


def filter_deduplicated_events(
    picks: Sequence[Pick],
    threshold: float | None,
) -> FilterOutcome:
    """在已经去重的长记录拾取上删除低分完整 P/S 对。"""

    _validate_threshold(threshold)
    baseline = list(picks)
    pairing = fifo_pair_events(baseline)
    dropped: set[int] = set()
    decisions: list[dict] = []
    for pair in pairing.pairs:
        p_pick = baseline[pair.p_index]
        s_pick = baseline[pair.s_index]
        p_conf = float(p_pick.confidence)
        s_conf = float(s_pick.confidence)
        nonfinite = not (math.isfinite(p_conf) and math.isfinite(s_conf))
        event_confidence: float | None
        if nonfinite:
            event_confidence = None
            keep = True
            decision = "keep_nonfinite_confidence"
        else:
            event_confidence = math.sqrt(max(p_conf, 0.0) * max(s_conf, 0.0))
            if threshold is None:
                keep = True
                decision = "keep_off"
            elif event_confidence < threshold:
                keep = False
                decision = "drop_below_threshold"
                dropped.update((pair.p_index, pair.s_index))
            else:
                keep = True
                decision = "keep_at_or_above_threshold"
        decisions.append(
            {
                "p_index": pair.p_index,
                "s_index": pair.s_index,
                "p_time_s": float(p_pick.time_utc),
                "s_time_s": float(s_pick.time_utc),
                "sp_s": float(s_pick.time_utc - p_pick.time_utc),
                "p_confidence": _finite_or_none(p_conf),
                "s_confidence": _finite_or_none(s_conf),
                "event_confidence": event_confidence,
                "nonfinite_confidence_keep": nonfinite,
                "keep": keep,
                "decision": decision,
            }
        )

    dropped_indices = tuple(sorted(dropped))
    output = [pick for index, pick in enumerate(baseline) if index not in dropped]
    return FilterOutcome(
        picks=output,
        baseline_picks=baseline,
        path="long_dedup_then_event_filter",
        threshold=threshold,
        pairing=pairing,
        pair_decisions=decisions,
        dropped_indices=dropped_indices,
    )


def apply_event_confidence_filter(
    picks: Sequence[Pick],
    *,
    duration_s: float,
    threshold: float | None,
) -> FilterOutcome:
    """应用严格的 300 秒作用域；短路径不去重、不重排、不复制 Pick。"""

    _validate_threshold(threshold)
    if float(duration_s) <= SHORT_MAX_DURATION_S:
        unchanged = list(picks)
        return FilterOutcome(
            picks=unchanged,
            baseline_picks=unchanged,
            path="short_unchanged",
            threshold=threshold,
            pairing=PairingResult((), (), ()),
            pair_decisions=[],
            dropped_indices=(),
        )
    deduplicated = dedup_cached_picks(picks)
    return filter_deduplicated_events(deduplicated, threshold)


def _truth_pairs(truth: Task1Result) -> list[tuple[str, float]]:
    return [
        *(("P", float(value)) for value in truth.p_times_s),
        *(("S", float(value)) for value in truth.s_times_s),
    ]


def _prediction_pairs(picks: Sequence[Pick]) -> list[tuple[str, float]]:
    # 冻结基线来自正式 T1.an；submission_writer 会先以 ``.2f`` 写出，再由
    # parser 回读评分。这里必须复刻这道官方提交边界，不能直接用缓存中的
    # 亚采样高精度时间，否则 OFF 路径会出现毫秒级伪差异。
    return [
        (pick.phase.value, float(f"{float(pick.time_utc):.{SUBMISSION_NDIGITS}f}"))
        for pick in picks
    ]


def _phase_error_record(
    pred_times: Sequence[float],
    true_times: Sequence[float],
    phase: str,
) -> dict:
    matched = match_phases(pred_times, true_times, phase)
    return {
        "false_positives": len(matched.unmatched_pred),
        "false_negatives": len(matched.unmatched_true),
        "matched": len(matched.matched),
    }


def score_file_all_modes(
    picks: Sequence[Pick],
    truth: Task1Result,
) -> tuple[dict, dict[str, ScoreReport]]:
    prediction = _prediction_pairs(picks)
    truth_pairs = _truth_pairs(truth)
    reports = {
        mode: score_file(prediction, truth_pairs, penalty_mode=mode)
        for mode in PENALTY_MODES
    }
    representative = reports["merged_file_floor0"]
    pred_p = [time_s for phase, time_s in prediction if phase == "P"]
    pred_s = [time_s for phase, time_s in prediction if phase == "S"]
    record = {
        "n_pred_p": representative.n_pred_p,
        "n_true_p": representative.n_true_p,
        "n_pred_s": representative.n_pred_s,
        "n_true_s": representative.n_true_s,
        "p_time_score": representative.p_time_score,
        "s_time_score": representative.s_time_score,
        "p_residuals_s": list(representative.p_residuals),
        "s_residuals_s": list(representative.s_residuals),
        "false_positives": representative.n_false_pos,
        "false_negatives": representative.n_false_neg,
        "phase_errors": {
            "P": _phase_error_record(pred_p, truth.p_times_s, "P"),
            "S": _phase_error_record(pred_s, truth.s_times_s, "S"),
        },
        "modes": {
            mode: {
                "total_score": report.total_score,
                "count_penalty": report.count_penalty,
            }
            for mode, report in reports.items()
        },
    }
    return record, reports


def pseudo_report_from_frozen(entry: Mapping, mode: str) -> ScoreReport:
    mode_record = entry["modes"][mode]
    return ScoreReport(
        total_score=float(mode_record["total_score"]),
        p_time_score=float(entry["p_time_score"]),
        s_time_score=float(entry["s_time_score"]),
        count_penalty=float(mode_record["count_penalty"]),
        n_pred_p=int(entry["n_pred_p"]),
        n_true_p=int(entry["n_true_p"]),
        n_pred_s=int(entry["n_pred_s"]),
        n_true_s=int(entry["n_true_s"]),
        p_residuals=[float(value) for value in entry["p_residuals_s"]],
        s_residuals=[float(value) for value in entry["s_residuals_s"]],
        n_false_pos=int(entry["false_positives"]),
        n_false_neg=int(entry["false_negatives"]),
    )


def _sum_phase_errors(
    per_file: Mapping[str, Mapping],
    file_ids: Sequence[str],
    side: str,
) -> dict:
    result = {
        "p_false_positives": 0,
        "s_false_positives": 0,
        "false_positives": 0,
        "p_false_negatives": 0,
        "s_false_negatives": 0,
        "false_negatives": 0,
    }
    for file_id in file_ids:
        record = per_file[file_id][side]
        p_error = record["phase_errors"]["P"]
        s_error = record["phase_errors"]["S"]
        result["p_false_positives"] += p_error["false_positives"]
        result["s_false_positives"] += s_error["false_positives"]
        result["p_false_negatives"] += p_error["false_negatives"]
        result["s_false_negatives"] += s_error["false_negatives"]
    result["false_positives"] = (
        result["p_false_positives"] + result["s_false_positives"]
    )
    result["false_negatives"] = (
        result["p_false_negatives"] + result["s_false_negatives"]
    )
    return result


def evaluate_package(
    *,
    spec: PackageSpec,
    baseline_dataset: Mapping,
    truths: Mapping[str, Task1Result],
    cached_picks: Mapping[str, Sequence[Pick]],
    threshold: float | None,
    active_file_ids: Sequence[str] | None = None,
    include_pair_details: bool = True,
) -> dict:
    """评估一个阈值；非 active 长文件保持 OFF，完整包仍包含全部文件。"""

    _validate_threshold(threshold)
    active_ids = tuple(active_file_ids or spec.long_file_ids)
    active_set = set(active_ids)
    if not active_set.issubset(spec.long_file_ids):
        raise ValueError("active_file_ids 含未预注册的长文件")

    frozen_files = baseline_dataset["tasks"]["T1"]["per_file"]
    per_file: dict[str, dict] = {}
    candidate_reports: dict[str, dict[str, ScoreReport]] = {}
    for file_id in spec.long_file_ids:
        if file_id not in truths:
            raise KeyError(f"{spec.label} 答案缺少 {file_id}")
        if file_id not in cached_picks:
            raise KeyError(f"{spec.label} 缓存缺少 {file_id}")
        raw = cached_picks[file_id]
        baseline_outcome = apply_event_confidence_filter(
            raw,
            duration_s=LONG_DURATION_SENTINEL_S,
            threshold=None,
        )
        candidate_threshold = threshold if file_id in active_set else None
        candidate_outcome = apply_event_confidence_filter(
            raw,
            duration_s=LONG_DURATION_SENTINEL_S,
            threshold=candidate_threshold,
        )
        baseline_metrics, _ = score_file_all_modes(
            baseline_outcome.picks, truths[file_id]
        )
        candidate_metrics, reports = score_file_all_modes(
            candidate_outcome.picks, truths[file_id]
        )
        candidate_reports[file_id] = reports
        per_file[file_id] = {
            "active": file_id in active_set,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta": {
                "p_time_score": (
                    candidate_metrics["p_time_score"]
                    - baseline_metrics["p_time_score"]
                ),
                "s_time_score": (
                    candidate_metrics["s_time_score"]
                    - baseline_metrics["s_time_score"]
                ),
                "false_positives": (
                    candidate_metrics["false_positives"]
                    - baseline_metrics["false_positives"]
                ),
                "false_negatives": (
                    candidate_metrics["false_negatives"]
                    - baseline_metrics["false_negatives"]
                ),
                "modes": {
                    mode: (
                        candidate_metrics["modes"][mode]["total_score"]
                        - baseline_metrics["modes"][mode]["total_score"]
                    )
                    for mode in PENALTY_MODES
                },
            },
            "filter": candidate_outcome.to_record(
                include_pair_details=include_pair_details
            ),
            "output_is_baseline_subset": all(
                id(pick) in {id(base_pick) for base_pick in candidate_outcome.baseline_picks}
                for pick in candidate_outcome.picks
            ),
        }

    package_modes: dict[str, dict] = {}
    for mode in PENALTY_MODES:
        reports: list[ScoreReport] = []
        for file_id, frozen_entry in frozen_files.items():
            if file_id in candidate_reports:
                reports.append(candidate_reports[file_id][mode])
            else:
                reports.append(pseudo_report_from_frozen(frozen_entry, mode))
        candidate_total, candidate_penalty = exam_total_score(reports, mode)
        frozen_mode = baseline_dataset["tasks"]["T1"]["penalty_modes"][mode]
        baseline_total = float(frozen_mode["total_score"])
        delta = candidate_total - baseline_total
        package_modes[mode] = {
            "baseline_total_score": baseline_total,
            "candidate_total_score": candidate_total,
            "delta": delta,
            "baseline_exam_count_penalty": float(
                frozen_mode["exam_count_penalty"]
            ),
            "candidate_exam_count_penalty": candidate_penalty,
        }

    long_truth_count = sum(
        len(truths[file_id].p_times_s) + len(truths[file_id].s_times_s)
        for file_id in active_ids
    )
    for mode_record in package_modes.values():
        mode_record["normalized_gain"] = (
            mode_record["delta"] / long_truth_count if long_truth_count else 0.0
        )
    worst_gain = min(
        mode_record["normalized_gain"] for mode_record in package_modes.values()
    )
    baseline_errors = _sum_phase_errors(per_file, active_ids, "baseline")
    candidate_errors = _sum_phase_errors(per_file, active_ids, "candidate")
    return {
        "package": spec.label,
        "threshold": threshold,
        "active_file_ids": list(active_ids),
        "long_truth_phase_count": long_truth_count,
        "per_file": per_file,
        "package_modes": package_modes,
        "worst_normalized_gain": worst_gain,
        "errors": {
            "baseline": baseline_errors,
            "candidate": candidate_errors,
            "delta": {
                key: candidate_errors[key] - baseline_errors[key]
                for key in baseline_errors
            },
        },
        "deleted_events": sum(
            per_file[file_id]["filter"]["deleted_events"] for file_id in active_ids
        ),
    }


def _close(actual: float, expected: float, tolerance: float = FLOAT_TOLERANCE) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def _append_mismatch(
    issues: list[str],
    label: str,
    actual,
    expected,
    *,
    tolerance: float = FLOAT_TOLERANCE,
) -> None:
    if isinstance(actual, (int, np.integer)) and isinstance(
        expected, (int, np.integer)
    ):
        if int(actual) != int(expected):
            issues.append(f"{label}: {actual} != {expected}")
        return
    if not _close(float(actual), float(expected), tolerance):
        issues.append(f"{label}: {actual!r} != {expected!r}")


def check_baseline_reproduction(
    *,
    spec: PackageSpec,
    baseline_dataset: Mapping,
    off_evaluation: Mapping,
) -> dict:
    issues: list[str] = []
    frozen_files = baseline_dataset["tasks"]["T1"]["per_file"]
    for file_id in spec.long_file_ids:
        actual = off_evaluation["per_file"][file_id]["candidate"]
        frozen = frozen_files[file_id]
        expected_p, expected_s = spec.expected_counts[file_id]
        _append_mismatch(
            issues, f"{file_id}.n_pred_p", actual["n_pred_p"], expected_p
        )
        _append_mismatch(
            issues, f"{file_id}.n_pred_s", actual["n_pred_s"], expected_s
        )
        for key in (
            "n_pred_p",
            "n_pred_s",
            "n_true_p",
            "n_true_s",
            "false_positives",
            "false_negatives",
            "p_time_score",
            "s_time_score",
        ):
            _append_mismatch(
                issues, f"{file_id}.{key}", actual[key], frozen[key]
            )
        for phase_key in ("p_residuals_s", "s_residuals_s"):
            actual_values = actual[phase_key]
            expected_values = frozen[phase_key]
            if len(actual_values) != len(expected_values):
                issues.append(
                    f"{file_id}.{phase_key}.length: "
                    f"{len(actual_values)} != {len(expected_values)}"
                )
            else:
                for index, (value, expected) in enumerate(
                    zip(actual_values, expected_values)
                ):
                    _append_mismatch(
                        issues,
                        f"{file_id}.{phase_key}[{index}]",
                        value,
                        expected,
                    )
        for mode in PENALTY_MODES:
            for key in ("total_score", "count_penalty"):
                _append_mismatch(
                    issues,
                    f"{file_id}.{mode}.{key}",
                    actual["modes"][mode][key],
                    frozen["modes"][mode][key],
                )

    _append_mismatch(
        issues,
        f"{spec.label}.long_truth_phase_count",
        off_evaluation["long_truth_phase_count"],
        spec.expected_long_truth_phase_count,
    )
    for mode in PENALTY_MODES:
        actual_mode = off_evaluation["package_modes"][mode]
        frozen_mode = baseline_dataset["tasks"]["T1"]["penalty_modes"][mode]
        _append_mismatch(
            issues,
            f"{spec.label}.{mode}.total_score",
            actual_mode["candidate_total_score"],
            frozen_mode["total_score"],
        )
        _append_mismatch(
            issues,
            f"{spec.label}.{mode}.exam_count_penalty",
            actual_mode["candidate_exam_count_penalty"],
            frozen_mode["exam_count_penalty"],
        )
    return {"pass": not issues, "issues": issues}


def source_eligibility(evaluation: Mapping) -> dict:
    """执行 active 源包阈值的全部合取条件。"""

    if evaluation.get("threshold") is None:
        return {"eligible": False, "reasons": ["OFF 不是 active 候选"]}
    reasons: list[str] = []
    for file_id in evaluation["active_file_ids"]:
        record = evaluation["per_file"][file_id]
        baseline = record["baseline"]
        candidate = record["candidate"]
        for phase in ("P", "S"):
            baseline_fn = baseline["phase_errors"][phase]["false_negatives"]
            candidate_fn = candidate["phase_errors"][phase]["false_negatives"]
            if candidate_fn > baseline_fn:
                reasons.append(f"{file_id}.{phase}.FN 增加")
        for key in ("p_time_score", "s_time_score"):
            if candidate[key] + FLOAT_TOLERANCE < baseline[key]:
                reasons.append(f"{file_id}.{key} 下降")
        for mode in ("merged_file_floor0", "per_phase_floor0"):
            if (
                candidate["modes"][mode]["total_score"] + FLOAT_TOLERANCE
                < baseline["modes"][mode]["total_score"]
            ):
                reasons.append(f"{file_id}.{mode} 文件分下降")
        if not record["output_is_baseline_subset"]:
            reasons.append(f"{file_id} 候选不是 baseline 子集")

    for mode, mode_record in evaluation["package_modes"].items():
        if not mode_record["delta"] > 0.0:
            reasons.append(f"{mode} 完整包分数未严格上升")
    baseline_fp = evaluation["errors"]["baseline"]["false_positives"]
    candidate_fp = evaluation["errors"]["candidate"]["false_positives"]
    if not candidate_fp < baseline_fp:
        reasons.append("源包 false positives 未严格减少")
    if (
        evaluation["worst_normalized_gain"]
        < SOURCE_MIN_NORMALIZED_GAIN - 1e-12
    ):
        reasons.append("worst_normalized_gain 低于 0.01")
    return {"eligible": not reasons, "reasons": reasons}


def select_source_threshold(evaluations: Mapping[str, Mapping]) -> dict:
    assessments: dict[str, dict] = {}
    eligible: list[tuple[float, float]] = []
    for threshold in ACTIVE_THRESHOLDS:
        key = threshold_key(threshold)
        if key not in evaluations:
            continue
        assessment = source_eligibility(evaluations[key])
        assessment = {
            **assessment,
            "worst_normalized_gain": evaluations[key]["worst_normalized_gain"],
        }
        assessments[key] = assessment
        if not assessment["eligible"]:
            continue
        eligible.append((threshold, float(assessment["worst_normalized_gain"])))

    selected: float | None = None
    selected_gain: float | None = None
    if eligible:
        best_gain = max(gain for _, gain in eligible)
        # 先确定全局最大值，再在与最大值相差 <=1e-4 的候选中取最低 tau；
        # 不能用逐步比较，否则链式近并列可能错误选中更高阈值。
        tied = [
            (threshold, gain)
            for threshold, gain in eligible
            if best_gain - gain <= SELECTION_TIE_TOLERANCE
        ]
        selected, selected_gain = min(tied, key=lambda item: item[0])
    return {
        "selected_tau": selected,
        "selected_key": threshold_key(selected),
        "selected_worst_normalized_gain": selected_gain,
        "threshold_assessments": assessments,
    }


def assess_selection_stability(
    full_selected_tau: float | None,
    lofo_selected: Mapping[str, float | None],
) -> dict:
    reasons: list[str] = []
    values: list[float] = []
    if full_selected_tau is None:
        reasons.append("full-source 选择 OFF")
    else:
        values.append(float(full_selected_tau))
    for omitted, threshold in lofo_selected.items():
        if threshold is None:
            reasons.append(f"移除 {omitted} 后选择 OFF")
        else:
            values.append(float(threshold))
    span = max(values) - min(values) if values else None
    if span is not None and span > SELECTION_MAX_SPAN + 1e-12:
        reasons.append(f"full/LOFO 阈值跨度 {span:.6f} > 0.05")
    return {"pass": not reasons, "span": span, "reasons": reasons}


def evaluate_active_candidates_after_reproduction(
    reproduction_pass: bool,
    evaluator: Callable[[float], Mapping],
) -> dict[str, Mapping]:
    """基线复现失败时绝不调用 active evaluator。"""

    if not reproduction_pass:
        return {}
    return {
        threshold_key(threshold): evaluator(threshold)
        for threshold in ACTIVE_THRESHOLDS
    }


def run_leave_one_file_out(
    *,
    spec: PackageSpec,
    full_selection: Mapping,
    evaluator: Callable[[float, Sequence[str]], Mapping],
) -> dict:
    records: dict[str, dict] = {}
    selected: dict[str, float | None] = {}
    for omitted in spec.long_file_ids:
        active = tuple(file_id for file_id in spec.long_file_ids if file_id != omitted)
        evaluations = {
            threshold_key(threshold): evaluator(threshold, active)
            for threshold in ACTIVE_THRESHOLDS
        }
        selection = select_source_threshold(evaluations)
        selected[omitted] = selection["selected_tau"]
        records[omitted] = {
            "active_file_ids": list(active),
            "selection": selection,
            "evaluations": evaluations,
        }
    stability = assess_selection_stability(
        full_selection["selected_tau"], selected
    )
    return {
        "records": records,
        "selected_by_omitted_file": selected,
        "stability": stability,
    }


def _target_assessment(
    selected_tau: float | None,
    target_evaluations: Mapping[str, Mapping],
) -> dict:
    if selected_tau is None:
        return {
            "selected_tau": None,
            "pass": False,
            "reasons": ["源包选择 OFF，无法执行 active 跨包验证"],
            "evaluation": None,
        }
    evaluation = target_evaluations[threshold_key(selected_tau)]
    assessment = source_eligibility(evaluation)
    return {
        "selected_tau": selected_tau,
        "pass": assessment["eligible"],
        "reasons": assessment["reasons"],
        "evaluation": evaluation,
    }


def development_decision(
    *,
    round2_selection: Mapping,
    final08_selection: Mapping,
    round2_stability: Mapping,
    final08_stability: Mapping,
    round2_evaluations: Mapping[str, Mapping],
    final08_evaluations: Mapping[str, Mapping],
    safety: Mapping,
) -> dict:
    """执行双向、共同阈值与安全性全部合取门槛。"""

    r2_tau = round2_selection["selected_tau"]
    final_tau = final08_selection["selected_tau"]
    r2_to_final = _target_assessment(r2_tau, final08_evaluations)
    final_to_r2 = _target_assessment(final_tau, round2_evaluations)
    bidirectional_pass = r2_to_final["pass"] and final_to_r2["pass"]

    common_tau: float | None = None
    common_record: dict = {
        "tau_common": None,
        "pass": False,
        "reasons": [],
        "round2": None,
        "final08": None,
        "combined": None,
    }
    common_reasons: list[str] = []
    if r2_tau is None or final_tau is None:
        common_reasons.append("至少一个源包选择 OFF")
    if not round2_stability["pass"]:
        common_reasons.append("R2 源选择不稳定")
    if not final08_stability["pass"]:
        common_reasons.append("08 源选择不稳定")
    if not common_reasons:
        common_tau = min(float(r2_tau), float(final_tau))
        r2_eval = round2_evaluations[threshold_key(common_tau)]
        final_eval = final08_evaluations[threshold_key(common_tau)]
        r2_assessment = source_eligibility(r2_eval)
        final_assessment = source_eligibility(final_eval)
        if not r2_assessment["eligible"]:
            common_reasons.extend(
                f"R2 common: {reason}" for reason in r2_assessment["reasons"]
            )
        if not final_assessment["eligible"]:
            common_reasons.extend(
                f"08 common: {reason}" for reason in final_assessment["reasons"]
            )
        baseline_fp = (
            r2_eval["errors"]["baseline"]["false_positives"]
            + final_eval["errors"]["baseline"]["false_positives"]
        )
        candidate_fp = (
            r2_eval["errors"]["candidate"]["false_positives"]
            + final_eval["errors"]["candidate"]["false_positives"]
        )
        baseline_fn = (
            r2_eval["errors"]["baseline"]["false_negatives"]
            + final_eval["errors"]["baseline"]["false_negatives"]
        )
        candidate_fn = (
            r2_eval["errors"]["candidate"]["false_negatives"]
            + final_eval["errors"]["candidate"]["false_negatives"]
        )
        deleted_events = r2_eval["deleted_events"] + final_eval["deleted_events"]
        if deleted_events <= 0:
            common_reasons.append("共同候选没有删除任何事件")
        if not candidate_fp < baseline_fp:
            common_reasons.append("共同候选合计 FP 未严格减少")
        if candidate_fn > baseline_fn:
            common_reasons.append("共同候选合计 FN 增加")
        common_record.update(
            {
                "tau_common": common_tau,
                "round2": {
                    "pass": r2_assessment["eligible"],
                    "reasons": r2_assessment["reasons"],
                    "evaluation": r2_eval,
                },
                "final08": {
                    "pass": final_assessment["eligible"],
                    "reasons": final_assessment["reasons"],
                    "evaluation": final_eval,
                },
                "combined": {
                    "baseline_false_positives": baseline_fp,
                    "candidate_false_positives": candidate_fp,
                    "baseline_false_negatives": baseline_fn,
                    "candidate_false_negatives": candidate_fn,
                    "deleted_events": deleted_events,
                },
            }
        )
    common_record["reasons"] = common_reasons
    common_record["pass"] = not common_reasons

    failure_reasons: list[str] = []
    if r2_tau is None:
        failure_reasons.append("R2 源包没有合格 active tau")
    if final_tau is None:
        failure_reasons.append("08 源包没有合格 active tau")
    if not round2_stability["pass"]:
        failure_reasons.extend(
            f"R2 LOFO: {reason}" for reason in round2_stability["reasons"]
        )
    if not final08_stability["pass"]:
        failure_reasons.extend(
            f"08 LOFO: {reason}" for reason in final08_stability["reasons"]
        )
    if not r2_to_final["pass"]:
        failure_reasons.extend(
            f"R2→08: {reason}" for reason in r2_to_final["reasons"]
        )
    if not final_to_r2["pass"]:
        failure_reasons.extend(
            f"08→R2: {reason}" for reason in final_to_r2["reasons"]
        )
    if not common_record["pass"]:
        failure_reasons.extend(
            f"common: {reason}" for reason in common_record["reasons"]
        )
    if not safety["pass"]:
        failure_reasons.extend(
            f"safety: {reason}" for reason in safety["reasons"]
        )
    return {
        "development_pass": not failure_reasons,
        "source_selection_pass": (
            r2_tau is not None
            and final_tau is not None
            and round2_stability["pass"]
            and final08_stability["pass"]
        ),
        "bidirectional_pass": bidirectional_pass,
        "r2_to_08": r2_to_final,
        "08_to_r2": final_to_r2,
        "common_candidate": common_record,
        "failure_reasons": failure_reasons,
    }


def run_safety_checks(
    cached_packages: Mapping[str, Mapping[str, Sequence[Pick]]],
) -> dict:
    reasons: list[str] = []
    short_input = [
        Pick(PhaseType.S, 12.0, 0.4, station="X"),
        Pick(PhaseType.P, 1.0, 0.4, station="X"),
        Pick(PhaseType.P, 1.01, 0.9, station="X"),
    ]
    short_outcome = apply_event_confidence_filter(
        short_input, duration_s=300.0, threshold=0.45
    )
    if len(short_outcome.picks) != len(short_input) or any(
        before is not after
        for before, after in zip(short_input, short_outcome.picks)
    ):
        reasons.append("duration_s <= 300 未逐对象保持原列表")
    if [_pick_token(pick) for pick in short_outcome.picks] != [
        _pick_token(pick) for pick in short_input
    ]:
        reasons.append("duration_s <= 300 内容或顺序改变")

    long_boundary = apply_event_confidence_filter(
        short_input, duration_s=300.000001, threshold=None
    )
    if long_boundary.path != "long_dedup_then_event_filter":
        reasons.append("300.000001 秒未进入长路径")

    deterministic = True
    subset_pass = True
    timings_ms: list[float] = []
    for package in cached_packages.values():
        for raw in package.values():
            for threshold in candidate_grid():
                first = apply_event_confidence_filter(
                    raw,
                    duration_s=LONG_DURATION_SENTINEL_S,
                    threshold=threshold,
                )
                second = apply_event_confidence_filter(
                    raw,
                    duration_s=LONG_DURATION_SENTINEL_S,
                    threshold=threshold,
                )
                if (
                    [_pick_token(pick) for pick in first.picks]
                    != [_pick_token(pick) for pick in second.picks]
                    or first.to_record() != second.to_record()
                ):
                    deterministic = False
                baseline_ids = {id(pick) for pick in first.baseline_picks}
                if any(id(pick) not in baseline_ids for pick in first.picks):
                    subset_pass = False
            for _ in range(100):
                started = time.perf_counter_ns()
                apply_event_confidence_filter(
                    raw,
                    duration_s=LONG_DURATION_SENTINEL_S,
                    threshold=0.45,
                )
                timings_ms.append((time.perf_counter_ns() - started) / 1e6)
    if not deterministic:
        reasons.append("相同输入重复运行结果不确定")
    if not subset_pass:
        reasons.append("候选输出不是 baseline picks 子集")
    sorted_timings = sorted(timings_ms)
    p95_index = max(0, math.ceil(0.95 * len(sorted_timings)) - 1)
    p95_ms = sorted_timings[p95_index] if sorted_timings else 0.0
    if p95_ms >= POSTPROCESS_P95_LIMIT_MS:
        reasons.append(
            f"后处理 P95 {p95_ms:.6f}ms 不小于 {POSTPROCESS_P95_LIMIT_MS}ms"
        )
    return {
        "pass": not reasons,
        "reasons": reasons,
        "short_300_unchanged": short_outcome.path == "short_unchanged",
        "long_300_000001_active": (
            long_boundary.path == "long_dedup_then_event_filter"
        ),
        "output_subset_pass": subset_pass,
        "deterministic_pass": deterministic,
        "postprocess_samples": len(timings_ms),
        "postprocess_p95_ms": p95_ms,
        "postprocess_mean_ms": (
            statistics.fmean(timings_ms) if timings_ms else 0.0
        ),
        "postprocess_limit_ms": POSTPROCESS_P95_LIMIT_MS,
    }


def _load_cache(path: Path) -> dict[str, dict[str, list[Pick]]]:
    raw_cache = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, list[Pick]]] = {}
    for spec in PACKAGE_SPECS.values():
        package: dict[str, list[Pick]] = {}
        for file_id in spec.long_file_ids:
            key = f"{spec.cache_prefix}{file_id}"
            rows = raw_cache[key]
            package[file_id] = [
                Pick(
                    phase=PhaseType(row["phase"]),
                    time_utc=float(row["t"]),
                    confidence=float(row["conf"]),
                )
                for row in rows
            ]
        result[spec.key] = package
    return result


def _git_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def run(
    *,
    baseline_path: Path,
    cache_path: Path,
    round2_answers: Path,
    final08_answers: Path,
) -> dict:
    started = time.perf_counter()
    baseline_hash = sha256_file(baseline_path)
    cache_hash = sha256_file(cache_path)
    round2_hash = sha256_file(round2_answers)
    final08_hash = sha256_file(final08_answers)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    cached_packages = _load_cache(cache_path)
    truths = {
        "round2": read_package_answers(str(round2_answers), ExamTask.T1),
        "final08": read_package_answers(str(final08_answers), ExamTask.T1),
    }

    identity_issues: list[str] = []
    if baseline_hash != BASELINE_SHA256:
        identity_issues.append(
            f"冻结基线 SHA-256 不匹配：{baseline_hash} != {BASELINE_SHA256}"
        )
    if cache_hash != CACHE_SHA256:
        identity_issues.append(
            f"长记录缓存 SHA-256 不匹配：{cache_hash} != {CACHE_SHA256}"
        )
    expected_round2_hash = baseline["datasets"]["round2"].get(
        "answer_sha256"
    ) or baseline["datasets"]["round2"]["exam_sha256"]
    expected_final08_hash = baseline["datasets"]["final08"]["answer_sha256"]
    if round2_hash != expected_round2_hash:
        identity_issues.append(
            f"R2 答案包 SHA-256 不匹配：{round2_hash} != {expected_round2_hash}"
        )
    if final08_hash != expected_final08_hash:
        identity_issues.append(
            f"08 答案包 SHA-256 不匹配：{final08_hash} != {expected_final08_hash}"
        )

    off_evaluations: dict[str, dict] = {}
    reproduction_by_package: dict[str, dict] = {}
    for key, spec in PACKAGE_SPECS.items():
        off = evaluate_package(
            spec=spec,
            baseline_dataset=baseline["datasets"][key],
            truths=truths[key],
            cached_picks=cached_packages[key],
            threshold=None,
        )
        off_evaluations[key] = off
        reproduction_by_package[key] = check_baseline_reproduction(
            spec=spec,
            baseline_dataset=baseline["datasets"][key],
            off_evaluation=off,
        )
    reproduction_pass = not identity_issues and all(
        record["pass"] for record in reproduction_by_package.values()
    )
    reproduction = {
        "pass": reproduction_pass,
        "identity_issues": identity_issues,
        "packages": reproduction_by_package,
    }

    active_evaluations: dict[str, dict[str, Mapping]] = {}
    for key, spec in PACKAGE_SPECS.items():
        active_evaluations[key] = evaluate_active_candidates_after_reproduction(
            reproduction_pass,
            lambda threshold, key=key, spec=spec: evaluate_package(
                spec=spec,
                baseline_dataset=baseline["datasets"][key],
                truths=truths[key],
                cached_picks=cached_packages[key],
                threshold=threshold,
            ),
        )

    safety = run_safety_checks(cached_packages)
    result: dict = {
        "experiment_id": EXPERIMENT_ID,
        "baseline_commit": BASELINE_COMMIT,
        "git_head": _git_head(),
        "seed": SEED,
        "inputs": {
            "baseline": {
                "basename": baseline_path.name,
                "sha256": baseline_hash,
                "expected_sha256": BASELINE_SHA256,
            },
            "long_pick_cache": {
                "basename": cache_path.name,
                "sha256": cache_hash,
                "expected_sha256": CACHE_SHA256,
            },
            "round2_answers": {
                "basename": round2_answers.name,
                "sha256": round2_hash,
                "expected_sha256": expected_round2_hash,
            },
            "final08_answers": {
                "basename": final08_answers.name,
                "sha256": final08_hash,
                "expected_sha256": expected_final08_hash,
            },
        },
        "protocol": {
            "duration_scope": {
                "short_unchanged_max_s": SHORT_MAX_DURATION_S,
                "long_active_when": "duration_s > 300.0",
            },
            "long_dedup_s": LONG_DEDUP_S,
            "pairing": "FIFO non-crossing P-to-S",
            "sp_min_s": SP_MIN_S,
            "sp_max_s": SP_MAX_S,
            "event_score": "sqrt(max(P_conf,0)*max(S_conf,0))",
            "threshold_grid": candidate_grid(),
            "submission_time_ndigits": SUBMISSION_NDIGITS,
            "drop_rule": "event_confidence < tau",
            "nonfinite_confidence": "keep pair",
            "source_min_worst_normalized_gain": SOURCE_MIN_NORMALIZED_GAIN,
            "selection_tie_tolerance": SELECTION_TIE_TOLERANCE,
            "lofo_max_tau_span": SELECTION_MAX_SPAN,
        },
        "baseline_reproduction": reproduction,
        "baseline_reproduction_pass": reproduction_pass,
        "off_evaluations": off_evaluations,
        "candidate_evaluations": active_evaluations,
        "safety": safety,
        "source_selections": None,
        "leave_one_file_out": None,
        "decision": None,
        "development_pass": False,
    }
    if not reproduction_pass:
        result["decision"] = {
            "development_pass": False,
            "failure_reasons": [
                "冻结 OFF 基线复现失败；按预注册停止，未运行 active 阈值"
            ],
        }
    else:
        all_evaluations = {
            key: {
                "OFF": off_evaluations[key],
                **active_evaluations[key],
            }
            for key in PACKAGE_SPECS
        }
        selections = {
            key: select_source_threshold(all_evaluations[key])
            for key in PACKAGE_SPECS
        }
        lofo: dict[str, dict] = {}
        for key, spec in PACKAGE_SPECS.items():
            lofo[key] = run_leave_one_file_out(
                spec=spec,
                full_selection=selections[key],
                evaluator=lambda threshold, active, key=key, spec=spec: evaluate_package(
                    spec=spec,
                    baseline_dataset=baseline["datasets"][key],
                    truths=truths[key],
                    cached_picks=cached_packages[key],
                    threshold=threshold,
                    active_file_ids=active,
                    include_pair_details=False,
                ),
            )
        decision = development_decision(
            round2_selection=selections["round2"],
            final08_selection=selections["final08"],
            round2_stability=lofo["round2"]["stability"],
            final08_stability=lofo["final08"]["stability"],
            round2_evaluations=all_evaluations["round2"],
            final08_evaluations=all_evaluations["final08"],
            safety=safety,
        )
        result["source_selections"] = selections
        result["leave_one_file_out"] = lofo
        result["decision"] = decision
        result["development_pass"] = decision["development_pass"]

    result["runtime_seconds"] = float(time.perf_counter() - started)
    result["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="outputs/frozen_baseline/baseline_full_profile_prod_20260811.json",
    )
    parser.add_argument(
        "--cache",
        default="outputs/port_verify/_long_picks_cache.json",
    )
    parser.add_argument(
        "--round2-answers",
        required=True,
        help="第 2 轮官方答案所在 zip；输出只记录 basename 与哈希",
    )
    parser.add_argument(
        "--final08-answers",
        required=True,
        help="去年决赛 T1 答案 zip；输出只记录 basename 与哈希",
    )
    parser.add_argument(
        "--output",
        default="outputs/experiments/round04_t1_long_event_confidence.json",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    result = run(
        baseline_path=Path(args.baseline),
        cache_path=Path(args.cache),
        round2_answers=Path(args.round2_answers),
        final08_answers=Path(args.final08_answers),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output": str(output),
        "baseline_reproduction_pass": result["baseline_reproduction_pass"],
        "source_selections": result["source_selections"],
        "development_pass": result["development_pass"],
        "failure_reasons": result["decision"]["failure_reasons"],
        "safety": result["safety"],
        "runtime_seconds": result["runtime_seconds"],
    }
    if result["decision"] and "common_candidate" in result["decision"]:
        summary["tau_common"] = result["decision"]["common_candidate"][
            "tau_common"
        ]
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["baseline_reproduction_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
