#!/usr/bin/env python3
"""审计 T1 贪心配对对官方时差分的敏感性。

当前生产评分器按残差从小到大贪心认领同相位预测/真值。本脚本不改变默认
评分口径，只用两个精确二分匹配目标复核冻结预测：

1. ``max_time_score``：直接最大化官方 P/S 时差分之和；
2. ``max_cardinality_min_residual``：先最大化容差内匹配数，再最小化总残差。

示例：

    python scripts/audit_t1_matching_sensitivity.py \
      --case "round1=C:/data/round1.zip::outputs/port_verify/r1_g7.an" \
      --case "round2=C:/data/round2.zip::outputs/port_verify/r2_g7.an" \
      --case "final08=C:/data/08-an.zip::outputs/port_verify/f08_g7.an" \
      --output outputs/frozen_baseline/t1_matching_sensitivity.json

输出只记录包/预测哈希与聚合指标，不写官方答案、逐文件 ID 或机器绝对路径。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_baseline import read_text_lines  # noqa: E402
from phasepicker.io.official_answers import parse_task1_answer_lines  # noqa: E402
from phasepicker.io.official_waveforms import read_package_answers  # noqa: E402
from phasepicker.scoring.scorer import (  # noqa: E402
    MatchResult,
    match_phases,
    phase_time_score,
)
from phasepicker.types import ExamTask, Task1Result  # noqa: E402


SCHEMA_VERSION = 1
ZERO_SCORE_LIMIT = {"P": 1.0, "S": 2.0}
OBJECTIVES = ("max_time_score", "max_cardinality_min_residual")


@dataclass(frozen=True)
class CaseSpec:
    name: str
    answer_path: Path
    prediction_path: Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_case(raw: str) -> CaseSpec:
    if "=" not in raw:
        raise ValueError(f"--case 必须是 NAME=ANSWERS::PREDICTIONS，收到 {raw!r}")
    name, value = raw.split("=", 1)
    parts = value.split("::", 1)
    if not name.strip() or len(parts) != 2 or not all(part.strip() for part in parts):
        raise ValueError(f"--case 必须是 NAME=ANSWERS::PREDICTIONS，收到 {raw!r}")
    return CaseSpec(
        name=name.strip(),
        answer_path=Path(os.path.expanduser(parts[0].strip())).resolve(),
        prediction_path=Path(os.path.expanduser(parts[1].strip())).resolve(),
    )


def match_phases_exact(
    pred_times: Sequence[float],
    true_times: Sequence[float],
    phase_type: str,
    objective: str = "max_time_score",
) -> MatchResult:
    """用 Hungarian assignment 求精确同相位匹配，不改变生产评分器。"""
    if phase_type not in ZERO_SCORE_LIMIT:
        raise ValueError(f"未知震相类型 {phase_type!r}")
    if objective not in OBJECTIVES:
        raise ValueError(f"未知精确匹配目标 {objective!r}")

    n_pred, n_true = len(pred_times), len(true_times)
    if not n_pred or not n_true:
        return MatchResult(
            unmatched_pred=list(range(n_pred)),
            unmatched_true=list(range(n_true)),
        )

    zero = ZERO_SCORE_LIMIT[phase_type]
    size = n_pred + n_true
    weights = np.zeros((size, size), dtype=np.float64)
    real_weights = np.full((n_pred, n_true), -1e12, dtype=np.float64)
    cardinality_base = (max(n_pred, n_true) + 1) * zero

    for pred_idx, pred_time in enumerate(pred_times):
        for true_idx, true_time in enumerate(true_times):
            residual = abs(float(pred_time) - float(true_time))
            if residual >= zero:
                continue
            if objective == "max_time_score":
                real_weights[pred_idx, true_idx] = phase_time_score(
                    residual, phase_type
                )
            else:
                # 每多匹配一对的收益大于全部可能残差之和，形成严格字典序：
                # 先最大匹配数，再最小总残差。
                real_weights[pred_idx, true_idx] = cardinality_base - residual

    weights[:n_pred, :n_true] = real_weights
    rows, columns = linear_sum_assignment(weights, maximize=True)

    matched = []
    used_pred: set[int] = set()
    used_true: set[int] = set()
    for pred_idx, true_idx in zip(rows.tolist(), columns.tolist()):
        if pred_idx >= n_pred or true_idx >= n_true:
            continue
        residual = abs(float(pred_times[pred_idx]) - float(true_times[true_idx]))
        if residual >= zero:
            continue
        matched.append((pred_idx, true_idx, residual))
        used_pred.add(pred_idx)
        used_true.add(true_idx)

    matched.sort(key=lambda item: (item[2], item[0], item[1]))
    return MatchResult(
        matched=matched,
        unmatched_pred=[idx for idx in range(n_pred) if idx not in used_pred],
        unmatched_true=[idx for idx in range(n_true) if idx not in used_true],
    )


def match_score(matches: Iterable[tuple[int, int, float]], phase_type: str) -> float:
    return sum(phase_time_score(residual, phase_type) for _, _, residual in matches)


def _times(result: Task1Result | None, phase_type: str) -> list[float]:
    if result is None:
        return []
    values = result.p_times_s if phase_type == "P" else result.s_times_s
    return [float(value) for value in values]


def audit_case(spec: CaseSpec) -> dict[str, object]:
    if not spec.answer_path.is_file():
        raise FileNotFoundError(f"答案包不存在：{spec.answer_path.name}")
    if not spec.prediction_path.is_file():
        raise FileNotFoundError(f"预测文件不存在：{spec.prediction_path.name}")

    answers = read_package_answers(str(spec.answer_path), ExamTask.T1)
    predictions = parse_task1_answer_lines(read_text_lines(str(spec.prediction_path)))
    phase_summaries: dict[str, object] = {}

    for phase_type in ("P", "S"):
        dense_files = 0
        greedy_matches = 0
        objective_summaries = {
            objective: {
                "different_files": 0,
                "dense_different_files": 0,
                "cardinality_different_files": 0,
                "exact_matches": 0,
                "score_delta": 0.0,
                "max_abs_file_score_delta": 0.0,
            }
            for objective in OBJECTIVES
        }

        for file_id in sorted(answers):
            pred_times = _times(predictions.get(file_id), phase_type)
            true_times = _times(answers[file_id], phase_type)
            is_dense = len(pred_times) > 1 or len(true_times) > 1
            dense_files += int(is_dense)

            greedy = match_phases(pred_times, true_times, phase_type)
            greedy_score = match_score(greedy.matched, phase_type)
            greedy_matches += len(greedy.matched)

            for objective in OBJECTIVES:
                exact = match_phases_exact(
                    pred_times, true_times, phase_type, objective=objective
                )
                exact_score = match_score(exact.matched, phase_type)
                delta = exact_score - greedy_score
                summary = objective_summaries[objective]
                summary["exact_matches"] += len(exact.matched)
                summary["score_delta"] += delta
                summary["max_abs_file_score_delta"] = max(
                    summary["max_abs_file_score_delta"], abs(delta)
                )
                if abs(delta) > 1e-12:
                    summary["different_files"] += 1
                    summary["dense_different_files"] += int(is_dense)
                if len(exact.matched) != len(greedy.matched):
                    summary["cardinality_different_files"] += 1

        phase_summaries[phase_type] = {
            "files": len(answers),
            "dense_files": dense_files,
            "greedy_matches": greedy_matches,
            "objectives": objective_summaries,
        }

    passed = all(
        objective["different_files"] == 0
        and objective["cardinality_different_files"] == 0
        for phase in phase_summaries.values()
        for objective in phase["objectives"].values()
    )
    return {
        "name": spec.name,
        "answer_basename": spec.answer_path.name,
        "answer_sha256": sha256_file(spec.answer_path),
        "prediction_basename": spec.prediction_path.name,
        "prediction_sha256": sha256_file(spec.prediction_path),
        "answer_files": len(answers),
        "prediction_files": len(predictions),
        "phases": phase_summaries,
        "pass": passed,
    }


def synthetic_guard() -> dict[str, object]:
    """证明审计器能识别贪心并非普遍最优，而不是恒等比较。"""
    pred_times = [0.0, 0.1]
    true_times = [0.1, 0.2]
    greedy = match_phases(pred_times, true_times, "P")
    exact = match_phases_exact(pred_times, true_times, "P", "max_time_score")
    greedy_score = match_score(greedy.matched, "P")
    exact_score = match_score(exact.matched, "P")
    return {
        "greedy_score": greedy_score,
        "exact_score": exact_score,
        "delta": exact_score - greedy_score,
        "detects_difference": exact_score > greedy_score + 1e-12,
    }


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="NAME=ANSWERS_ZIP::PREDICTION_AN，可重复",
    )
    parser.add_argument("--output", required=True, help="ignored JSON 输出路径")
    args = parser.parse_args()

    specs = [parse_case(raw) for raw in args.case]
    if len({spec.name for spec in specs}) != len(specs):
        raise SystemExit("--case 名称不能重复")

    guard = synthetic_guard()
    cases = [audit_case(spec) for spec in specs]
    passed = bool(guard["detects_difference"]) and all(case["pass"] for case in cases)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "objectives": list(OBJECTIVES),
        "synthetic_guard": guard,
        "cases": cases,
        "historical_predictions_insensitive": passed,
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for case in cases:
        print(f"{case['name']}: pass={case['pass']} files={case['answer_files']}")
    print(f"synthetic_guard_delta={guard['delta']:.12f}")
    print(f"historical_predictions_insensitive={passed}")
    print(f"output={output.name}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
