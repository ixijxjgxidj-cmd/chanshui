#!/usr/bin/env python3
"""审计 T1 候选在三套冻结数据包和四种数量罚口径下的稳健性。

本脚本只做已有候选预测的评分审计，不训练模型，也不改变生产评分器。
每个 ``--case`` 的格式为：

``NAME=ANSWER_TEXT::CANDIDATE=prediction.an::OTHER=other.an``

答案和预测可以是仅含 T1 行的最小文本，也可以是官方答案 zip。输出只保存
basename、SHA-256 和聚合指标；不会把输入绝对路径、答案内容或文件 ID 写入
JSON。以 ``g7`` 为生产基线，候选必须在全部 3×4=12 个单元相对基线不下降，
且至少一个单元严格上升，才通过稳健准入。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_baseline import read_text_lines  # noqa: E402
from phasepicker.io.official_answers import parse_task1_answer_lines  # noqa: E402
from phasepicker.io.official_waveforms import read_package_answers  # noqa: E402
from phasepicker.scoring.scorer import (  # noqa: E402
    PENALTY_MODES,
    ScoreReport,
    exam_total_score,
    score_file,
)
from phasepicker.types import ExamTask, Task1Result  # noqa: E402


SCHEMA_VERSION = 1
BASELINE_NAME = "g7"
TOLERANCE = 1e-9


@dataclass(frozen=True)
class CaseSpec:
    name: str
    answer_path: Path
    predictions: Mapping[str, Path]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_case(raw: str) -> CaseSpec:
    """解析 NAME=ANSWER::CANDIDATE=PATH::...，拒绝歧义输入。"""
    parts = raw.split("::")
    if len(parts) < 2 or "=" not in parts[0]:
        raise ValueError(
            "--case 必须是 NAME=ANSWER::CANDIDATE=PATH::...，至少包含一个候选"
        )
    name, answer = parts[0].split("=", 1)
    name, answer = name.strip(), answer.strip()
    if not name or not answer:
        raise ValueError("数据包名称和答案路径不能为空")

    predictions: dict[str, Path] = {}
    for item in parts[1:]:
        if "=" not in item:
            raise ValueError(f"候选必须是 NAME=PATH，收到 {item!r}")
        candidate, path = item.split("=", 1)
        candidate, path = candidate.strip(), path.strip()
        if not candidate or not path:
            raise ValueError(f"候选名称和路径不能为空，收到 {item!r}")
        if candidate in predictions:
            raise ValueError(f"候选名称重复：{candidate}")
        predictions[candidate] = Path(os.path.expanduser(path)).resolve()

    if BASELINE_NAME not in predictions:
        raise ValueError(f"每个数据包都必须包含 {BASELINE_NAME!r} 基线")
    return CaseSpec(
        name=name,
        answer_path=Path(os.path.expanduser(answer)).resolve(),
        predictions=predictions,
    )


def read_t1_answers(path: Path) -> dict[str, Task1Result]:
    """读取 zip 或最小 T1 文本。"""
    if not path.is_file():
        raise FileNotFoundError(f"输入不存在：{path.name}")
    if path.suffix.lower() == ".zip":
        return read_package_answers(str(path), ExamTask.T1)  # type: ignore[return-value]
    return parse_task1_answer_lines(read_text_lines(str(path)))


def pairs(result: Task1Result | None) -> list[tuple[str, float]]:
    if result is None:
        return []
    return [("P", float(value)) for value in result.p_times_s] + [
        ("S", float(value)) for value in result.s_times_s
    ]


def score_dataset(
    predictions: Mapping[str, Task1Result], answers: Mapping[str, Task1Result]
) -> dict[str, object]:
    """严格按答案文件集合评分，并保留每种口径的 ScoreReport。"""
    truth_ids = sorted(answers)
    missing = sorted(set(answers) - set(predictions))
    extra = sorted(set(predictions) - set(answers))
    reports_by_mode: dict[str, list[ScoreReport]] = {mode: [] for mode in PENALTY_MODES}
    per_file: dict[str, dict[str, ScoreReport]] = {}

    for file_id in truth_ids:
        pred = predictions.get(file_id)
        truth = answers[file_id]
        per_file[file_id] = {}
        for mode in PENALTY_MODES:
            report = score_file(pairs(pred), pairs(truth), penalty_mode=mode)
            per_file[file_id][mode] = report
            reports_by_mode[mode].append(report)

    package_modes: dict[str, dict[str, float]] = {}
    for mode in PENALTY_MODES:
        total, exam_penalty = exam_total_score(reports_by_mode[mode], mode)
        file_penalty = sum(report.count_penalty for report in reports_by_mode[mode])
        package_modes[mode] = {
            "total_score": float(total),
            "mean_score": float(total / len(truth_ids)) if truth_ids else 0.0,
            "exam_count_penalty": float(exam_penalty),
            "effective_count_penalty": float(
                exam_penalty if mode.endswith("_exam") else file_penalty
            ),
        }

    default_reports = reports_by_mode[PENALTY_MODES[0]]
    return {
        "coverage": {
            "answers": len(answers),
            "predictions": len(predictions),
            "matched": len(set(answers) & set(predictions)),
            "missing": len(missing),
            "extra": len(extra),
            "complete": not missing and not extra,
        },
        "package_modes": package_modes,
        "per_file": per_file,
        "errors": {
            "false_positives": sum(report.n_false_pos for report in default_reports),
            "false_negatives": sum(report.n_false_neg for report in default_reports),
        },
        "phase_scores": {
            "p_time_score": float(
                sum(report.p_time_score for report in default_reports)
            ),
            "s_time_score": float(
                sum(report.s_time_score for report in default_reports)
            ),
        },
    }


def _delta(candidate: float, baseline: float) -> float:
    return float(candidate - baseline)


def compare_candidate(
    candidate_name: str,
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, object]:
    """生成不含文件 ID 的候选相对基线诊断。"""
    base_modes = baseline["package_modes"]
    cand_modes = candidate["package_modes"]
    mode_deltas = {
        mode: {
            "baseline_total_score": float(base_modes[mode]["total_score"]),
            "candidate_total_score": float(cand_modes[mode]["total_score"]),
            "delta": _delta(
                cand_modes[mode]["total_score"], base_modes[mode]["total_score"]
            ),
            "baseline_count_penalty": float(
                base_modes[mode]["effective_count_penalty"]
            ),
            "candidate_count_penalty": float(
                cand_modes[mode]["effective_count_penalty"]
            ),
            "count_penalty_delta": _delta(
                cand_modes[mode]["effective_count_penalty"],
                base_modes[mode]["effective_count_penalty"],
            ),
        }
        for mode in PENALTY_MODES
    }

    base_files = baseline["per_file"]
    cand_files = candidate["per_file"]
    damaged_by_mode = {mode: 0 for mode in PENALTY_MODES}
    worst_by_mode = {mode: 0.0 for mode in PENALTY_MODES}
    p_time_damaged = s_time_damaged = 0
    fn_increased = 0
    file_count = 0
    for file_id, base_report_modes in base_files.items():
        file_count += 1
        cand_report_modes = cand_files[file_id]
        for mode in PENALTY_MODES:
            diff = _delta(
                cand_report_modes[mode].total_score,
                base_report_modes[mode].total_score,
            )
            if diff < -TOLERANCE:
                damaged_by_mode[mode] += 1
            worst_by_mode[mode] = min(worst_by_mode[mode], diff)
        base_default = base_report_modes[PENALTY_MODES[0]]
        cand_default = cand_report_modes[PENALTY_MODES[0]]
        p_time_damaged += int(
            cand_default.p_time_score + TOLERANCE < base_default.p_time_score
        )
        s_time_damaged += int(
            cand_default.s_time_score + TOLERANCE < base_default.s_time_score
        )
        fn_increased += int(cand_default.n_false_neg > base_default.n_false_neg)

    deltas = [mode_deltas[mode]["delta"] for mode in PENALTY_MODES]
    coverage_complete = bool(candidate["coverage"]["complete"])
    all_non_decreasing = all(delta >= -TOLERANCE for delta in deltas)
    strict_improvement = any(delta > TOLERANCE for delta in deltas)
    robust_pass = (
        candidate_name != BASELINE_NAME
        and coverage_complete
        and all_non_decreasing
        and strict_improvement
    )
    return {
        "candidate": candidate_name,
        "coverage": candidate["coverage"],
        "package_modes": mode_deltas,
        "worst_delta": float(min(deltas)) if deltas else 0.0,
        "mean_delta": float(sum(deltas) / len(deltas)) if deltas else 0.0,
        "file_diagnostics": {
            "files": file_count,
            "damaged_files_by_mode": damaged_by_mode,
            "worst_file_delta_by_mode": worst_by_mode,
            "p_time_damaged_files": p_time_damaged,
            "s_time_damaged_files": s_time_damaged,
            "fn_increased_files": fn_increased,
        },
        "errors": {
            "baseline_false_positives": int(baseline["errors"]["false_positives"]),
            "candidate_false_positives": int(candidate["errors"]["false_positives"]),
            "false_positive_delta": int(
                candidate["errors"]["false_positives"]
                - baseline["errors"]["false_positives"]
            ),
            "baseline_false_negatives": int(baseline["errors"]["false_negatives"]),
            "candidate_false_negatives": int(candidate["errors"]["false_negatives"]),
            "false_negative_delta": int(
                candidate["errors"]["false_negatives"]
                - baseline["errors"]["false_negatives"]
            ),
        },
        "phase_scores": {
            phase: {
                "baseline": float(baseline["phase_scores"][phase]),
                "candidate": float(candidate["phase_scores"][phase]),
                "delta": _delta(
                    candidate["phase_scores"][phase],
                    baseline["phase_scores"][phase],
                ),
            }
            for phase in ("p_time_score", "s_time_score")
        },
        "all_modes_non_decreasing": all_non_decreasing,
        "has_strict_improvement": strict_improvement,
        "robust_pass": robust_pass,
    }


def audit_case(spec: CaseSpec) -> dict[str, object]:
    answers = read_t1_answers(spec.answer_path)
    parsed = {
        name: read_t1_answers(path) for name, path in spec.predictions.items()
    }
    scored = {name: score_dataset(pred, answers) for name, pred in parsed.items()}
    baseline = scored[BASELINE_NAME]
    candidates = {
        name: compare_candidate(name, scored[name], baseline)
        for name in sorted(scored)
    }
    return {
        "name": spec.name,
        "answer_basename": spec.answer_path.name,
        "answer_sha256": sha256_file(spec.answer_path),
        "answer_files": len(answers),
        "predictions": {
            name: {
                "basename": spec.predictions[name].name,
                "sha256": sha256_file(spec.predictions[name]),
                "coverage": scored[name]["coverage"],
            }
            for name in sorted(spec.predictions)
        },
        "baseline": {
            "candidate": BASELINE_NAME,
            "package_modes": baseline["package_modes"],
            "errors": baseline["errors"],
        },
        "candidates": candidates,
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
        "--case", action="append", required=True, help="NAME=ANSWER::CAND=PATH::..."
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    specs = [parse_case(raw) for raw in args.case]
    if len({spec.name for spec in specs}) != len(specs):
        raise SystemExit("数据包名称不能重复")
    cases = [audit_case(spec) for spec in specs]

    all_candidates = sorted(
        {
            candidate
            for case in cases
            for candidate in case["candidates"]
            if candidate != BASELINE_NAME
        }
    )
    robust: dict[str, dict[str, object]] = {}
    for candidate in all_candidates:
        records = [case["candidates"].get(candidate) for case in cases]
        deltas = [
            record["package_modes"][mode]["delta"]
            for record in records
            if record is not None
            for mode in PENALTY_MODES
        ]
        complete = len(records) == len(cases) and all(
            record is not None and record["coverage"]["complete"]
            for record in records
        )
        all_non_decreasing = (
            len(deltas) == len(cases) * len(PENALTY_MODES)
            and all(delta >= -TOLERANCE for delta in deltas)
        )
        strict_improvement = any(delta > TOLERANCE for delta in deltas)
        robust[candidate] = {
            "cells_present": len(deltas),
            "coverage_complete": complete,
            "worst_delta": float(min(deltas)) if deltas else None,
            "mean_delta": float(sum(deltas) / len(deltas)) if deltas else None,
            "all_cells_non_decreasing": all_non_decreasing,
            "has_strict_improvement": strict_improvement,
            "robust_pass": complete and all_non_decreasing and strict_improvement,
        }
    ranking = sorted(
        all_candidates,
        key=lambda candidate: (
            not robust[candidate]["robust_pass"],
            -float(
                robust[candidate]["worst_delta"]
                if robust[candidate]["worst_delta"] is not None
                else float("-inf")
            ),
            -float(
                robust[candidate]["mean_delta"]
                if robust[candidate]["mean_delta"] is not None
                else float("-inf")
            ),
            candidate,
        ),
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "baseline": BASELINE_NAME,
        "penalty_modes": list(PENALTY_MODES),
        "criteria": {
            "cells": len(cases) * len(PENALTY_MODES),
            "non_decrease_tolerance": TOLERANCE,
            "requires_strict_improvement": True,
            "requires_complete_coverage": True,
        },
        "cases": cases,
        "robust_candidates": robust,
        "ranking": ranking,
        "any_robust_candidate": any(
            record["robust_pass"] for record in robust.values()
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for candidate in ranking:
        record = robust[candidate]
        print(
            f"{candidate}: robust={record['robust_pass']} "
            f"worst_delta={record['worst_delta']:.12f} "
            f"mean_delta={record['mean_delta']:.12f}"
        )
    print(f"cases={len(cases)} cells={len(cases) * len(PENALTY_MODES)}")
    print(f"any_robust_candidate={result['any_robust_candidate']}")
    print(f"output={output.name}")
    return 0 if result["any_robust_candidate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
