"""官方任务评估（Official task evaluation）——把预测与官方答案比对出分.

三个任务各有各的评估口径：
- T1：复用现有官方评分器 scoring.scorer.score_file（P/S 到时得分 + 数量罚）。
  这里只做"结构转换"：把 Task1Result 的 P/S 列表拍平成 score_file 认识的
  ``[("P", t), ("S", t), ...]``，然后逐文件汇总。
- T2：震级回归，报 MAE（平均绝对误差）+ 已匹配数 + 缺失/多余数。
- T3：事件分类，报 accuracy + 混淆矩阵 + 缺失/多余数。

"缺失/多余"指预测集合与答案集合的 file_id 差集：answer 有而 pred 没有 = missing；
pred 有而 answer 没有 = extra。默认只对交集文件计入分数（--limit 调试友好），
T1 可用 strict_missing=True 把缺失按 0 分计入均分分母（官方对缺失必然记 0）。

返回普通 dataclass（字段皆为内置类型），便于测试断言与直接 print。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..types import Task1Result, Task2Result, Task3Result
from ..scoring.scorer import (
    DEFAULT_PENALTY_MODE,
    PENALTY_MODES,
    ScoreReport,
    exam_total_score,
    score_file,
)


def _common_keys(pred: Dict[str, object], truth: Dict[str, object]) -> Tuple[List[str], List[str], List[str]]:
    """返回 (共有键 已排序, 缺失键 truth有pred无, 多余键 pred有truth无)。"""
    pk, tk = set(pred), set(truth)
    common = sorted(pk & tk)
    missing = sorted(tk - pk)
    extra = sorted(pk - tk)
    return common, missing, extra


# ============================ T1 ============================

@dataclass
class Task1EvalReport:
    """T1 评估汇总。新增字段一律带默认值，保持既有构造/断言不受影响。"""

    total_score: float
    mean_score: float
    n_files: int
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)
    per_file: Dict[str, float] = field(default_factory=dict)
    penalty_mode: str = DEFAULT_PENALTY_MODE
    strict_missing: bool = False
    exam_count_penalty: float = 0.0
    """全卷读法(*_exam)下在卷级统一扣掉的数量罚；文件级读法恒 0。"""

    def summary(self) -> str:
        head = (
            f"T1: 均分={self.mean_score:.3f} 总分={self.total_score:.3f} "
            f"文件数={self.n_files} 缺失={len(self.missing)} 多余={len(self.extra)}"
        )
        if self.penalty_mode != DEFAULT_PENALTY_MODE:
            head += f" [数量罚读法={self.penalty_mode}"
            if self.penalty_mode.endswith("_exam"):
                head += f", 卷级数量罚={self.exam_count_penalty:.1f}"
            head += "]"
        if self.missing:
            head += (
                f"\n⚠️  警告: 答案侧有 {len(self.missing)} 个文件没有预测"
                + (
                    "，已按 0 分计入均分分母（strict 口径）"
                    if self.strict_missing
                    else "，未计入均分（默认口径，--limit 调试属正常；"
                    "正式评估请核对覆盖率或用 strict 口径按 0 分计入）"
                )
            )
        return head


def _task1_to_pairs(r: Task1Result) -> List[Tuple[str, float]]:
    """Task1Result → score_file 认识的 [("P", t), ("S", t), ...]。"""
    pairs: List[Tuple[str, float]] = [("P", float(t)) for t in r.p_times_s]
    pairs += [("S", float(t)) for t in r.s_times_s]
    return pairs


def evaluate_task1(
    predictions: Dict[str, Task1Result],
    answers: Dict[str, Task1Result],
    penalty_mode: str = DEFAULT_PENALTY_MODE,
    strict_missing: bool = False,
) -> Task1EvalReport:
    """对 T1 逐文件评分并汇总。

    Args:
        penalty_mode: 数量罚读法（见 scoring.scorer.PENALTY_MODES），逐文件与
            卷级聚合使用同一读法。
        strict_missing: True 时，答案有而预测无的文件按"空预测"参与评分——
            计入均分分母（per_file 记 0 分），*_exam 读法下其真值数也计入
            卷级数量罚。默认 False 保持历史口径（只对共有文件计分），
            以免打爆 --limit 调试工作流。
    """
    common, missing, extra = _common_keys(predictions, answers)
    scored: Dict[str, ScoreReport] = {}
    for fid in common:
        scored[fid] = score_file(
            _task1_to_pairs(predictions[fid]),
            _task1_to_pairs(answers[fid]),
            penalty_mode=penalty_mode,
        )
    if strict_missing:
        for fid in missing:
            scored[fid] = score_file(
                [], _task1_to_pairs(answers[fid]), penalty_mode=penalty_mode
            )
    ordered_fids = sorted(scored)
    total, exam_pen = exam_total_score(
        [scored[fid] for fid in ordered_fids], penalty_mode
    )
    n = len(ordered_fids)
    return Task1EvalReport(
        total_score=total,
        mean_score=(total / n) if n else 0.0,
        n_files=n,
        missing=missing,
        extra=extra,
        per_file={fid: scored[fid].total_score for fid in ordered_fids},
        penalty_mode=penalty_mode,
        strict_missing=strict_missing,
        exam_count_penalty=exam_pen,
    )


def evaluate_task1_all_modes(
    predictions: Dict[str, Task1Result],
    answers: Dict[str, Task1Result],
    strict_missing: bool = False,
) -> Dict[str, Task1EvalReport]:
    """一次性按全部数量罚读法各评一遍，返回 {读法: 报告}（键序同 PENALTY_MODES）。"""
    return {
        mode: evaluate_task1(
            predictions, answers, penalty_mode=mode, strict_missing=strict_missing
        )
        for mode in PENALTY_MODES
    }


def penalty_modes_table(reports: Dict[str, Task1EvalReport]) -> str:
    """把 evaluate_task1_all_modes 的结果排成对照表（官方规则读法敏感度一览）。"""
    lines = ["数量罚读法对照（官方规则原文缺失，读法间差异越大越需向组委会确认口径）:"]
    lines.append(f"  {'读法':<22} {'均分':>8} {'总分':>10} {'卷级罚':>8}")
    for mode in PENALTY_MODES:
        rep = reports.get(mode)
        if rep is None:
            continue
        lines.append(
            f"  {mode:<22} {rep.mean_score:>8.4f} {rep.total_score:>10.3f} "
            f"{rep.exam_count_penalty:>8.1f}"
        )
    return "\n".join(lines)


# ============================ T2 ============================

@dataclass
class Task2EvalReport:
    """T2 评估汇总（震级 MAE）。"""

    mae: float
    count: int
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)
    max_abs_error: float = 0.0

    def summary(self) -> str:
        return (
            f"T2: MAE={self.mae:.4f} maxAE={self.max_abs_error:.4f} "
            f"count={self.count} 缺失={len(self.missing)} 多余={len(self.extra)}"
        )


def evaluate_task2(
    predictions: Dict[str, Task2Result],
    answers: Dict[str, Task2Result],
) -> Task2EvalReport:
    """对 T2 计算 MAE、最大绝对误差、计数与缺失/多余。"""
    common, missing, extra = _common_keys(predictions, answers)
    abs_errors: List[float] = []
    for fid in common:
        err = abs(float(predictions[fid].magnitude) - float(answers[fid].magnitude))
        abs_errors.append(err)
    count = len(abs_errors)
    mae = (sum(abs_errors) / count) if count else 0.0
    return Task2EvalReport(
        mae=mae,
        count=count,
        missing=missing,
        extra=extra,
        max_abs_error=max(abs_errors) if abs_errors else 0.0,
    )


# ============================ T3 ============================

@dataclass
class Task3EvalReport:
    """T3 评估汇总（分类 accuracy + 混淆矩阵）。"""

    accuracy: float
    correct: int
    count: int
    confusion: Dict[Tuple[int, int], int] = field(default_factory=dict)
    """混淆矩阵：{(真值类别, 预测类别): 计数}。"""
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"T3: acc={self.accuracy:.4f} ({self.correct}/{self.count}) "
            f"缺失={len(self.missing)} 多余={len(self.extra)}"
        )


def evaluate_task3(
    predictions: Dict[str, Task3Result],
    answers: Dict[str, Task3Result],
) -> Task3EvalReport:
    """对 T3 计算 accuracy、混淆矩阵与缺失/多余。"""
    common, missing, extra = _common_keys(predictions, answers)
    confusion: Dict[Tuple[int, int], int] = {}
    correct = 0
    for fid in common:
        true_label = int(answers[fid].label)
        pred_label = int(predictions[fid].label)
        key = (true_label, pred_label)
        confusion[key] = confusion.get(key, 0) + 1
        if true_label == pred_label:
            correct += 1
    count = len(common)
    return Task3EvalReport(
        accuracy=(correct / count) if count else 0.0,
        correct=correct,
        count=count,
        confusion=confusion,
        missing=missing,
        extra=extra,
    )
