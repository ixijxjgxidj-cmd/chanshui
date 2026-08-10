#!/usr/bin/env python3
"""冻结官方历史包的可复现基线。

这个入口只做只读审计与评估，不训练模型、不修改官方包：

* 固定输入包、答案包和预测文件的 SHA-256；
* 对 T1 同时计算四种数量罚读法，并保存逐文件残差、FP/FN 与数量罚；
* 对 T2 计算 MAE、偏差和误差分位数；
* 对 T3 计算准确率、每类召回与混淆矩阵；
* 可选逐条读取 MiniSEED，统计采样率、时长、台站、通道、缺分量和 gap；
* 记录评估/画像耗时、吞吐、单文件延迟分位数和进程峰值 RSS。

示例（08 的输入与答案分包）：

    python scripts/freeze_baseline.py \
      --dataset "round1=C:/data/round1.zip" \
      --dataset "round2=C:/data/round2.zip" \
      --dataset "final08=C:/data/08-exam.zip::C:/data/08-an.zip" \
      --t1-pred "round1=outputs/port_verify/r1_g7.an" \
      --t1-pred "round2=outputs/port_verify/r2_g7.an" \
      --t1-pred "final08=outputs/port_verify/f08_g7.an" \
      --t2-pred "final08=outputs/final08/an/T2.an" \
      --t3-pred "final08=outputs/final08/an/T3.an" \
      --profile-waveforms --output outputs/frozen_baseline/baseline.json

``NAME=EXAM[::ANSWERS]`` 中省略 ANSWERS 时，答案从 EXAM 同一个包读取。
输出 JSON 应放在被忽略的 ``outputs/``；仓库只提交摘要 scoreboard，绝不提交
原始波形、官方答案全文或逐文件敏感产物。
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.io.official_answers import (  # noqa: E402
    parse_task1_answer_lines,
    parse_task2_answer_lines,
    parse_task3_answer_lines,
)
from phasepicker.io.official_exam import scan_exam_input  # noqa: E402
from phasepicker.io.official_waveforms import (  # noqa: E402
    read_mseed_stream,
    read_package_answers,
)
from phasepicker.scoring.scorer import (  # noqa: E402
    PENALTY_MODES,
    ScoreReport,
    exam_total_score,
    score_file,
)
from phasepicker.types import (  # noqa: E402
    ExamSample,
    ExamTask,
    Task1Result,
    Task2Result,
    Task3Result,
)


SCHEMA_VERSION = 1
COMPONENT_MAP = {"Z": "Z", "N": "N", "E": "E", "1": "N", "2": "E", "3": "Z"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    exam_path: str
    answer_path: str


def parse_named_value(raw: str, option: str) -> Tuple[str, str]:
    """解析 ``NAME=VALUE``，给出可读错误。"""
    if "=" not in raw:
        raise ValueError(f"{option} 必须是 NAME=VALUE，收到：{raw!r}")
    name, value = raw.split("=", 1)
    name, value = name.strip(), value.strip()
    if not name or not value:
        raise ValueError(f"{option} 的名称和值都不能为空：{raw!r}")
    return name, value


def parse_dataset(raw: str) -> DatasetSpec:
    name, value = parse_named_value(raw, "--dataset")
    parts = value.split("::", 1)
    exam = os.path.abspath(os.path.expanduser(parts[0]))
    answers = os.path.abspath(os.path.expanduser(parts[1] if len(parts) == 2 else parts[0]))
    return DatasetSpec(name=name, exam_path=exam, answer_path=answers)


def parse_named_paths(values: Sequence[str], option: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in values:
        name, value = parse_named_value(raw, option)
        if name in out:
            raise ValueError(f"{option} 重复数据集名称：{name}")
        out[name] = os.path.abspath(os.path.expanduser(value))
    return out


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_text_lines(path: str) -> List[str]:
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return raw.decode(encoding).splitlines()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").splitlines()


def quantiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """JSON 友好的基础分布；空输入全部为 None。"""
    if not values:
        return {k: None for k in ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _linux_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if not status.exists():
        return 0
    for line in status.read_text(encoding="ascii", errors="ignore").splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            return int(fields[1]) * 1024
    return 0


def _windows_rss_bytes() -> int:
    """不依赖 psutil 获取当前 WorkingSetSize。"""

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
    return int(counters.WorkingSetSize) if ok else 0


def current_rss_bytes() -> int:
    try:
        if os.name == "nt":
            return _windows_rss_bytes()
        return _linux_rss_bytes()
    except Exception:  # pragma: no cover - 仅影响诊断字段
        return 0


class PeakRSSSampler:
    """后台采样峰值 RSS；不引入新的运行依赖。"""

    def __init__(self, interval_s: float = 0.02):
        self.interval_s = interval_s
        self.start_rss = 0
        self.end_rss = 0
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        self.start_rss = current_rss_bytes()
        self.peak_rss = self.start_rss

        def _sample() -> None:
            while not self._stop.wait(self.interval_s):
                self.peak_rss = max(self.peak_rss, current_rss_bytes())

        self._thread = threading.Thread(target=_sample, name="peak-rss", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.end_rss = current_rss_bytes()
        self.peak_rss = max(self.peak_rss, self.end_rss)

    def as_dict(self) -> Dict[str, int]:
        return {
            "start_rss_bytes": self.start_rss,
            "end_rss_bytes": self.end_rss,
            "peak_rss_bytes": self.peak_rss,
            "peak_increase_bytes": max(0, self.peak_rss - self.start_rss),
        }


def _pairs(result: Task1Result) -> List[Tuple[str, float]]:
    return [("P", float(t)) for t in result.p_times_s] + [
        ("S", float(t)) for t in result.s_times_s
    ]


def _score_report_dict(rep: ScoreReport) -> Dict[str, object]:
    return {
        "total_score": rep.total_score,
        "p_time_score": rep.p_time_score,
        "s_time_score": rep.s_time_score,
        "count_penalty": rep.count_penalty,
        "n_pred_p": rep.n_pred_p,
        "n_true_p": rep.n_true_p,
        "n_pred_s": rep.n_pred_s,
        "n_true_s": rep.n_true_s,
        "p_residuals_s": rep.p_residuals,
        "s_residuals_s": rep.s_residuals,
        "false_positives": rep.n_false_pos,
        "false_negatives": rep.n_false_neg,
    }


def evaluate_t1_frozen(
    predictions: Mapping[str, Task1Result],
    answers: Mapping[str, Task1Result],
) -> Dict[str, object]:
    """严格覆盖答案集合；缺失预测按空结果计分。"""
    truth_ids = sorted(answers)
    missing = sorted(set(answers) - set(predictions))
    extra = sorted(set(predictions) - set(answers))
    per_file: Dict[str, Dict[str, object]] = {
        fid: {"modes": {}} for fid in truth_ids
    }
    mode_summaries: Dict[str, object] = {}

    for mode in PENALTY_MODES:
        reports: List[ScoreReport] = []
        for fid in truth_ids:
            pred = predictions.get(fid, Task1Result(file_id=fid))
            rep = score_file(_pairs(pred), _pairs(answers[fid]), penalty_mode=mode)
            reports.append(rep)
            per_file[fid]["modes"][mode] = {
                "total_score": rep.total_score,
                "count_penalty": rep.count_penalty,
            }
            if mode == PENALTY_MODES[0]:
                per_file[fid].update(_score_report_dict(rep))
        total, exam_penalty = exam_total_score(reports, mode)
        n = len(reports)
        mode_summaries[mode] = {
            "total_score": total,
            "mean_score": total / n if n else 0.0,
            "exam_count_penalty": exam_penalty,
            "n_files": n,
        }

    default_reports = [
        score_file(
            _pairs(predictions.get(fid, Task1Result(file_id=fid))),
            _pairs(answers[fid]),
            penalty_mode=PENALTY_MODES[0],
        )
        for fid in truth_ids
    ]
    p_res = [r for rep in default_reports for r in rep.p_residuals]
    s_res = [r for rep in default_reports for r in rep.s_residuals]
    return {
        "coverage": {
            "answers": len(answers),
            "predictions": len(predictions),
            "matched": len(set(answers) & set(predictions)),
            "missing": missing,
            "extra": extra,
        },
        "truth_counts": {
            "p": quantiles([len(r.p_times_s) for r in answers.values()]),
            "s": quantiles([len(r.s_times_s) for r in answers.values()]),
            "files_with_no_p": sum(not r.p_times_s for r in answers.values()),
            "files_with_no_s": sum(not r.s_times_s for r in answers.values()),
            "files_with_multi_event": sum(
                len(r.p_times_s) > 1 or len(r.s_times_s) > 1 for r in answers.values()
            ),
        },
        "penalty_modes": mode_summaries,
        "residuals_s": {
            "p": quantiles(p_res),
            "s": quantiles(s_res),
            "p_full_score_rate": (
                sum(r <= 0.1 for r in p_res) / len(p_res) if p_res else None
            ),
            "s_full_score_rate": (
                sum(r <= 0.2 for r in s_res) / len(s_res) if s_res else None
            ),
        },
        "errors": {
            "false_positives": sum(r.n_false_pos for r in default_reports),
            "false_negatives": sum(r.n_false_neg for r in default_reports),
        },
        "per_file": per_file,
    }


def evaluate_t2_frozen(
    predictions: Mapping[str, Task2Result],
    answers: Mapping[str, Task2Result],
) -> Dict[str, object]:
    common = sorted(set(predictions) & set(answers))
    missing = sorted(set(answers) - set(predictions))
    extra = sorted(set(predictions) - set(answers))
    signed = [
        float(predictions[fid].magnitude) - float(answers[fid].magnitude) for fid in common
    ]
    absolute = [abs(x) for x in signed]
    exact_matches = sum(x == 0.0 for x in signed)
    return {
        "coverage": {
            "answers": len(answers),
            "predictions": len(predictions),
            "matched": len(common),
            "missing": missing,
            "extra": extra,
        },
        "mae": float(np.mean(absolute)) if absolute else None,
        "mean_signed_error": float(np.mean(signed)) if signed else None,
        "exact_match_count": exact_matches,
        "suspicious_prediction_equals_answers": bool(
            common and len(common) == len(answers) == len(predictions) and exact_matches == len(common)
        ),
        "absolute_error": quantiles(absolute),
        "truth_magnitude": quantiles([float(r.magnitude) for r in answers.values()]),
        "per_file": {
            fid: {
                "truth": float(answers[fid].magnitude),
                "prediction": float(predictions[fid].magnitude),
                "signed_error": signed[i],
                "absolute_error": absolute[i],
            }
            for i, fid in enumerate(common)
        },
    }


def evaluate_t3_frozen(
    predictions: Mapping[str, Task3Result],
    answers: Mapping[str, Task3Result],
) -> Dict[str, object]:
    common = sorted(set(predictions) & set(answers))
    missing = sorted(set(answers) - set(predictions))
    extra = sorted(set(predictions) - set(answers))
    confusion: Counter[Tuple[int, int]] = Counter()
    truth_counts: Counter[int] = Counter()
    correct_counts: Counter[int] = Counter()
    per_file: Dict[str, object] = {}
    for fid in common:
        truth = int(answers[fid].label)
        pred = int(predictions[fid].label)
        confusion[(truth, pred)] += 1
        truth_counts[truth] += 1
        correct_counts[truth] += int(truth == pred)
        per_file[fid] = {"truth": truth, "prediction": pred, "correct": truth == pred}
    labels = sorted({int(r.label) for r in answers.values()} | {int(r.label) for r in predictions.values()})
    correct = sum(v for (truth, pred), v in confusion.items() if truth == pred)
    return {
        "coverage": {
            "answers": len(answers),
            "predictions": len(predictions),
            "matched": len(common),
            "missing": missing,
            "extra": extra,
        },
        "accuracy": correct / len(common) if common else None,
        "correct": correct,
        "count": len(common),
        "suspicious_prediction_equals_answers": bool(
            common and len(common) == len(answers) == len(predictions) and correct == len(common)
        ),
        "labels_seen_in_truth": sorted({int(r.label) for r in answers.values()}),
        "truth_distribution": {str(k): v for k, v in sorted(truth_counts.items())},
        "per_class_recall": {
            str(label): (
                correct_counts[label] / truth_counts[label] if truth_counts[label] else None
            )
            for label in labels
        },
        "confusion": {
            f"{truth}->{pred}": count
            for (truth, pred), count in sorted(confusion.items())
        },
        "per_file": per_file,
    }


def _component(channel: str) -> Optional[str]:
    return COMPONENT_MAP.get(channel[-1:].upper()) if channel else None


def _trace_start_end(trace) -> Tuple[float, float]:
    return float(trace.stats.starttime.timestamp), float(trace.stats.endtime.timestamp)


def _station_gaps(
    traces: Sequence[object], station_start: float
) -> Tuple[List[Dict[str, float]], int]:
    """按通道计算正 gap；返回 gap 明细和重叠段数。"""
    by_channel: Dict[str, List[object]] = defaultdict(list)
    for tr in traces:
        by_channel[str(tr.stats.channel)].append(tr)
    gaps: List[Dict[str, float]] = []
    overlaps = 0
    for channel, channel_traces in by_channel.items():
        covered_end: Optional[float] = None
        for tr in sorted(channel_traces, key=lambda x: x.stats.starttime):
            start, end = _trace_start_end(tr)
            sr = float(tr.stats.sampling_rate)
            delta = 1.0 / sr if sr > 0 else 0.0
            if covered_end is not None:
                if start - covered_end > 1.5 * delta:
                    gaps.append(
                        {
                            "channel": channel,
                            "start_relative_s": covered_end - station_start,
                            "end_relative_s": start - station_start,
                            "duration_s": start - covered_end,
                        }
                    )
                elif start < covered_end - 1.5 * delta:
                    overlaps += 1
            covered_end = end if covered_end is None else max(covered_end, end)
    return gaps, overlaps


def profile_sample(sample: ExamSample) -> Dict[str, object]:
    t0 = time.perf_counter()
    stream = read_mseed_stream(sample.source_path)
    stations: Dict[str, List[object]] = defaultdict(list)
    channels: Counter[str] = Counter()
    sample_rates: Counter[str] = Counter()
    for tr in stream:
        station = f"{tr.stats.network}.{tr.stats.station}"
        stations[station].append(tr)
        channels[str(tr.stats.channel)] += 1
        sample_rates[f"{float(tr.stats.sampling_rate):g}"] += 1

    station_profiles: Dict[str, object] = {}
    file_durations: List[float] = []
    total_gaps = 0
    total_gap_duration = 0.0
    missing_station_count = 0
    overlap_count = 0
    for station, traces in stations.items():
        starts_ends = [_trace_start_end(tr) for tr in traces]
        station_start = min(x[0] for x in starts_ends)
        station_end = max(x[1] for x in starts_ends)
        duration = station_end - station_start
        file_durations.append(duration)
        present = sorted({c for tr in traces if (c := _component(str(tr.stats.channel)))})
        missing = sorted({"Z", "N", "E"} - set(present))
        gaps, overlaps = _station_gaps(traces, station_start)
        total_gaps += len(gaps)
        total_gap_duration += sum(float(g["duration_s"]) for g in gaps)
        overlap_count += overlaps
        missing_station_count += int(bool(missing))
        station_profiles[station] = {
            "duration_s": duration,
            "components_present": present,
            "components_missing": missing,
            "gaps": gaps,
            "overlap_segments": overlaps,
        }

    duration = max(file_durations) if file_durations else 0.0
    return {
        "file_id": sample.file_id,
        "task": sample.task.value,
        "duration_s": duration,
        "station_count": len(stations),
        "trace_count": len(stream),
        "sample_rates_hz": dict(sorted(sample_rates.items())),
        "channels": dict(sorted(channels.items())),
        "gap_count": total_gaps,
        "gap_duration_s": total_gap_duration,
        "stations_with_missing_components": missing_station_count,
        "overlap_segments": overlap_count,
        "stations": station_profiles,
        "latency_ms": (time.perf_counter() - t0) * 1000.0,
    }


def summarize_waveform_profiles(
    samples: Sequence[ExamSample], workers: int
) -> Dict[str, object]:
    profiles: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    start = time.perf_counter()
    with PeakRSSSampler() as mem:
        with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="freeze-profile") as ex:
            futures = [(sample, ex.submit(profile_sample, sample)) for sample in samples]
            for sample, future in futures:
                try:
                    profiles.append(future.result())
                except Exception as exc:  # noqa: BLE001 - 单文件失败不能终止全包画像
                    failures.append(
                        {
                            "file_id": sample.file_id,
                            "task": sample.task.value,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    elapsed = time.perf_counter() - start
    durations = [float(p["duration_s"]) for p in profiles]
    latencies = [float(p["latency_ms"]) for p in profiles]
    anomalous = {
        str(p["file_id"]): p
        for p in profiles
        if int(p["gap_count"])
        or int(p["stations_with_missing_components"])
        or int(p["overlap_segments"])
        or int(p["station_count"]) != 1
        or float(p["duration_s"]) < 5.0
        or float(p["duration_s"]) > 3600.0
    }
    sample_rates: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    task_quality: Dict[str, Counter[str]] = defaultdict(Counter)
    for p in profiles:
        task = str(p["task"])
        duration = float(p["duration_s"])
        task_counts[task] += 1
        task_quality[task]["files"] += 1
        task_quality[task]["too_short_lt_5"] += int(duration < 5.0)
        task_quality[task]["short_5_to_300"] += int(5.0 <= duration <= 300.0)
        task_quality[task]["long_300_to_3600"] += int(300.0 < duration <= 3600.0)
        task_quality[task]["ultralong_gt_3600"] += int(duration > 3600.0)
        task_quality[task]["files_with_gaps"] += int(int(p["gap_count"]) > 0)
        task_quality[task]["files_with_missing_components"] += int(
            int(p["stations_with_missing_components"]) > 0
        )
        task_quality[task]["files_with_overlaps"] += int(int(p["overlap_segments"]) > 0)
        sample_rates.update({k: int(v) for k, v in dict(p["sample_rates_hz"]).items()})
        channels.update({k: int(v) for k, v in dict(p["channels"]).items()})
    return {
        "files_requested": len(samples),
        "files_profiled": len(profiles),
        "failures": failures,
        "task_counts": dict(sorted(task_counts.items())),
        "task_quality": {
            task: dict(sorted(counts.items())) for task, counts in sorted(task_quality.items())
        },
        "sample_rates_trace_count": dict(sorted(sample_rates.items(), key=lambda x: float(x[0]))),
        "channel_trace_count": dict(sorted(channels.items())),
        "duration_s": quantiles(durations),
        "duration_buckets": {
            "too_short_lt_5": sum(x < 5.0 for x in durations),
            "short_5_to_300": sum(5.0 <= x <= 300.0 for x in durations),
            "long_300_to_3600": sum(300.0 < x <= 3600.0 for x in durations),
            "ultralong_gt_3600": sum(x > 3600.0 for x in durations),
        },
        "station_count": quantiles([float(p["station_count"]) for p in profiles]),
        "trace_count": quantiles([float(p["trace_count"]) for p in profiles]),
        "files_with_gaps": sum(int(p["gap_count"]) > 0 for p in profiles),
        "gap_count": sum(int(p["gap_count"]) for p in profiles),
        "gap_duration_s": sum(float(p["gap_duration_s"]) for p in profiles),
        "files_with_missing_components": sum(
            int(p["stations_with_missing_components"]) > 0 for p in profiles
        ),
        "stations_with_missing_components": sum(
            int(p["stations_with_missing_components"]) for p in profiles
        ),
        "files_with_overlaps": sum(int(p["overlap_segments"]) > 0 for p in profiles),
        "anomalous_files": anomalous,
        "runtime": {
            "elapsed_s": elapsed,
            "throughput_files_per_s": len(profiles) / elapsed if elapsed else None,
            "per_file_latency_ms": quantiles(latencies),
            **mem.as_dict(),
        },
    }


def _git_head() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # pragma: no cover - 非 Git 环境
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="冻结官方历史包画像与三任务基线")
    ap.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="NAME=EXAM[::ANSWERS]；可重复，输入与答案同包时省略 ::ANSWERS",
    )
    ap.add_argument("--t1-pred", action="append", default=[], help="NAME=T1.an；可重复")
    ap.add_argument("--t2-pred", action="append", default=[], help="NAME=T2.an；可重复")
    ap.add_argument("--t3-pred", action="append", default=[], help="NAME=T3.an；可重复")
    ap.add_argument("--profile-waveforms", action="store_true", help="逐文件读取 MiniSEED 做质量画像")
    ap.add_argument("--profile-task", choices=["all", "T1", "T2", "T3"], default="all")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--output", required=True, help="输出 JSON（建议 outputs/...）")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    try:
        datasets = [parse_dataset(x) for x in args.dataset]
        t1_paths = parse_named_paths(args.t1_pred, "--t1-pred")
        t2_paths = parse_named_paths(args.t2_pred, "--t2-pred")
        t3_paths = parse_named_paths(args.t3_pred, "--t3-pred")
    except ValueError as exc:
        ap.error(str(exc))

    names = [d.name for d in datasets]
    if len(names) != len(set(names)):
        ap.error("--dataset 名称不能重复")
    unknown_preds = (set(t1_paths) | set(t2_paths) | set(t3_paths)) - set(names)
    if unknown_preds:
        ap.error(f"预测引用了未定义数据集：{sorted(unknown_preds)}")

    for spec in datasets:
        for label, path in (("exam", spec.exam_path), ("answers", spec.answer_path)):
            if not os.path.isfile(path):
                ap.error(f"{spec.name} 的 {label} 包不存在：{path}")
    for task_name, mapping in (("T1", t1_paths), ("T2", t2_paths), ("T3", t3_paths)):
        for name, path in mapping.items():
            if not os.path.isfile(path):
                ap.error(f"{name} 的 {task_name} 预测不存在：{path}")

    output: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "penalty_modes": list(PENALTY_MODES),
        "datasets": {},
    }

    overall_start = time.perf_counter()
    with PeakRSSSampler() as overall_mem:
        for spec in datasets:
            print(f"[{spec.name}] 扫描与答案解析...", flush=True)
            samples = scan_exam_input(spec.exam_path)
            record: Dict[str, object] = {
                "exam_path_basename": os.path.basename(spec.exam_path),
                "answer_path_basename": os.path.basename(spec.answer_path),
                "exam_sha256": sha256_file(spec.exam_path),
                "answer_sha256": (
                    sha256_file(spec.answer_path)
                    if os.path.abspath(spec.answer_path) != os.path.abspath(spec.exam_path)
                    else None
                ),
                "sample_counts": dict(sorted(Counter(s.task.value for s in samples).items())),
                "tasks": {},
            }

            if spec.name in t1_paths:
                pred_path = t1_paths[spec.name]
                preds = parse_task1_answer_lines(read_text_lines(pred_path))
                answers = read_package_answers(spec.answer_path, ExamTask.T1)
                task = evaluate_t1_frozen(preds, answers)  # type: ignore[arg-type]
                task["prediction_sha256"] = sha256_file(pred_path)
                task["prediction_basename"] = os.path.basename(pred_path)
                record["tasks"]["T1"] = task
                default = task["penalty_modes"][PENALTY_MODES[0]]
                print(
                    f"[{spec.name}] T1 {PENALTY_MODES[0]} mean={default['mean_score']:.6f}",
                    flush=True,
                )

            if spec.name in t2_paths:
                pred_path = t2_paths[spec.name]
                preds = parse_task2_answer_lines(read_text_lines(pred_path))
                answers = read_package_answers(spec.answer_path, ExamTask.T2)
                task = evaluate_t2_frozen(preds, answers)  # type: ignore[arg-type]
                task["prediction_sha256"] = sha256_file(pred_path)
                task["prediction_basename"] = os.path.basename(pred_path)
                record["tasks"]["T2"] = task
                print(f"[{spec.name}] T2 MAE={task['mae']}", flush=True)
                if task["suspicious_prediction_equals_answers"]:
                    print(
                        f"[{spec.name}] [警告] T2 预测与答案逐条完全相同；"
                        "请确认没有把官方答案当成预测。",
                        file=sys.stderr,
                        flush=True,
                    )

            if spec.name in t3_paths:
                pred_path = t3_paths[spec.name]
                preds = parse_task3_answer_lines(read_text_lines(pred_path))
                answers = read_package_answers(spec.answer_path, ExamTask.T3)
                task = evaluate_t3_frozen(preds, answers)  # type: ignore[arg-type]
                task["prediction_sha256"] = sha256_file(pred_path)
                task["prediction_basename"] = os.path.basename(pred_path)
                record["tasks"]["T3"] = task
                print(f"[{spec.name}] T3 accuracy={task['accuracy']}", flush=True)
                if task["suspicious_prediction_equals_answers"]:
                    print(
                        f"[{spec.name}] [警告] T3 预测与答案逐条完全相同；"
                        "请确认没有把官方答案当成预测。",
                        file=sys.stderr,
                        flush=True,
                    )

            if args.profile_waveforms:
                selected = (
                    samples
                    if args.profile_task == "all"
                    else [s for s in samples if s.task.value == args.profile_task]
                )
                print(f"[{spec.name}] 波形画像 {len(selected)} 文件...", flush=True)
                record["waveform_profile"] = summarize_waveform_profiles(
                    selected, workers=args.workers
                )

            output["datasets"][spec.name] = record

    output["runtime"] = {
        "elapsed_s": time.perf_counter() - overall_start,
        **overall_mem.as_dict(),
    }
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"冻结基线已写出：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
