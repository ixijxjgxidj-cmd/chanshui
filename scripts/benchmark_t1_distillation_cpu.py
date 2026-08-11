#!/usr/bin/env python3
"""Benchmark whether T1 ensemble-to-PhaseNet distillation is practical on CPU.

This is an experiment-only entry point.  It reads official packages directly from
ZIP files, loads the frozen production teacher members, measures real annotation
throughput, and runs a short in-memory student forward/backward benchmark.  It
does not save a checkpoint and does not alter the production picker or service.

The output is a JSON evidence bundle containing only package basenames and sample
IDs.  Absolute data paths are intentionally omitted so the result can be safely
summarised in the public experiment log.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phasepicker.defaults import (  # noqa: E402
    DEFAULT_PRETRAINED,
    DEFAULT_P_THRESHOLD,
    DEFAULT_S_THRESHOLD,
)
from phasepicker.inference.picker import (  # noqa: E402
    PickerConfig,
    ProbEnsemblePicker,
)
from phasepicker.io.mseed_reader import load_waveforms  # noqa: E402
from phasepicker.io.official_exam import scan_exam_input  # noqa: E402
from phasepicker.io.official_waveforms import read_source_bytes  # noqa: E402
from phasepicker.types import ExamSample, ExamTask, Waveform  # noqa: E402

# Import the already-tested training semantics instead of duplicating PhaseNet
# probability/logit handling and the safe small-batch BN/Dropout policy.
from finetune_phasenet import phasenet_log_probs, set_safe_finetune_mode  # noqa: E402


DEFAULT_MEMBERS = (
    "guangxi",
    "jiangxi",
    "shandong",
    "weights/aug/exam_aug6_r2train_sd.pt",
    "weights/aug/crew_sp23_r2train_sd.pt",
    "weights/geofon/geofon_m1_last_sd.pt",
    "weights/geofon/geofon_m3_last_sd.pt",
)


@dataclass(frozen=True)
class InventoryRecord:
    package: str
    sample: ExamSample
    duration_s: float
    input_samples: int
    sampling_rate: float
    station_count: int


def _peak_rss_bytes() -> int:
    """Return process peak RSS, normalised to bytes on Linux/macOS."""

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return value
        return value * 1024
    except ImportError:  # Windows: used only by portable unit tests/local dry-runs
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except ImportError:
            return 0


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_sample_waveforms(sample: ExamSample) -> list[Waveform]:
    result = load_waveforms(read_source_bytes(sample.source_path))
    if not result.waveforms:
        reasons = "; ".join(w.reason for w in result.warnings) or "no waveform"
        raise RuntimeError(f"{sample.file_id}: {reasons}")
    return list(result.waveforms)


def _scan_inventory(package_paths: dict[str, str]) -> tuple[list[InventoryRecord], dict]:
    records: list[InventoryRecord] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    for package, path in package_paths.items():
        samples = [s for s in scan_exam_input(path) if s.task is ExamTask.T1]
        for sample in samples:
            try:
                waveforms = _load_sample_waveforms(sample)
            except Exception as exc:  # noqa: BLE001 - inventory must report bad files
                failures.append(
                    {"package": package, "file_id": sample.file_id, "error": repr(exc)}
                )
                continue
            duration = max(float(w.duration) for w in waveforms)
            input_samples = sum(int(w.n_samples) for w in waveforms)
            sampling_rate = float(waveforms[0].sampling_rate)
            records.append(
                InventoryRecord(
                    package=package,
                    sample=sample,
                    duration_s=duration,
                    input_samples=input_samples,
                    sampling_rate=sampling_rate,
                    station_count=len(waveforms),
                )
            )
            del waveforms
    return records, {
        "elapsed_s": time.perf_counter() - started,
        "failures": failures,
        "files": len(records),
    }


def _quantile_indices(n: int, count: int) -> list[int]:
    if n <= 0 or count <= 0:
        return []
    if count >= n:
        return list(range(n))
    # Avoid the extreme shortest/longest short records; use stable interior points.
    fractions = np.linspace(0.3, 0.7, count)
    return sorted({min(n - 1, max(0, int(round(float(q) * (n - 1))))) for q in fractions})


def select_representatives(
    inventory: Sequence[InventoryRecord],
    short_per_package: int,
    long_threshold_s: float,
    round2_long_limit: int,
    final08_long_limit: int,
) -> list[InventoryRecord]:
    selected: list[InventoryRecord] = []
    packages = sorted({record.package for record in inventory})
    for package in packages:
        group = [record for record in inventory if record.package == package]
        shorts = sorted(
            (record for record in group if record.duration_s <= long_threshold_s),
            key=lambda record: (record.duration_s, record.sample.file_id),
        )
        for index in _quantile_indices(len(shorts), short_per_package):
            selected.append(shorts[index])

        long_limit = 0
        if package == "round2":
            long_limit = round2_long_limit
        elif package == "final08":
            long_limit = final08_long_limit
        longs = sorted(
            (record for record in group if record.duration_s > long_threshold_s),
            key=lambda record: (-record.duration_s, record.sample.file_id),
        )
        selected.extend(longs[:long_limit])
    return selected


def estimate_sliding_windows(duration_s: float, window_s: float, overlap: float) -> int:
    """Conservative count of fixed windows needed for an annotate-like scan."""

    if duration_s <= window_s:
        return 1
    stride = max(window_s * (1.0 - overlap), 1e-6)
    return 1 + int(math.ceil((duration_s - window_s) / stride))


def _average_annotations(annotation_sets: Sequence) -> tuple[list[dict], int, float]:
    """Average aligned member traces and return serialisable trace arrays metadata."""

    if not annotation_sets:
        raise ValueError("annotation_sets is empty")
    started = time.perf_counter()
    base = annotation_sets[0]
    averaged: list[dict] = []
    total_bytes = 0
    for trace in base:
        arrays = [np.asarray(trace.data, dtype=np.float32)]
        for other in annotation_sets[1:]:
            matches = [
                item
                for item in other
                if item.id == trace.id
                and item.stats.starttime == trace.stats.starttime
                and len(item.data) == len(trace.data)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"annotation mismatch: {trace.id}@{trace.stats.starttime} "
                    f"matched {len(matches)} traces"
                )
            arrays.append(np.asarray(matches[0].data, dtype=np.float32))
        mean = np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float32)
        total_bytes += int(mean.nbytes)
        averaged.append(
            {
                "id": str(trace.id),
                "channel": str(trace.stats.channel),
                "starttime": float(trace.stats.starttime.timestamp),
                "sampling_rate": float(trace.stats.sampling_rate),
                "data": mean,
            }
        )
    return averaged, total_bytes, time.perf_counter() - started


def _resample_array(data: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    if abs(float(source_rate) - float(target_rate)) < 1e-9:
        return np.asarray(data, dtype=np.float32)
    from scipy.signal import resample_poly

    ratio = Fraction(float(target_rate) / float(source_rate)).limit_denominator(1000)
    return np.asarray(
        resample_poly(
            np.asarray(data, dtype=np.float32),
            ratio.numerator,
            ratio.denominator,
            axis=-1,
        ),
        dtype=np.float32,
    )


def fit_length(data: np.ndarray, length: int, pad_mode: str = "edge") -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.shape[-1] >= length:
        return np.ascontiguousarray(data[..., :length], dtype=np.float32)
    pad = length - data.shape[-1]
    if data.shape[-1] == 0:
        return np.zeros(data.shape[:-1] + (length,), dtype=np.float32)
    return np.asarray(np.pad(data, [(0, 0)] * (data.ndim - 1) + [(0, pad)], mode=pad_mode), dtype=np.float32)


def _normalise_waveform(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data = data - data.mean(axis=1, keepdims=True)
    scale = data.std(axis=1, keepdims=True) + np.float32(1e-6)
    return np.asarray(data / scale, dtype=np.float32)


def _student_example(
    waveform: Waveform,
    averaged_annotations: Sequence[dict],
    target_rate: float,
    input_length: int,
    label_order: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    x = _resample_array(np.asarray(waveform.data), waveform.sampling_rate, target_rate)
    x = _normalise_waveform(fit_length(x, input_length))

    curves: dict[str, np.ndarray] = {}
    for trace in averaged_annotations:
        phase = str(trace["channel"]).rsplit("_", 1)[-1].upper()
        if phase not in {"P", "S", "N"}:
            continue
        curve = _resample_array(
            np.asarray(trace["data"], dtype=np.float32),
            float(trace["sampling_rate"]),
            target_rate,
        )
        curves[phase] = fit_length(curve[None, :], input_length)[0]

    zeros = np.zeros(input_length, dtype=np.float32)
    p_curve = np.clip(curves.get("P", zeros), 0.0, 1.0)
    s_curve = np.clip(curves.get("S", zeros), 0.0, 1.0)
    n_curve = np.clip(curves.get("N", 1.0 - p_curve - s_curve), 0.0, 1.0)
    channels = []
    for label in label_order:
        upper = str(label).upper()
        channels.append(p_curve if upper.startswith("P") else s_curve if upper.startswith("S") else n_curve)
    y = np.vstack(channels).astype(np.float32)
    y /= np.maximum(y.sum(axis=0, keepdims=True), np.float32(1e-7))
    return x, y


def _benchmark_student(
    examples: Sequence[tuple[np.ndarray, np.ndarray]],
    pretrained: str,
    steps: int,
    warmup_steps: int,
    batch_size: int,
    learning_rate: float,
) -> dict:
    import seisbench.models as sbm
    import torch

    if not examples:
        raise ValueError("no student examples")
    model = sbm.PhaseNet.from_pretrained(pretrained).to("cpu")
    label_order = list(getattr(model, "labels", ["P", "S", "N"]))
    expected = len(label_order)
    if examples[0][1].shape[0] != expected:
        raise ValueError("teacher target channel count does not match student labels")
    set_safe_finetune_mode(model, update_bn=False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    x_all = torch.from_numpy(np.stack([item[0] for item in examples]))
    y_all = torch.from_numpy(np.stack([item[1] for item in examples]))

    measured: list[float] = []
    losses: list[float] = []
    total_steps = warmup_steps + steps
    for index in range(total_steps):
        indices = torch.arange(index * batch_size, (index + 1) * batch_size) % len(examples)
        x_batch = x_all[indices]
        y_batch = y_all[indices]
        started = time.perf_counter()
        output = model(x_batch)
        log_prob = phasenet_log_probs(output)
        loss = -(y_batch * log_prob).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        elapsed = time.perf_counter() - started
        if index >= warmup_steps:
            measured.append(elapsed)
            losses.append(float(loss.detach()))

    return {
        "pretrained": pretrained,
        "labels": label_order,
        "examples": len(examples),
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "measured_steps": steps,
        "step_seconds_mean": statistics.fmean(measured),
        "step_seconds_median": statistics.median(measured),
        "steps_per_second": 1.0 / statistics.fmean(measured),
        "examples_per_second": batch_size / statistics.fmean(measured),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _inventory_summary(inventory: Sequence[InventoryRecord], long_threshold_s: float) -> dict:
    packages: dict[str, dict] = {}
    for package in sorted({record.package for record in inventory}):
        group = [record for record in inventory if record.package == package]
        packages[package] = {
            "files": len(group),
            "stations": sum(record.station_count for record in group),
            "signal_seconds": sum(record.duration_s for record in group),
            "input_samples": sum(record.input_samples for record in group),
            "short_files": sum(record.duration_s <= long_threshold_s for record in group),
            "long_files": sum(record.duration_s > long_threshold_s for record in group),
        }
    return {
        "files": len(inventory),
        "packages": packages,
        "signal_seconds": sum(record.duration_s for record in inventory),
        "input_samples": sum(record.input_samples for record in inventory),
        "long_files": sum(record.duration_s > long_threshold_s for record in inventory),
    }


def _build_estimates(
    inventory: Sequence[InventoryRecord],
    teacher_records: Sequence[dict],
    student: dict,
    model_window_s: float,
    overlap: float,
    long_threshold_s: float,
    long_members: int,
    all_members: int,
    student_window_s: float,
    student_stride_s: float,
    student_batch: int,
    estimate_epochs: int,
) -> dict:
    unit_costs = []
    average_costs_per_sample = []
    probability_rates = []
    probability_trim_s = []
    for record in teacher_records:
        windows = estimate_sliding_windows(record["duration_s"], model_window_s, overlap)
        work_units = windows * record["active_members"]
        unit_costs.append(record["annotation_seconds"] / max(work_units, 1))
        average_costs_per_sample.append(
            record["average_seconds"] / max(record["effective_probability_samples"], 1)
        )
        probability_rate = float(record["probability_sampling_rate"])
        probability_rates.append(probability_rate)
        padded_duration = max(record["duration_s"], model_window_s + 1.0)
        probability_trim_s.append(
            padded_duration - record["effective_probability_samples"] / probability_rate
        )
    annotation_s_per_member_window = statistics.median(unit_costs)
    averaging_s_per_sample = statistics.median(average_costs_per_sample)
    probability_rate = statistics.median(probability_rates)
    annotation_trim_s = statistics.median(probability_trim_s)

    total_annotation_s = 0.0
    total_average_s = 0.0
    total_probability_samples = 0
    for record in inventory:
        active_members = long_members if record.duration_s > long_threshold_s else all_members
        windows = estimate_sliding_windows(record.duration_s, model_window_s, overlap)
        total_annotation_s += annotation_s_per_member_window * windows * active_members
        effective_duration = max(
            1.0 / probability_rate,
            max(record.duration_s, model_window_s + 1.0) - annotation_trim_s,
        )
        probability_samples = int(math.floor(effective_duration * probability_rate + 0.5))
        total_probability_samples += probability_samples * record.station_count
        total_average_s += averaging_s_per_sample * probability_samples * record.station_count
    teacher_total_s = total_annotation_s + total_average_s

    package_windows: dict[str, int] = {}
    for package in sorted({record.package for record in inventory}):
        total = 0
        for record in inventory:
            if record.package != package:
                continue
            total += estimate_sliding_windows(
                record.duration_s, student_window_s, 1.0 - student_stride_s / student_window_s
            )
        package_windows[package] = total

    lopo = {}
    packages = sorted(package_windows)
    step_seconds = float(student["step_seconds_mean"])
    for held_out in packages:
        train_windows = sum(value for key, value in package_windows.items() if key != held_out)
        steps_per_epoch = int(math.ceil(train_windows / student_batch))
        seconds = steps_per_epoch * estimate_epochs * step_seconds
        lopo[held_out] = {
            "train_windows": train_windows,
            "steps_per_epoch": steps_per_epoch,
            "epochs": estimate_epochs,
            "estimated_seconds": seconds,
            "estimated_hours": seconds / 3600.0,
        }

    cache = {
        "averaged_float16_bytes": total_probability_samples * 3 * 2,
        "averaged_float32_bytes": total_probability_samples * 3 * 4,
        # Upper bound: storing all seven members for every file. Production uses five
        # on long files, so also report the exact gated member count below.
        "all_seven_float16_upper_bound_bytes": total_probability_samples * 3 * 2 * all_members,
    }
    gated_member_samples = 0
    for record in inventory:
        effective_duration = max(
            1.0 / probability_rate,
            max(record.duration_s, model_window_s + 1.0) - annotation_trim_s,
        )
        samples = (
            int(math.floor(effective_duration * probability_rate + 0.5))
            * record.station_count
        )
        gated_member_samples += samples * (
            long_members if record.duration_s > long_threshold_s else all_members
        )
    cache["all_active_members_float16_bytes"] = gated_member_samples * 3 * 2

    max_lopo_hours = max((item["estimated_hours"] for item in lopo.values()), default=math.inf)
    teacher_hours = teacher_total_s / 3600.0
    phase_net_feasible = teacher_hours <= 12.0 and max_lopo_hours <= 24.0
    return {
        "teacher": {
            "annotation_seconds_per_member_window": annotation_s_per_member_window,
            "averaging_seconds_per_probability_sample": averaging_s_per_sample,
            "probability_sampling_rate": probability_rate,
            "annotation_trim_seconds": annotation_trim_s,
            "estimated_seconds": teacher_total_s,
            "estimated_hours": teacher_hours,
            "estimated_files_per_hour": len(inventory) / max(teacher_hours, 1e-12),
        },
        "cache": cache,
        "student_windows_by_package": package_windows,
        "leave_one_package_out": lopo,
        "decision": {
            "teacher_limit_hours": 12.0,
            "student_lopo_limit_hours": 24.0,
            "max_estimated_lopo_hours": max_lopo_hours,
            "phase_net_distillation_feasible": phase_net_feasible,
            "recommended_route": (
                "preregister_phase_net_distillation"
                if phase_net_feasible
                else "switch_to_lightweight_probability_feature_distillation"
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1", required=True, help="round-1 official ZIP")
    parser.add_argument("--round2", required=True, help="round-2 official ZIP")
    parser.add_argument("--final08", required=True, help="08 exam ZIP")
    parser.add_argument("--output", required=True, help="JSON evidence output path")
    parser.add_argument("--members", nargs="+", default=list(DEFAULT_MEMBERS))
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--annotation-batch", type=int, default=32)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--short-per-package", type=int, default=2)
    parser.add_argument("--long-threshold-s", type=float, default=300.0)
    parser.add_argument("--round2-long-limit", type=int, default=2)
    parser.add_argument("--final08-long-limit", type=int, default=1)
    parser.add_argument("--long-members", type=int, default=5)
    parser.add_argument("--student-examples", type=int, default=6)
    parser.add_argument("--student-batch", type=int, default=2)
    parser.add_argument("--student-steps", type=int, default=12)
    parser.add_argument("--student-warmup-steps", type=int, default=2)
    parser.add_argument("--student-lr", type=float, default=1e-4)
    parser.add_argument("--student-stride-s", type=float, default=30.0)
    parser.add_argument("--estimate-epochs", type=int, default=10)
    parser.add_argument("--skip-warmup", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.long_members <= 0 or args.long_members > len(args.members):
        raise SystemExit("--long-members must be in [1, len(--members)]")
    if args.student_batch <= 0 or args.student_steps <= 0:
        raise SystemExit("student batch/steps must be positive")

    import torch

    torch.set_num_threads(int(args.threads))
    package_paths = {
        "round1": args.round1,
        "round2": args.round2,
        "final08": args.final08,
    }
    inventory, inventory_runtime = _scan_inventory(package_paths)
    print(
        f"[inventory] files={len(inventory)} "
        f"elapsed={inventory_runtime['elapsed_s']:.2f}s",
        flush=True,
    )
    expected = {"round1": 1000, "round2": 915, "final08": 784}
    actual = {
        package: sum(record.package == package for record in inventory)
        for package in expected
    }
    if actual != expected or inventory_runtime["failures"]:
        raise SystemExit(
            f"T1 inventory coverage failed: expected={expected}, actual={actual}, "
            f"failures={len(inventory_runtime['failures'])}"
        )

    selected = select_representatives(
        inventory,
        short_per_package=args.short_per_package,
        long_threshold_s=args.long_threshold_s,
        round2_long_limit=args.round2_long_limit,
        final08_long_limit=args.final08_long_limit,
    )
    cfg = PickerConfig(
        device="cpu",
        pretrained=args.pretrained,
        p_threshold=DEFAULT_P_THRESHOLD,
        s_threshold=DEFAULT_S_THRESHOLD,
        num_threads=args.threads,
        batch_size=args.annotation_batch,
        overlap=args.overlap,
        ensemble_long_top_n=args.long_members,
        ensemble_long_max_duration_s=args.long_threshold_s,
    )
    load_started = time.perf_counter()
    picker = ProbEnsemblePicker.from_member_names(list(args.members), cfg)
    model_load_s = time.perf_counter() - load_started
    model_window_s = float(picker._model.in_samples) / float(picker._model.sampling_rate)
    print(
        f"[teacher] loaded_members={len(picker._members)} "
        f"elapsed={model_load_s:.2f}s",
        flush=True,
    )

    if not args.skip_warmup:
        warm = next(record for record in selected if record.duration_s <= args.long_threshold_s)
        warm_waveform = _load_sample_waveforms(warm.sample)[0]
        stream = picker._to_stream(warm_waveform)
        with torch.inference_mode():
            for member in picker._members:
                member.annotate(stream, batch_size=args.annotation_batch, overlap=args.overlap)
        del stream, warm_waveform
        gc.collect()

    teacher_records: list[dict] = []
    student_sources: list[tuple[Waveform, list[dict]]] = []
    for selected_record in selected:
        print(
            f"[teacher] start package={selected_record.package} "
            f"file={selected_record.sample.file_id} "
            f"duration={selected_record.duration_s:.2f}s",
            flush=True,
        )
        waveforms = _load_sample_waveforms(selected_record.sample)
        if len(waveforms) != 1:
            raise RuntimeError(
                f"representative {selected_record.sample.file_id} has "
                f"{len(waveforms)} stations; benchmark expects one"
            )
        waveform = waveforms[0]
        stream = picker._to_stream(waveform)
        active_members = (
            args.long_members
            if selected_record.duration_s > args.long_threshold_s
            else len(picker._members)
        )
        annotations = []
        timings = []
        with torch.inference_mode():
            for member in picker._members[:active_members]:
                started = time.perf_counter()
                annotation = member.annotate(
                    stream,
                    batch_size=args.annotation_batch,
                    overlap=args.overlap,
                )
                timings.append(time.perf_counter() - started)
                annotations.append(annotation)
        averaged, cache_bytes, average_s = _average_annotations(annotations)
        probability_samples = sum(len(item["data"]) for item in averaged) // max(len(averaged), 1)
        probability_sampling_rate = statistics.median(
            float(item["sampling_rate"]) for item in averaged
        )
        teacher_records.append(
            {
                "package": selected_record.package,
                "file_id": selected_record.sample.file_id,
                "duration_s": selected_record.duration_s,
                "active_members": active_members,
                "single_member_seconds": timings,
                "single_member_seconds_mean": statistics.fmean(timings),
                "annotation_seconds": sum(timings),
                "average_seconds": average_s,
                "teacher_seconds": sum(timings) + average_s,
                "files_per_hour": 3600.0 / (sum(timings) + average_s),
                "effective_probability_samples": probability_samples,
                "probability_sampling_rate": probability_sampling_rate,
                "averaged_cache_float32_bytes": cache_bytes,
                "averaged_cache_float16_bytes": cache_bytes // 2,
                "peak_rss_bytes": _peak_rss_bytes(),
            }
        )
        print(
            f"[teacher] done package={selected_record.package} "
            f"file={selected_record.sample.file_id} "
            f"members={active_members} elapsed={sum(timings) + average_s:.2f}s",
            flush=True,
        )
        if (
            selected_record.duration_s <= args.long_threshold_s
            and len(student_sources) < args.student_examples
        ):
            student_sources.append((waveform, averaged))
        del annotations, stream
        if not student_sources or student_sources[-1][0] is not waveform:
            del waveform
        gc.collect()

    import seisbench.models as sbm

    template_student = sbm.PhaseNet.from_pretrained(args.pretrained)
    student_rate = float(template_student.sampling_rate)
    student_length = int(template_student.in_samples)
    student_labels = list(getattr(template_student, "labels", ["P", "S", "N"]))
    del template_student
    examples = [
        _student_example(
            waveform,
            averaged,
            target_rate=student_rate,
            input_length=student_length,
            label_order=student_labels,
        )
        for waveform, averaged in student_sources
    ]
    del student_sources
    gc.collect()
    student = _benchmark_student(
        examples,
        pretrained=args.pretrained,
        steps=args.student_steps,
        warmup_steps=args.student_warmup_steps,
        batch_size=args.student_batch,
        learning_rate=args.student_lr,
    )
    print(
        f"[student] steps={student['measured_steps']} "
        f"mean_step={student['step_seconds_mean']:.4f}s",
        flush=True,
    )

    estimates = _build_estimates(
        inventory=inventory,
        teacher_records=teacher_records,
        student=student,
        model_window_s=model_window_s,
        overlap=args.overlap,
        long_threshold_s=args.long_threshold_s,
        long_members=args.long_members,
        all_members=len(args.members),
        student_window_s=student_length / student_rate,
        student_stride_s=args.student_stride_s,
        student_batch=args.student_batch,
        estimate_epochs=args.estimate_epochs,
    )

    output = {
        "schema_version": 1,
        "experiment": "t1_cpu_distillation_feasibility",
        "git_head": _git_head(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        },
        "packages": {name: os.path.basename(path) for name, path in package_paths.items()},
        "config": {
            "members": list(args.members),
            "long_members": args.long_members,
            "long_threshold_s": args.long_threshold_s,
            "annotation_batch": args.annotation_batch,
            "overlap": args.overlap,
            "student_pretrained": args.pretrained,
            "student_steps": args.student_steps,
            "student_batch": args.student_batch,
            "estimate_epochs": args.estimate_epochs,
            "checkpoint_saved": False,
        },
        "inventory": _inventory_summary(inventory, args.long_threshold_s),
        "inventory_runtime": inventory_runtime,
        "selected": [
            {
                "package": record.package,
                "file_id": record.sample.file_id,
                "duration_s": record.duration_s,
            }
            for record in selected
        ],
        "model_load_seconds": model_load_s,
        "teacher": teacher_records,
        "student": student,
        "estimates": estimates,
        "peak_rss_bytes": _peak_rss_bytes(),
        "limitations": [
            "CPU-only benchmark; no GPU throughput is inferred.",
            "Historical packages are not a blind test and must remain package-isolated.",
            "Runtime projection uses measured representative annotation-window cost.",
            "No checkpoint is saved and production inference is unchanged.",
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if os.name == "posix":
        output_path.chmod(0o600)
    print(json.dumps(output["estimates"]["decision"], indent=2, ensure_ascii=False))
    print(f"evidence: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
