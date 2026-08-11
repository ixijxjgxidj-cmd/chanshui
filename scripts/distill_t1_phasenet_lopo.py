#!/usr/bin/env python3
"""Train one preregistered T1 PhaseNet distillation fold.

The held-out package is a hard barrier: this process requires exactly the other
two waveform packages and, for ``kd-hard``, exactly the other two answer packages.
It never opens held-out waveforms or answers.  Evaluation is deliberately a later
process using ``run_official_task1.py`` so training cannot inspect held-out scores.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cache_t1_teacher_probabilities import (  # noqa: E402
    CANONICAL_LABELS,
    DEFAULT_MEMBERS,
    EXPECTED_PACKAGES,
    parse_named_paths,
    sha256_file,
)
from finetune_phasenet import phasenet_log_probs, set_safe_finetune_mode  # noqa: E402
from phasepicker.io.mseed_reader import load_waveforms  # noqa: E402
from phasepicker.io.official_exam import scan_exam_input  # noqa: E402
from phasepicker.io.official_waveforms import (  # noqa: E402
    read_package_answers,
    read_source_bytes,
)
from phasepicker.types import ExamSample, ExamTask, Task1Result, Waveform  # noqa: E402


SCHEMA_VERSION = 1
PRETRAINED = "diting"
SAMPLING_RATE = 50.0
INPUT_SAMPLES = 3001
BLIND_LEFT = 250
BLIND_RIGHT = 250
VALID_SAMPLES = INPUT_SAMPLES - BLIND_LEFT - BLIND_RIGHT
STRIDE_SAMPLES = 1500
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
HARD_WEIGHT = 0.3
KD_WEIGHT = 0.7
P_SIGMA_S = 0.2
S_SIGMA_S = 0.3
DEFAULT_SEED = 20260812
EXPECTED_TRAIN_WINDOWS = {"round1": 4698, "round2": 4221, "final08": 3939}


@dataclass(frozen=True)
class WindowSpec:
    record_index: int
    start_sample: int


@dataclass
class PreparedRecord:
    package: str
    file_id: str
    waveform: np.ndarray
    teacher: np.ndarray
    teacher_start_offset_s: float
    p_times_s: tuple[float, ...]
    s_times_s: tuple[float, ...]


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def peak_rss_bytes() -> int:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except ImportError:
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except ImportError:
            return 0


def resample_array(data: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if abs(float(source_rate) - float(target_rate)) < 1e-9:
        return np.ascontiguousarray(data, dtype=np.float32)
    from scipy.signal import resample_poly

    ratio = Fraction(float(target_rate) / float(source_rate)).limit_denominator(1000)
    return np.ascontiguousarray(
        resample_poly(
            data, ratio.numerator, ratio.denominator, axis=-1
        ).astype(np.float32),
        dtype=np.float32,
    )


def normalise_window(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data = data - data.mean(axis=1, keepdims=True)
    scale = data.std(axis=1, keepdims=True) + np.float32(1e-6)
    return np.ascontiguousarray(data / scale, dtype=np.float32)


def slice_or_edge_pad(data: np.ndarray, start: int, length: int) -> np.ndarray:
    if start < 0 or length <= 0:
        raise ValueError("start must be non-negative and length positive")
    data = np.asarray(data, dtype=np.float32)
    sliced = data[..., start : start + length]
    if sliced.shape[-1] == length:
        return np.ascontiguousarray(sliced, dtype=np.float32)
    if sliced.shape[-1] == 0:
        return np.zeros(data.shape[:-1] + (length,), dtype=np.float32)
    pad = length - sliced.shape[-1]
    return np.ascontiguousarray(
        np.pad(sliced, [(0, 0)] * (data.ndim - 1) + [(0, pad)], mode="edge"),
        dtype=np.float32,
    )


def window_starts(max_start: int, stride: int = STRIDE_SAMPLES) -> list[int]:
    if max_start < 0 or stride <= 0:
        raise ValueError("max_start must be non-negative and stride positive")
    starts = list(range(0, max_start + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def teacher_slice_start(
    window_start: int,
    teacher_start_offset_s: float,
    sampling_rate: float = SAMPLING_RATE,
) -> int:
    offset_samples = int(round(float(teacher_start_offset_s) * sampling_rate))
    return int(window_start + BLIND_LEFT - offset_samples)


def maximum_window_start(
    waveform_samples: int,
    teacher_samples: int,
    teacher_start_offset_s: float,
) -> int:
    if waveform_samples <= 0 or teacher_samples < VALID_SAMPLES:
        raise ValueError("waveform/teacher arrays are too short")
    offset_samples = int(round(float(teacher_start_offset_s) * SAMPLING_RATE))
    max_by_waveform = max(0, waveform_samples - INPUT_SAMPLES)
    max_by_teacher = teacher_samples - VALID_SAMPLES - BLIND_LEFT + offset_samples
    max_start = min(max_by_waveform, max_by_teacher)
    if max_start < 0:
        raise ValueError(
            "teacher curve cannot cover the model-defined central prediction region"
        )
    if teacher_slice_start(0, teacher_start_offset_s) < 0:
        raise ValueError("teacher curve begins after the first central prediction sample")
    return int(max_start)


def _phase_curve(
    length: int,
    relative_times_s: Sequence[float],
    target_start_s: float,
    sigma_s: float,
) -> np.ndarray:
    if not relative_times_s:
        return np.zeros(length, dtype=np.float32)
    sample_axis = np.arange(length, dtype=np.float32)
    sigma_samples = np.float32(sigma_s * SAMPLING_RATE)
    curves = []
    for relative_time in relative_times_s:
        center = np.float32((float(relative_time) - target_start_s) * SAMPLING_RATE)
        curves.append(np.exp(-0.5 * ((sample_axis - center) / sigma_samples) ** 2))
    return np.maximum.reduce(curves).astype(np.float32)


def make_hard_target(
    p_times_s: Sequence[float],
    s_times_s: Sequence[float],
    target_start_s: float,
    length: int = VALID_SAMPLES,
) -> np.ndarray:
    p_curve = _phase_curve(length, p_times_s, target_start_s, P_SIGMA_S)
    s_curve = _phase_curve(length, s_times_s, target_start_s, S_SIGMA_S)
    n_curve = np.clip(1.0 - p_curve - s_curve, 0.0, 1.0).astype(np.float32)
    target = np.stack([n_curve, p_curve, s_curve], axis=0).astype(np.float32)
    target /= np.maximum(target.sum(axis=0, keepdims=True), np.float32(1e-7))
    return target


def reorder_channels(array: np.ndarray, source_labels: Sequence[str], target_labels: Sequence[str]) -> np.ndarray:
    source = [str(label).upper() for label in source_labels]
    indices = []
    for label in target_labels:
        upper = str(label).upper()
        if upper not in source:
            raise ValueError(f"target label {upper!r} not present in {source}")
        indices.append(source.index(upper))
    return np.asarray(array, dtype=np.float32)[indices]


def _load_waveform(sample: ExamSample) -> Waveform:
    result = load_waveforms(read_source_bytes(sample.source_path))
    if len(result.waveforms) != 1:
        raise RuntimeError(
            f"{sample.file_id}: expected one station, got {len(result.waveforms)}"
        )
    return result.waveforms[0]


def _read_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported teacher cache schema")
    if not manifest.get("complete"):
        raise ValueError("teacher cache is incomplete")
    if set(manifest.get("packages", {})) != set(EXPECTED_PACKAGES):
        raise ValueError("teacher cache does not cover the three formal packages")
    config = manifest.get("config", {})
    if tuple(config.get("teacher_labels", [])) != CANONICAL_LABELS:
        raise ValueError("teacher cache labels are not canonical N/P/S")
    if config.get("probability_dtype") != "float16":
        raise ValueError("teacher cache must use float16 probabilities")
    if tuple(config.get("members", [])) != DEFAULT_MEMBERS:
        raise ValueError("teacher cache member order differs from frozen production")
    if config.get("pretrained") != PRETRAINED:
        raise ValueError("teacher cache pretrained base differs from preregistration")
    if abs(float(config.get("overlap", -1.0)) - 0.5) > 1e-12:
        raise ValueError("teacher cache overlap differs from preregistration")
    if abs(float(config.get("long_threshold_s", -1.0)) - 300.0) > 1e-12:
        raise ValueError("teacher cache long threshold differs from preregistration")
    if int(config.get("long_members", -1)) != 5:
        raise ValueError("teacher cache long-member gate differs from preregistration")
    if config.get("member_curves_saved") is not False:
        raise ValueError("formal cache must not contain member-level curves")
    return manifest


def _validate_package_barrier(
    held_out: str,
    packages: Mapping[str, Path],
    answers: Mapping[str, Path],
    loss_name: str,
) -> tuple[str, str]:
    if held_out not in EXPECTED_PACKAGES:
        raise ValueError(f"unknown held-out package: {held_out}")
    training = sorted(set(EXPECTED_PACKAGES) - {held_out})
    if sorted(packages) != training:
        raise ValueError(
            f"held-out barrier requires exactly training packages {training}, "
            f"got {sorted(packages)}"
        )
    if loss_name == "kd-hard" and sorted(answers) != training:
        raise ValueError(
            f"kd-hard requires exactly training answer packages {training}, "
            f"got {sorted(answers)}"
        )
    if loss_name == "kd-only" and answers:
        raise ValueError("kd-only must not receive answer packages")
    return training[0], training[1]


def _ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {path.name}; choose a new fold directory"
        )
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        path.chmod(0o700)


def _verify_source_packages(
    packages: Mapping[str, Path], manifest: Mapping[str, object]
) -> dict[str, dict]:
    output: dict[str, dict] = {}
    manifest_packages = manifest["packages"]
    assert isinstance(manifest_packages, Mapping)
    for name, path in sorted(packages.items()):
        expected = manifest_packages[name]
        actual_hash = sha256_file(path)
        if path.name != expected["basename"] or actual_hash != expected["sha256"]:
            raise ValueError(f"{name}: package basename/hash does not match teacher cache")
        output[name] = {"basename": path.name, "sha256": actual_hash}
    return output


def _scan_training_samples(packages: Mapping[str, Path]) -> dict[str, dict[str, ExamSample]]:
    result: dict[str, dict[str, ExamSample]] = {}
    for name, path in sorted(packages.items()):
        samples = {
            item.file_id: item
            for item in scan_exam_input(str(path))
            if item.task is ExamTask.T1
        }
        if len(samples) != EXPECTED_PACKAGES[name]:
            raise RuntimeError(
                f"{name}: expected {EXPECTED_PACKAGES[name]} T1 samples, got {len(samples)}"
            )
        result[name] = samples
    return result


def _load_training_answers(
    answer_paths: Mapping[str, Path], loss_name: str
) -> tuple[dict[str, dict[str, Task1Result]], dict[str, dict]]:
    if loss_name == "kd-only":
        return {}, {}
    answers: dict[str, dict[str, Task1Result]] = {}
    metadata: dict[str, dict] = {}
    for name, path in sorted(answer_paths.items()):
        parsed = read_package_answers(str(path), ExamTask.T1)
        answers[name] = dict(parsed)  # type: ignore[arg-type]
        metadata[name] = {"basename": path.name, "sha256": sha256_file(path)}
    return answers, metadata


def _prepare_records(
    cache_dir: Path,
    manifest: Mapping[str, object],
    training_packages: Sequence[str],
    samples: Mapping[str, Mapping[str, ExamSample]],
    answers: Mapping[str, Mapping[str, Task1Result]],
    loss_name: str,
) -> tuple[list[PreparedRecord], list[WindowSpec], dict[str, int]]:
    records: list[PreparedRecord] = []
    specs: list[WindowSpec] = []
    windows_by_package = {name: 0 for name in training_packages}

    manifest_records = [
        item for item in manifest["records"] if item["package"] in training_packages
    ]
    expected_records = sum(EXPECTED_PACKAGES[name] for name in training_packages)
    if len(manifest_records) != expected_records:
        raise ValueError(
            f"teacher cache training subset has {len(manifest_records)} records, "
            f"expected {expected_records}"
        )

    for item in manifest_records:
        package = item["package"]
        file_id = item["file_id"]
        sample = samples[package].get(file_id)
        if sample is None:
            raise ValueError(f"{package}/{file_id}: waveform missing from source package")
        cache_path = cache_dir / item["cache_path"]
        if not cache_path.is_file() or sha256_file(cache_path) != item["cache_sha256"]:
            raise ValueError(f"{package}/{file_id}: teacher cache file hash mismatch")
        teacher = np.load(cache_path, allow_pickle=False)
        if teacher.dtype != np.float16 or list(teacher.shape) != item["teacher_shape"]:
            raise ValueError(f"{package}/{file_id}: teacher cache dtype/shape mismatch")
        teacher = reorder_channels(
            teacher.astype(np.float32), item["teacher_labels"], CANONICAL_LABELS
        )
        teacher = np.clip(teacher, 0.0, 1.0)
        teacher /= np.maximum(
            teacher.sum(axis=0, keepdims=True), np.float32(1e-7)
        )
        if not np.isfinite(teacher).all():
            raise ValueError(f"{package}/{file_id}: non-finite teacher probabilities")

        waveform = _load_waveform(sample)
        if (
            int(waveform.n_samples) != int(item["waveform_samples"])
            or abs(float(waveform.sampling_rate) - float(item["waveform_sampling_rate"]))
            > 1e-9
        ):
            raise ValueError(f"{package}/{file_id}: waveform metadata differs from cache")
        if abs(float(item["probability_sampling_rate"]) - SAMPLING_RATE) > 1e-9:
            raise ValueError(f"{package}/{file_id}: teacher probability rate is not 50 Hz")
        expected_offset = BLIND_LEFT / SAMPLING_RATE
        if abs(float(item["probability_start_offset_s"]) - expected_offset) > 1e-6:
            raise ValueError(
                f"{package}/{file_id}: teacher start offset does not match PhaseNet blinding"
            )
        wave = resample_array(waveform.data, waveform.sampling_rate, SAMPLING_RATE)
        truth = answers.get(package, {}).get(file_id)
        if loss_name == "kd-hard" and truth is None:
            raise ValueError(f"{package}/{file_id}: hard-label answer missing")
        p_times = tuple(float(value) for value in (truth.p_times_s if truth else ()))
        s_times = tuple(float(value) for value in (truth.s_times_s if truth else ()))

        prepared = PreparedRecord(
            package=package,
            file_id=file_id,
            waveform=wave,
            teacher=teacher,
            teacher_start_offset_s=float(item["probability_start_offset_s"]),
            p_times_s=p_times,
            s_times_s=s_times,
        )
        record_index = len(records)
        max_start = maximum_window_start(
            wave.shape[-1], teacher.shape[-1], prepared.teacher_start_offset_s
        )
        starts = window_starts(max_start)
        for start in starts:
            cache_start = teacher_slice_start(start, prepared.teacher_start_offset_s)
            if cache_start < 0 or cache_start + VALID_SAMPLES > teacher.shape[-1]:
                raise ValueError(f"{package}/{file_id}: window-to-teacher alignment overflow")
            specs.append(WindowSpec(record_index=record_index, start_sample=start))
        windows_by_package[package] += len(starts)
        records.append(prepared)

    return records, specs, windows_by_package


def _batch_arrays(
    records: Sequence[PreparedRecord],
    batch_specs: Sequence[WindowSpec],
    model_labels: Sequence[str],
    include_hard: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x_batch = []
    teacher_batch = []
    hard_batch = []
    for spec in batch_specs:
        record = records[spec.record_index]
        x = slice_or_edge_pad(record.waveform, spec.start_sample, INPUT_SAMPLES)
        x_batch.append(normalise_window(x))

        target_start = teacher_slice_start(
            spec.start_sample, record.teacher_start_offset_s
        )
        teacher = record.teacher[:, target_start : target_start + VALID_SAMPLES]
        teacher_batch.append(
            reorder_channels(teacher, CANONICAL_LABELS, model_labels)
        )
        if include_hard:
            target_start_s = (spec.start_sample + BLIND_LEFT) / SAMPLING_RATE
            hard = make_hard_target(
                record.p_times_s, record.s_times_s, target_start_s
            )
            hard_batch.append(reorder_channels(hard, CANONICAL_LABELS, model_labels))
    return (
        np.stack(x_batch).astype(np.float32),
        np.stack(teacher_batch).astype(np.float32),
        np.stack(hard_batch).astype(np.float32) if include_hard else None,
    )


def _save_checkpoint_private(path: Path, payload: dict) -> None:
    import torch

    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    if os.name == "posix":
        path.chmod(0o600)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--held-out", required=True, choices=sorted(EXPECTED_PACKAGES))
    parser.add_argument("--loss", required=True, choices=["kd-only", "kd-hard"])
    parser.add_argument(
        "--package",
        action="append",
        required=True,
        help="training waveform package NAME=PATH; held-out package is forbidden",
    )
    parser.add_argument(
        "--answer-package",
        action="append",
        default=[],
        help="training answer package NAME=PATH; required only for kd-hard",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package_paths = parse_named_paths(args.package)
    answer_paths = parse_named_paths(args.answer_package) if args.answer_package else {}
    training_packages = _validate_package_barrier(
        args.held_out, package_paths, answer_paths, args.loss
    )
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")

    output_dir = Path(args.output_dir).resolve()
    _ensure_empty_output(output_dir)
    manifest_path = Path(args.cache_manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cache manifest does not exist: {manifest_path.name}")
    manifest = _read_manifest(manifest_path)
    current_head = git_head()
    if current_head is not None and manifest.get("git_head") != current_head:
        raise ValueError(
            "teacher cache git_head differs from current training code; regenerate cache"
        )
    manifest_hash = sha256_file(manifest_path)
    cache_dir = manifest_path.parent

    source_metadata = _verify_source_packages(package_paths, manifest)
    sample_maps = _scan_training_samples(package_paths)
    answer_maps, answer_metadata = _load_training_answers(answer_paths, args.loss)
    prepare_started = time.perf_counter()
    records, specs, windows_by_package = _prepare_records(
        cache_dir,
        manifest,
        training_packages,
        sample_maps,
        answer_maps,
        args.loss,
    )
    if len(specs) != EXPECTED_TRAIN_WINDOWS[args.held_out]:
        raise ValueError(
            f"held-out {args.held_out}: got {len(specs)} windows, "
            f"expected preregistered {EXPECTED_TRAIN_WINDOWS[args.held_out]}"
        )
    prepare_seconds = time.perf_counter() - prepare_started

    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    import seisbench.models as sbm

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = sbm.PhaseNet.from_pretrained(PRETRAINED).to(device)
    if int(model.in_samples) != INPUT_SAMPLES or float(model.sampling_rate) != SAMPLING_RATE:
        raise ValueError("student PhaseNet geometry differs from preregistration")
    model_labels = [str(label).upper() for label in list(model.labels)]
    if set(model_labels) != set(CANONICAL_LABELS):
        raise ValueError(f"unexpected student labels: {model_labels}")
    default_blinding = tuple(int(value) for value in model.default_args.get("blinding", ()))
    if default_blinding != (BLIND_LEFT, BLIND_RIGHT):
        raise ValueError(
            f"student blinding {default_blinding} differs from preregistered "
            f"{(BLIND_LEFT, BLIND_RIGHT)}"
        )
    set_safe_finetune_mode(model, update_bn=False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    include_hard = args.loss == "kd-hard"
    epoch_metrics: list[dict] = []
    train_started = time.perf_counter()
    for epoch in range(EPOCHS):
        set_safe_finetune_mode(model, update_bn=False)
        order = np.random.RandomState(args.seed + epoch).permutation(len(specs))
        total_loss = 0.0
        total_kd = 0.0
        total_hard = 0.0
        steps = 0
        examples = 0
        epoch_started = time.perf_counter()
        for offset in range(0, len(order), BATCH_SIZE):
            selected = [specs[int(index)] for index in order[offset : offset + BATCH_SIZE]]
            x_np, teacher_np, hard_np = _batch_arrays(
                records, selected, model_labels, include_hard
            )
            x = torch.from_numpy(x_np).to(device)
            teacher = torch.from_numpy(teacher_np).to(device)
            output = model(x)
            log_prob = phasenet_log_probs(output)
            central = log_prob[:, :, BLIND_LEFT : INPUT_SAMPLES - BLIND_RIGHT]
            if central.shape != teacher.shape:
                raise RuntimeError(
                    f"student/teacher central shapes differ: {central.shape} vs {teacher.shape}"
                )
            kd_loss = -(teacher * central).sum(dim=1).mean()
            if include_hard:
                assert hard_np is not None
                hard = torch.from_numpy(hard_np).to(device)
                hard_loss = -(hard * central).sum(dim=1).mean()
                loss = KD_WEIGHT * kd_loss + HARD_WEIGHT * hard_loss
            else:
                hard_loss = torch.zeros((), device=device)
                loss = kd_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite gradient norm")
            optimizer.step()

            batch_examples = len(selected)
            total_loss += float(loss.detach()) * batch_examples
            total_kd += float(kd_loss.detach()) * batch_examples
            total_hard += float(hard_loss.detach()) * batch_examples
            steps += 1
            examples += batch_examples

        metrics = {
            "epoch": epoch + 1,
            "examples": examples,
            "steps": steps,
            "loss": total_loss / examples,
            "kd_loss": total_kd / examples,
            "hard_loss": total_hard / examples if include_hard else None,
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "peak_rss_bytes": peak_rss_bytes(),
        }
        epoch_metrics.append(metrics)
        print(
            f"[train] epoch={epoch + 1}/{EPOCHS} loss={metrics['loss']:.6f} "
            f"kd={metrics['kd_loss']:.6f} elapsed={metrics['elapsed_seconds']:.2f}s",
            flush=True,
        )

    training_seconds = time.perf_counter() - train_started
    sanitized_config = {
        "held_out": args.held_out,
        "training_packages": list(training_packages),
        "loss": args.loss,
        "student": f"PhaseNet({PRETRAINED})",
        "sampling_rate": SAMPLING_RATE,
        "input_samples": INPUT_SAMPLES,
        "blinding": [BLIND_LEFT, BLIND_RIGHT],
        "stride_samples": STRIDE_SAMPLES,
        "stride_seconds": STRIDE_SAMPLES / SAMPLING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "kd_weight": 1.0 if args.loss == "kd-only" else KD_WEIGHT,
        "hard_weight": 0.0 if args.loss == "kd-only" else HARD_WEIGHT,
        "p_sigma_s": P_SIGMA_S if include_hard else None,
        "s_sigma_s": S_SIGMA_S if include_hard else None,
        "batchnorm": "frozen_eval",
        "dropout": "eval",
        "seed": args.seed,
    }
    checkpoint_path = output_dir / "student.pt"
    _save_checkpoint_private(
        checkpoint_path,
        {
            "model": model.state_dict(),
            "epoch": EPOCHS,
            "loss": epoch_metrics[-1]["loss"],
            "config": sanitized_config,
            "git_head": current_head,
            "teacher_manifest_sha256": manifest_hash,
            "epoch_metrics": epoch_metrics,
        },
    )
    checkpoint_hash = sha256_file(checkpoint_path)

    evidence = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "t1_phasenet_lopo_distillation",
        "git_head": current_head,
        "teacher_manifest": {
            "basename": manifest_path.name,
            "sha256": manifest_hash,
            "git_head": manifest.get("git_head"),
        },
        "source_packages": source_metadata,
        "answer_packages": answer_metadata,
        "config": sanitized_config,
        "data": {
            "records": len(records),
            "windows": len(specs),
            "windows_by_package": windows_by_package,
            "held_out_opened": False,
            "held_out_answer_opened": False,
        },
        "runtime": {
            "prepare_seconds": prepare_seconds,
            "training_seconds": training_seconds,
            "peak_rss_bytes": peak_rss_bytes(),
            "device": str(device),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "epochs": epoch_metrics,
        "checkpoint": {
            "basename": checkpoint_path.name,
            "sha256": checkpoint_hash,
            "bytes": checkpoint_path.stat().st_size,
            "mode": oct(checkpoint_path.stat().st_mode & 0o777),
        },
        "limitations": [
            "This process does not open held-out waveforms or answers.",
            "Historical packages are regression sets, not a blind test.",
            "The checkpoint is experimental and is not deployed by this script.",
            "Held-out inference and four-mode scoring must run in a later process.",
        ],
    }
    evidence_path = output_dir / "training.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if os.name == "posix":
        evidence_path.chmod(0o600)
    print(
        json.dumps(
            {
                "held_out": args.held_out,
                "loss": args.loss,
                "windows": len(specs),
                "final_loss": epoch_metrics[-1]["loss"],
                "checkpoint_sha256": checkpoint_hash,
                "evidence_sha256": sha256_file(evidence_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    del records, specs, model, optimizer
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
