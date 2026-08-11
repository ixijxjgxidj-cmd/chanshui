#!/usr/bin/env python3
"""Cache frozen T1 ensemble probabilities for package-isolated distillation.

The cache contains only the active-member average in canonical ``N/P/S`` order.
Member-level curves, model checkpoints, absolute package paths, and answer data are
never written.  Long records use the first five production members, matching the
frozen inference rule.  This is an experiment-only entry point and does not touch
the API, deployment configuration, or any running service.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phasepicker.defaults import (  # noqa: E402
    DEFAULT_PRETRAINED,
    DEFAULT_P_THRESHOLD,
    DEFAULT_S_THRESHOLD,
)
from phasepicker.inference.picker import PickerConfig, ProbEnsemblePicker  # noqa: E402
from phasepicker.io.mseed_reader import load_waveforms  # noqa: E402
from phasepicker.io.official_exam import scan_exam_input  # noqa: E402
from phasepicker.io.official_waveforms import read_source_bytes  # noqa: E402
from phasepicker.types import ExamSample, ExamTask, Waveform  # noqa: E402


SCHEMA_VERSION = 1
CANONICAL_LABELS = ("N", "P", "S")
EXPECTED_PACKAGES = {"round1": 1000, "round2": 915, "final08": 784}
DEFAULT_MEMBERS = (
    "guangxi",
    "jiangxi",
    "shandong",
    "weights/aug/exam_aug6_r2train_sd.pt",
    "weights/aug/crew_sp23_r2train_sd.pt",
    "weights/geofon/geofon_m1_last_sd.pt",
    "weights/geofon/geofon_m3_last_sd.pt",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"expected NAME=PATH, got {raw!r}")
        name, value = raw.split("=", 1)
        name = name.strip()
        path = Path(os.path.expanduser(value.strip())).resolve()
        if not name or not value.strip():
            raise ValueError(f"empty package name/path in {raw!r}")
        if name in parsed:
            raise ValueError(f"duplicate package name: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"package does not exist: {path.name}")
        parsed[name] = path
    return parsed


def _load_waveform(sample: ExamSample) -> Waveform:
    result = load_waveforms(read_source_bytes(sample.source_path))
    if len(result.waveforms) != 1:
        reasons = "; ".join(item.reason for item in result.warnings)
        raise RuntimeError(
            f"{sample.file_id}: expected exactly one station, got "
            f"{len(result.waveforms)} ({reasons or 'no warning'})"
        )
    return result.waveforms[0]


def canonical_probabilities(annotation_sets: Sequence) -> tuple[np.ndarray, dict]:
    """Average aligned annotation streams and return float16 N/P/S probabilities."""

    if not annotation_sets:
        raise ValueError("annotation_sets is empty")
    base = annotation_sets[0]
    by_phase: dict[str, list[np.ndarray]] = {label: [] for label in CANONICAL_LABELS}
    metadata: dict[str, tuple[float, float, int]] = {}

    for trace in base:
        phase = str(trace.stats.channel).rsplit("_", 1)[-1].upper()
        if phase not in by_phase:
            continue
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
        by_phase[phase].append(
            np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float32)
        )
        metadata[phase] = (
            float(trace.stats.starttime.timestamp),
            float(trace.stats.sampling_rate),
            len(trace.data),
        )

    missing = [phase for phase, arrays in by_phase.items() if len(arrays) != 1]
    if missing:
        raise RuntimeError(f"expected one annotation trace per phase, bad phases={missing}")
    reference = metadata[CANONICAL_LABELS[0]]
    if any(metadata[label] != reference for label in CANONICAL_LABELS[1:]):
        raise RuntimeError(f"N/P/S annotation metadata does not align: {metadata}")

    probabilities = np.stack(
        [by_phase[label][0] for label in CANONICAL_LABELS], axis=0
    ).astype(np.float32)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= np.maximum(
        probabilities.sum(axis=0, keepdims=True), np.float32(1e-7)
    )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("teacher probabilities contain non-finite values")
    return probabilities.astype(np.float16), {
        "starttime_utc": reference[0],
        "sampling_rate": reference[1],
        "samples": reference[2],
    }


def write_npy_private(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if os.name == "posix":
        path.chmod(0o600)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        action="append",
        required=True,
        help="official input package as NAME=PATH; formal run uses round1/round2/final08",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--members", nargs="+", default=list(DEFAULT_MEMBERS))
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--annotation-batch", type=int, default=32)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--long-threshold-s", type=float, default=300.0)
    parser.add_argument("--long-members", type=int, default=5)
    parser.add_argument("--limit-per-package", type=int, default=None)
    return parser


def _validate_formal_config(
    packages: Mapping[str, Path],
    members: Sequence[str],
    pretrained: str,
    overlap: float,
    long_threshold_s: float,
    long_members: int,
) -> None:
    if set(packages) != set(EXPECTED_PACKAGES):
        raise ValueError(
            f"formal cache requires packages {sorted(EXPECTED_PACKAGES)}, "
            f"got {sorted(packages)}"
        )
    if tuple(members) != DEFAULT_MEMBERS:
        raise ValueError("formal cache requires the frozen seven-member order")
    if pretrained != DEFAULT_PRETRAINED:
        raise ValueError(f"formal cache requires pretrained={DEFAULT_PRETRAINED!r}")
    if abs(float(overlap) - 0.5) > 1e-12:
        raise ValueError("formal cache requires overlap=0.5")
    if abs(float(long_threshold_s) - 300.0) > 1e-12:
        raise ValueError("formal cache requires long_threshold_s=300")
    if long_members != 5:
        raise ValueError("formal cache requires first five members on >300s records")


def _ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {path.name}; choose a new evidence directory"
        )
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        path.chmod(0o700)


def _manifest_package_metadata(packages: Mapping[str, Path]) -> dict[str, dict]:
    return {
        name: {
            "basename": path.name,
            "sha256": sha256_file(path),
            "expected_files": EXPECTED_PACKAGES[name],
        }
        for name, path in sorted(packages.items())
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    packages = parse_named_paths(args.package)
    _validate_formal_config(
        packages,
        args.members,
        args.pretrained,
        args.overlap,
        args.long_threshold_s,
        args.long_members,
    )
    if args.annotation_batch <= 0 or args.threads <= 0:
        raise SystemExit("annotation batch and threads must be positive")
    if args.limit_per_package is not None and args.limit_per_package <= 0:
        raise SystemExit("--limit-per-package must be positive")

    output_dir = Path(args.output_dir).resolve()
    _ensure_empty_output(output_dir)
    package_metadata = _manifest_package_metadata(packages)

    import torch

    torch.set_num_threads(args.threads)
    cfg = PickerConfig(
        device=args.device,
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
    model_load_seconds = time.perf_counter() - load_started

    started = time.perf_counter()
    records: list[dict] = []
    package_counts: dict[str, int] = {}
    total_probability_bytes = 0
    total_active_member_annotations = 0
    total_files = sum(
        min(EXPECTED_PACKAGES[name], args.limit_per_package)
        if args.limit_per_package is not None
        else EXPECTED_PACKAGES[name]
        for name in packages
    )
    completed = 0

    for package, package_path in sorted(packages.items()):
        samples = sorted(
            (item for item in scan_exam_input(str(package_path)) if item.task is ExamTask.T1),
            key=lambda item: item.file_id,
        )
        if args.limit_per_package is None and len(samples) != EXPECTED_PACKAGES[package]:
            raise RuntimeError(
                f"{package}: expected {EXPECTED_PACKAGES[package]} T1 files, got {len(samples)}"
            )
        if args.limit_per_package is not None:
            samples = samples[: args.limit_per_package]
        package_counts[package] = len(samples)

        for package_index, sample in enumerate(samples):
            file_started = time.perf_counter()
            waveform = _load_waveform(sample)
            stream = picker._to_stream(waveform)
            active_members = (
                args.long_members
                if waveform.duration > args.long_threshold_s
                else len(picker._members)
            )
            annotation_sets = []
            with torch.inference_mode():
                for member in picker._members[:active_members]:
                    annotation_sets.append(
                        member.annotate(
                            stream,
                            batch_size=args.annotation_batch,
                            overlap=args.overlap,
                        )
                    )
            probabilities, probability_meta = canonical_probabilities(annotation_sets)
            probability_offset = probability_meta["starttime_utc"] - waveform.starttime_utc
            relative = Path("records") / package / f"{package_index:05d}.npy"
            cache_path = output_dir / relative
            write_npy_private(cache_path, probabilities)
            cache_hash = sha256_file(cache_path)

            record = {
                "package": package,
                "file_id": sample.file_id,
                "cache_path": relative.as_posix(),
                "cache_sha256": cache_hash,
                "teacher_labels": list(CANONICAL_LABELS),
                "teacher_shape": list(probabilities.shape),
                "probability_sampling_rate": probability_meta["sampling_rate"],
                "probability_start_offset_s": probability_offset,
                "waveform_sampling_rate": float(waveform.sampling_rate),
                "waveform_samples": int(waveform.n_samples),
                "duration_s": float(waveform.duration),
                "active_members": active_members,
                "probability_bytes": int(probabilities.nbytes),
                "cache_file_bytes": int(cache_path.stat().st_size),
                "elapsed_s": time.perf_counter() - file_started,
            }
            records.append(record)
            total_probability_bytes += int(probabilities.nbytes)
            total_active_member_annotations += active_members
            completed += 1
            if completed == total_files or completed % 50 == 0:
                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 1e-9)
                eta = (total_files - completed) / max(rate, 1e-9)
                print(
                    f"[cache] {completed}/{total_files} rate={rate:.2f} files/s "
                    f"eta={eta:.1f}s",
                    flush=True,
                )
            del annotation_sets, probabilities, stream, waveform
            gc.collect()

    complete = args.limit_per_package is None and package_counts == EXPECTED_PACKAGES
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "t1_teacher_probability_cache",
        "git_head": git_head(),
        "complete": complete,
        "created_at_unix": time.time(),
        "packages": package_metadata,
        "config": {
            "members": list(args.members),
            "pretrained": args.pretrained,
            "teacher_labels": list(CANONICAL_LABELS),
            "probability_dtype": "float16",
            "long_threshold_s": args.long_threshold_s,
            "long_members": args.long_members,
            "annotation_batch": args.annotation_batch,
            "overlap": args.overlap,
            "checkpoint_saved": False,
            "member_curves_saved": False,
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "device": args.device,
        },
        "summary": {
            "files": len(records),
            "package_counts": package_counts,
            "active_member_annotations": total_active_member_annotations,
            "probability_bytes": total_probability_bytes,
            "model_load_seconds": model_load_seconds,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "records": records,
        "limitations": [
            "Historical packages are regression sets, not a blind test.",
            "Only averaged active-member probabilities are cached.",
            "The cache does not contain answers or absolute package paths.",
            "No model checkpoint is produced and production inference is unchanged.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if os.name == "posix":
        manifest_path.chmod(0o600)
    print(
        json.dumps(
            {
                "complete": complete,
                "files": len(records),
                "probability_bytes": total_probability_bytes,
                "manifest_sha256": sha256_file(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
