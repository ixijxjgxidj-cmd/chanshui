#!/usr/bin/env python3
"""Round 05: falsify a final-list mask for zero-filled waveform gaps.

This is an isolated research script.  It deliberately does not modify the
production picker, API, defaults, or deployment configuration.  The active
candidate is a deterministic stable-subsequence filter applied to final
``Pick`` objects.  Model inference is used only to determine whether that
small mechanism is safe under the preregistered synthetic gap perturbations.

Tracked output must never contain local archive paths.  CLI paths are reduced
to basename + SHA-256 before being written to JSON.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phasepicker.inference.picker import PickerConfig, ProbEnsemblePicker  # noqa: E402
from phasepicker.io.mseed_reader import load_waveforms  # noqa: E402
from phasepicker.io.official_exam import scan_exam_input  # noqa: E402
from phasepicker.io.official_waveforms import read_source_bytes  # noqa: E402
from phasepicker.types import CHANNEL_ORDER, Pick, PhaseType, Waveform  # noqa: E402


SEED = 20260811
ACTIVE_MARGINS_S = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
REMOTE_DISTANCE_S = 10.0
PICK_TOLERANCE_S = {PhaseType.P: 0.10, PhaseType.S: 0.20}
PERFORMANCE_MAX_P95_MS = 5.0
PERFORMANCE_PICKS = 10_000
PERFORMANCE_GAPS = 100
PERFORMANCE_RUNS = 500
LONG_CROP_S = 360.0
LONG_CONTEXT_S = 180.0

EXPECTED_PACKAGES = {
    "round1": {
        "sha256": "beb93b7544718c3b05be9cd5f4f3cbf78f7be8a32c125a696adeb87c5d3a524e",
        "label": "R1",
    },
    "round2": {
        "sha256": "d5ffc69223ab75815618e7647e4212d39e4c0e756a91c1e6ee04cb55c29f54e6",
        "label": "R2",
    },
    "final08": {
        "sha256": "560145f74f2d8861bb344a7396c15793a54f0ce3f4ad056dfa115cd7f756bb3b",
        "label": "08",
    },
}

DEVELOPMENT_FILES = {
    "round1": (
        "T1.A.Q0001.mseed",
        "T1.B.Q0001.mseed",
        "T1.C.Q0001.mseed",
        "T1.D.Q0001.mseed",
    ),
    "round2": (
        "T1.A.Q0001.mseed",
        "T1.A.Q0501.mseed",
        "T1.B.Q0326.mseed",
        "T1.C.Q0021.mseed",
        "T1.D.Q0051.mseed",
    ),
}

HOLDOUT_FILES = (
    "T1.A.Q0001.mseed",
    "T1.A.Q0943.mseed",
    "T1.B.Q0501.mseed",
    "T1.C.Q0501.mseed",
    "T1.D.Q0131.mseed",
)

LONG_FILE_IDS = {
    ("round2", "T1.A.Q0001.mseed"),
    ("final08", "T1.A.Q0001.mseed"),
}

SENTINEL_KEYS = {
    "R1/T1.A.Q0001.mseed",
    "R2/T1.A.Q0501.mseed",
    "R2/T1.A.Q0001.mseed@360s",
    "noise-short",
    "noise-long",
}


@dataclass(frozen=True)
class PreparedSample:
    key: str
    package_key: str
    file_id: str
    waveform: Waveform
    source_kind: str
    crop: Mapping[str, object] | None = None
    ingest_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class InjectionResult:
    waveform: Waveform
    variant_id: str
    components: tuple[int, ...]
    intervals_samples: tuple[tuple[int, int], ...]
    gaps_utc: tuple[tuple[float, float], ...]
    array_sha256: str
    identity_pass: bool
    identity_reasons: tuple[str, ...]


def candidate_grid() -> list[float | None]:
    """The preregistered grid. ``None`` is the OFF/raw control."""

    return [None, *ACTIVE_MARGINS_S]


def margin_key(margin_s: float | None) -> str:
    return "OFF" if margin_s is None else f"{float(margin_s):.1f}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(data: object) -> str:
    array = np.ascontiguousarray(np.asarray(data))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_margin(margin_s: float) -> float:
    margin = float(margin_s)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(f"margin_s must be finite and >= 0, got {margin_s!r}")
    return margin


def normalize_gaps(
    gaps: Sequence[Sequence[float]], margin_s: float = 0.0
) -> list[tuple[float, float]]:
    """Drop invalid gaps, expand, sort, and merge overlapping/touching spans."""

    margin = _validate_margin(margin_s)
    valid: list[tuple[float, float]] = []
    for gap in gaps:
        try:
            start, end = float(gap[0]), float(gap[1])
        except (IndexError, TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        valid.append((start - margin, end + margin))
    merged: list[tuple[float, float]] = []
    for start, end in sorted(valid):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _time_is_masked(time_utc: float, intervals: Sequence[tuple[float, float]]) -> bool:
    if not intervals or not math.isfinite(time_utc):
        return False
    starts = [start for start, _ in intervals]
    idx = bisect.bisect_right(starts, time_utc) - 1
    return idx >= 0 and time_utc <= intervals[idx][1]


def mask_gap_picks(
    picks: list[Pick], gaps: Sequence[Sequence[float]], margin_s: float
) -> list[Pick]:
    """Return a stable object-identical subsequence outside expanded gaps.

    With an empty gap list the *same list object* is returned.  Finite pick
    times are tested against closed intervals.  Non-finite times are retained
    so the experiment can report, rather than silently erase, upstream faults.
    """

    margin = _validate_margin(margin_s)
    if not gaps:
        return picks
    intervals = normalize_gaps(gaps, margin)
    if not intervals:
        return picks

    finite_times = [float(pick.time_utc) for pick in picks if math.isfinite(float(pick.time_utc))]
    nondecreasing = all(a <= b for a, b in zip(finite_times, finite_times[1:]))
    kept: list[Pick] = []
    if nondecreasing:
        interval_idx = 0
        for pick in picks:
            value = float(pick.time_utc)
            if not math.isfinite(value):
                kept.append(pick)
                continue
            while interval_idx < len(intervals) and value > intervals[interval_idx][1]:
                interval_idx += 1
            if interval_idx < len(intervals):
                start, end = intervals[interval_idx]
                if start <= value <= end:
                    continue
            kept.append(pick)
        return kept

    starts = [start for start, _ in intervals]
    for pick in picks:
        value = float(pick.time_utc)
        if not math.isfinite(value):
            kept.append(pick)
            continue
        idx = bisect.bisect_right(starts, value) - 1
        if idx >= 0 and value <= intervals[idx][1]:
            continue
        kept.append(pick)
    return kept


def _phase_value(phase: object) -> str:
    return phase.value if isinstance(phase, PhaseType) else str(phase).upper()


def pick_token(pick: Pick) -> tuple[object, ...]:
    return (
        _phase_value(pick.phase),
        float(pick.time_utc),
        float(pick.confidence),
        pick.station,
        pick.sample_index,
    )


def is_stable_identity_subsequence(candidate: Sequence[Pick], raw: Sequence[Pick]) -> bool:
    cursor = 0
    for item in candidate:
        while cursor < len(raw) and raw[cursor] is not item:
            cursor += 1
        if cursor >= len(raw):
            return False
        cursor += 1
    return True


def match_pick_lists(
    reference: Sequence[Pick], other: Sequence[Pick]
) -> dict[str, object]:
    """Greedy preregistered same-phase matching by (abs(dt), ref, other)."""

    edges: list[tuple[float, int, int]] = []
    for ref_idx, ref_pick in enumerate(reference):
        phase = PhaseType(_phase_value(ref_pick.phase))
        tolerance = PICK_TOLERANCE_S[phase]
        ref_time = float(ref_pick.time_utc)
        if not math.isfinite(ref_time):
            continue
        for other_idx, other_pick in enumerate(other):
            if _phase_value(other_pick.phase) != phase.value:
                continue
            other_time = float(other_pick.time_utc)
            if not math.isfinite(other_time):
                continue
            delta = abs(other_time - ref_time)
            if delta <= tolerance + 1e-12:
                edges.append((delta, ref_idx, other_idx))
    matches: list[tuple[int, int, float]] = []
    claimed_reference: set[int] = set()
    claimed_other: set[int] = set()
    for delta, ref_idx, other_idx in sorted(edges):
        if ref_idx in claimed_reference or other_idx in claimed_other:
            continue
        claimed_reference.add(ref_idx)
        claimed_other.add(other_idx)
        matches.append((ref_idx, other_idx, delta))
    matches.sort(key=lambda row: (row[0], row[1]))
    induced = [idx for idx in range(len(other)) if idx not in claimed_other]
    lost = [idx for idx in range(len(reference)) if idx not in claimed_reference]
    return {
        "matches": matches,
        "induced_indices": induced,
        "lost_indices": lost,
        "pass": not induced and not lost and len(reference) == len(other),
    }


def distance_to_gaps(time_utc: float, gaps: Sequence[Sequence[float]]) -> float:
    intervals = normalize_gaps(gaps, 0.0)
    value = float(time_utc)
    if not math.isfinite(value) or not intervals:
        return math.inf
    starts = [start for start, _ in intervals]
    idx = bisect.bisect_right(starts, value) - 1
    if idx >= 0 and value <= intervals[idx][1]:
        return 0.0
    distances: list[float] = []
    if idx >= 0:
        distances.append(value - intervals[idx][1])
    next_idx = idx + 1
    if next_idx < len(intervals):
        distances.append(intervals[next_idx][0] - value)
    return min(distances) if distances else math.inf


def analyze_candidate(
    reference: Sequence[Pick],
    raw_gapped: list[Pick],
    gaps: Sequence[Sequence[float]],
    margin_s: float,
) -> dict[str, object]:
    before_tokens = [pick_token(pick) for pick in raw_gapped]
    candidate = mask_gap_picks(raw_gapped, gaps, margin_s)
    after_tokens = [pick_token(pick) for pick in raw_gapped]
    matching = match_pick_lists(reference, raw_gapped)
    matches = matching["matches"]
    induced_indices = matching["induced_indices"]
    lost_indices = matching["lost_indices"]
    physical = normalize_gaps(gaps, 0.0)
    expanded = normalize_gaps(gaps, margin_s)

    candidate_raw_indices: list[int] = []
    cursor = 0
    for item in candidate:
        while cursor < len(raw_gapped) and raw_gapped[cursor] is not item:
            cursor += 1
        if cursor >= len(raw_gapped):
            candidate_raw_indices = []
            break
        candidate_raw_indices.append(cursor)
        cursor += 1
    candidate_index_set = set(candidate_raw_indices)

    stable_outside: list[tuple[int, int]] = []
    for ref_idx, raw_idx, _ in matches:
        ref_time = float(reference[ref_idx].time_utc)
        raw_time = float(raw_gapped[raw_idx].time_utc)
        if not _time_is_masked(ref_time, physical) and not _time_is_masked(raw_time, physical):
            stable_outside.append((ref_idx, raw_idx))

    residual = [idx for idx in induced_indices if idx in candidate_index_set]
    collateral = [pair for pair in stable_outside if pair[1] not in candidate_index_set]
    remote_induced = [
        idx
        for idx in induced_indices
        if distance_to_gaps(float(raw_gapped[idx].time_utc), physical) > REMOTE_DISTANCE_S
    ]
    remote_lost = [
        idx
        for idx in lost_indices
        if distance_to_gaps(float(reference[idx].time_utc), physical) > REMOTE_DISTANCE_S
    ]
    inside_expanded = [
        idx
        for idx, pick in zip(candidate_raw_indices, candidate)
        if _time_is_masked(float(pick.time_utc), expanded)
    ]

    subset_pass = (
        bool(candidate_raw_indices) or not candidate
    ) and is_stable_identity_subsequence(candidate, raw_gapped)
    object_unchanged = before_tokens == after_tokens
    return {
        "margin_s": float(margin_s),
        "raw_count": len(raw_gapped),
        "candidate_count": len(candidate),
        "matched_stable_count": len(matches),
        "induced_new_indices": list(induced_indices),
        "lost_reference_indices": list(lost_indices),
        "remote_induced_new_indices": remote_induced,
        "remote_lost_reference_indices": remote_lost,
        "stable_outside_physical_gap": [list(pair) for pair in stable_outside],
        "collateral_deleted": [list(pair) for pair in collateral],
        "residual_induced_new_indices": residual,
        "candidate_inside_expanded_gap_indices": inside_expanded,
        "candidate_raw_indices": candidate_raw_indices,
        "subset_and_order_pass": subset_pass,
        "object_fields_unchanged_pass": object_unchanged,
        "pass": (
            subset_pass
            and object_unchanged
            and not residual
            and not remote_induced
            and not remote_lost
            and not collateral
            and not inside_expanded
        ),
    }


def _bounded_interval(center_s: float, length_s: float, duration_s: float) -> tuple[float, float]:
    if duration_s < length_s + 10.0:
        raise ValueError("waveform too short for preregistered 5 s boundary guard")
    low = 5.0
    latest_start = duration_s - 5.0 - length_s
    start = min(max(float(center_s) - length_s / 2.0, low), latest_start)
    return start, start + length_s


def variant_specs(
    *, sample_key: str, duration_s: float, anchor_s: float
) -> list[tuple[str, tuple[int, ...], tuple[tuple[float, float], ...]]]:
    all_components = (0, 1, 2)
    mid = duration_s / 2.0
    single_component = hashlib.sha256(sample_key.encode("utf-8")).digest()[0] % 3
    edge_end = anchor_s - 0.5
    edge_start = edge_end - 2.0
    if edge_start < 5.0:
        edge_start = 5.0
        edge_end = 7.0
    if edge_end > duration_s - 5.0:
        edge_end = duration_s - 5.0
        edge_start = edge_end - 2.0
    return [
        ("mid-0p5-all", all_components, (_bounded_interval(mid, 0.5, duration_s),)),
        ("mid-2-all", all_components, (_bounded_interval(mid, 2.0, duration_s),)),
        ("mid-10-all", all_components, (_bounded_interval(mid, 10.0, duration_s),)),
        (
            "anchor-center-2-all",
            all_components,
            (_bounded_interval(anchor_s, 2.0, duration_s),),
        ),
        (
            "anchor-edge-2-all",
            all_components,
            ((edge_start, edge_end),),
        ),
        (
            "double-2-10-all",
            all_components,
            (
                _bounded_interval(duration_s / 3.0, 2.0, duration_s),
                _bounded_interval(2.0 * duration_s / 3.0, 10.0, duration_s),
            ),
        ),
        (
            "anchor-center-2-one",
            (single_component,),
            (_bounded_interval(anchor_s, 2.0, duration_s),),
        ),
    ]


def inject_gaps(
    waveform: Waveform,
    *,
    variant_id: str,
    components: Sequence[int],
    relative_intervals_s: Sequence[tuple[float, float]],
) -> InjectionResult:
    source = np.asarray(waveform.data)
    data = np.array(source, dtype=np.float32, copy=True, order="C")
    rate = float(waveform.sampling_rate)
    n_samples = int(data.shape[-1])
    component_tuple = tuple(int(value) for value in components)
    interval_samples: list[tuple[int, int]] = []
    gaps_utc: list[tuple[float, float]] = []
    reasons: list[str] = []

    for start_s, end_s in relative_intervals_s:
        i0 = max(0, min(n_samples - 1, int(round(float(start_s) * rate))))
        i1 = max(i0 + 1, min(n_samples, int(round(float(end_s) * rate))))
        interval_samples.append((i0, i1))
        gaps_utc.append(
            (
                float(waveform.starttime_utc) + i0 / rate,
                float(waveform.starttime_utc) + i1 / rate,
            )
        )
        for component in component_tuple:
            if component not in (0, 1, 2):
                raise ValueError(f"invalid component index: {component}")
            data[component, i0:i1] = np.float32(0.0)

    if data.shape != source.shape:
        reasons.append("shape changed")
    if data.dtype != np.float32:
        reasons.append("injected dtype is not float32")
    for component in range(3):
        target = component in component_tuple
        covered = np.zeros(n_samples, dtype=bool)
        for i0, i1 in interval_samples:
            covered[i0:i1] = True
        if target:
            if not np.all(data[component, covered] == np.float32(0.0)):
                reasons.append(f"component {component} target samples are not exactly zero")
            if not np.array_equal(data[component, ~covered], source[component, ~covered]):
                reasons.append(f"component {component} changed outside gaps")
        elif not np.array_equal(data[component], source[component]):
            reasons.append(f"untargeted component {component} changed")

    injected = Waveform(
        data=data,
        sampling_rate=rate,
        starttime_utc=float(waveform.starttime_utc),
        station=waveform.station,
        gaps=list(gaps_utc),
    )
    if injected.sampling_rate != waveform.sampling_rate:
        reasons.append("sampling rate changed")
    if injected.starttime_utc != waveform.starttime_utc:
        reasons.append("start time changed")
    if injected.station != waveform.station:
        reasons.append("station changed")

    repeat = np.array(source, dtype=np.float32, copy=True, order="C")
    for i0, i1 in interval_samples:
        repeat[list(component_tuple), i0:i1] = np.float32(0.0)
    if sha256_array(repeat) != sha256_array(data):
        reasons.append("repeat injection hash changed")

    return InjectionResult(
        waveform=injected,
        variant_id=variant_id,
        components=component_tuple,
        intervals_samples=tuple(interval_samples),
        gaps_utc=tuple(gaps_utc),
        array_sha256=sha256_array(data),
        identity_pass=not reasons,
        identity_reasons=tuple(reasons),
    )


def select_anchor(reference: Sequence[Pick], waveform: Waveform) -> float:
    duration = float(waveform.duration)
    eligible = sorted(
        float(pick.time_utc) - float(waveform.starttime_utc)
        for pick in reference
        if math.isfinite(float(pick.time_utc))
        and 15.0 <= float(pick.time_utc) - float(waveform.starttime_utc) <= duration - 15.0
    )
    return eligible[0] if eligible else duration / 2.0


def crop_long_waveform(
    waveform: Waveform, full_reference: Sequence[Pick]
) -> tuple[Waveform, dict[str, object]]:
    duration = float(waveform.duration)
    if duration < LONG_CROP_S:
        raise ValueError(f"long source shorter than {LONG_CROP_S}s")
    central = sorted(
        float(pick.time_utc) - float(waveform.starttime_utc)
        for pick in full_reference
        if math.isfinite(float(pick.time_utc))
        and LONG_CONTEXT_S
        <= float(pick.time_utc) - float(waveform.starttime_utc)
        <= duration - LONG_CONTEXT_S
    )
    full_anchor = central[0] if central else duration / 2.0
    requested_start = min(
        max(full_anchor - LONG_CONTEXT_S, 0.0), duration - LONG_CROP_S
    )
    rate = float(waveform.sampling_rate)
    i0 = int(round(requested_start * rate))
    length = int(round(LONG_CROP_S * rate))
    i0 = max(0, min(i0, waveform.n_samples - length))
    i1 = i0 + length
    data = np.array(waveform.data[:, i0:i1], dtype=np.float32, copy=True, order="C")
    cropped = Waveform(
        data=data,
        sampling_rate=rate,
        starttime_utc=float(waveform.starttime_utc) + i0 / rate,
        station=waveform.station,
        gaps=[],
    )
    if abs(cropped.duration - LONG_CROP_S) > 1.0 / rate:
        raise AssertionError("360 s crop length drifted")
    return cropped, {
        "source_duration_s": duration,
        "full_reference_pick_count": len(full_reference),
        "full_anchor_relative_s": full_anchor,
        "crop_start_relative_s": i0 / rate,
        "crop_duration_s": cropped.duration,
        "array_sha256": sha256_array(cropped.data),
    }


def make_noise_samples() -> list[PreparedSample]:
    rng = np.random.default_rng(SEED)
    result: list[PreparedSample] = []
    for key, duration in (("noise-short", 60.0), ("noise-long", 360.0)):
        data = rng.standard_normal((3, int(duration * 100.0)), dtype=np.float32)
        result.append(
            PreparedSample(
                key=key,
                package_key="synthetic",
                file_id=key,
                waveform=Waveform(
                    data=data,
                    sampling_rate=100.0,
                    starttime_utc=1_700_000_000.0,
                    station=key,
                    gaps=[],
                ),
                source_kind="gaussian-noise",
            )
        )
    return result


def _load_one_waveform(archive: Path, file_id: str) -> tuple[Waveform, tuple[str, ...]]:
    samples = [sample for sample in scan_exam_input(str(archive)) if sample.file_id == file_id]
    if len(samples) != 1:
        raise RuntimeError(f"expected one {file_id}, found {len(samples)}")
    ingest = load_waveforms(read_source_bytes(samples[0].source_path))
    if len(ingest.waveforms) != 1:
        raise RuntimeError(
            f"expected one station waveform for {file_id}, found {len(ingest.waveforms)}"
        )
    warnings = tuple(warning.reason for warning in ingest.warnings)
    return ingest.waveforms[0], warnings


def _sample_key(package_key: str, file_id: str, is_long: bool) -> str:
    label = EXPECTED_PACKAGES[package_key]["label"]
    suffix = "@360s" if is_long else ""
    return f"{label}/{file_id}{suffix}"


def prepare_real_samples(
    *,
    picker: ProbEnsemblePicker,
    archive: Path,
    package_key: str,
    file_ids: Sequence[str],
) -> list[PreparedSample]:
    prepared: list[PreparedSample] = []
    for file_id in file_ids:
        waveform, warnings = _load_one_waveform(archive, file_id)
        if waveform.gaps:
            raise RuntimeError(f"frozen sample unexpectedly contains a real gap: {file_id}")
        is_long = (package_key, file_id) in LONG_FILE_IDS
        crop: Mapping[str, object] | None = None
        if is_long:
            full_reference = picker.pick(waveform)
            waveform, crop = crop_long_waveform(waveform, full_reference)
        prepared.append(
            PreparedSample(
                key=_sample_key(package_key, file_id, is_long),
                package_key=package_key,
                file_id=file_id,
                waveform=waveform,
                source_kind="real",
                crop=crop,
                ingest_warnings=warnings,
            )
        )
    return prepared


def _relative_pick_records(picks: Sequence[Pick], waveform: Waveform) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for pick in picks:
        relative = float(pick.time_utc) - float(waveform.starttime_utc)
        records.append(
            {
                "phase": _phase_value(pick.phase),
                "relative_time_s": relative if math.isfinite(relative) else None,
                "confidence": (
                    float(pick.confidence) if math.isfinite(float(pick.confidence)) else None
                ),
            }
        )
    return records


def _relative_gap_records(
    gaps: Sequence[Sequence[float]], waveform: Waveform
) -> list[list[float]]:
    return [
        [float(start) - waveform.starttime_utc, float(end) - waveform.starttime_utc]
        for start, end in gaps
    ]


def _matching_record(
    matching: Mapping[str, object],
    reference: Sequence[Pick],
    other: Sequence[Pick],
    waveform: Waveform,
) -> dict[str, object]:
    return {
        "pass": bool(matching["pass"]),
        "matched_count": len(matching["matches"]),
        "induced_count": len(matching["induced_indices"]),
        "lost_count": len(matching["lost_indices"]),
        "matches": [
            {
                "reference_index": ref_idx,
                "other_index": other_idx,
                "abs_dt_s": delta,
            }
            for ref_idx, other_idx, delta in matching["matches"]
        ],
        "induced": [
            {
                "index": idx,
                "phase": _phase_value(other[idx].phase),
                "relative_time_s": float(other[idx].time_utc) - waveform.starttime_utc,
            }
            for idx in matching["induced_indices"]
        ],
        "lost": [
            {
                "index": idx,
                "phase": _phase_value(reference[idx].phase),
                "relative_time_s": float(reference[idx].time_utc) - waveform.starttime_utc,
            }
            for idx in matching["lost_indices"]
        ],
    }


def _exact_pick_sequence(left: Sequence[Pick], right: Sequence[Pick]) -> bool:
    return [pick_token(pick) for pick in left] == [pick_token(pick) for pick in right]


def _candidate_single_batch(
    single_raw: list[Pick],
    batch_raw: list[Pick],
    gaps: Sequence[Sequence[float]],
    margin_s: float,
) -> dict[str, object]:
    single = mask_gap_picks(single_raw, gaps, margin_s)
    batch = mask_gap_picks(batch_raw, gaps, margin_s)
    matching = match_pick_lists(single, batch)
    return {
        "pass": bool(matching["pass"]),
        "single_count": len(single),
        "batch_count": len(batch),
        "matched_count": len(matching["matches"]),
    }


def _make_injections(
    sample: PreparedSample, reference: Sequence[Pick]
) -> tuple[float, dict[str, InjectionResult]]:
    anchor = select_anchor(reference, sample.waveform)
    injections: dict[str, InjectionResult] = {}
    for variant_id, components, intervals in variant_specs(
        sample_key=sample.key,
        duration_s=sample.waveform.duration,
        anchor_s=anchor,
    ):
        injections[variant_id] = inject_gaps(
            sample.waveform,
            variant_id=variant_id,
            components=components,
            relative_intervals_s=intervals,
        )
    return anchor, injections


def run_group(
    *,
    picker: ProbEnsemblePicker,
    samples: Sequence[PreparedSample],
    references: Mapping[str, list[Pick]],
) -> tuple[dict[str, object], list[str]]:
    """Run single path once and batch path twice for a sample group."""

    injections_by_sample: dict[str, dict[str, InjectionResult]] = {}
    anchors: dict[str, float] = {}
    entries: list[tuple[str, str, Waveform]] = []
    for sample in samples:
        anchor, injections = _make_injections(sample, references[sample.key])
        anchors[sample.key] = anchor
        injections_by_sample[sample.key] = injections
        entries.append((sample.key, "reference", sample.waveform))
        for variant_id, injection in injections.items():
            entries.append((sample.key, variant_id, injection.waveform))

    single_outputs: dict[tuple[str, str], list[Pick]] = {}
    for sample_key, variant_id, waveform in entries:
        if variant_id == "reference":
            single_outputs[(sample_key, variant_id)] = references[sample_key]
        else:
            single_outputs[(sample_key, variant_id)] = picker.pick(waveform)

    waveforms = [waveform for _, _, waveform in entries]
    batch_first = picker.pick_batch(waveforms)
    batch_second = picker.pick_batch(waveforms)
    if len(batch_first) != len(entries) or len(batch_second) != len(entries):
        raise RuntimeError("pick_batch output length mismatch")

    batch_first_map = {
        (sample_key, variant_id): picks
        for (sample_key, variant_id, _), picks in zip(entries, batch_first)
    }
    batch_second_map = {
        (sample_key, variant_id): picks
        for (sample_key, variant_id, _), picks in zip(entries, batch_second)
    }

    structural_reasons: list[str] = []
    records: dict[str, object] = {}
    for sample in samples:
        reference_single = references[sample.key]
        reference_batch = batch_first_map[(sample.key, "reference")]
        reference_repeat = batch_second_map[(sample.key, "reference")]
        no_gap_identity = {
            margin_key(margin): (
                mask_gap_picks(reference_single, [], margin) is reference_single
                and mask_gap_picks(reference_batch, [], margin) is reference_batch
            )
            for margin in ACTIVE_MARGINS_S
        }
        reference_single_batch = match_pick_lists(reference_single, reference_batch)
        reference_batch_repeat = _exact_pick_sequence(reference_batch, reference_repeat)
        if not all(no_gap_identity.values()):
            structural_reasons.append(f"{sample.key}: no-gap list identity failed")
        if not reference_single_batch["pass"]:
            structural_reasons.append(f"{sample.key}: reference single/batch mismatch")
        if not reference_batch_repeat:
            structural_reasons.append(f"{sample.key}: reference batch repeat changed")

        sample_record: dict[str, object] = {
            "package": sample.package_key,
            "file_id": sample.file_id,
            "source_kind": sample.source_kind,
            "sampling_rate_hz": sample.waveform.sampling_rate,
            "duration_s": sample.waveform.duration,
            "array_sha256": sha256_array(sample.waveform.data),
            "anchor_relative_s": anchors[sample.key],
            "crop": sample.crop,
            "ingest_warning_reasons": list(sample.ingest_warnings),
            "reference": {
                "single": _relative_pick_records(reference_single, sample.waveform),
                "batch": _relative_pick_records(reference_batch, sample.waveform),
                "single_batch": _matching_record(
                    reference_single_batch,
                    reference_single,
                    reference_batch,
                    sample.waveform,
                ),
                "batch_repeat_exact": reference_batch_repeat,
                "no_gap_identity": no_gap_identity,
            },
            "variants": {},
        }

        for variant_id, injection in injections_by_sample[sample.key].items():
            single_raw = single_outputs[(sample.key, variant_id)]
            batch_raw = batch_first_map[(sample.key, variant_id)]
            batch_repeat = batch_second_map[(sample.key, variant_id)]
            raw_single_batch = match_pick_lists(single_raw, batch_raw)
            repeat_exact = _exact_pick_sequence(batch_raw, batch_repeat)
            if not injection.identity_pass:
                structural_reasons.append(
                    f"{sample.key}/{variant_id}: injection identity failed"
                )
            if not raw_single_batch["pass"]:
                structural_reasons.append(
                    f"{sample.key}/{variant_id}: raw single/batch mismatch"
                )
            if not repeat_exact:
                structural_reasons.append(
                    f"{sample.key}/{variant_id}: batch repeat changed"
                )

            margins: dict[str, object] = {}
            for margin in ACTIVE_MARGINS_S:
                single_analysis = analyze_candidate(
                    reference_single, single_raw, injection.gaps_utc, margin
                )
                batch_analysis = analyze_candidate(
                    reference_batch, batch_raw, injection.gaps_utc, margin
                )
                candidate_single_batch = _candidate_single_batch(
                    single_raw, batch_raw, injection.gaps_utc, margin
                )
                if not single_analysis["subset_and_order_pass"]:
                    structural_reasons.append(
                        f"{sample.key}/{variant_id}/{margin_key(margin)}: "
                        "single candidate is not an identity subsequence"
                    )
                if not batch_analysis["subset_and_order_pass"]:
                    structural_reasons.append(
                        f"{sample.key}/{variant_id}/{margin_key(margin)}: "
                        "batch candidate is not an identity subsequence"
                    )
                if not single_analysis["object_fields_unchanged_pass"]:
                    structural_reasons.append(
                        f"{sample.key}/{variant_id}/{margin_key(margin)}: "
                        "single raw objects were mutated"
                    )
                if not batch_analysis["object_fields_unchanged_pass"]:
                    structural_reasons.append(
                        f"{sample.key}/{variant_id}/{margin_key(margin)}: "
                        "batch raw objects were mutated"
                    )
                margins[margin_key(margin)] = {
                    "single": single_analysis,
                    "batch": batch_analysis,
                    "single_batch_candidate": candidate_single_batch,
                }

            sample_record["variants"][variant_id] = {
                "components": [CHANNEL_ORDER[idx] for idx in injection.components],
                "intervals_samples": [list(pair) for pair in injection.intervals_samples],
                "gaps_relative_s": _relative_gap_records(
                    injection.gaps_utc, sample.waveform
                ),
                "array_sha256": injection.array_sha256,
                "input_identity_pass": injection.identity_pass,
                "input_identity_reasons": list(injection.identity_reasons),
                "raw_single": _relative_pick_records(single_raw, sample.waveform),
                "raw_batch": _relative_pick_records(batch_raw, sample.waveform),
                "raw_single_batch": _matching_record(
                    raw_single_batch,
                    single_raw,
                    batch_raw,
                    sample.waveform,
                ),
                "batch_repeat_exact": repeat_exact,
                "margins": margins,
            }
        records[sample.key] = sample_record
    return records, sorted(set(structural_reasons))


def assess_development_margins(
    records: Mapping[str, object]
) -> tuple[dict[str, object], float | None]:
    assessments: dict[str, object] = {}
    selected: float | None = None
    for margin in ACTIVE_MARGINS_S:
        reasons: list[str] = []
        key = margin_key(margin)
        for sample_key, sample_record in records.items():
            reference = sample_record["reference"]
            if not reference["no_gap_identity"][key]:
                reasons.append(f"{sample_key}: no-gap identity")
            if not reference["single_batch"]["pass"]:
                reasons.append(f"{sample_key}: reference single/batch")
            if not reference["batch_repeat_exact"]:
                reasons.append(f"{sample_key}: reference determinism")
            for variant_id, variant in sample_record["variants"].items():
                if not variant["input_identity_pass"]:
                    reasons.append(f"{sample_key}/{variant_id}: input identity")
                if not variant["raw_single_batch"]["pass"]:
                    reasons.append(f"{sample_key}/{variant_id}: raw single/batch")
                if not variant["batch_repeat_exact"]:
                    reasons.append(f"{sample_key}/{variant_id}: determinism")
                margin_record = variant["margins"][key]
                for path_name in ("single", "batch"):
                    analysis = margin_record[path_name]
                    if not analysis["subset_and_order_pass"]:
                        reasons.append(
                            f"{sample_key}/{variant_id}/{path_name}: not subset"
                        )
                    if not analysis["object_fields_unchanged_pass"]:
                        reasons.append(
                            f"{sample_key}/{variant_id}/{path_name}: object mutated"
                        )
                    if analysis["candidate_inside_expanded_gap_indices"]:
                        reasons.append(
                            f"{sample_key}/{variant_id}/{path_name}: pick inside gap"
                        )
                    if analysis["residual_induced_new_indices"]:
                        reasons.append(
                            f"{sample_key}/{variant_id}/{path_name}: residual induced"
                        )
                    if analysis["remote_induced_new_indices"]:
                        reasons.append(
                            f"{sample_key}/{variant_id}/{path_name}: remote induced"
                        )
                    if analysis["remote_lost_reference_indices"]:
                        reasons.append(
                            f"{sample_key}/{variant_id}/{path_name}: remote lost"
                        )
                    if analysis["collateral_deleted"]:
                        reasons.append(
                            f"{sample_key}/{variant_id}/{path_name}: collateral deleted"
                        )
                if not margin_record["single_batch_candidate"]["pass"]:
                    reasons.append(
                        f"{sample_key}/{variant_id}: candidate single/batch"
                    )
        unique_reasons = sorted(set(reasons))
        eligible = not unique_reasons
        assessments[key] = {
            "margin_s": margin,
            "eligible": eligible,
            "failure_reasons": unique_reasons,
        }
        if eligible and selected is None:
            selected = margin
    return assessments, selected


def assess_holdout(
    records: Mapping[str, object], active_margin_s: float
) -> dict[str, object]:
    key = margin_key(active_margin_s)
    reasons: list[str] = []
    for sample_key, sample_record in records.items():
        if not sample_record["reference"]["no_gap_identity"][key]:
            reasons.append(f"{sample_key}: no-gap identity")
        for variant_id, variant in sample_record["variants"].items():
            analysis = variant["margins"][key]["single"]
            if not variant["input_identity_pass"]:
                reasons.append(f"{sample_key}/{variant_id}: input identity")
            if not analysis["subset_and_order_pass"]:
                reasons.append(f"{sample_key}/{variant_id}: not subset")
            if not analysis["object_fields_unchanged_pass"]:
                reasons.append(f"{sample_key}/{variant_id}: object mutated")
            if analysis["candidate_inside_expanded_gap_indices"]:
                reasons.append(f"{sample_key}/{variant_id}: pick inside gap")
            if analysis["residual_induced_new_indices"]:
                reasons.append(f"{sample_key}/{variant_id}: residual induced")
            if analysis["remote_induced_new_indices"]:
                reasons.append(f"{sample_key}/{variant_id}: remote induced")
            if analysis["remote_lost_reference_indices"]:
                reasons.append(f"{sample_key}/{variant_id}: remote lost")
            if analysis["collateral_deleted"]:
                reasons.append(f"{sample_key}/{variant_id}: collateral deleted")
    unique = sorted(set(reasons))
    return {"pass": not unique, "failure_reasons": unique}


def _strip_unselected_holdout_margins(
    records: Mapping[str, object], active_margin_s: float
) -> dict[str, object]:
    """The 08 JSON exposes only raw/OFF and the frozen active margin."""

    key = margin_key(active_margin_s)
    for sample_record in records.values():
        for variant in sample_record["variants"].values():
            variant["margins"] = {key: variant["margins"][key]}
    return dict(records)


def benchmark_mask() -> dict[str, object]:
    picks = [
        Pick(
            phase=PhaseType.P if idx % 2 == 0 else PhaseType.S,
            time_utc=float(idx) * 0.4,
            confidence=0.5,
            station="bench",
        )
        for idx in range(PERFORMANCE_PICKS)
    ]
    gaps = [
        (float(idx) * 40.0 + 10.0, float(idx) * 40.0 + 10.5)
        for idx in range(PERFORMANCE_GAPS)
    ]
    for _ in range(20):
        mask_gap_picks(picks, gaps, 10.0)
    timings_ms: list[float] = []
    last: list[Pick] = []
    for _ in range(PERFORMANCE_RUNS):
        started = time.perf_counter_ns()
        last = mask_gap_picks(picks, gaps, 10.0)
        timings_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    ordered = sorted(timings_ms)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "pick_count": PERFORMANCE_PICKS,
        "gap_count": PERFORMANCE_GAPS,
        "runs": PERFORMANCE_RUNS,
        "margin_s": 10.0,
        "p50_ms": ordered[len(ordered) // 2],
        "p95_ms": p95,
        "max_ms": ordered[-1],
        "output_count": len(last),
        "stable_subsequence_pass": is_stable_identity_subsequence(last, picks),
        "threshold_ms": PERFORMANCE_MAX_P95_MS,
        "pass": p95 < PERFORMANCE_MAX_P95_MS
        and is_stable_identity_subsequence(last, picks),
    }


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _package_identities(paths: Mapping[str, Path]) -> tuple[dict[str, object], bool]:
    records: dict[str, object] = {}
    passed = True
    for key, path in paths.items():
        actual = sha256_file(path)
        expected = EXPECTED_PACKAGES[key]["sha256"]
        identity = actual == expected
        passed = passed and identity
        records[key] = {
            "basename": path.name,
            "sha256": actual,
            "expected_sha256": expected,
            "identity_pass": identity,
        }
    return records, passed


def _load_manifest(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _weight_identities(manifest: Mapping[str, object]) -> tuple[list[dict[str, object]], bool]:
    production = manifest["production_config"]
    names = [value.strip() for value in production["weights_cli"].split(",") if value.strip()]
    tracked = {item["path"]: item for item in manifest["assets"]["tracked"]}
    relative_paths: list[str] = ["weights/phasenet_diting_weights.tar.gz"]
    for name in names:
        if name.endswith(".pt") or "/" in name or "\\" in name:
            relative_paths.append(name.replace("\\", "/"))
        else:
            relative_paths.append(f"weights/ustc_pickers/{name}_sd.pt")
    records: list[dict[str, object]] = []
    passed = True
    for relative in relative_paths:
        expected = tracked.get(relative)
        path = ROOT / Path(relative)
        actual_sha = sha256_file(path) if path.is_file() else None
        actual_size = path.stat().st_size if path.is_file() else None
        identity = bool(
            expected
            and actual_sha == expected["sha256"]
            and actual_size == expected["size_bytes"]
        )
        passed = passed and identity
        records.append(
            {
                "path": relative,
                "size_bytes": actual_size,
                "sha256": actual_sha,
                "identity_pass": identity,
            }
        )
    return records, passed


def build_production_picker(
    manifest: Mapping[str, object], *, device: str
) -> ProbEnsemblePicker:
    production = manifest["production_config"]
    cfg = PickerConfig(
        device=None if device == "auto" else device,
        pretrained=production["pretrained"],
        p_threshold=float(production["p_threshold"]),
        s_threshold=float(production["s_threshold"]),
        batch_size=256,
        overlap=0.5,
        use_fp16=False,
        num_threads=int(production["threads"]),
        compile_model=False,
        p_merge_window_s=float(production["p_merge_window_s"]),
        s_merge_window_s=float(production["s_merge_window_s"]),
        subsample_refine=True,
        long_snr_threshold_db=float(production["long_snr_db"]),
        long_snr_min_duration_s=float(production["long_snr_min_s"]),
        force_pair_max_duration_s=float(production["force_pair_short_s"]),
        force_pair_floor=float(production["force_pair_floor"]),
        force_pair_conditional=production["force_pair_mode"] == "conditional",
        long_dedup_p_window_s=float(production["long_dedup_s"]),
        long_dedup_s_window_s=float(production["long_dedup_s"]),
        tta_polarity_flip=bool(production["tta_polarity_flip"]),
        ensemble_long_top_n=int(production["ensemble_long_members"]),
    )
    members = [
        value.strip() for value in production["weights_cli"].split(",") if value.strip()
    ]
    return ProbEnsemblePicker.from_member_names(members, cfg)


def _structural_sentinel(
    *,
    reasons: Sequence[str],
    package_identity_pass: bool,
    weight_identity_pass: bool,
    picker: ProbEnsemblePicker,
) -> dict[str, object]:
    all_reasons = list(reasons)
    member_count = len(getattr(picker, "_members", []))
    if not package_identity_pass:
        all_reasons.append("input package identity failed")
    if not weight_identity_pass:
        all_reasons.append("production weight identity failed")
    if member_count != 7:
        all_reasons.append(f"expected 7 ensemble members, found {member_count}")
    unique = sorted(set(all_reasons))
    return {
        "pass": not unique,
        "ensemble_member_count": member_count,
        "failure_reasons": unique,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run(
    *,
    round1: Path,
    round2: Path,
    final08: Path,
    manifest_path: Path,
    output: Path,
    device: str,
) -> dict[str, object]:
    archive_paths = {"round1": round1, "round2": round2, "final08": final08}
    package_records, package_identity_pass = _package_identities(archive_paths)
    manifest = _load_manifest(manifest_path)
    weight_records, weight_identity_pass = _weight_identities(manifest)
    performance = benchmark_mask()

    base_result: dict[str, object] = {
        "schema_version": 1,
        "round": 5,
        "experiment": "t1-gap-mask-robustness",
        "git_head": _git_head(),
        "seed": SEED,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _version("torch"),
            "obspy": _version("obspy"),
            "seisbench": _version("seisbench"),
            "requested_device": device,
        },
        "input_packages": package_records,
        "input_identity_pass": package_identity_pass,
        "weights": weight_records,
        "weight_identity_pass": weight_identity_pass,
        "preregistered_margins_s": [margin_key(value) for value in candidate_grid()],
        "performance": performance,
        "sentinel": None,
        "development": None,
        "holdout08": None,
        "development_margin_selected": "OFF",
        "development_pass": False,
        "production_eligible": False,
    }

    if not package_identity_pass or not weight_identity_pass:
        base_result["sentinel"] = {
            "pass": False,
            "failure_reasons": [
                reason
                for reason, passed in (
                    ("input package identity failed", package_identity_pass),
                    ("production weight identity failed", weight_identity_pass),
                )
                if not passed
            ],
        }
        base_result["decision"] = {
            "status": "structural-stop",
            "failure_reasons": base_result["sentinel"]["failure_reasons"],
            "enter_production": False,
        }
        _write_json(output, _json_safe(base_result))
        return base_result

    picker = build_production_picker(manifest, device=device)
    try:
        actual_device = next(picker._model.parameters()).device.type
    except Exception:  # noqa: BLE001
        actual_device = "unknown"
    base_result["environment"]["actual_device"] = actual_device
    base_result["production_config"] = {
        key: manifest["production_config"][key]
        for key in (
            "pretrained",
            "weights_cli",
            "p_threshold",
            "s_threshold",
            "p_merge_window_s",
            "s_merge_window_s",
            "long_snr_db",
            "long_snr_min_s",
            "force_pair_short_s",
            "force_pair_floor",
            "force_pair_mode",
            "long_dedup_s",
            "ensemble_long_members",
            "tta_polarity_flip",
        )
    }

    development_samples = []
    development_samples.extend(
        prepare_real_samples(
            picker=picker,
            archive=round1,
            package_key="round1",
            file_ids=DEVELOPMENT_FILES["round1"],
        )
    )
    development_samples.extend(
        prepare_real_samples(
            picker=picker,
            archive=round2,
            package_key="round2",
            file_ids=DEVELOPMENT_FILES["round2"],
        )
    )
    development_samples.extend(make_noise_samples())
    references = {
        sample.key: picker.pick(sample.waveform) for sample in development_samples
    }

    sentinel_samples = [
        sample for sample in development_samples if sample.key in SENTINEL_KEYS
    ]
    if {sample.key for sample in sentinel_samples} != SENTINEL_KEYS:
        missing = sorted(SENTINEL_KEYS - {sample.key for sample in sentinel_samples})
        raise RuntimeError(f"sentinel sample selection incomplete: {missing}")
    sentinel_records, sentinel_reasons = run_group(
        picker=picker,
        samples=sentinel_samples,
        references=references,
    )
    sentinel = _structural_sentinel(
        reasons=sentinel_reasons,
        package_identity_pass=package_identity_pass,
        weight_identity_pass=weight_identity_pass,
        picker=picker,
    )
    sentinel["samples"] = sentinel_records
    base_result["sentinel"] = sentinel
    if not sentinel["pass"]:
        base_result["decision"] = {
            "status": "structural-stop",
            "failure_reasons": sentinel["failure_reasons"],
            "enter_production": False,
        }
        _write_json(output, _json_safe(base_result))
        return base_result

    remaining_samples = [
        sample for sample in development_samples if sample.key not in SENTINEL_KEYS
    ]
    remaining_records, remaining_reasons = run_group(
        picker=picker,
        samples=remaining_samples,
        references=references,
    )
    if remaining_reasons:
        base_result["development"] = {
            "records": {**sentinel_records, **remaining_records},
            "structural_pass": False,
            "structural_failure_reasons": remaining_reasons,
        }
        base_result["decision"] = {
            "status": "structural-stop",
            "failure_reasons": remaining_reasons,
            "enter_production": False,
        }
        _write_json(output, _json_safe(base_result))
        return base_result

    development_records = {**sentinel_records, **remaining_records}
    assessments, selected = assess_development_margins(development_records)
    base_result["development"] = {
        "structural_pass": True,
        "sample_count": len(development_samples),
        "records": development_records,
        "margin_assessments": assessments,
    }
    base_result["development_margin_selected"] = margin_key(selected)

    holdout_records: dict[str, object] | None = None
    holdout_assessment: dict[str, object] = {
        "pass": False,
        "not_run_reason": "no active development margin selected",
        "failure_reasons": [],
    }
    if selected is not None:
        holdout_samples = prepare_real_samples(
            picker=picker,
            archive=final08,
            package_key="final08",
            file_ids=HOLDOUT_FILES,
        )
        holdout_references = {
            sample.key: picker.pick(sample.waveform) for sample in holdout_samples
        }
        holdout_all_records, holdout_structural_reasons = run_group(
            picker=picker,
            samples=holdout_samples,
            references=holdout_references,
        )
        if holdout_structural_reasons:
            holdout_assessment = {
                "pass": False,
                "failure_reasons": holdout_structural_reasons,
                "structural_pass": False,
            }
        else:
            holdout_assessment = assess_holdout(holdout_all_records, selected)
            holdout_assessment["structural_pass"] = True
        holdout_records = _strip_unselected_holdout_margins(
            holdout_all_records, selected
        )
        base_result["holdout08"] = {
            "active_margin_s": selected,
            "assessment": holdout_assessment,
            "records": holdout_records,
            "reported_candidates": ["OFF", margin_key(selected)],
        }
    else:
        base_result["holdout08"] = {
            "active_margin_s": None,
            "assessment": holdout_assessment,
            "records": None,
            "reported_candidates": ["OFF"],
        }

    holdout_pass = bool(holdout_assessment.get("pass"))
    development_pass = bool(
        package_identity_pass
        and weight_identity_pass
        and sentinel["pass"]
        and selected is not None
        and holdout_pass
        and performance["pass"]
    )
    failure_reasons: list[str] = []
    if selected is None:
        failure_reasons.append("no active margin satisfied all R1/R2 criteria")
    if selected is not None and not holdout_pass:
        failure_reasons.extend(holdout_assessment.get("failure_reasons", []))
    if not performance["pass"]:
        failure_reasons.append("mask P95 performance threshold failed")
    base_result["development_pass"] = development_pass
    base_result["production_eligible"] = False
    base_result["decision"] = {
        "status": "qualified-for-production-review" if development_pass else "rejected",
        "development_pass": development_pass,
        "failure_reasons": sorted(set(failure_reasons)),
        "enter_production": False,
        "reason_enter_production_false": (
            "isolated experiment only; production integration, default-off switch, "
            "three-package zero-diff hashes and full regressions remain mandatory"
            if development_pass
            else "preregistered mechanism failed"
        ),
    }
    _write_json(output, _json_safe(base_result))
    return base_result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Round 05 isolated T1 zero-filled gap-mask robustness experiment"
    )
    parser.add_argument("--round1", required=True, help="R1 official archive")
    parser.add_argument("--round2", required=True, help="R2 official archive")
    parser.add_argument("--final08", required=True, help="08 official exam archive")
    parser.add_argument(
        "--manifest",
        default="deploy/production_release_manifest.json",
        help="production release manifest",
    )
    parser.add_argument(
        "--output",
        default="outputs/experiments/round05_t1_gap_mask.json",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    result = run(
        round1=Path(args.round1).resolve(),
        round2=Path(args.round2).resolve(),
        final08=Path(args.final08).resolve(),
        manifest_path=(ROOT / args.manifest).resolve()
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest),
        output=(ROOT / args.output).resolve()
        if not Path(args.output).is_absolute()
        else Path(args.output),
        device=args.device,
    )
    summary = {
        "sentinel_pass": bool(result["sentinel"]["pass"]),
        "development_margin_selected": result["development_margin_selected"],
        "holdout08_pass": bool(
            result.get("holdout08")
            and result["holdout08"].get("assessment", {}).get("pass")
        ),
        "performance_pass": bool(result["performance"]["pass"]),
        "development_pass": bool(result["development_pass"]),
        "decision": result["decision"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
