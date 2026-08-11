#!/usr/bin/env python3
"""Round 06: audit and falsify a local gap-aware annotation mask.

This is an isolated research script.  It reproduces the production seven-member
ensemble annotations, measures whether zero-filled gaps change P/S probability
outside a preregistered 10 second neighborhood, and only then applies a local
zero mask before normal-threshold and conditional force-pair peak extraction.

The production picker, API, defaults, release manifest, and deployment files are
not modified.  Archive paths are reduced to basename plus SHA-256 in JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from phasepicker.inference.picker import ProbEnsemblePicker  # noqa: E402
from phasepicker.postprocess.dedup import dedup_picks  # noqa: E402
from phasepicker.types import Pick, PhaseType, Waveform  # noqa: E402
from scripts import experiment_t1_gap_mask as round05  # noqa: E402


SEED = round05.SEED
ACTIVE_GUARDS_S = round05.ACTIVE_MARGINS_S
REMOTE_DISTANCE_S = round05.REMOTE_DISTANCE_S
PROBABILITY_TOLERANCE = 1e-6
REPLAY_TIME_TOLERANCE_S = 1e-6
REPLAY_CONFIDENCE_TOLERANCE = 1e-7
PERFORMANCE_MAX_P95_MS = 10.0
PERFORMANCE_SAMPLES = 400_000
PERFORMANCE_GAPS = 100
PERFORMANCE_RUNS = 200


def guard_key(guard_s: float | None) -> str:
    return "OFF" if guard_s is None else f"{float(guard_s):.1f}"


def _validate_guard(guard_s: float) -> float:
    guard = float(guard_s)
    if not math.isfinite(guard) or guard < 0:
        raise ValueError("guard_s must be finite and non-negative")
    return guard


def _epoch(value: object) -> float:
    timestamp = getattr(value, "timestamp", None)
    return float(timestamp if timestamp is not None else value)


def _phase_from_channel(channel: object) -> str | None:
    text = str(channel or "").upper()
    if text.endswith("_P"):
        return "P"
    if text.endswith("_S"):
        return "S"
    return None


def _closed_index_spans(trace, intervals: Sequence[tuple[float, float]]) -> list[tuple[int, int]]:
    """Map closed UTC intervals to half-open sample spans for one annotation trace."""

    n = int(len(trace.data))
    if n == 0:
        return []
    start = _epoch(trace.stats.starttime)
    delta = float(trace.stats.delta)
    if not math.isfinite(start) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("annotation trace has invalid starttime/delta")
    spans: list[tuple[int, int]] = []
    # Epoch floats around 1e9 have sub-microsecond representation error.  The
    # epsilon is far below the 20 ms DiTing output grid but makes exact endpoints
    # deterministic across ObsPy/Python builds.
    eps = 1e-7
    for left, right in intervals:
        lo = int(math.ceil((float(left) - start) / delta - eps))
        hi = int(math.floor((float(right) - start) / delta + eps)) + 1
        lo = max(0, min(n, lo))
        hi = max(0, min(n, hi))
        if lo < hi:
            spans.append((lo, hi))
    return spans


def mask_phase_annotations(annotations, gaps: Sequence[Sequence[float]], guard_s: float):
    """Return a copy with P/S probabilities set to zero in expanded gaps.

    With no legal gap the exact input Stream object is returned.  With legal
    gaps, ObsPy's deep ``Stream.copy`` protects the raw annotation arrays and
    metadata.  Only channels ending in ``_P`` or ``_S`` are changed.
    """

    guard = _validate_guard(guard_s)
    if not gaps:
        return annotations
    intervals = round05.normalize_gaps(gaps, guard)
    if not intervals:
        return annotations
    candidate = annotations.copy()
    for trace in candidate:
        if _phase_from_channel(getattr(trace.stats, "channel", "")) is None:
            continue
        for lo, hi in _closed_index_spans(trace, intervals):
            trace.data[lo:hi] = np.asarray(0.0, dtype=np.asarray(trace.data).dtype)
    return candidate


def _trace_metadata(trace) -> tuple[object, ...]:
    stats = trace.stats
    return (
        str(trace.id),
        str(getattr(stats, "network", "") or ""),
        str(getattr(stats, "station", "") or ""),
        str(getattr(stats, "location", "") or ""),
        str(getattr(stats, "channel", "") or ""),
        _epoch(stats.starttime),
        float(stats.delta),
        int(len(trace.data)),
        str(np.asarray(trace.data).dtype),
    )


def _trace_record(trace) -> dict[str, object]:
    return {
        "id": str(trace.id),
        "channel": str(getattr(trace.stats, "channel", "") or ""),
        "starttime_utc": _epoch(trace.stats.starttime),
        "delta_s": float(trace.stats.delta),
        "n_samples": int(len(trace.data)),
        "dtype": str(np.asarray(trace.data).dtype),
        "sha256": round05.sha256_array(trace.data),
    }


def annotation_records(annotations) -> list[dict[str, object]]:
    return [_trace_record(trace) for trace in annotations]


def annotation_hash(annotations) -> str:
    digest = hashlib.sha256()
    for trace in annotations:
        digest.update(repr(_trace_metadata(trace)).encode("utf-8"))
        digest.update(np.ascontiguousarray(trace.data).tobytes())
    return digest.hexdigest()


def validate_mask_structure(raw, candidate, gaps, guard_s: float) -> dict[str, object]:
    intervals = round05.normalize_gaps(gaps, guard_s)
    reasons: list[str] = []
    if len(raw) != len(candidate):
        reasons.append("trace count changed")
    raw_by_id = {str(trace.id): trace for trace in raw}
    candidate_by_id = {str(trace.id): trace for trace in candidate}
    if set(raw_by_id) != set(candidate_by_id):
        reasons.append("trace ids changed")
    for trace_id in sorted(set(raw_by_id) & set(candidate_by_id)):
        before = raw_by_id[trace_id]
        after = candidate_by_id[trace_id]
        if _trace_metadata(before) != _trace_metadata(after):
            reasons.append(f"metadata changed: {trace_id}")
            continue
        expected = np.array(before.data, copy=True)
        if _phase_from_channel(getattr(before.stats, "channel", "")) is not None:
            for lo, hi in _closed_index_spans(before, intervals):
                expected[lo:hi] = np.asarray(0.0, dtype=expected.dtype)
        if not np.array_equal(expected, np.asarray(after.data), equal_nan=True):
            reasons.append(f"unexpected data change: {trace_id}")
    return {"pass": not reasons, "failure_reasons": sorted(set(reasons))}


def ensemble_annotations(picker: ProbEnsemblePicker, waveform: Waveform):
    """Reproduce only the production ensemble annotation-averaging stage."""

    stream = picker._to_stream(waveform)
    streams = [stream]
    if picker._cfg.tta_polarity_flip:
        flipped = stream.copy()
        for trace in flipped:
            trace.data = -trace.data
        streams.append(flipped)
    annotations = [
        member.annotate(
            item,
            batch_size=picker._cfg.batch_size,
            overlap=picker._cfg.overlap,
        )
        for member in picker._members
        for item in streams
    ]
    if not annotations:
        raise RuntimeError("ensemble has no member annotations")

    top_n = picker._cfg.ensemble_long_top_n
    n_aug = len(streams)
    long_stations: set[str] = set()
    if top_n is not None and 0 < top_n < len(picker._members):
        for trace in stream:
            duration = float(trace.stats.endtime - trace.stats.starttime)
            if duration > picker._cfg.ensemble_long_max_duration_s:
                long_stations.add(str(trace.stats.station))

    base = annotations[0]
    for trace in base:
        n_keep = len(annotations)
        if str(trace.stats.station) in long_stations:
            n_keep = int(top_n) * n_aug
        stack = [np.asarray(trace.data, dtype=np.float64)]
        for other in annotations[1:n_keep]:
            matches = [
                item
                for item in other
                if item.id == trace.id
                and item.stats.starttime == trace.stats.starttime
                and len(item.data) == len(trace.data)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"ensemble annotation mismatch: {trace.id}@{trace.stats.starttime} "
                    f"matched {len(matches)} traces"
                )
            stack.append(np.asarray(matches[0].data, dtype=np.float64))
        trace.data = np.mean(stack, axis=0)
    return base


def _phase_annotations(picker: ProbEnsemblePicker, annotations, phase: str):
    prefix = picker._model.__class__.__name__
    return annotations.select(channel=f"{prefix}_{phase}")


def _convert_model_picks(
    picker: ProbEnsemblePicker,
    waveform: Waveform,
    model_picks: Sequence[object],
) -> list[Pick]:
    end_guard = picker._end_guard_utc(waveform)
    converted: list[Pick] = []
    for model_pick in model_picks:
        phase = picker._normalize_phase(getattr(model_pick, "phase", ""))
        if phase is None:
            continue
        time_utc = _epoch(getattr(model_pick, "peak_time"))
        if time_utc > end_guard:
            continue
        converted.append(
            Pick(
                phase=phase,
                time_utc=time_utc,
                confidence=float(getattr(model_pick, "peak_value", 0.0) or 0.0),
                station=waveform.station,
            )
        )
    return converted


def threshold_peak_sets(
    picker: ProbEnsemblePicker,
    waveform: Waveform,
    annotations,
    thresholds: Mapping[str, float],
) -> dict[str, list[Pick]]:
    result: dict[str, list[Pick]] = {}
    for phase in ("P", "S"):
        phase_annotations = _phase_annotations(picker, annotations, phase)
        model_picks = picker._model.picks_from_annotations(
            phase_annotations, float(thresholds[phase]), phase
        )
        for model_pick in model_picks:
            picker._refine_peak_inplace(model_pick, phase_annotations)
        result[phase] = _convert_model_picks(picker, waveform, model_picks)
    return result


def final_picks_from_annotations(
    picker: ProbEnsemblePicker, waveform: Waveform, annotations
) -> list[Pick]:
    stream = picker._to_stream(waveform)
    thresholds = {"P": picker._cfg.p_threshold, "S": picker._cfg.s_threshold}
    phase_annotations: dict[str, object] = {}
    normal_model_picks: dict[str, list[object]] = {}
    all_model_picks: list[object] = []
    for phase in ("P", "S"):
        selected = _phase_annotations(picker, annotations, phase)
        picks = list(
            picker._model.picks_from_annotations(
                selected, float(thresholds[phase]), phase
            )
        )
        for model_pick in picks:
            picker._refine_peak_inplace(model_pick, selected)
        phase_annotations[phase] = selected
        normal_model_picks[phase] = picks
    for phase in ("P", "S"):
        all_model_picks.extend(normal_model_picks[phase])
        all_model_picks.extend(
            picker._fallback_lowth_picks(
                phase, phase_annotations, normal_model_picks, stream
            )
        )
    converted = _convert_model_picks(
        picker,
        waveform,
        sorted(all_model_picks, key=lambda item: _epoch(getattr(item, "peak_time"))),
    )
    standard = dedup_picks(converted, picker._dedup_cfg)
    return picker._long_dedup(
        waveform, picker._filter_long_snr(waveform, standard)
    )


def _pick_replay_record(reference: Sequence[Pick], replay: Sequence[Pick]) -> dict[str, object]:
    if len(reference) != len(replay):
        return {
            "pass": False,
            "reference_count": len(reference),
            "replay_count": len(replay),
            "max_abs_time_delta_s": None,
            "max_abs_confidence_delta": None,
            "failure_reasons": ["pick count changed"],
        }
    reasons: list[str] = []
    time_deltas: list[float] = []
    confidence_deltas: list[float] = []
    for index, (left, right) in enumerate(zip(reference, replay)):
        if round05._phase_value(left.phase) != round05._phase_value(right.phase):
            reasons.append(f"phase changed at {index}")
        if left.station != right.station:
            reasons.append(f"station changed at {index}")
        time_delta = abs(float(left.time_utc) - float(right.time_utc))
        confidence_delta = abs(float(left.confidence) - float(right.confidence))
        time_deltas.append(time_delta)
        confidence_deltas.append(confidence_delta)
        if time_delta > REPLAY_TIME_TOLERANCE_S:
            reasons.append(f"time drift at {index}")
        if confidence_delta > REPLAY_CONFIDENCE_TOLERANCE:
            reasons.append(f"confidence drift at {index}")
    return {
        "pass": not reasons,
        "reference_count": len(reference),
        "replay_count": len(replay),
        "max_abs_time_delta_s": max(time_deltas, default=0.0),
        "max_abs_confidence_delta": max(confidence_deltas, default=0.0),
        "failure_reasons": sorted(set(reasons)),
    }


def _annotation_map(annotations) -> dict[str, object]:
    return {
        str(trace.id): trace
        for trace in annotations
        if _phase_from_channel(getattr(trace.stats, "channel", "")) is not None
    }


def probability_difference(
    reference,
    gapped,
    waveform: Waveform,
    gaps: Sequence[Sequence[float]],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    reference_map = _annotation_map(reference)
    gapped_map = _annotation_map(gapped)
    reasons: list[str] = []
    if set(reference_map) != set(gapped_map):
        reasons.append("P/S trace ids differ")
    expanded = round05.normalize_gaps(gaps, REMOTE_DISTANCE_S)
    real_start = float(waveform.starttime_utc)
    real_end = real_start + waveform.n_samples / waveform.sampling_rate
    per_trace: list[dict[str, object]] = []
    totals = {
        "remote_samples": 0,
        "remote_count_gt_1e6": 0,
        "remote_normal_threshold_crossing_count": 0,
        "remote_floor_crossing_count": 0,
    }
    weighted_abs_sum = 0.0
    remote_max = 0.0
    for trace_id in sorted(set(reference_map) & set(gapped_map)):
        left = reference_map[trace_id]
        right = gapped_map[trace_id]
        if _trace_metadata(left) != _trace_metadata(right):
            reasons.append(f"trace alignment changed: {trace_id}")
            continue
        left_data = np.asarray(left.data, dtype=np.float64)
        right_data = np.asarray(right.data, dtype=np.float64)
        start = _epoch(left.stats.starttime)
        delta = float(left.stats.delta)
        times = start + np.arange(len(left_data), dtype=np.float64) * delta
        remote = (times >= real_start) & (times < real_end)
        for gap_start, gap_end in expanded:
            remote &= ~((times >= gap_start) & (times <= gap_end))
        absolute = np.abs(right_data - left_data)
        selected = absolute[remote]
        phase = _phase_from_channel(getattr(left.stats, "channel", ""))
        normal_threshold = float(thresholds[str(phase)])
        count = int(np.count_nonzero(remote))
        count_gt = int(np.count_nonzero(selected > PROBABILITY_TOLERANCE))
        normal_crossings = int(
            np.count_nonzero(
                ((left_data >= normal_threshold) != (right_data >= normal_threshold)) & remote
            )
        )
        floor_crossings = int(
            np.count_nonzero(((left_data >= 0.03) != (right_data >= 0.03)) & remote)
        )
        trace_max = float(np.max(selected)) if count else 0.0
        trace_mean = float(np.mean(selected)) if count else 0.0
        totals["remote_samples"] += count
        totals["remote_count_gt_1e6"] += count_gt
        totals["remote_normal_threshold_crossing_count"] += normal_crossings
        totals["remote_floor_crossing_count"] += floor_crossings
        weighted_abs_sum += float(np.sum(selected))
        remote_max = max(remote_max, trace_max)
        per_trace.append(
            {
                "id": trace_id,
                "phase": phase,
                "remote_samples": count,
                "remote_max_abs_delta": trace_max,
                "remote_mean_abs_delta": trace_mean,
                "remote_count_gt_1e6": count_gt,
                "remote_normal_threshold_crossing_count": normal_crossings,
                "remote_floor_crossing_count": floor_crossings,
            }
        )
    total_remote = int(totals["remote_samples"])
    totals["remote_max_abs_delta"] = remote_max
    totals["remote_mean_abs_delta"] = (
        weighted_abs_sum / total_remote if total_remote else 0.0
    )
    remote_pass = bool(
        not reasons
        and totals["remote_count_gt_1e6"] == 0
        and totals["remote_normal_threshold_crossing_count"] == 0
        and totals["remote_floor_crossing_count"] == 0
    )
    return {
        "alignment_pass": not reasons,
        "failure_reasons": sorted(set(reasons)),
        "remote_probability_pass": remote_pass,
        "tolerance": PROBABILITY_TOLERANCE,
        "totals": totals,
        "traces": per_trace,
    }


def _flatten_peak_sets(peaks: Mapping[str, Sequence[Pick]]) -> list[Pick]:
    return sorted(
        [pick for phase in ("P", "S") for pick in peaks.get(phase, [])],
        key=lambda pick: (float(pick.time_utc), round05._phase_value(pick.phase)),
    )


def pick_change_record(
    reference: Sequence[Pick],
    other: Sequence[Pick],
    gaps: Sequence[Sequence[float]],
) -> dict[str, object]:
    matching = round05.match_pick_lists(reference, other)
    remote_induced = [
        index
        for index in matching["induced_indices"]
        if round05.distance_to_gaps(float(other[index].time_utc), gaps)
        > REMOTE_DISTANCE_S
    ]
    remote_lost = [
        index
        for index in matching["lost_indices"]
        if round05.distance_to_gaps(float(reference[index].time_utc), gaps)
        > REMOTE_DISTANCE_S
    ]
    return {
        "pass": bool(matching["pass"]),
        "reference_count": len(reference),
        "other_count": len(other),
        "matched_count": len(matching["matches"]),
        "induced_count": len(matching["induced_indices"]),
        "lost_count": len(matching["lost_indices"]),
        "remote_induced_count": len(remote_induced),
        "remote_lost_count": len(remote_lost),
        "induced": [
            {
                "phase": round05._phase_value(other[index].phase),
                "relative_time_s": float(other[index].time_utc),
                "remote": index in remote_induced,
            }
            for index in matching["induced_indices"]
        ],
        "lost": [
            {
                "phase": round05._phase_value(reference[index].phase),
                "relative_time_s": float(reference[index].time_utc),
                "remote": index in remote_lost,
            }
            for index in matching["lost_indices"]
        ],
        "matching": matching,
    }


def _relative_change_record(record: Mapping[str, object], waveform: Waveform) -> dict[str, object]:
    copied = {key: value for key, value in record.items() if key != "matching"}
    for key in ("induced", "lost"):
        copied[key] = [
            {
                **item,
                "relative_time_s": float(item["relative_time_s"])
                - float(waveform.starttime_utc),
            }
            for item in copied[key]
        ]
    return copied


def collateral_changed_count(
    reference: Sequence[Pick],
    raw_gapped: Sequence[Pick],
    candidate: Sequence[Pick],
    gaps: Sequence[Sequence[float]],
) -> int:
    raw_matching = round05.match_pick_lists(reference, raw_gapped)
    candidate_matching = round05.match_pick_lists(reference, candidate)
    candidate_reference_indices = {
        int(reference_index)
        for reference_index, _, _ in candidate_matching["matches"]
    }
    stable_reference_indices = {
        int(reference_index)
        for reference_index, raw_index, _ in raw_matching["matches"]
        if round05.distance_to_gaps(float(reference[reference_index].time_utc), gaps) > 0
        and round05.distance_to_gaps(float(raw_gapped[raw_index].time_utc), gaps) > 0
    }
    return len(stable_reference_indices - candidate_reference_indices)


def _count_inside_mask(picks: Sequence[Pick], gaps, guard_s: float) -> int:
    intervals = round05.normalize_gaps(gaps, guard_s)
    return sum(
        1
        for pick in picks
        if any(left <= float(pick.time_utc) <= right for left, right in intervals)
    )


def analyze_guard(
    *,
    picker: ProbEnsemblePicker,
    sample: round05.PreparedSample,
    injection: round05.InjectionResult,
    raw_annotations,
    reference_final: Sequence[Pick],
    raw_final: Sequence[Pick],
    stage_a_pass: bool,
    guard_s: float,
) -> dict[str, object]:
    raw_hash_before = annotation_hash(raw_annotations)
    candidate_annotations = mask_phase_annotations(
        raw_annotations, injection.gaps_utc, guard_s
    )
    raw_hash_after = annotation_hash(raw_annotations)
    structure = validate_mask_structure(
        raw_annotations, candidate_annotations, injection.gaps_utc, guard_s
    )
    if raw_hash_before != raw_hash_after:
        structure["pass"] = False
        structure["failure_reasons"] = sorted(
            set([*structure["failure_reasons"], "raw annotations mutated"])
        )
    normal = threshold_peak_sets(
        picker,
        injection.waveform,
        candidate_annotations,
        {"P": picker._cfg.p_threshold, "S": picker._cfg.s_threshold},
    )
    floor = threshold_peak_sets(
        picker,
        injection.waveform,
        candidate_annotations,
        {"P": picker._cfg.force_pair_floor, "S": picker._cfg.force_pair_floor},
    )
    normal_flat = _flatten_peak_sets(normal)
    floor_flat = _flatten_peak_sets(floor)
    final = final_picks_from_annotations(
        picker, injection.waveform, candidate_annotations
    )
    final_change = pick_change_record(reference_final, final, injection.gaps_utc)
    collateral = collateral_changed_count(
        reference_final, raw_final, final, injection.gaps_utc
    )
    normal_inside = _count_inside_mask(normal_flat, injection.gaps_utc, guard_s)
    floor_inside = _count_inside_mask(floor_flat, injection.gaps_utc, guard_s)
    reasons: list[str] = []
    if not stage_a_pass:
        reasons.append("ineligible_remote_probability")
    if not structure["pass"]:
        reasons.append("mask_structure_failed")
    if normal_inside:
        reasons.append("normal_peak_inside_mask")
    if floor_inside:
        reasons.append("floor_peak_inside_mask")
    if final_change["induced_count"]:
        reasons.append("residual_induced_new")
    if final_change["lost_count"]:
        reasons.append("lost_reference")
    if collateral:
        reasons.append("collateral_changed")
    eligible = not reasons
    return {
        "guard_s": float(guard_s),
        "eligible": eligible,
        "failure_reasons": sorted(set(reasons)),
        "structure": structure,
        "normal_peak_inside_mask_count": normal_inside,
        "floor_peak_inside_mask_count": floor_inside,
        "normal_peak_count": len(normal_flat),
        "floor_peak_count": len(floor_flat),
        "final_picks": round05._relative_pick_records(final, sample.waveform),
        "final_change": _relative_change_record(final_change, sample.waveform),
        "collateral_changed_count": collateral,
        "annotation_sha256": annotation_hash(candidate_annotations),
    }


def audit_no_gap_annotations(
    *,
    picker: ProbEnsemblePicker,
    waveform: Waveform,
    annotations,
    reference_final: Sequence[Pick],
    guard_s: float,
) -> dict[str, object]:
    """Verify the no-gap fast path is an exact identity at runtime."""

    raw_hash_before = annotation_hash(annotations)
    candidate = mask_phase_annotations(annotations, [], guard_s)
    replay = final_picks_from_annotations(picker, waveform, candidate)
    raw_hash_after = annotation_hash(annotations)
    structure = validate_mask_structure(annotations, candidate, [], guard_s)
    pick_replay = _pick_replay_record(reference_final, replay)
    same_stream_object = candidate is annotations
    annotation_unchanged = raw_hash_before == raw_hash_after
    passed = bool(
        same_stream_object
        and annotation_unchanged
        and structure["pass"]
        and pick_replay["pass"]
    )
    return {
        "guard_s": float(guard_s),
        "pass": passed,
        "same_stream_object": same_stream_object,
        "annotation_unchanged": annotation_unchanged,
        "annotation_sha256": raw_hash_after,
        "structure": structure,
        "final_pick_replay": pick_replay,
    }


def run_sample(
    *,
    picker: ProbEnsemblePicker,
    sample: round05.PreparedSample,
    guards: Sequence[float],
    direct_replay: bool,
    repeat_annotation: bool,
) -> dict[str, object]:
    reference_annotations = ensemble_annotations(picker, sample.waveform)
    reference_final = final_picks_from_annotations(
        picker, sample.waveform, reference_annotations
    )
    no_gap_identity = {
        guard_key(guard): audit_no_gap_annotations(
            picker=picker,
            waveform=sample.waveform,
            annotations=reference_annotations,
            reference_final=reference_final,
            guard_s=guard,
        )
        for guard in guards
    }
    replay = None
    if direct_replay:
        replay = _pick_replay_record(picker.pick(sample.waveform), reference_final)
    deterministic = None
    if repeat_annotation:
        repeat = ensemble_annotations(picker, sample.waveform)
        deterministic = {
            "pass": annotation_hash(reference_annotations) == annotation_hash(repeat),
            "first_sha256": annotation_hash(reference_annotations),
            "second_sha256": annotation_hash(repeat),
        }
    anchor_s, injections = round05._make_injections(sample, reference_final)
    variant_records: list[dict[str, object]] = []
    thresholds = {"P": picker._cfg.p_threshold, "S": picker._cfg.s_threshold}
    reference_normal = threshold_peak_sets(
        picker, sample.waveform, reference_annotations, thresholds
    )
    reference_floor = threshold_peak_sets(
        picker,
        sample.waveform,
        reference_annotations,
        {"P": picker._cfg.force_pair_floor, "S": picker._cfg.force_pair_floor},
    )
    for variant_id in sorted(injections):
        injection = injections[variant_id]
        print(f"[round06] {sample.key} {variant_id}", flush=True)
        gapped_annotations = ensemble_annotations(picker, injection.waveform)
        probability = probability_difference(
            reference_annotations,
            gapped_annotations,
            sample.waveform,
            injection.gaps_utc,
            thresholds,
        )
        raw_normal = threshold_peak_sets(
            picker, sample.waveform, gapped_annotations, thresholds
        )
        raw_floor = threshold_peak_sets(
            picker,
            sample.waveform,
            gapped_annotations,
            {"P": picker._cfg.force_pair_floor, "S": picker._cfg.force_pair_floor},
        )
        normal_change = pick_change_record(
            _flatten_peak_sets(reference_normal),
            _flatten_peak_sets(raw_normal),
            injection.gaps_utc,
        )
        floor_change = pick_change_record(
            _flatten_peak_sets(reference_floor),
            _flatten_peak_sets(raw_floor),
            injection.gaps_utc,
        )
        raw_final = final_picks_from_annotations(
            picker, injection.waveform, gapped_annotations
        )
        final_change = pick_change_record(
            reference_final, raw_final, injection.gaps_utc
        )
        remote_peak_pass = bool(
            normal_change["remote_induced_count"] == 0
            and normal_change["remote_lost_count"] == 0
            and floor_change["remote_induced_count"] == 0
            and floor_change["remote_lost_count"] == 0
        )
        remote_final_pass = bool(
            final_change["remote_induced_count"] == 0
            and final_change["remote_lost_count"] == 0
        )
        stage_a_pass = bool(
            probability["remote_probability_pass"]
            and remote_peak_pass
            and remote_final_pass
        )
        guard_records = {
            guard_key(guard): analyze_guard(
                picker=picker,
                sample=sample,
                injection=injection,
                raw_annotations=gapped_annotations,
                reference_final=reference_final,
                raw_final=raw_final,
                stage_a_pass=stage_a_pass,
                guard_s=guard,
            )
            for guard in guards
        }
        variant_records.append(
            {
                "variant_id": variant_id,
                "components": list(injection.components),
                "intervals_samples": [list(value) for value in injection.intervals_samples],
                "gaps_relative_s": round05._relative_gap_records(
                    injection.gaps_utc, sample.waveform
                ),
                "array_sha256": injection.array_sha256,
                "injection_identity_pass": injection.identity_pass,
                "injection_identity_reasons": list(injection.identity_reasons),
                "raw_annotation_sha256": annotation_hash(gapped_annotations),
                "probability": probability,
                "normal_peak_change": _relative_change_record(
                    normal_change, sample.waveform
                ),
                "floor_peak_change": _relative_change_record(
                    floor_change, sample.waveform
                ),
                "raw_final_picks": round05._relative_pick_records(
                    raw_final, sample.waveform
                ),
                "raw_final_change": _relative_change_record(
                    final_change, sample.waveform
                ),
                "remote_peak_pass": remote_peak_pass,
                "remote_final_pick_pass": remote_final_pass,
                "stage_a_pass": stage_a_pass,
                "guards": guard_records,
            }
        )
    return {
        "key": sample.key,
        "package_key": sample.package_key,
        "file_id": sample.file_id,
        "source_kind": sample.source_kind,
        "sampling_rate": float(sample.waveform.sampling_rate),
        "duration_s": float(sample.waveform.duration),
        "array_sha256": round05.sha256_array(sample.waveform.data),
        "crop": sample.crop,
        "ingest_warnings": list(sample.ingest_warnings),
        "anchor_relative_s": anchor_s,
        "reference_annotations": annotation_records(reference_annotations),
        "reference_annotation_sha256": annotation_hash(reference_annotations),
        "reference_final_picks": round05._relative_pick_records(
            reference_final, sample.waveform
        ),
        "no_gap_identity": no_gap_identity,
        "annotation_to_production_replay": replay,
        "annotation_determinism": deterministic,
        "variants": variant_records,
    }


def aggregate_development(
    records: Sequence[Mapping[str, object]],
    guards_s: Sequence[float] = ACTIVE_GUARDS_S,
) -> dict[str, object]:
    variants = [variant for sample in records for variant in sample["variants"]]
    guard_values = tuple(float(guard) for guard in guards_s)
    if not records or not variants:
        raise ValueError("development aggregation requires records and variants")
    if len({guard_key(guard) for guard in guard_values}) != len(guard_values):
        raise ValueError("guards_s contains duplicate guard keys")
    guards: dict[str, object] = {}
    for guard in guard_values:
        key = guard_key(guard)
        candidate_records = [variant["guards"][key] for variant in variants]
        reasons = sorted(
            {
                reason
                for record in candidate_records
                for reason in record["failure_reasons"]
            }
        )
        guards[key] = {
            "guard_s": float(guard),
            "eligible": all(record["eligible"] for record in candidate_records),
            "failure_reasons": reasons,
            "residual_induced_count": sum(
                int(record["final_change"]["induced_count"])
                for record in candidate_records
            ),
            "lost_reference_count": sum(
                int(record["final_change"]["lost_count"])
                for record in candidate_records
            ),
            "collateral_changed_count": sum(
                int(record["collateral_changed_count"])
                for record in candidate_records
            ),
        }
    eligible = [
        float(guard)
        for guard in guard_values
        if guards[guard_key(guard)]["eligible"]
    ]
    probability_totals = {
        "remote_samples": 0,
        "remote_count_gt_1e6": 0,
        "remote_normal_threshold_crossing_count": 0,
        "remote_floor_crossing_count": 0,
    }
    remote_max = 0.0
    weighted_sum = 0.0
    for variant in variants:
        totals = variant["probability"]["totals"]
        count = int(totals["remote_samples"])
        for key in probability_totals:
            probability_totals[key] += int(totals[key])
        remote_max = max(remote_max, float(totals["remote_max_abs_delta"]))
        weighted_sum += float(totals["remote_mean_abs_delta"]) * count
    remote_samples = probability_totals["remote_samples"]
    probability_totals["remote_max_abs_delta"] = remote_max
    probability_totals["remote_mean_abs_delta"] = (
        weighted_sum / remote_samples if remote_samples else 0.0
    )
    raw_final = {
        "induced_count": sum(
            int(variant["raw_final_change"]["induced_count"]) for variant in variants
        ),
        "lost_count": sum(
            int(variant["raw_final_change"]["lost_count"]) for variant in variants
        ),
        "remote_induced_count": sum(
            int(variant["raw_final_change"]["remote_induced_count"])
            for variant in variants
        ),
        "remote_lost_count": sum(
            int(variant["raw_final_change"]["remote_lost_count"])
            for variant in variants
        ),
    }
    return {
        "sample_count": len(records),
        "variant_count": len(variants),
        "injection_identity_pass": all(
            variant["injection_identity_pass"] for variant in variants
        ),
        "annotation_alignment_pass": all(
            variant["probability"]["alignment_pass"] for variant in variants
        ),
        "no_gap_identity_pass": all(
            sample["no_gap_identity"][guard_key(guard)]["pass"]
            for sample in records
            for guard in guard_values
        ),
        "remote_probability_pass": all(
            variant["probability"]["remote_probability_pass"] for variant in variants
        ),
        "remote_peak_pass": all(variant["remote_peak_pass"] for variant in variants),
        "remote_final_pick_pass": all(
            variant["remote_final_pick_pass"] for variant in variants
        ),
        "stage_a_pass": all(variant["stage_a_pass"] for variant in variants),
        "probability_totals": probability_totals,
        "raw_final_totals": raw_final,
        "guards": guards,
        "selected_guard_s": min(eligible) if eligible else None,
    }


def benchmark_mask() -> dict[str, object]:
    from obspy import Stream, Trace, UTCDateTime

    start = 1_700_000_000.0
    traces = []
    for phase in ("P", "S"):
        trace = Trace(data=np.ones(PERFORMANCE_SAMPLES, dtype=np.float32))
        trace.stats.starttime = UTCDateTime(start)
        trace.stats.sampling_rate = 100.0
        trace.stats.network = "XB"
        trace.stats.station = "BENCH"
        trace.stats.channel = f"PhaseNet_{phase}"
        traces.append(trace)
    annotations = Stream(traces)
    total_duration = PERFORMANCE_SAMPLES / 100.0
    gaps = [
        (start + 20.0 + index * (total_duration - 40.0) / PERFORMANCE_GAPS,
         start + 20.5 + index * (total_duration - 40.0) / PERFORMANCE_GAPS)
        for index in range(PERFORMANCE_GAPS)
    ]
    for _ in range(10):
        mask_phase_annotations(annotations, gaps, 1.0)
    durations: list[float] = []
    last = None
    for _ in range(PERFORMANCE_RUNS):
        before = time.perf_counter()
        last = mask_phase_annotations(annotations, gaps, 1.0)
        durations.append((time.perf_counter() - before) * 1000.0)
    p95 = float(np.percentile(np.asarray(durations), 95))
    structure = validate_mask_structure(annotations, last, gaps, 1.0)
    return {
        "pass": bool(structure["pass"] and p95 < PERFORMANCE_MAX_P95_MS),
        "p95_ms": p95,
        "limit_ms": PERFORMANCE_MAX_P95_MS,
        "runs": PERFORMANCE_RUNS,
        "samples_per_phase": PERFORMANCE_SAMPLES,
        "gap_count": PERFORMANCE_GAPS,
        "structure": structure,
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


def _checkpoint_write(output: Path, payload: Mapping[str, object]) -> None:
    round05._write_json(output, round05._json_safe(payload))


def run(
    *,
    round1: Path,
    round2: Path,
    final08: Path,
    manifest_path: Path,
    output: Path,
    device: str,
) -> dict[str, object]:
    package_paths = {"round1": round1, "round2": round2, "final08": final08}
    package_records, package_pass = round05._package_identities(package_paths)
    manifest = round05._load_manifest(manifest_path)
    weight_records, weight_pass = round05._weight_identities(manifest)
    picker = round05.build_production_picker(manifest, device=device)

    base: dict[str, object] = {
        "status": "running",
        "round": 6,
        "seed": SEED,
        "git_head": _git_head(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _version("torch"),
            "obspy": _version("obspy"),
            "seisbench": _version("seisbench"),
            "device_requested": device,
            "device_actual": picker._device_type(),
        },
        "packages": package_records,
        "package_identity_pass": package_pass,
        "weights": weight_records,
        "weight_identity_pass": weight_pass,
        "ensemble_member_count": len(getattr(picker, "_members", [])),
        "guards_s": list(ACTIVE_GUARDS_S),
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "remote_distance_s": REMOTE_DISTANCE_S,
        "development": {"records": []},
        "holdout08": {"records": None},
    }
    _checkpoint_write(output, base)
    if not package_pass or not weight_pass or base["ensemble_member_count"] != 7:
        base["status"] = "structural_failure"
        base["decision"] = "aborted"
        _checkpoint_write(output, base)
        return base

    development_samples: list[round05.PreparedSample] = []
    development_samples.extend(
        round05.prepare_real_samples(
            picker=picker,
            archive=round1,
            package_key="round1",
            file_ids=round05.DEVELOPMENT_FILES["round1"],
        )
    )
    development_samples.extend(
        round05.prepare_real_samples(
            picker=picker,
            archive=round2,
            package_key="round2",
            file_ids=round05.DEVELOPMENT_FILES["round2"],
        )
    )
    development_samples.extend(round05.make_noise_samples())

    development_records: list[dict[str, object]] = []
    for sample in development_samples:
        sentinel = sample.key in round05.SENTINEL_KEYS
        record = run_sample(
            picker=picker,
            sample=sample,
            guards=ACTIVE_GUARDS_S,
            direct_replay=sentinel,
            repeat_annotation=sentinel,
        )
        development_records.append(record)
        base["development"] = {"records": development_records}
        _checkpoint_write(output, base)

    assessment = aggregate_development(development_records, ACTIVE_GUARDS_S)
    replay_records = [
        sample["annotation_to_production_replay"]
        for sample in development_records
        if sample["annotation_to_production_replay"] is not None
    ]
    determinism_records = [
        sample["annotation_determinism"]
        for sample in development_records
        if sample["annotation_determinism"] is not None
    ]
    replay_pass = bool(replay_records and all(item["pass"] for item in replay_records))
    determinism_pass = bool(
        determinism_records and all(item["pass"] for item in determinism_records)
    )
    performance = benchmark_mask()
    selected_guard = assessment["selected_guard_s"]

    holdout_record: dict[str, object] = {"records": None}
    holdout_pass = False
    if selected_guard is not None:
        holdout_samples = round05.prepare_real_samples(
            picker=picker,
            archive=final08,
            package_key="final08",
            file_ids=round05.HOLDOUT_FILES,
        )
        holdout_records = [
            run_sample(
                picker=picker,
                sample=sample,
                guards=(float(selected_guard),),
                direct_replay=sample.key in round05.SENTINEL_KEYS,
                repeat_annotation=False,
            )
            for sample in holdout_samples
        ]
        holdout_assessment = aggregate_development(
            holdout_records, (float(selected_guard),)
        )
        selected_key = guard_key(float(selected_guard))
        holdout_pass = bool(
            holdout_assessment["stage_a_pass"]
            and holdout_assessment["guards"][selected_key]["eligible"]
        )
        holdout_record = {
            "records": holdout_records,
            "assessment": holdout_assessment,
            "pass": holdout_pass,
        }

    sentinel_reasons: list[str] = []
    if not assessment["injection_identity_pass"]:
        sentinel_reasons.append("injection identity failed")
    if not assessment["annotation_alignment_pass"]:
        sentinel_reasons.append("annotation alignment failed")
    if not assessment["no_gap_identity_pass"]:
        sentinel_reasons.append("no-gap annotation identity failed")
    if not replay_pass:
        sentinel_reasons.append("annotation-to-production replay failed")
    if not determinism_pass:
        sentinel_reasons.append("annotation determinism failed")
    sentinel = round05._structural_sentinel(
        reasons=sentinel_reasons,
        package_identity_pass=package_pass,
        weight_identity_pass=weight_pass,
        picker=picker,
    )
    development_pass = bool(
        sentinel["pass"]
        and assessment["stage_a_pass"]
        and selected_guard is not None
        and replay_pass
        and determinism_pass
        and performance["pass"]
        and holdout_pass
    )
    base.update(
        {
            "status": "complete",
            "development": {
                "records": development_records,
                "assessment": assessment,
            },
            "sentinel": sentinel,
            "annotation_to_production_replay_pass": replay_pass,
            "determinism_pass": determinism_pass,
            "performance": performance,
            "development_guard_selected": (
                float(selected_guard) if selected_guard is not None else "OFF"
            ),
            "holdout08": holdout_record,
            "holdout08_pass": holdout_pass,
            "development_pass": development_pass,
            "production_eligible": development_pass,
            "decision": "adopt_candidate_for_production_review"
            if development_pass
            else "rejected",
        }
    )
    _checkpoint_write(output, base)
    return base


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Round 06 isolated T1 gap-aware annotation audit"
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
        default="outputs/experiments/round06_t1_gap_annotation.json",
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
        "sentinel_pass": bool(result.get("sentinel", {}).get("pass")),
        "development_guard_selected": result.get("development_guard_selected"),
        "remote_probability_pass": bool(
            result.get("development", {})
            .get("assessment", {})
            .get("remote_probability_pass")
        ),
        "remote_peak_pass": bool(
            result.get("development", {})
            .get("assessment", {})
            .get("remote_peak_pass")
        ),
        "remote_final_pick_pass": bool(
            result.get("development", {})
            .get("assessment", {})
            .get("remote_final_pick_pass")
        ),
        "holdout08_pass": bool(result.get("holdout08_pass")),
        "performance_pass": bool(result.get("performance", {}).get("pass")),
        "development_pass": bool(result.get("development_pass")),
        "decision": result.get("decision"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
