from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_t1_distillation_cpu.py"
SPEC = importlib.util.spec_from_file_location("benchmark_t1_distillation_cpu", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _record(package: str, file_id: str, duration_s: float):
    sample = MODULE.ExamSample(
        file_id=file_id,
        task=MODULE.ExamTask.T1,
        source_path=f"{package}.zip!{file_id}",
    )
    return MODULE.InventoryRecord(
        package=package,
        sample=sample,
        duration_s=duration_s,
        input_samples=int(duration_s * 100),
        sampling_rate=100.0,
        station_count=1,
    )


def test_estimate_sliding_windows_handles_short_exact_and_overlap():
    assert MODULE.estimate_sliding_windows(10.0, 60.0, 0.5) == 1
    assert MODULE.estimate_sliding_windows(60.0, 60.0, 0.5) == 1
    assert MODULE.estimate_sliding_windows(60.1, 60.0, 0.5) == 2
    assert MODULE.estimate_sliding_windows(120.0, 60.0, 0.5) == 3


def test_select_representatives_uses_package_specific_long_limits():
    inventory = []
    for package in ("round1", "round2", "final08"):
        inventory.extend(
            _record(package, f"{package}-s{i}.mseed", float(40 + i)) for i in range(5)
        )
    inventory.extend(
        [
            _record("round2", "r2-long-1.mseed", 3600.0),
            _record("round2", "r2-long-2.mseed", 3500.0),
            _record("final08", "f08-long-1.mseed", 4000.0),
            _record("final08", "f08-long-2.mseed", 3900.0),
        ]
    )
    selected = MODULE.select_representatives(
        inventory,
        short_per_package=2,
        long_threshold_s=300.0,
        round2_long_limit=2,
        final08_long_limit=1,
    )
    assert sum(item.duration_s <= 300 for item in selected) == 6
    assert {item.sample.file_id for item in selected if item.package == "round2" and item.duration_s > 300} == {
        "r2-long-1.mseed",
        "r2-long-2.mseed",
    }
    assert [
        item.sample.file_id
        for item in selected
        if item.package == "final08" and item.duration_s > 300
    ] == ["f08-long-1.mseed"]
    assert not any(item.package == "round1" and item.duration_s > 300 for item in selected)


def test_fit_length_crops_and_edge_pads_last_axis():
    short = np.array([[1.0, 2.0]], dtype=np.float32)
    padded = MODULE.fit_length(short, 4)
    assert padded.dtype == np.float32
    assert padded.tolist() == [[1.0, 2.0, 2.0, 2.0]]

    long = np.arange(10, dtype=np.float32).reshape(2, 5)
    cropped = MODULE.fit_length(long, 3)
    assert cropped.tolist() == [[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]]


def test_resample_array_preserves_component_axis():
    data = np.arange(30, dtype=np.float32).reshape(3, 10)
    resampled = MODULE._resample_array(data, source_rate=100.0, target_rate=50.0)
    assert resampled.shape == (3, 5)


def test_quantile_indices_are_deterministic_and_interior():
    assert MODULE._quantile_indices(0, 2) == []
    assert MODULE._quantile_indices(2, 2) == [0, 1]
    assert MODULE._quantile_indices(10, 2) == [3, 6]
