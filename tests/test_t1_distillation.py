from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CACHE = _load_script("cache_t1_teacher_probabilities")
DISTILL = _load_script("distill_t1_phasenet_lopo")


class _Time:
    def __init__(self, value: float):
        self.timestamp = value

    def __eq__(self, other):
        return isinstance(other, _Time) and self.timestamp == other.timestamp


class _Stats:
    def __init__(self, channel: str, start: float = 105.0, rate: float = 50.0):
        self.channel = channel
        self.starttime = _Time(start)
        self.sampling_rate = rate


class _Trace:
    def __init__(self, phase: str, data):
        self.id = f"XB.STA..PhaseNet_{phase}"
        self.stats = _Stats(f"PhaseNet_{phase}")
        self.data = np.asarray(data, dtype=np.float32)


def _annotations(nps_values):
    return [
        _Trace(phase, np.full(5, value, dtype=np.float32))
        for phase, value in zip(("N", "P", "S"), nps_values)
    ]


def test_canonical_probabilities_average_and_normalise_nps():
    probabilities, metadata = CACHE.canonical_probabilities(
        [_annotations((0.8, 0.1, 0.1)), _annotations((0.6, 0.2, 0.2))]
    )
    assert probabilities.dtype == np.float16
    assert probabilities.shape == (3, 5)
    np.testing.assert_allclose(
        probabilities.astype(np.float32)[:, 0], [0.7, 0.15, 0.15], atol=5e-4
    )
    np.testing.assert_allclose(
        probabilities.astype(np.float32).sum(axis=0), 1.0, atol=5e-4
    )
    assert metadata == {"starttime_utc": 105.0, "sampling_rate": 50.0, "samples": 5}


def test_canonical_probabilities_reject_missing_phase():
    incomplete = _annotations((0.8, 0.1, 0.1))[:2]
    with pytest.raises(RuntimeError, match="bad phases"):
        CACHE.canonical_probabilities([incomplete])


def test_private_npy_roundtrip(tmp_path):
    path = tmp_path / "record.npy"
    array = np.arange(12, dtype=np.float16).reshape(3, 4)
    CACHE.write_npy_private(path, array)
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), array)
    assert not path.with_name(path.name + ".tmp").exists()
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


def test_window_starts_use_nominal_stride_and_boundary_window():
    assert DISTILL.window_starts(0) == [0]
    assert DISTILL.window_starts(1500) == [0, 1500]
    assert DISTILL.window_starts(1501) == [0, 1500, 1501]
    starts = DISTILL.window_starts(176999)
    assert len(starts) == 119
    assert starts[-2:] == [175500, 176999]


def test_teacher_alignment_matches_phasenet_blinding():
    assert DISTILL.teacher_slice_start(0, 5.0) == 0
    assert DISTILL.teacher_slice_start(1500, 5.0) == 1500
    assert DISTILL.maximum_window_start(2575, 2551, 5.0) == 0
    assert DISTILL.maximum_window_start(180000, 179500, 5.0) == 176999


def test_hard_target_supports_multiple_p_and_s_and_sums_to_one():
    target = DISTILL.make_hard_target(
        p_times_s=[15.0, 30.0],
        s_times_s=[20.0, 40.0],
        target_start_s=5.0,
    )
    assert target.shape == (3, DISTILL.VALID_SAMPLES)
    np.testing.assert_allclose(target.sum(axis=0), 1.0, atol=1e-6)
    assert int(np.argmax(target[1, :800])) == 500
    assert int(np.argmax(target[2, :1000])) == 750
    assert target[0, 0] > 0.999


def test_reorder_channels_uses_explicit_labels():
    data = np.stack(
        [np.full(4, 1), np.full(4, 2), np.full(4, 3)], axis=0
    ).astype(np.float32)
    reordered = DISTILL.reorder_channels(data, ["N", "P", "S"], ["P", "S", "N"])
    assert reordered[:, 0].tolist() == [2.0, 3.0, 1.0]


def test_held_out_barrier_requires_only_the_two_training_packages():
    packages = {"round2": Path("r2.zip"), "final08": Path("f08.zip")}
    answers = {"round2": Path("r2.zip"), "final08": Path("a08.zip")}
    assert DISTILL._validate_package_barrier(
        "round1", packages, answers, "kd-hard"
    ) == ("final08", "round2")

    with pytest.raises(ValueError, match="held-out barrier"):
        DISTILL._validate_package_barrier(
            "round1",
            {**packages, "round1": Path("r1.zip")},
            answers,
            "kd-hard",
        )
    with pytest.raises(ValueError, match="must not receive answer"):
        DISTILL._validate_package_barrier(
            "round1", packages, answers, "kd-only"
        )


def test_batch_arrays_align_teacher_and_hard_targets():
    waveform = np.vstack(
        [
            np.linspace(0.0, 1.0, 2575, dtype=np.float32),
            np.linspace(1.0, 2.0, 2575, dtype=np.float32),
            np.linspace(2.0, 3.0, 2575, dtype=np.float32),
        ]
    )
    teacher = np.zeros((3, 2551), dtype=np.float32)
    teacher[0] = 0.8
    teacher[1] = 0.1
    teacher[2] = 0.1
    record = DISTILL.PreparedRecord(
        package="round1",
        file_id="sample.mseed",
        waveform=waveform,
        teacher=teacher,
        teacher_start_offset_s=5.0,
        p_times_s=(15.0,),
        s_times_s=(25.0,),
    )
    x, soft, hard = DISTILL._batch_arrays(
        [record], [DISTILL.WindowSpec(0, 0)], ["N", "P", "S"], True
    )
    assert x.shape == (1, 3, 3001)
    assert soft.shape == (1, 3, 2501)
    assert hard is not None and hard.shape == (1, 3, 2501)
    np.testing.assert_allclose(soft.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(hard.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(x.mean(axis=2), 0.0, atol=2e-6)
