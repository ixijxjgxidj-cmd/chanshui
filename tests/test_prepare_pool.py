import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import prepare_pool as pp  # noqa: E402
from phasepicker.training.data_policy import assert_experiment_path_allowed  # noqa: E402


def test_event_hash_split_keeps_event_together():
    rows = [
        pd.Series({"source_event_id": "event-a"}),
        pd.Series({"source_event_id": "event-a"}),
        pd.Series({"source_event_id": "event-b"}),
    ]
    buckets = [pp._split_bucket(row, False, "event-hash", 0.5) for row in rows]
    assert buckets[0] == buckets[1]
    assert buckets[2] in {"train", "dev"}


def test_event_hash_split_rejects_missing_event_id():
    try:
        pp._split_bucket(pd.Series({"trace_name": "x"}), False, "event-hash", 0.1)
        assert False, "missing event id must not silently degrade to window split"
    except ValueError as exc:
        assert "event-hash" in str(exc)


def test_official_split_is_preserved():
    assert pp._split_bucket(pd.Series({"split": "train"}), True, "official", 0.1) == "train"
    assert pp._split_bucket(pd.Series({"split": "dev"}), True, "official", 0.1) == "dev"
    assert pp._split_bucket(pd.Series({"split": "test"}), True, "official", 0.1) == "dev"


def test_sp_seconds():
    assert pp._sp_seconds(100, 300, 100.0) == 2.0
    assert pp._sp_seconds(-1, 300, 100.0) is None
    assert pp._sp_seconds(300, 100, 100.0) is None


def test_sealed_08_paths_are_rejected():
    for path in (
        "/data/08-exam.zip",
        "/cache/final08/features.h5",
        r"C:\data\08_an\predictions",
    ):
        try:
            assert_experiment_path_allowed(path)
            assert False, f"sealed path must fail: {path}"
        except ValueError:
            pass
    assert_experiment_path_allowed("/data/cwa/train.h5")
