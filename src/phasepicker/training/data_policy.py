"""Hard experiment-data boundaries shared by data preparation and training tools."""

from __future__ import annotations

from pathlib import Path


_SEALED_08_MARKERS = (
    "08-exam",
    "08_exam",
    "08-an",
    "08_an",
    "final08",
    "final_08",
)


def assert_experiment_path_allowed(path: str | Path, label: str = "data") -> None:
    if not path:
        return
    normalized = str(Path(path)).replace("\\", "/").lower()
    marker = next((item for item in _SEALED_08_MARKERS if item in normalized), None)
    if marker is not None:
        raise ValueError(
            f"{label} path contains sealed 08 final-test marker {marker!r}: {path}"
        )
