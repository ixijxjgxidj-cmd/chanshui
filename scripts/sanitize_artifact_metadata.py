#!/usr/bin/env python3
"""移除模型产物元数据中的机器绝对路径，同时证明模型数值内容未改变。

仅处理显式传入的 PyTorch checkpoint 或本仓 BaselineModelBundle。PyTorch 文件只
清洗顶层 args 元数据，并逐张量比较重写前后内容；joblib 文件只清洗
trained_on/metrics，并比较固定探针预测。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MACHINE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/(?:home|root|Users)/)")


def sanitize_string(value: str) -> str:
    if not MACHINE_PATH.match(value):
        return value
    normalized = value.replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1]


def sanitize_metadata(value):
    if isinstance(value, str):
        return sanitize_string(value)
    if isinstance(value, dict):
        return {key: sanitize_metadata(child) for key, child in value.items()}
    if isinstance(value, list):
        return [sanitize_metadata(child) for child in value]
    if isinstance(value, tuple):
        return tuple(sanitize_metadata(child) for child in value)
    return value


def _tensor_map(value, prefix: str = "root") -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor):
        result[prefix] = value.detach().cpu()
    elif isinstance(value, Mapping):
        for key, child in value.items():
            result.update(_tensor_map(child, f"{prefix}[{key!r}]"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.update(_tensor_map(child, f"{prefix}[{index}]"))
    return result


def _replace_atomically(temp_path: Path, target_path: Path) -> None:
    os.replace(temp_path, target_path)


def sanitize_torch_checkpoint(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "args" not in payload:
        raise ValueError(f"{path} 不是含 args 的预期 checkpoint")
    before_tensors = _tensor_map(payload)
    original_args = payload["args"]
    payload["args"] = sanitize_metadata(original_args)
    if payload["args"] == original_args:
        return 0

    temp_path = path.with_name(f".{path.name}.sanitize.tmp")
    try:
        torch.save(payload, temp_path)
        rewritten = torch.load(temp_path, map_location="cpu", weights_only=False)
        after_tensors = _tensor_map(rewritten)
        if before_tensors.keys() != after_tensors.keys():
            raise RuntimeError(f"{path} 重写前后张量键集合变化")
        for key, before in before_tensors.items():
            if not torch.equal(before, after_tensors[key]):
                raise RuntimeError(f"{path} 重写导致张量变化: {key}")
        if rewritten["args"] != payload["args"]:
            raise RuntimeError(f"{path} 清洗后的 args 未能稳定回读")
        _replace_atomically(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return len(before_tensors)


def _probe_predictions(bundle) -> np.ndarray:
    width = len(bundle.feature_names)
    probes = np.vstack(
        [
            np.zeros(width, dtype=np.float64),
            np.ones(width, dtype=np.float64),
            np.linspace(-1.0, 1.0, width, dtype=np.float64),
        ]
    )
    return np.asarray([bundle.predict_one(row) for row in probes])


def sanitize_joblib_bundle(path: Path) -> int:
    bundle = joblib.load(path)
    before_predictions = _probe_predictions(bundle)
    changed = 0
    for attribute in ("trained_on", "metrics"):
        if not hasattr(bundle, attribute):
            continue
        original = getattr(bundle, attribute)
        cleaned = sanitize_metadata(original)
        if cleaned != original:
            setattr(bundle, attribute, cleaned)
            changed += 1
    temp_path = path.with_name(f".{path.name}.sanitize.tmp")
    try:
        joblib.dump(bundle, temp_path, compress=3)
        rewritten = joblib.load(temp_path)
        after_predictions = _probe_predictions(rewritten)
        if not np.allclose(
            before_predictions,
            after_predictions,
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise RuntimeError(f"{path} 重写导致模型预测变化")
        _replace_atomically(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return changed


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--torch-checkpoint", action="append", default=[], type=Path
    )
    parser.add_argument("--joblib-bundle", action="append", default=[], type=Path)
    return parser


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    if not args.torch_checkpoint and not args.joblib_bundle:
        raise SystemExit("至少传入一个待清洗产物")
    for path in args.torch_checkpoint:
        count = sanitize_torch_checkpoint(path)
        print(f"torch metadata sanitized: {path} | tensors verified={count}")
    for path in args.joblib_bundle:
        count = sanitize_joblib_bundle(path)
        print(f"joblib metadata sanitized: {path} | fields changed={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
