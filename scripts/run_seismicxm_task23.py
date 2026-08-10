#!/usr/bin/env python3
"""用同一个 SeismicXM 编码器离线生成 T2/T3 冻结预测。

与旧 ``run_official_task23.py`` 的手工特征基线不同，本脚本走当前生产模型的
完整路径：MiniSEED → 多窗 TTA → SeismicXM middle 1024 维特征 → joblib
pipeline。T2/T3 共用一个编码器实例，输出高精度诊断预测和运行元数据，供
``freeze_baseline.py`` 评分。

输出默认保留 6 位小数，目的是复现模型本身的 MAE；正式官方文本若要求一位或
API 两位小数，应另外评估对应量化版本，不能把量化误差混进模型能力结论。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np

from freeze_baseline import PeakRSSSampler, quantiles, sha256_file
from phasepicker.io.official_exam import scan_exam_input
from phasepicker.io.official_waveforms import read_mseed_stream
from phasepicker.io.submission_writer import write_task2_submission, write_task3_submission
from phasepicker.tasks.seismicxm_features import SeismicXMEncoder
from phasepicker.types import ExamTask, Task2Result, Task3Result


def _load_pipeline(path: str, task: str):
    bundle = joblib.load(path)
    if bundle.get("task") != task or "pipeline" not in bundle:
        raise ValueError(f"{path} 不是有效的 {task} SeismicXM bundle")
    return bundle["pipeline"], bundle


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="生成生产 SeismicXM T2/T3 冻结预测")
    ap.add_argument("--input", required=True, help="官方输入 zip/目录")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--weights", default="weights/seismicxm/seismicxm.middle.pt")
    ap.add_argument("--t2-model", default="weights/official_r1_to_r2/t2_seismicxm_r1r2.joblib")
    ap.add_argument("--t3-model", default="weights/official_r1_to_r2/t3_seismicxm_r1r2.joblib")
    ap.add_argument("--skip-t2", action="store_true")
    ap.add_argument("--skip-t3", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--t2-ndigits", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="仅冒烟；正式冻结不要设置")
    ap.add_argument("--progress-every", type=int, default=10)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    if args.skip_t2 and args.skip_t3:
        ap.error("--skip-t2 与 --skip-t3 不能同时使用")
    for label, path in (("input", args.input), ("weights", args.weights)):
        if not os.path.exists(path):
            ap.error(f"{label} 不存在：{path}")
    if not args.skip_t2 and not os.path.isfile(args.t2_model):
        ap.error(f"T2 bundle 不存在：{args.t2_model}")
    if not args.skip_t3 and not os.path.isfile(args.t3_model):
        ap.error(f"T3 bundle 不存在：{args.t3_model}")

    device = None if args.device == "auto" else args.device
    print(f"加载 SeismicXM 编码器：{args.weights} device={device or 'auto'}", flush=True)
    encoder = SeismicXMEncoder(args.weights, device=device)
    t2_pipeline = t3_pipeline = None
    t2_bundle: Dict[str, object] = {}
    t3_bundle: Dict[str, object] = {}
    if not args.skip_t2:
        t2_pipeline, t2_bundle = _load_pipeline(args.t2_model, "T2")
    if not args.skip_t3:
        t3_pipeline, t3_bundle = _load_pipeline(args.t3_model, "T3")

    samples = scan_exam_input(args.input)
    os.makedirs(args.output_dir, exist_ok=True)
    metadata: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_basename": os.path.basename(args.input),
        "input_sha256": sha256_file(args.input) if os.path.isfile(args.input) else None,
        "encoder_sha256": sha256_file(args.weights),
        "encoder_device": str(encoder.device),
        "tasks": {},
    }

    def _run(task: ExamTask, pipeline, model_path: str):
        selected = [s for s in samples if s.task is task]
        if args.limit is not None:
            selected = selected[: args.limit]
        results: List[object] = []
        failures: List[Dict[str, str]] = []
        latencies: List[float] = []
        started = time.perf_counter()
        with PeakRSSSampler() as mem:
            for i, sample in enumerate(selected, 1):
                t0 = time.perf_counter()
                try:
                    stream = read_mseed_stream(sample.source_path)
                    vec = encoder.encode_stream(stream)
                    pred = pipeline.predict(vec[None])[0]
                    if task is ExamTask.T2:
                        results.append(
                            Task2Result(
                                file_id=sample.file_id,
                                magnitude=float(np.clip(float(pred), 0.0, 9.9)),
                            )
                        )
                    else:
                        results.append(Task3Result(file_id=sample.file_id, label=int(pred)))
                except Exception as exc:  # noqa: BLE001 - 单样本失败继续全包
                    failures.append(
                        {
                            "file_id": sample.file_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                latencies.append((time.perf_counter() - t0) * 1000.0)
                if i % max(1, args.progress_every) == 0 or i == len(selected):
                    elapsed = time.perf_counter() - started
                    print(
                        f"{task.value} {i}/{len(selected)} "
                        f"{i / max(elapsed, 1e-9):.2f} 文件/秒",
                        flush=True,
                    )
        elapsed = time.perf_counter() - started
        out_path = os.path.join(args.output_dir, f"{task.value}.pred.an")
        results = sorted(results, key=lambda x: x.file_id)
        if task is ExamTask.T2:
            write_task2_submission(results, out_path, prefix="", ndigits=args.t2_ndigits)
        else:
            write_task3_submission(results, out_path, prefix="")
        return {
            "model_basename": os.path.basename(model_path),
            "model_sha256": sha256_file(model_path),
            "bundle_train": (t2_bundle if task is ExamTask.T2 else t3_bundle).get("train"),
            "samples_requested": len(selected),
            "predictions_written": len(results),
            "failures": failures,
            "prediction_path": os.path.abspath(out_path),
            "prediction_sha256": sha256_file(out_path),
            "elapsed_s": elapsed,
            "throughput_files_per_s": len(selected) / elapsed if elapsed else None,
            "latency_ms": quantiles(latencies),
            **mem.as_dict(),
        }

    if t2_pipeline is not None:
        metadata["tasks"]["T2"] = _run(ExamTask.T2, t2_pipeline, args.t2_model)
    if t3_pipeline is not None:
        metadata["tasks"]["T3"] = _run(ExamTask.T3, t3_pipeline, args.t3_model)

    meta_path = os.path.join(args.output_dir, "seismicxm_prediction_manifest.json")
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"预测与运行清单已写出：{os.path.abspath(meta_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
