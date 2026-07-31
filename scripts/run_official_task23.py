#!/usr/bin/env python
"""用已训练模型生成官方 T2.an 与 T3.an（并行特征提取 + 整批预测版）。

性能设计（相对旧版逐样本循环）：
- 读 mseed + 提特征放进线程池并行：ObsPy 解析与 numpy FFT 都在 C 层释放 GIL，
  多线程近线性提速；官方 zip 读取配合 official_waveforms 的归档缓存，
  免去每样本重复解压整个内层 zip。
- 预测改为一次矩阵调用（BaselineModelBundle.predict_many）：几百棵树的
  ExtraTrees 逐样本 predict 的固定开销 × N 是纯浪费，合批后一次遍历完成。
行为与旧版一致：失败样本记录并跳过、输出按 file_id 排序、格式不变。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.io.official_exam import scan_exam_input  # noqa: E402
from phasepicker.io.official_waveforms import read_mseed_stream  # noqa: E402
from phasepicker.io.submission_writer import write_task2_submission, write_task3_submission  # noqa: E402
from phasepicker.tasks.baseline_models import load_bundle  # noqa: E402
from phasepicker.types import EventClass, ExamSample, ExamTask, Task2Result, Task3Result  # noqa: E402


def _extract_features_parallel(
    samples: List[ExamSample],
    workers: int,
    metadata_encoding: str,
    tag: str,
) -> Tuple[List[int], np.ndarray, List[str]]:
    """并行读波形 + 提特征。

    Returns:
        (成功样本下标列表, 特征矩阵 X[n_ok, n_feat], 失败描述列表)。
    """
    from phasepicker.tasks.waveform_features import FEATURE_NAMES, extract_waveform_features

    n = len(samples)
    feats: List[Optional[np.ndarray]] = [None] * n
    failures: List[str] = []

    def _one(i: int) -> Tuple[int, Optional[np.ndarray], Optional[str]]:
        s = samples[i]
        try:
            stream = read_mseed_stream(s.source_path, metadata_encoding)
            return i, extract_waveform_features(stream), None
        except Exception as exc:  # noqa: BLE001 —— 单样本失败不拖垮整批
            return i, None, f"{s.file_id}: {type(exc).__name__}: {exc}"

    t0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix=f"{tag}feat") as ex:
        for i, feat, err in ex.map(_one, range(n)):
            if err is not None:
                failures.append(err)
            else:
                feats[i] = feat
            done += 1
            if done % 50 == 0 or done == n:
                rate = done / max(1e-9, time.perf_counter() - t0)
                print(f"{tag} 特征 {done}/{n}（{rate:.1f} 样本/秒）")

    ok_idx = [i for i, f in enumerate(feats) if f is not None]
    X = (
        np.stack([feats[i] for i in ok_idx], axis=0)
        if ok_idx
        else np.zeros((0, len(FEATURE_NAMES)))
    )
    return ok_idx, X, failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成官方 T2/T3 提交答案")
    ap.add_argument("--input", required=True, help="官方输入目录或 zip（支持嵌套 zip）")
    ap.add_argument("--t2-model", help="t2_magnitude_baseline.joblib")
    ap.add_argument("--t3-model", help="t3_event_baseline.joblib")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--t2-prefix", default="./T2-Q/")
    ap.add_argument("--t3-prefix", default="./T3-Q/")
    ap.add_argument("--metadata-encoding", default="gbk")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="并行特征提取线程数")
    args = ap.parse_args(argv)

    if not args.t2_model and not args.t3_model:
        ap.error("--t2-model 与 --t3-model 至少给一个")
    if not os.path.exists(args.input):
        print(f"找不到输入：{args.input}", file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)
    samples = scan_exam_input(args.input, metadata_encoding=args.metadata_encoding)

    if args.t2_model:
        bundle = load_bundle(args.t2_model, expected_task="T2")
        t2_samples = [s for s in samples if s.task is ExamTask.T2]
        ok_idx, X, failures = _extract_features_parallel(
            t2_samples, args.workers, args.metadata_encoding, "T2"
        )
        results: List[Task2Result] = []
        if ok_idx:
            preds = bundle.predict_many(X)
            for i, mag in zip(ok_idx, preds):
                # 去掉树模型极少数外推异常，并限制在官方历史合理范围外留少量余量
                magnitude = min(9.9, max(0.0, float(mag)))
                results.append(Task2Result(file_id=t2_samples[i].file_id, magnitude=magnitude))
        out = os.path.join(args.output_dir, "T2.an")
        write_task2_submission(sorted(results, key=lambda r: r.file_id), out, prefix=args.t2_prefix)
        print(f"T2 写出 {len(results)} 条：{out}；失败 {len(failures)}")
        for line in failures[:5]:
            print(f"  - {line}")

    if args.t3_model:
        bundle = load_bundle(args.t3_model, expected_task="T3")
        t3_samples = [s for s in samples if s.task is ExamTask.T3]
        ok_idx, X, failures = _extract_features_parallel(
            t3_samples, args.workers, args.metadata_encoding, "T3"
        )
        results3: List[Task3Result] = []
        valid = {int(v) for v in EventClass}
        if ok_idx:
            preds = bundle.predict_many(X)
            probas = bundle.predict_proba_many(X)
            for k, (i, label) in enumerate(zip(ok_idx, preds)):
                label = int(label)
                if label not in valid:
                    failures.append(f"{t3_samples[i].file_id}: 模型输出非法 T3 类别 {label}")
                    continue
                confidence = float(np.max(probas[k])) if probas is not None else None
                results3.append(
                    Task3Result(file_id=t3_samples[i].file_id, label=label, confidence=confidence)
                )
        out = os.path.join(args.output_dir, "T3.an")
        write_task3_submission(sorted(results3, key=lambda r: r.file_id), out, prefix=args.t3_prefix)
        print(f"T3 写出 {len(results3)} 条：{out}；失败 {len(failures)}")
        for line in failures[:5]:
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
