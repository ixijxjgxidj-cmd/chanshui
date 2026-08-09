#!/usr/bin/env python3
"""概率级多模型集成拾取 A/B —— 对"未知区域"分布偏移的对冲方案.

与此前失败的"拾取级投票"（离散、稀释最强成员）不同：这里在 annotate 概率
曲线层面做逐点平均，再统一挑峰+亚采样精细化。理论依据：软集成能平均掉
单一区域权重的系统性偏差，在任何新分布上接近其最优成员——正是"今年考题
区域未知"场景要的性质。

用法：
    python scripts/ensemble_prob_t1.py --input <官方zip> \
        --members guangxi,huanan,sichuan,diting --output out.an
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from phasepicker.types import ExamTask
from phasepicker.defaults import DEFAULT_PRETRAINED
from phasepicker.inference.picker import PickerConfig, ProbEnsemblePicker
from phasepicker.io.official_exam import scan_exam_input
from phasepicker.io.official_waveforms import read_package_answers
from phasepicker.io.submission_writer import write_task1_results
from phasepicker.eval.official_eval import evaluate_task1
from phasepicker.tasks.task1_runner import run_task1_samples_fast


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--members", required=True, help="逗号分隔，如 guangxi,huanan,diting")
    ap.add_argument("--output", default="outputs/pnsn_compare/T1_ensemble.an")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    names = [x.strip() for x in args.members.split(",") if x.strip()]
    picker = ProbEnsemblePicker.from_member_names(names, PickerConfig())
    print(f"概率集成成员({len(names)}): {names}")

    samples = [s for s in scan_exam_input(args.input) if s.task is ExamTask.T1]

    from phasepicker.io.official_waveforms import read_source_bytes
    from phasepicker.io.mseed_reader import load_waveforms

    def load_fn(sample):
        return load_waveforms(read_source_bytes(sample.source_path)).waveforms

    import time
    t0 = time.perf_counter()
    results = run_task1_samples_fast(samples, load_fn, picker, io_workers=args.workers)
    print(f"推理完成 {len(samples)} 文件 {time.perf_counter()-t0:.0f}s")
    ordered = [results[s.file_id] for s in samples if s.file_id in results]
    write_task1_results(ordered, args.output)
    answers = read_package_answers(args.input, ExamTask.T1)
    report = evaluate_task1(results, answers)
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
