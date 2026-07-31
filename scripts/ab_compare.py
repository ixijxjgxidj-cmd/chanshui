#!/usr/bin/env python3
"""A/B 对照：验证"提速开关/换权重/升级依赖"前后 **picks 是否同分**.

任何速度优化（--overlap 调小、--fp16、--compile、升级 seisbench、换微调权重）
在用于正式提交前，都必须过这道闸：对同一批输入分别跑 A、B 两套配置，
对比每个文件的 P/S 到时（数量 + 最大时间差），可选再对官方答案各打一分。

用法示例（B 相对 A 只开 fp16 与更小 overlap）：
    python scripts/ab_compare.py --input round2.zip \\
        --a "--device cuda" \\
        --b "--device cuda --fp16 --overlap 0.2" \\
        --answer-package round2.zip --limit 100

判定：
    - 数量差异文件数 = 0 且 全局最大 |Δt| <= --tol（默认 0.01s） → "可以启用 B"
    - 否则列出差异明细；若给了答案，还会打印 A/B 各自的官方评分供权衡。
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.types import ExamTask, Task1Result  # noqa: E402


# ---------------------------------------------------------------------------
# 纯逻辑：两组 Task1Result 的差异统计（单元测试覆盖）
# ---------------------------------------------------------------------------
def diff_results(
    a: Dict[str, Task1Result],
    b: Dict[str, Task1Result],
    tol: float = 0.01,
) -> dict:
    """对比 A/B 两组结果。

    Returns:
        {
          "n_files": 全部文件数,
          "count_mismatch": [(file_id, "P", nA, nB), ...],
          "max_dt": 全局最大 |Δt|（对齐比较，仅数量一致的序列）,
          "max_dt_file": 出现处,
          "over_tol": [(file_id, phase, dt), ...] 超容差的对齐差异,
          "same": 是否可判定等价,
        }
    """
    count_mismatch: List[Tuple[str, str, int, int]] = []
    over_tol: List[Tuple[str, str, float]] = []
    max_dt = 0.0
    max_dt_file = ""
    keys = sorted(set(a.keys()) | set(b.keys()))
    for fid in keys:
        ra = a.get(fid, Task1Result(file_id=fid))
        rb = b.get(fid, Task1Result(file_id=fid))
        for phase, ta, tb in (("P", ra.p_times_s, rb.p_times_s), ("S", ra.s_times_s, rb.s_times_s)):
            if len(ta) != len(tb):
                count_mismatch.append((fid, phase, len(ta), len(tb)))
                continue
            for x, y in zip(sorted(ta), sorted(tb)):
                dt = abs(float(x) - float(y))
                if dt > max_dt:
                    max_dt, max_dt_file = dt, f"{fid}:{phase}"
                if dt > tol:
                    over_tol.append((fid, phase, dt))
    return {
        "n_files": len(keys),
        "count_mismatch": count_mismatch,
        "max_dt": max_dt,
        "max_dt_file": max_dt_file,
        "over_tol": over_tol,
        "same": not count_mismatch and not over_tol,
    }


def _parse_variant(spec: str):
    """解析一套配置串（与 run_official_task1 的性能旋钮同名同义）。"""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--pretrained", default="stead")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--p-threshold", type=float, default=0.3)
    ap.add_argument("--s-threshold", type=float, default=0.3)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--file-batch", type=int, default=16)
    return ap.parse_args(shlex.split(spec))


def _run_variant(tag: str, spec: str, samples, load_fn) -> Tuple[Dict[str, Task1Result], float]:
    from run_official_task1 import _make_picker  # 复用同一构建逻辑，保证口径一致
    from phasepicker.tasks.task1_runner import run_task1_samples_fast

    v = _parse_variant(spec)
    print(f"[{tag}] 配置: {spec.strip() or '(全默认)'}")
    picker = _make_picker(
        v.weights, v.device, v.p_threshold, v.s_threshold, v.pretrained,
        use_fp16=v.fp16, num_threads=v.threads, batch_size=v.batch_size,
        overlap=v.overlap, compile_model=v.compile,
    )
    t0 = time.perf_counter()
    results = run_task1_samples_fast(
        samples, load_fn, picker, io_workers=v.workers, file_batch=v.file_batch
    )
    dt = time.perf_counter() - t0
    print(f"[{tag}] 用时 {dt:.1f}s（{len(samples)/max(dt,1e-9):.2f} 文件/秒）")
    return results, dt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="速度开关 A/B 同分验证")
    ap.add_argument("--input", required=True, help="官方输入目录或 zip")
    ap.add_argument("--a", required=True, help='A 配置串，如 "--device cuda"')
    ap.add_argument("--b", required=True, help='B 配置串，如 "--device cuda --fp16"')
    ap.add_argument("--tol", type=float, default=0.01, help="到时等价容差（秒）")
    ap.add_argument("--limit", type=int, default=None, help="只对比前 N 个文件")
    ap.add_argument("--answer-package", default=None, help="可选：官方 zip，附带各打一分")
    args = ap.parse_args(argv)

    from phasepicker.io.official_exam import scan_exam_input
    from run_official_task1 import _make_load_waveforms_fn

    samples = [s for s in scan_exam_input(args.input) if s.task == ExamTask.T1]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print("没有 T1 样本", file=sys.stderr)
        return 2
    print(f"对比样本数: {len(samples)}")
    load_fn = _make_load_waveforms_fn()

    res_a, t_a = _run_variant("A", args.a, samples, load_fn)
    res_b, t_b = _run_variant("B", args.b, samples, load_fn)

    d = diff_results(res_a, res_b, tol=args.tol)
    print("-" * 64)
    print(f"文件数 {d['n_files']}  加速比 A/B = ×{t_a/max(t_b,1e-9):.2f}")
    print(f"最大 |Δt| = {d['max_dt']*1000:.1f}ms @ {d['max_dt_file'] or '-'}（容差 {args.tol*1000:.0f}ms）")
    if d["count_mismatch"]:
        print(f"数量不一致 {len(d['count_mismatch'])} 处（最伤分！），示例：")
        for fid, ph, na, nb in d["count_mismatch"][:8]:
            print(f"  - {fid} {ph}: A={na} B={nb}")
    if d["over_tol"]:
        print(f"超容差对齐差异 {len(d['over_tol'])} 处，示例：")
        for fid, ph, dt in d["over_tol"][:8]:
            print(f"  - {fid} {ph}: Δ{dt*1000:.1f}ms")

    if args.answer_package:
        from phasepicker.io.official_waveforms import read_package_answers
        from phasepicker.eval.official_eval import evaluate_task1

        answers = read_package_answers(args.answer_package, ExamTask.T1)
        for tag, res in (("A", res_a), ("B", res_b)):
            rep = evaluate_task1(res, answers)
            print(f"[{tag}] 官方规则评分：")
            print(rep.summary())

    if d["same"]:
        print("结论：A/B 同分等价 —— B 的提速开关可以启用 ✅")
        return 0
    print("结论：A/B 存在差异 —— 先看上面明细/分数再决定 ⚠️")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
