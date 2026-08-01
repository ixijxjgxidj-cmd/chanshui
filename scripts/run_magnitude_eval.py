#!/usr/bin/env python
"""PSDelta 震级系数拟合（r1）+ r2 留出评测：psdelta vs 去年 joblib 基线.

===== 干什么 =====
1. ``--fit``：在第 1 轮真题 200 条 T2 上，用默认拾取管线（diting + defaults.py
   阈值）拿 P/S，取主事件的 (log10 幅值, log10 S-P 秒差)，最小二乘拟合
       M = a·log10(A) + b·log10(Δt_SP) + c        （有 S-P 配对）
       M = amp_a·log10(A) + amp_c                 （无配对退化式）
   并打印可直接粘回 src/phasepicker/magnitude.py::PSDeltaCoefficients 的默认值。
2. 评测：在第 2 轮真题 200 条 T2 上报三个 MAE 对比：
   baseline joblib（manifest 记载 0.817）/ psdelta / 常数基线。

这是"拟合小模型 + 真题推理评测"量级，允许本机执行（硬约束允许项）。

===== 用法 =====
    PYTHONUTF8=1 python scripts/run_magnitude_eval.py \
        --fit --train-zip "C:/.../第1轮比赛试题与答案_....zip" \
        --eval-zip "C:/.../第2轮比赛试题与答案_....zip" [--limit 50] [--device cpu]
不带 --fit 时只评测（psdelta 用 magnitude.py 里已固化的默认系数）。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.io.mseed_reader import load_waveforms  # noqa: E402
from phasepicker.io.official_exam import scan_exam_input  # noqa: E402
from phasepicker.io.official_waveforms import (  # noqa: E402
    read_mseed_stream,
    read_package_answers,
    read_source_bytes,
)
from phasepicker.magnitude import (  # noqa: E402
    BaselineJoblibMagnitude,
    EventGroup,
    PSDeltaCoefficients,
    PSDeltaMagnitude,
    group_picks_into_events,
)
from phasepicker.types import ExamTask, Waveform  # noqa: E402

DEFAULT_BASELINE_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "weights", "official_r1_to_r2", "t2_magnitude_baseline.joblib"
)


def select_primary_event(events: List[EventGroup]) -> Optional[EventGroup]:
    """单事件文件（去年 T2）的主事件选择：优先有 S-P 配对的最早事件，
    其次任何最早事件；没有拾取返回 None（走常数兜底）。"""
    paired = [e for e in events if e.sp_delta is not None]
    if paired:
        return paired[0]
    return events[0] if events else None


def _iter_t2_waveforms(zip_path: str, limit: Optional[int]):
    """逐个产出 (file_id, 单台站 Waveform 或 None)。用与 API 相同的
    load_waveforms 路径解析，保证特征取数与在线一致。"""
    samples = [s for s in scan_exam_input(zip_path) if s.task is ExamTask.T2]
    if limit:
        samples = samples[:limit]
    for s in samples:
        try:
            raw = read_source_bytes(s.source_path)
            wfs = load_waveforms(raw).waveforms
            yield s, (wfs[0] if wfs else None)
        except Exception as exc:  # noqa: BLE001 —— 单文件失败不拖垮整评
            print(f"  跳过 {s.file_id}: {type(exc).__name__}: {exc}")
            yield s, None


def _build_picker(device: str):
    from phasepicker.inference.picker import PickerConfig, SeisBenchPicker

    cfg = PickerConfig(device=None if device == "auto" else device)
    return SeisBenchPicker.from_config(cfg)


def collect_features(
    zip_path: str,
    picker,
    psd: PSDeltaMagnitude,
    limit: Optional[int],
    tag: str,
) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], Dict[str, float]]:
    """每文件跑拾取 → 主事件 (logA, logdt)。返回 (特征表, 真值表)。"""
    answers = read_package_answers(zip_path, ExamTask.T2)
    feats: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    t0 = time.perf_counter()
    n_done = 0
    for sample, wf in _iter_t2_waveforms(zip_path, limit):
        if wf is None:
            feats[sample.file_id] = (None, None)
        else:
            try:
                picks = picker.pick(wf)
            except Exception as exc:  # noqa: BLE001
                print(f"  拾取失败 {sample.file_id}: {exc}")
                picks = []
            event = select_primary_event(group_picks_into_events(picks))
            feats[sample.file_id] = (
                psd.event_features(wf, event) if event is not None else (None, None)
            )
        n_done += 1
        if n_done % 50 == 0:
            rate = n_done / (time.perf_counter() - t0)
            print(f"  {tag} {n_done} 条（{rate:.1f} 条/秒）")
    truth = {fid: r.magnitude for fid, r in answers.items() if fid in feats}
    return feats, truth


def fit_coefficients(
    feats: Dict[str, Tuple[Optional[float], Optional[float]]],
    truth: Dict[str, float],
) -> PSDeltaCoefficients:
    """最小二乘拟合两套公式 + 常数兜底。

    实测边界（2026-08 全量核实）：去年 T2 波形是"55s 噪声 + 尾部 P 起跳"的
    60s 短窗，S 波在文件末尾之外——**配对分支在官方 T2 数据上无样本可拟合**。
    此时退化为物理先验锚定：a 取幅值分支斜率（同一幅值敏感度），b=1.0
    （S-P 越大→越远→同幅值下震级越大，Δt 十倍→约 +1 级的保守先验），
    c 锚定在 Δt=10s（本地震典型 S-P）处与幅值分支等值：c = amp_c − b·log10(10)。
    今年官方样例（长连续多事件记录）到手后重跑 --fit 即可得到真拟合值。"""
    full_x, full_y, amp_x, amp_y, all_y = [], [], [], [], []
    for fid, (log_a, log_dt) in feats.items():
        if fid not in truth:
            continue
        y = truth[fid]
        all_y.append(y)
        if log_a is not None:
            amp_x.append(log_a)
            amp_y.append(y)
            if log_dt is not None:
                full_x.append((log_a, log_dt))
                full_y.append(y)

    fallback_m = float(np.mean(all_y)) if all_y else 4.5

    def _lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef

    if len(full_x) >= 3:
        X = np.column_stack([np.asarray(full_x), np.ones(len(full_x))])
        a, b, c = (float(v) for v in _lstsq(X, np.asarray(full_y)))
    else:
        a, b, c = 0.0, 0.0, fallback_m
    if len(amp_x) >= 2:
        X = np.column_stack([np.asarray(amp_x), np.ones(len(amp_x))])
        amp_a, amp_c = (float(v) for v in _lstsq(X, np.asarray(amp_y)))
    else:
        amp_a, amp_c = 0.0, fallback_m

    print(
        f"拟合样本：S-P 配对 {len(full_x)} / 有幅值 {len(amp_x)} / 总数 {len(all_y)}"
    )
    return PSDeltaCoefficients(
        a=round(a, 4), b=round(b, 4), c=round(c, 4),
        amp_a=round(amp_a, 4), amp_c=round(amp_c, 4),
        fallback_m=round(fallback_m, 3),
    )


def psdelta_predict(
    feats: Dict[str, Tuple[Optional[float], Optional[float]]],
    coef: PSDeltaCoefficients,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for fid, (log_a, log_dt) in feats.items():
        if log_a is None:
            m = coef.fallback_m
        elif log_dt is None:
            m = coef.amp_a * log_a + coef.amp_c
        else:
            m = coef.a * log_a + coef.b * log_dt + coef.c
        out[fid] = min(9.9, max(0.0, float(m)))
    return out


def mae(pred: Dict[str, float], truth: Dict[str, float]) -> Tuple[float, int]:
    errs = [abs(pred[f] - truth[f]) for f in pred if f in truth]
    return (float(np.mean(errs)) if errs else float("nan")), len(errs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PSDelta 震级拟合 + r2 留出评测")
    ap.add_argument("--train-zip", help="第 1 轮官方 zip（--fit 时必填）")
    ap.add_argument("--eval-zip", required=True, help="第 2 轮官方 zip")
    ap.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    ap.add_argument("--fit", action="store_true", help="先在 train-zip 上拟合系数")
    ap.add_argument("--limit", type=int, default=None, help="每轮最多取多少条（快速自测）")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args(argv)
    if args.fit and not args.train_zip:
        ap.error("--fit 需要 --train-zip")

    print("加载拾取模型 …", flush=True)
    picker = _build_picker(args.device)
    psd = PSDeltaMagnitude()

    coef = psd.coef
    if args.fit:
        print(f"[拟合] 扫描 {args.train_zip}")
        train_feats, train_truth = collect_features(args.train_zip, picker, psd, args.limit, "r1")
        coef = fit_coefficients(train_feats, train_truth)
        print("拟合系数（粘回 src/phasepicker/magnitude.py::PSDeltaCoefficients）：")
        print(f"  a={coef.a}, b={coef.b}, c={coef.c}")
        print(f"  amp_a={coef.amp_a}, amp_c={coef.amp_c}, fallback_m={coef.fallback_m}")
        train_mae, n = mae(psdelta_predict(train_feats, coef), train_truth)
        print(f"[拟合] r1 训练内 MAE={train_mae:.4f}（n={n}，仅供过拟合自查）")

    print(f"[评测] 扫描 {args.eval_zip}")
    eval_feats, eval_truth = collect_features(args.eval_zip, picker, psd, args.limit, "r2")
    psd_mae, n_psd = mae(psdelta_predict(eval_feats, coef), eval_truth)

    baseline = BaselineJoblibMagnitude(args.baseline_model)
    base_pred: Dict[str, float] = {}
    samples = [s for s in scan_exam_input(args.eval_zip) if s.task is ExamTask.T2]
    if args.limit:
        samples = samples[: args.limit]
    for s in samples:
        try:
            base_pred[s.file_id] = baseline.magnitude_for_stream(read_mseed_stream(s.source_path))
        except Exception as exc:  # noqa: BLE001
            print(f"  baseline 失败 {s.file_id}: {exc}")
    base_mae, n_base = mae(base_pred, eval_truth)

    const = coef.fallback_m
    const_mae, _ = mae({f: const for f in eval_truth}, eval_truth)

    print("\n===== r2 留出集 MAE 对比 =====")
    print(f"baseline joblib : {base_mae:.4f}（n={n_base}，manifest 记载 0.8167）")
    print(f"psdelta         : {psd_mae:.4f}（n={n_psd}）")
    print(f"常数 {const:.3f}     : {const_mae:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
