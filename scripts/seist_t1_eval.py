#!/usr/bin/env python3
"""SeisT(dpk, DiTing 预训练) 在官方 T1 上的 A/B 评测.

预处理按 SeisT demo_predict.py：ZNE 通道序、逐道 demean+std 归一化、50Hz。
官方 100Hz 波形先 resample_poly(1,2) 降到 50Hz。滑窗 8192 点（163.84s）、
步长 4096，重叠区概率取 max；scipy.find_peaks 提峰后换算相对秒。

用法：
    python scripts/seist_t1_eval.py --input <官方zip> --repo <SeisT仓库> \
        --ckpt <seist_l_dpk_diting.pth> --model seist_l_dpk
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from scipy.signal import resample_poly, find_peaks

from phasepicker.types import ExamTask, Task1Result
from phasepicker.io.official_exam import scan_exam_input
from phasepicker.io.official_waveforms import read_package_answers, read_source_bytes
from phasepicker.io.mseed_reader import load_waveforms
from phasepicker.eval.official_eval import evaluate_task1

WIN, STRIDE, SR = 8192, 4096, 50.0


def predict_curves(model, device, data_zne: np.ndarray) -> np.ndarray:
    """(3, n)@50Hz 已归一化 → (3, n) 概率曲线（det, P, S），滑窗取 max。"""
    n = data_zne.shape[1]
    out = np.zeros((3, n), dtype=np.float32)
    starts = list(range(0, max(1, n - WIN + 1), STRIDE))
    if not starts or starts[-1] + WIN < n:
        starts.append(max(0, n - WIN))
    batch = []
    for s in starts:
        seg = data_zne[:, s:s + WIN]
        if seg.shape[1] < WIN:
            seg = np.pad(seg, ((0, 0), (0, WIN - seg.shape[1])))
        batch.append(seg)
    x = torch.tensor(np.stack(batch), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(x).cpu().numpy()  # (B, 3, WIN)
    for i, s in enumerate(starts):
        w = min(WIN, n - s)
        out[:, s:s + w] = np.maximum(out[:, s:s + w], pred[i, :, :w])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="seist_l_dpk")
    ap.add_argument("--outdir", default="outputs/seist_t1")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from models import create_model, load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name=args.model, in_channels=3, in_samples=WIN)
    ckpt = load_checkpoint(args.ckpt, device=device)
    model.load_state_dict(ckpt["model_dict"] if "model_dict" in ckpt else ckpt)
    model.to(device).eval()
    print(f"{args.model} 加载完成 device={device}")

    answers = read_package_answers(args.input, ExamTask.T1)
    samples = [s for s in scan_exam_input(args.input) if s.task is ExamTask.T1]
    print(f"T1 样本 {len(samples)} 个")

    os.makedirs(args.outdir, exist_ok=True)
    # file_id -> (p_curvepeaks, s_curvepeaks)：存峰位置与峰值，评分时再按阈值筛
    peaks_store: dict[str, dict] = {}
    t0 = time.perf_counter()
    for i, sample in enumerate(samples, 1):
        entry = {"P": [], "S": []}
        try:
            ingest = load_waveforms(read_source_bytes(sample.source_path))
            for wf in ingest.waveforms:
                # wf.data (3,n) [Z,N,E]@原始采样率 → 50Hz + std 归一化
                sr0 = float(wf.sampling_rate)
                data = np.asarray(wf.data, dtype=np.float64)
                if abs(sr0 - SR) > 1e-6:
                    up, down = 1, int(round(sr0 / SR))
                    if abs(sr0 / SR - down) > 1e-6:
                        up, down = int(round(SR)), int(round(sr0))
                    data = resample_poly(data, up, down, axis=1)
                data = data - data.mean(axis=1, keepdims=True)
                std = data.std(axis=1, keepdims=True)
                std[std == 0] = 1
                data = (data / std).astype(np.float32)
                curves = predict_curves(model, device, data)
                for ci, ph in ((1, "P"), (2, "S")):
                    idx, props = find_peaks(curves[ci], height=0.05, distance=int(SR * 1.0))
                    entry[ph].extend((float(t / SR), float(h))
                                     for t, h in zip(idx, props["peak_heights"]))
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {sample.file_id}: {exc!r}", file=sys.stderr)
        peaks_store[sample.file_id] = entry
        if i % 100 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}  {i/max(1e-9,time.perf_counter()-t0):.1f} 文件/秒", flush=True)

    import json
    with open(os.path.join(args.outdir, f"{args.model}_peaks.json"), "w", encoding="utf-8") as f:
        json.dump(peaks_store, f)

    print("\n阈值扫描（P阈/S阈 → 均分）:")
    best = None
    for pth in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        for sth in (0.1, 0.15, 0.2, 0.3, 0.4):
            res = {fid: Task1Result(
                file_id=fid,
                p_times_s=sorted(t for t, h in e["P"] if h >= pth),
                s_times_s=sorted(t for t, h in e["S"] if h >= sth),
            ) for fid, e in peaks_store.items()}
            r = evaluate_task1(res, answers)
            mean = r.total_score / max(1, r.n_files)
            if best is None or mean > best[0]:
                best = (mean, pth, sth)
    print(f"最优: P>={best[1]} S>={best[2]} 均分={best[0]:.4f}（对照 guangxi=1.716 diting=1.669）")
    # 打印最优附近的网格
    for pth in (best[1],):
        for sth in (0.1, 0.15, 0.2, 0.3, 0.4):
            res = {fid: Task1Result(
                file_id=fid,
                p_times_s=sorted(t for t, h in e["P"] if h >= pth),
                s_times_s=sorted(t for t, h in e["S"] if h >= sth),
            ) for fid, e in peaks_store.items()}
            r = evaluate_task1(res, answers)
            print(f"  P>={pth} S>={sth}: {r.total_score/max(1,r.n_files):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
