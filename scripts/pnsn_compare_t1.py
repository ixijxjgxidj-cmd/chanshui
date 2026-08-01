#!/usr/bin/env python3
"""用 seismicx-catalog-skill 自带的 PNSN v3 TorchScript 模型跑官方 T1 并打分.

目的：与本地 PhaseNet(diting) 在同一份真题、同一评分器下对比。
预处理完全按 skill 脚本 torchscript_pnsn_pick_files 的做法：
E/N/Z 按列堆叠 (npts,3)、逐道 demean、不滤波、不归一化、100Hz 原始波形。
输出行 [phase_idx, sample, conf]，0=Pg 1=Sg 2=Pn 3=Sn → Pg/Pn 记 P，Sg/Sn 记 S。

用法：
    python scripts/pnsn_compare_t1.py --input <官方zip> --model <pnsn.v3.jit> \
        --outdir outputs/pnsn_compare --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from phasepicker.types import ExamTask, Task1Result
from phasepicker.io.official_exam import scan_exam_input
from phasepicker.io.official_waveforms import read_source_bytes, read_package_answers
from phasepicker.io.mseed_reader import load_waveforms
from phasepicker.io.submission_writer import write_task1_results
from phasepicker.eval.official_eval import evaluate_task1

PHASE_MAP = {0: "P", 1: "S", 2: "P", 3: "S"}  # Pg,Sg,Pn,Sn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="官方试题 zip（含答案则同时用于打分）")
    ap.add_argument("--model", required=True, help="pnsn.v3.jit 路径")
    ap.add_argument("--outdir", default="outputs/pnsn_compare")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--thresholds", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    model = torch.jit.load(args.model, map_location=device)
    model.eval()

    samples = [s for s in scan_exam_input(args.input) if s.task == ExamTask.T1]
    print(f"扫描到 {len(samples)} 个 T1 样本，device={device}")

    # file_id -> list[(phase, rel_s, conf)]
    all_picks: dict[str, list[tuple[str, float, float]]] = {}
    n_fail = 0
    import time
    t0 = time.perf_counter()
    for i, sample in enumerate(samples, 1):
        picks: list[tuple[str, float, float]] = []
        try:
            ingest = load_waveforms(read_source_bytes(sample.source_path))
            for wf in ingest.waveforms:
                # wf.data 形状 (3,n)，通道顺序 [Z,N,E]；PNSN 要 (n,3) 列序 [E,N,Z]
                z, n_, e = (np.asarray(wf.data[k], dtype=np.float32) for k in range(3))
                cols = [c - float(np.mean(c)) for c in (e, n_, z)]
                data = np.stack(cols, axis=1).astype(np.float32)
                npts = data.shape[0]
                if npts == 0:
                    continue
                sr = float(wf.sampling_rate)
                with torch.no_grad():
                    out = model(torch.tensor(data, device=device)).detach().cpu().numpy()
                if out.size == 0:
                    continue
                if out.ndim == 1:
                    out = out.reshape(1, -1)
                for row in out:
                    if len(row) < 3:
                        continue
                    phase = PHASE_MAP.get(int(row[0]))
                    idx = int(round(float(row[1])))
                    if phase is None or idx < 0 or idx >= npts:
                        continue
                    picks.append((phase, idx / sr, float(row[2])))
        except Exception as exc:  # noqa: BLE001 —— 单文件失败不拖垮整批
            n_fail += 1
            print(f"[失败] {sample.file_id}: {exc!r}", file=sys.stderr)
        all_picks[sample.file_id] = picks
        if i % 100 == 0 or i == len(samples):
            rate = i / max(1e-9, time.perf_counter() - t0)
            print(f"进度 {i}/{len(samples)}  {rate:.1f} 文件/秒", flush=True)

    with open(os.path.join(args.outdir, "pnsn_raw_picks.json"), "w", encoding="utf-8") as f:
        json.dump(all_picks, f, ensure_ascii=False)
    n_p = sum(1 for v in all_picks.values() for ph, _, _ in v if ph == "P")
    n_s = sum(1 for v in all_picks.values() for ph, _, _ in v if ph == "S")
    confs = [c for v in all_picks.values() for _, _, c in v]
    print(f"原始拾取：P={n_p} S={n_s} 失败文件={n_fail}")
    if confs:
        qs = np.percentile(confs, [0, 25, 50, 75, 100])
        print("置信度分布 min/q25/med/q75/max:", " ".join(f"{q:.3f}" for q in qs))

    answers = read_package_answers(args.input, ExamTask.T1)

    best = None
    for th in [float(x) for x in args.thresholds.split(",")]:
        results = {}
        for fid, picks in all_picks.items():
            kept = [(ph, t, c) for ph, t, c in picks if c >= th]
            results[fid] = Task1Result(
                file_id=fid,
                p_times_s=sorted(t for ph, t, _ in kept if ph == "P"),
                s_times_s=sorted(t for ph, t, _ in kept if ph == "S"),
            )
        report = evaluate_task1(results, answers)
        mean = report.total_score / max(1, report.n_files)
        kept_n = sum(len(r.p_times_s) + len(r.s_times_s) for r in results.values())
        print(f"阈值 {th:.2f}: 均分={mean:.3f} 总分={report.total_score:.1f} 保留拾取={kept_n}")
        if best is None or mean > best[1]:
            best = (th, mean, results)

    th, mean, results = best
    ordered = [results[s.file_id] for s in samples if s.file_id in results]
    out_an = os.path.join(args.outdir, "T1_pnsn.an")
    write_task1_results(ordered, out_an)
    print(f"最优阈值 {th:.2f} 均分 {mean:.3f}，已写出 {out_an}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
