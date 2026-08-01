#!/usr/bin/env python3
"""训练 T2 SeismicXM 深度特征震级回归并产出可部署 joblib.

A/B 依据（2026-08-01，scripts/seismicxm_t2_features.py）：deep1024 +
StandardScaler+Ridge(alpha=30) 第1轮训练→第2轮盲测 MAE 0.621（baseline
0.817，常数 0.884）。振幅/手工特征拼接均更差，纯深度特征即最优。

产出：
  t2_seismicxm_r1.joblib   ：仅第1轮训练——验收用（r2 盲测应≈0.621）
  t2_seismicxm_r1r2.joblib ：两轮合训——部署默认
--verify 走完整部署路径（mseed→SeismicXMEncoder→pipeline）在第2轮端到端复评。

用法：
    python scripts/train_seismicxm_t2.py --verify --eval-zip "<第2轮zip>"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np


def build_pipeline():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge(alpha=30.0))


def save_bundle(path, pipeline, meta):
    joblib.dump({"task": "T2", "kind": "seismicxm-deep1024", "pipeline": pipeline, **meta}, path)
    print(f"已保存 {path}（{os.path.getsize(path)/1024:.0f} KB）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/seismicxm_t2/features_t2.npz")
    ap.add_argument("--outdir", default="weights/official_r1_to_r2")
    ap.add_argument("--weights", default="weights/seismicxm/seismicxm.middle.pt")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--eval-zip", default=None)
    args = ap.parse_args()

    z = np.load(args.features)
    Xd_tr, y_tr, Xd_ev, y_ev = z["Xd_tr"], z["y_tr"], z["Xd_ev"], z["y_ev"]
    print(f"特征载入：r1 {Xd_tr.shape} / r2 {Xd_ev.shape}")
    os.makedirs(args.outdir, exist_ok=True)
    meta = {"feature_dim": int(Xd_tr.shape[1]),
            "encoder_weights": "weights/seismicxm/seismicxm.middle.pt",
            "notes": "SeismicXM middle hidden[:,:,0] 1024维 + Ridge(a=30)；预处理见 tasks/seismicxm_features.py"}

    p1 = build_pipeline().fit(Xd_tr, y_tr)
    pred = np.clip(p1.predict(Xd_ev), 0.0, 9.9)
    mae = float(np.mean(np.abs(pred - y_ev)))
    print(f"r1训练 → r2缓存特征 MAE={mae:.4f}（A/B 基准 0.621，baseline 0.8167）")
    save_bundle(os.path.join(args.outdir, "t2_seismicxm_r1.joblib"), p1,
                {**meta, "train": "round1", "holdout_round2_mae": mae})

    p2 = build_pipeline().fit(np.vstack([Xd_tr, Xd_ev]), np.concatenate([y_tr, y_ev]))
    save_bundle(os.path.join(args.outdir, "t2_seismicxm_r1r2.joblib"), p2,
                {**meta, "train": "round1+round2", "holdout_round2_mae_of_r1_model": mae})

    if args.verify:
        if not args.eval_zip:
            ap.error("--verify 需要 --eval-zip")
        from phasepicker.types import ExamTask
        from phasepicker.io.official_exam import scan_exam_input
        from phasepicker.io.official_waveforms import read_package_answers, read_mseed_stream
        from phasepicker.tasks.seismicxm_features import SeismicXMEncoder
        from phasepicker.tasks.waveform_features import stream_to_components
        from phasepicker.tasks.seismicxm_features import prep_window

        enc = SeismicXMEncoder(args.weights)
        answers = read_package_answers(args.eval_zip, ExamTask.T2)
        samples = [s for s in scan_exam_input(args.eval_zip)
                   if s.task is ExamTask.T2 and s.file_id in answers]
        errs = []
        import time
        t0 = time.perf_counter()
        for i, s in enumerate(samples, 1):
            components, sr = stream_to_components(read_mseed_stream(s.source_path))
            vec = enc.encode_window(prep_window(components, sr))
            m = float(np.clip(p1.predict(vec[None])[0], 0.0, 9.9))
            errs.append(abs(m - float(answers[s.file_id].magnitude)))
            if i % 50 == 0 or i == len(samples):
                print(f"  端到端 {i}/{len(samples)} 当前 MAE={np.mean(errs):.4f} "
                      f"{i/max(1e-9,time.perf_counter()-t0):.1f} 文件/秒", flush=True)
        mae_e2e = float(np.mean(errs))
        diff = abs(mae_e2e - mae)
        print(f"端到端 r2 MAE：{mae_e2e:.6f}（缓存 {mae:.6f}，差 {diff:.2e}）")
        # torch CPU 多线程规约顺序不定，float32 前向存在 1e-4 量级抖动；
        # 超过 0.005（震级半个刻度的 1%）才视为预处理真的分歧
        if diff > 0.005:
            print("!! 端到端与缓存特征结果不一致——训练/推理预处理有分歧，禁止部署", file=sys.stderr)
            return 2
        print("验收通过：部署代码路径与训练特征一致（差异在浮点抖动范围内）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
