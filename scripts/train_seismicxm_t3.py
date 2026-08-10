#!/usr/bin/env python3
"""训练 T3 SeismicXM 深度特征分类器并产出可部署 joblib.

产出两个包（都含 sklearn Pipeline + 元数据）：
  - t3_seismicxm_r1.joblib   ：仅第1轮训练——用于诚实验收（第2轮盲测应≈98.9%）；
  - t3_seismicxm_r1r2.joblib ：两轮合训——部署默认（数据最大化）。

特征优先读 scripts/seismicxm_t3_features.py 的缓存（outputs/seismicxm_t3/
features.npz）；--verify 会走**完整部署代码路径**（mseed→SeismicXMEncoder→
pipeline）在第2轮上端到端复评，确保上线路径与训练特征一致。

用法：
    python scripts/train_seismicxm_t3.py --verify \
        --eval-zip "<第2轮zip>" --weights weights/seismicxm/seismicxm.middle.pt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np


def build_pipeline():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    # 2026-08-01 二次 A/B：TTA 特征 + 余弦 kNN(k=5)，r1→r2 盲测 98.94%
    # （k=3..9 高原全 ≥98.4%，与 logreg+TTA 收敛同一结果；logreg 单窗仅 94.2%）
    return make_pipeline(Normalizer(), KNeighborsClassifier(5, metric="cosine"))


def save_bundle(path: str, pipeline, meta: dict) -> None:
    joblib.dump({"task": "T3", "kind": "seismicxm-deep1024", "pipeline": pipeline, **meta}, path)
    print(f"已保存 {path}（{os.path.getsize(path)/1024:.0f} KB）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/seismicxm_t3/features_tta.npz")
    ap.add_argument("--outdir", default="weights/official_r1_to_r2")
    ap.add_argument("--weights", default="weights/seismicxm/seismicxm.middle.pt")
    ap.add_argument("--verify", action="store_true", help="用部署代码路径在第2轮上端到端复评")
    ap.add_argument("--eval-zip", default=None, help="--verify 时的第2轮官方 zip")
    args = ap.parse_args()

    z = np.load(args.features)
    if "X1t3" in z:  # TTA 缓存（多窗平均）
        Xd_tr, y_tr, Xd_ev, y_ev = z["X1t3"], z["y1t3"], z["X2t3"], z["y2t3"]
    else:  # 旧单窗缓存
        Xd_tr, y_tr, Xd_ev, y_ev = z["Xd_tr"], z["y_tr"], z["Xd_ev"], z["y_ev"]
    print(f"特征载入：r1 {Xd_tr.shape} / r2 {Xd_ev.shape}")

    os.makedirs(args.outdir, exist_ok=True)
    meta_common = {
        "feature_dim": int(Xd_tr.shape[1]),
        "encoder_weights": "weights/seismicxm/seismicxm.middle.pt",
        "notes": "SeismicXM middle hidden[:,:,0] 1024维；预处理见 tasks/seismicxm_features.py",
    }

    # 1) 验收包：仅 r1 训练，r2 上用 TTA 缓存特征打分（应复现 0.9894）
    p1 = build_pipeline().fit(Xd_tr, y_tr)
    acc_cached = float(np.mean(p1.predict(Xd_ev) == y_ev))
    print(f"r1训练 → r2缓存特征评分：{acc_cached:.4f}（A/B 基准 0.9894，joblib 基线 0.8148）")
    save_bundle(os.path.join(args.outdir, "t3_seismicxm_r1.joblib"), p1,
                {**meta_common, "train": "round1", "holdout_round2_acc": acc_cached})

    # 2) 部署包：r1+r2 合训
    p2 = build_pipeline().fit(np.vstack([Xd_tr, Xd_ev]), np.concatenate([y_tr, y_ev]))
    save_bundle(os.path.join(args.outdir, "t3_seismicxm_r1r2.joblib"), p2,
                {**meta_common, "train": "round1+round2", "holdout_round2_acc_of_r1_model": acc_cached})

    if args.verify:
        if not args.eval_zip:
            ap.error("--verify 需要 --eval-zip")
        from phasepicker.types import ExamTask
        from phasepicker.io.official_exam import scan_exam_input
        from phasepicker.io.official_waveforms import read_package_answers, read_mseed_stream
        from phasepicker.tasks.seismicxm_features import SeismicXMEncoder

        enc = SeismicXMEncoder(args.weights)
        answers = read_package_answers(args.eval_zip, ExamTask.T3)
        samples = [s for s in scan_exam_input(args.eval_zip)
                   if s.task is ExamTask.T3 and s.file_id in answers]
        correct = total = 0
        import time
        t0 = time.perf_counter()
        for i, sample in enumerate(samples, 1):
            vec = enc.encode_stream(read_mseed_stream(sample.source_path))
            pred = int(p1.predict(vec[None])[0])
            correct += int(pred == int(answers[sample.file_id].label))
            total += 1
            if i % 50 == 0 or i == len(samples):
                print(f"  端到端复评 {i}/{len(samples)}  当前 acc={correct/total:.4f}  "
                      f"{i/max(1e-9,time.perf_counter()-t0):.1f} 文件/秒", flush=True)
        acc_e2e = correct / max(1, total)
        print(f"端到端（mseed→编码器→pipeline）r2 准确率：{acc_e2e:.4f}")
        # 浮点抖动可能翻转 kNN 边界上的个别样本，容差放 1 个样本
        if abs(acc_e2e - acc_cached) > 1.5 / max(1, total):
            print("!! 端到端与缓存特征结果不一致——训练/推理预处理有分歧，禁止部署", file=sys.stderr)
            return 2
        print("验收通过：部署代码路径与训练特征完全一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
