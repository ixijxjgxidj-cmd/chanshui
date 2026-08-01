#!/usr/bin/env python3
"""T2 震级回归：SeismicXM 深度特征 + 振幅统计量 A/B（第1轮训练→第2轮盲测）.

关键点：SeismicXM 输入做了 max 归一化，深度特征里**没有绝对振幅**；而震级
物理上主要由振幅决定。因此变体设计：
  hand60          ：现有 baseline 同款特征（对照，≈MAE 0.817）
  deep1024        ：纯深度特征（预期差——无振幅）
  amp12           ：仅逐道 log 振幅统计（std/peak/p2p/q99 × ZNE）
  deep+amp        ：深度特征 + 振幅
  deep+hand       ：深度特征 + 全部手工特征
回归器：Ridge 与 GradientBoosting 两档。

用法：
    python scripts/seismicxm_t2_features.py --train-zip <r1> --eval-zip <r2> \
        --weights weights/seismicxm/seismicxm.middle.pt --repo <seismicxm仓库>
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from phasepicker.types import ExamTask
from phasepicker.io.official_exam import scan_exam_input
from phasepicker.io.official_waveforms import read_package_answers, read_mseed_stream
from phasepicker.tasks.waveform_features import extract_waveform_features, stream_to_components
from phasepicker.tasks.seismicxm_features import prep_window, SeismicXMEncoder


def amp_features(components, default_sr) -> np.ndarray:
    """逐道对数振幅统计（不归一化——这是震级的物理载体）。"""
    out = []
    for comp in ("Z", "N", "E"):
        _, data = components.get(comp, (default_sr, np.zeros(1)))
        x = np.asarray(data, dtype=np.float64).reshape(-1)
        x = np.where(np.isfinite(x), x, 0.0)
        x = x - x.mean()
        eps = 1e-12
        out.extend([
            np.log10(np.std(x) + eps),
            np.log10(np.abs(x).max() + eps),
            np.log10(np.ptp(x) + eps),
            np.log10(np.percentile(np.abs(x), 99) + eps),
        ])
    return np.asarray(out)


def extract(zip_path, enc):
    answers = read_package_answers(zip_path, ExamTask.T2)
    samples = [s for s in scan_exam_input(zip_path) if s.task is ExamTask.T2 and s.file_id in answers]
    print(f"{os.path.basename(zip_path)}: T2 样本 {len(samples)}")
    deep, hand, amp, y = [], [], [], []
    t0 = time.perf_counter()
    for i, s in enumerate(samples, 1):
        try:
            stream = read_mseed_stream(s.source_path)
            components, sr = stream_to_components(stream)
            hand.append(extract_waveform_features(stream))
            amp.append(amp_features(components, sr))
            deep.append(enc.encode_window(prep_window(components, sr)))
            y.append(float(answers[s.file_id].magnitude))
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {s.file_id}: {exc!r}", file=sys.stderr)
        if i % 50 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}  {i/max(1e-9,time.perf_counter()-t0):.1f} 文件/秒", flush=True)
    return map(np.asarray, (deep, hand, amp, y))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-zip", required=True)
    ap.add_argument("--eval-zip", required=True)
    ap.add_argument("--weights", default="weights/seismicxm/seismicxm.middle.pt")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--outdir", default="outputs/seismicxm_t2")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    os.makedirs(args.outdir, exist_ok=True)
    cache = os.path.join(args.outdir, "features_t2.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        Xd_tr, Xh_tr, Xa_tr, y_tr = z["Xd_tr"], z["Xh_tr"], z["Xa_tr"], z["y_tr"]
        Xd_ev, Xh_ev, Xa_ev, y_ev = z["Xd_ev"], z["Xh_ev"], z["Xa_ev"], z["y_ev"]
        print("已从缓存加载")
    else:
        enc = SeismicXMEncoder(args.weights)
        Xd_tr, Xh_tr, Xa_tr, y_tr = extract(args.train_zip, enc)
        Xd_ev, Xh_ev, Xa_ev, y_ev = extract(args.eval_zip, enc)
        np.savez_compressed(cache, Xd_tr=Xd_tr, Xh_tr=Xh_tr, Xa_tr=Xa_tr, y_tr=y_tr,
                            Xd_ev=Xd_ev, Xh_ev=Xh_ev, Xa_ev=Xa_ev, y_ev=y_ev)
        print(f"特征缓存到 {cache}")

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    def ridge(): return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    def gbr(): return GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                learning_rate=0.05, random_state=0)

    variants = {
        "hand60": (Xh_tr, Xh_ev),
        "deep1024": (Xd_tr, Xd_ev),
        "amp12": (Xa_tr, Xa_ev),
        "deep+amp": (np.hstack([Xd_tr, Xa_tr]), np.hstack([Xd_ev, Xa_ev])),
        "deep+hand": (np.hstack([Xd_tr, Xh_tr]), np.hstack([Xd_ev, Xh_ev])),
        "hand+amp": (np.hstack([Xh_tr, Xa_tr]), np.hstack([Xh_ev, Xa_ev])),
    }
    print(f"\n训练 {len(y_tr)} / 盲测 {len(y_ev)}；baseline MAE=0.8167，全猜常数 MAE=0.8835")
    for name, (xtr, xev) in variants.items():
        for rname, mk in (("ridge", ridge), ("gbr", gbr)):
            m = mk().fit(xtr, y_tr)
            pred = np.clip(m.predict(xev), 0.0, 10.0)
            mae = float(np.mean(np.abs(pred - y_ev)))
            print(f"  {name:12s} {rname:6s} MAE={mae:.4f} maxAE={np.max(np.abs(pred-y_ev)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
