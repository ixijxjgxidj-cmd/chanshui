#!/usr/bin/env python3
"""SeismicXM(middle) 特征向量能否提升 T3 事件分类 —— A/B 验证脚本.

流程：对两轮官方包的 T3 样本各提两组特征：
  1) deep-1024：SeismicXM middle 前向的 hidden[:, :, 0]（README 推荐用法）；
  2) hand-60：现有 joblib baseline 同款 extract_waveform_features。
第1轮(200样本)训练，第2轮(189样本)评测，与 joblib 基线 81.5% 对比。

预处理按作者 makejit.picker.py 的约定：ENZ 通道序、逐道 demean、除以最大
绝对值；窗口取 10240 点（不足零填充，超长取以 Z 道能量峰为中心的窗）。

用法：
    python scripts/seismicxm_t3_features.py --train-zip <第1轮zip> --eval-zip <第2轮zip> \
        --weights weights/seismicxm/seismicxm.middle.pt --repo <seismicxm仓库路径>
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from phasepicker.types import ExamTask
from phasepicker.io.official_exam import scan_exam_input
from phasepicker.io.official_waveforms import read_package_answers, read_mseed_stream
from phasepicker.tasks.waveform_features import extract_waveform_features, stream_to_components

WIN = 10240


def prep_window(components: dict, default_sr: float) -> np.ndarray:
    """按作者约定拼 (3, 10240) 输入，通道序 E,N,Z。"""
    chans = []
    z_sr, z_data = components.get("Z", (default_sr, np.zeros(1)))
    z = np.asarray(z_data, dtype=np.float64)
    # 以 Z 道能量峰为中心定窗（对所有通道用同一窗口，保持对齐）
    n = z.size
    if n > WIN:
        peak = int(np.argmax(np.abs(z))) if n else 0
        start = min(max(0, peak - WIN // 2), n - WIN)
    else:
        start = 0
    for comp in ("E", "N", "Z"):
        _, data = components.get(comp, (default_sr, np.zeros(1)))
        x = np.asarray(data, dtype=np.float64).reshape(-1)
        x = np.where(np.isfinite(x), x, 0.0)
        seg = x[start:start + WIN] if x.size > WIN else x
        out = np.zeros(WIN, dtype=np.float32)
        out[: seg.size] = seg[:WIN]
        out -= out.mean()
        m = np.abs(out).max()
        out /= (m + 1e-6)
        chans.append(out)
    return np.stack(chans, axis=0)  # (3, WIN)


def extract_package(zip_path: str, model, device, batch: int = 8):
    answers = read_package_answers(zip_path, ExamTask.T3)
    samples = [s for s in scan_exam_input(zip_path) if s.task is ExamTask.T3 and s.file_id in answers]
    print(f"{os.path.basename(zip_path)}: T3 样本 {len(samples)} 个（有答案）")
    deep, hand, event_logits, y, ids = [], [], [], [], []
    buf, meta = [], []
    import time
    t0 = time.perf_counter()

    def flush():
        if not buf:
            return
        x = torch.tensor(np.stack(buf), dtype=torch.float32, device=device)
        with torch.no_grad():
            _, _, ev, _, hidden = model(x)
        deep.extend(hidden[:, :, 0].cpu().numpy())
        event_logits.extend(ev.cpu().numpy())
        buf.clear()

    for i, sample in enumerate(samples, 1):
        try:
            stream = read_mseed_stream(sample.source_path)
            components, default_sr = stream_to_components(stream)
            hand.append(extract_waveform_features(stream))
            buf.append(prep_window(components, default_sr))
            y.append(int(answers[sample.file_id].label))
            ids.append(sample.file_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {sample.file_id}: {exc!r}", file=sys.stderr)
        if len(buf) >= batch:
            flush()
        if i % 25 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}  {i/max(1e-9,time.perf_counter()-t0):.1f} 文件/秒", flush=True)
    flush()
    return (np.asarray(deep, np.float32), np.asarray(hand, np.float64),
            np.asarray(event_logits, np.float32), np.asarray(y), ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-zip", required=True)
    ap.add_argument("--eval-zip", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--repo", required=True, help="seismicxm 源码仓库根目录")
    ap.add_argument("--outdir", default="outputs/seismicxm_t3")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from seismicxm.middle import SeismicXM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeismicXM()
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()
    print(f"SeismicXM middle 加载完成，device={device}")

    os.makedirs(args.outdir, exist_ok=True)
    cache = os.path.join(args.outdir, "features.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        Xd_tr, Xh_tr, y_tr = z["Xd_tr"], z["Xh_tr"], z["y_tr"]
        Xd_ev, Xh_ev, y_ev = z["Xd_ev"], z["Xh_ev"], z["y_ev"]
        print("已从缓存加载特征")
    else:
        Xd_tr, Xh_tr, _, y_tr, _ = extract_package(args.train_zip, model, device)
        Xd_ev, Xh_ev, _, y_ev, _ = extract_package(args.eval_zip, model, device)
        np.savez_compressed(cache, Xd_tr=Xd_tr, Xh_tr=Xh_tr, y_tr=y_tr,
                            Xd_ev=Xd_ev, Xh_ev=Xh_ev, y_ev=y_ev)
        print(f"特征已缓存到 {cache}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    variants = {
        "deep1024_logreg": (Xd_tr, Xd_ev, make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))),
        "deep1024_rf": (Xd_tr, Xd_ev, RandomForestClassifier(
            n_estimators=500, random_state=0, class_weight="balanced")),
        "hand60_logreg": (Xh_tr, Xh_ev, make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000))),
        "deep+hand_logreg": (
            np.hstack([Xd_tr, Xh_tr]), np.hstack([Xd_ev, Xh_ev]),
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))),
        "deep+hand_rf": (
            np.hstack([Xd_tr, Xh_tr]), np.hstack([Xd_ev, Xh_ev]),
            RandomForestClassifier(n_estimators=500, random_state=0, class_weight="balanced")),
    }
    print(f"\n训练 {len(y_tr)} 样本 / 评测 {len(y_ev)} 样本；joblib 基线=0.8148")
    for name, (xtr, xev, clf) in variants.items():
        clf.fit(xtr, y_tr)
        pred = clf.predict(xev)
        acc = float(np.mean(pred == y_ev))
        # 混淆里最关心 1→3/4 是否被救回
        conf = {}
        for t, p in zip(y_ev, pred):
            if t != p:
                conf[f"{t}->{p}"] = conf.get(f"{t}->{p}", 0) + 1
        top_err = sorted(conf.items(), key=lambda kv: -kv[1])[:4]
        print(f"  {name:20s} acc={acc:.4f} ({int(acc*len(y_ev))}/{len(y_ev)})  主要错误: {top_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
