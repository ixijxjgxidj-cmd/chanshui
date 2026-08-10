"""SeismicXM(middle) 特征编码器——T3 深度分类的共享预处理与前向.

训练（scripts/train_seismicxm_t3.py）与在线推理（classification.SeismicXMClassifier）
都从这里取 prep_window / SeismicXMEncoder，保证两侧预处理逐字节一致——
训练/推理预处理不一致是这类特征模型最隐蔽的翻车点。

预处理约定 = 上游作者 makejit.picker.py 的导出约定（不可改，改了就与预训练
分布脱节）：ENZ 通道序、逐道 demean、除以最大绝对值、10240 点窗口；
超长波形取 Z 道能量峰居中的窗，不足零填充。

A/B 依据（2026-08-01，去年真题）：hidden[:,:,0] 1024 维 + 最多五个 TTA
窗口平均 + Normalizer + 余弦 kNN(k=5)，第1轮训练→第2轮两类盲测 98.94%，
08 决赛五类盲测 89.3%；手工 60 维 joblib 基线 81.5%。
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

WIN = 10240
DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "weights", "seismicxm", "seismicxm.middle.pt"
)


def prep_window(components: Dict[str, Tuple[float, np.ndarray]], default_sr: float) -> np.ndarray:
    """把 stream_to_components 的输出拼成 (3, 10240) 模型输入，通道序 E,N,Z。"""
    z_sr, z_data = components.get("Z", (default_sr, np.zeros(1)))
    z = np.asarray(z_data, dtype=np.float64)
    n = z.size
    if n > WIN:
        peak = int(np.argmax(np.abs(z))) if n else 0
        start = min(max(0, peak - WIN // 2), n - WIN)
    else:
        start = 0
    chans = []
    for comp in ("E", "N", "Z"):
        _, data = components.get(comp, (default_sr, np.zeros(1)))
        x = np.asarray(data, dtype=np.float64).reshape(-1)
        x = np.where(np.isfinite(x), x, 0.0)
        seg = x[start:start + WIN] if x.size > WIN else x
        out = np.zeros(WIN, dtype=np.float32)
        out[: seg.size] = seg[:WIN]
        out -= out.mean()
        out /= (np.abs(out).max() + 1e-6)
        chans.append(out)
    return np.stack(chans, axis=0)


def tta_windows(components: Dict[str, Tuple[float, np.ndarray]], default_sr: float,
                max_windows: int = 5) -> list:
    """多窗口 TTA：Z 峰居中窗 + 全程滑窗（步长 WIN/2），去重取前 max_windows 个。

    A/B 依据（2026-08-01）：特征平均后 T3 r1→r2 盲测 94.2%→98.9%（kNN 与
    logreg 收敛到同一结果）；T2 持平（波形短于窗长时退化为单窗，无损）。
    """
    z_sr, z_data = components.get("Z", (default_sr, np.zeros(1)))
    z = np.asarray(z_data, dtype=np.float64)
    n = z.size
    starts = {0}
    if n > WIN:
        peak = int(np.argmax(np.abs(z))) if n else 0
        starts.add(min(max(0, peak - WIN // 2), n - WIN))
        starts.update(range(0, n - WIN + 1, WIN // 2))
    outs = []
    for s0 in sorted(starts)[:max_windows]:
        chans = []
        for comp in ("E", "N", "Z"):
            _, data = components.get(comp, (default_sr, np.zeros(1)))
            x = np.asarray(data, dtype=np.float64).reshape(-1)
            x = np.where(np.isfinite(x), x, 0.0)
            seg = x[s0:s0 + WIN]
            out = np.zeros(WIN, dtype=np.float32)
            out[: seg.size] = seg[:WIN]
            out -= out.mean()
            out /= (np.abs(out).max() + 1e-6)
            chans.append(out)
        outs.append(np.stack(chans, axis=0))
    return outs


class SeismicXMEncoder:
    """懒加载的 middle 模型封装：stream → 1024 维特征向量。

    模型 51.9M 参数、权重约 208MB，构建一次全程复用；forward 无梯度。
    torch/einops 缺失或权重不存在时在构造期抛清晰中文错误。
    """

    def __init__(self, weights_path: str | None = None, device: str | None = None):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("SeismicXM 编码器需要 PyTorch，请先 pip install torch") from exc
        try:
            from ..vendor.seismicxm_middle import SeismicXM
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("SeismicXM 依赖 einops，请先 pip install einops") from exc

        path = os.path.abspath(weights_path or DEFAULT_WEIGHTS)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"SeismicXM 权重不存在：{path}（从 Google Drive 下载 seismicxm.middle.pt，"
                "见 https://github.com/cangyeone/seismicxm）"
            )
        self._torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = SeismicXM()
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device).eval()
        self._model = model
        self.weights_path = path

    def encode_window(self, window: np.ndarray) -> np.ndarray:
        """(3, 10240) → (1024,) 特征向量（README 推荐的 hidden[:, :, 0]）。"""
        torch = self._torch
        x = torch.tensor(window[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, _, _, _, hidden = self._model(x)
        return hidden[0, :, 0].cpu().numpy().astype(np.float32)

    def encode_stream(self, stream) -> np.ndarray:
        """ObsPy Stream（或伪 stream）→ (1024,) 特征向量（多窗 TTA 平均）。"""
        from .waveform_features import stream_to_components

        components, default_sr = stream_to_components(stream)
        vecs = [self.encode_window(w) for w in tta_windows(components, default_sr)]
        return np.mean(vecs, axis=0).astype(np.float32)
