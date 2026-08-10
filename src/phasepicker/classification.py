"""事件分类器（Event classifiers）——供 /classify API 与离线评测复用.

===== 背景 =====
官网记分板列出「基础分 | 震相拾取 | 地震分类 | 震级估计 | 总分」，而此前 API 没有
分类端点——若复赛按此计分该列为 0。去年 T3 资产（ExtraTrees + 60 维波形特征，
r2 留出准确率 81.5%，weights/official_r1_to_r2/t3_event_baseline.joblib）现成可用，
本模块把它包装成在线估计器。类别语义沿用去年答案：1..5 的整数
（1=天然地震 earthquake、2=爆破 explosion 等，去年 T3 答案尾缀单词与之对应）。

输出格式官方未公布：serve_api 用可配 formatter 组装 JSON，格式公布后只改外层。
与 magnitude 同一契约：输入 MagnitudeInput（waveforms + picks_per_wf 下标对齐），
输出与 waveforms 对齐的 List[List[int]]（baseline 整文件一个类别，赋给每台站）。
异常向上抛，由 API 层统一降级。
"""

from __future__ import annotations

from typing import List, Optional

from .magnitude import MagnitudeInput, _waveforms_to_pseudo_stream


class BaselineJoblibClassifier:
    """去年 T3 基线（ExtraTrees + 60 维特征）的在线封装。

    **局限（诚实边界）**：按"整文件单事件 60s 短窗"训练，对长连续多事件记录
    只能整文件出一个类别；今年真实类别体系若与去年 1..5 不同需重训。
    """

    name = "baseline"
    needs_picks = False

    def __init__(self, model_path: str):
        from .tasks.baseline_models import load_bundle

        self._bundle = load_bundle(model_path, expected_task="T3")

    def class_for_stream(self, stream) -> int:
        """单文件 → 一个类别整数（与离线 run_official_task23 同源同参）。"""
        from .tasks.waveform_features import extract_waveform_features

        return int(self._bundle.predict_one(extract_waveform_features(stream)))

    def estimate(self, inp: MagnitudeInput) -> List[List[int]]:
        if not inp.waveforms:
            return []
        stream = inp.stream if inp.stream is not None else _waveforms_to_pseudo_stream(inp.waveforms)
        cls = self.class_for_stream(stream)
        return [[cls] for _ in inp.waveforms]


class SeismicXMClassifier:
    """SeismicXM(middle) 多窗 TTA 深度特征 + 余弦 kNN 的在线封装。

    历史包验收：第1轮训练→第2轮匹配集 98.94%（joblib 基线 81.5%）；
    08 包 183/205=89.3%，但现有 08 答案实际只出现标签 1..4，不能据此
    宣称第五类泛化；第1轮才含标签 1..5。见 scripts/train_seismicxm_t3.py。bundle
    由该脚本产出，内含 sklearn Pipeline
    与 encoder_weights 相对路径；编码器（51.9M Transformer，CPU 最多五窗约
    0.8s/文件）构建一次全程复用。类别语义与 baseline 相同（1..5 整数）。
    """

    name = "seismicxm"
    needs_picks = False

    def __init__(self, bundle_path: str, encoder_weights: Optional[str] = None):
        import joblib

        bundle = joblib.load(bundle_path)
        if bundle.get("task") != "T3" or "pipeline" not in bundle:
            raise ValueError(f"不是有效的 T3 seismicxm bundle：{bundle_path}")
        self._pipeline = bundle["pipeline"]
        from .tasks.seismicxm_features import SeismicXMEncoder

        self._encoder = SeismicXMEncoder(encoder_weights or bundle.get("encoder_weights"))

    def class_for_stream(self, stream) -> int:
        vec = self._encoder.encode_stream(stream)
        return int(self._pipeline.predict(vec[None])[0])

    def estimate(self, inp: MagnitudeInput) -> List[List[int]]:
        if not inp.waveforms:
            return []
        stream = inp.stream if inp.stream is not None else _waveforms_to_pseudo_stream(inp.waveforms)
        cls = self.class_for_stream(stream)
        return [[cls] for _ in inp.waveforms]


def build_classifier(kind: str, model_path: Optional[str] = None):
    """serve_api CLI --cls-model 的接线点；``off``/空 返回 None（API 层 501）。"""
    kind = (kind or "off").strip().lower()
    if kind in {"off", "none", ""}:
        return None
    if kind == "baseline":
        if not model_path:
            raise ValueError("baseline 分类模型需要 --cls-weights 指向 t3_event_baseline.joblib")
        return BaselineJoblibClassifier(model_path)
    if kind == "seismicxm":
        if not model_path:
            raise ValueError("seismicxm 分类模型需要 --cls-weights 指向 t3_seismicxm_*.joblib")
        return SeismicXMClassifier(model_path)
    raise ValueError(f"未知分类模型：{kind}（可选 seismicxm / baseline / off）")
