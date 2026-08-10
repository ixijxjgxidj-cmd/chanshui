"""震相拾取推理封装（Inference）——SeisBench 模型的统一接口.

设计目标（对应原方案"可扩展、不硬编码模型结构"）：
- 用一个抽象基类 ``BasePicker`` 定义统一契约：输入 Waveform，输出 List[Pick]。
- ``SeisBenchPicker`` 是默认实现，通过"模型名 + 权重名"从 SeisBench 加载，
  **不硬编码网络结构**。要换 PhaseNet→EQTransformer，或换成微调后的权重，
  只改配置，不改代码。
- 设备（cpu/cuda）可配置：本地用 cuda，云上若是 CPU 主机自动退化到 cpu。
  PhaseNet 很小，单条波形 CPU 推理完全可接受，这正是可用性架构的底气。

显存说明（RTX4060 8GB）：
- PhaseNet 参数量极小（~几 MB），推理显存主要由输入长度决定。
- 采用 SeisBench 的 ``annotate`` 滑窗机制处理长波形，避免一次性喂入超长序列爆显存。
- 微调时 batch_size 建议从 64 起（100Hz、30s 窗口），8GB 足够；OOM 则减半。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..types import Pick, PhaseType, Waveform, CHANNEL_ORDER
from ..utils.timing import sample_to_utc
from ..postprocess.dedup import dedup_picks, DedupConfig
from ..defaults import (
    DEFAULT_PRETRAINED,
    DEFAULT_P_MERGE_WINDOW_S,
    DEFAULT_P_THRESHOLD,
    DEFAULT_S_MERGE_WINDOW_S,
    DEFAULT_S_THRESHOLD,
)

logger = logging.getLogger(__name__)

# 末端护栏容差（秒）：_to_stream 会把短波形尾部边缘复制补齐到模型窗长，
# 补齐区触发的 pick 到时落在真实数据末尾之后，必然是错的（真值都在文件内），
# 上报即吃数量罚。超过"真实末尾 + 容差"的 pick 一律丢弃；容差吸收重采样取整。
END_GUARD_TOLERANCE_S = 0.5


@dataclass
class PickerConfig:
    """推理配置。全部可配置，供参数搜索与部署环境切换。

    Attributes:
        model_name: SeisBench 模型类名，如 "PhaseNet" / "EQTransformer"。
        pretrained: 预训练权重名，如 "original" / "stead" / "instance"，
            或指向本地微调权重的标识（见 from_config 的加载逻辑）。
        device: "cuda" 或 "cpu"。None 表示自动探测。
        p_threshold: P 波触发概率阈值。**调高偏向高精确率**（少误报），
            这是应对"数量误差每超1个扣0.5分"的关键旋钮。
        s_threshold: S 波触发概率阈值。
        batch_size: annotate 滑窗的 batch 大小，受显存限制。
        overlap: 滑窗重叠比例，边界震相不漏检。
        local_weights_path: 本地微调权重 (.pt) 路径。给定则优先加载，
            这就是"数据一到位即可切换到微调模型"的可插拔入口。
        use_fp16: CUDA 下用 autocast 半精度推理（吞吐更高、显存更省）。
            概率数值会有细微差异，正式提交前务必用本地评分脚本验证同分后再开。
        num_threads: CPU 推理线程数；None = 交给 torch 默认。云 CPU 容器里
            torch 有时探测不到全部核心，显式设满核常有 2~4 倍差距。
        compile_model: 试用 torch.compile 加速（PyTorch 2.x）。首次调用有编译
            开销、老显卡（如 sm_61）可能不支持，失败会自动回退不报错。
        p_merge_window_s: P 波去重合并窗口（秒）。None = 用 defaults.py 全局默认。
        s_merge_window_s: S 波去重合并窗口（秒）。None = 用 defaults.py 全局默认。
            这两个是 P3 网格产物接入部署的管道：调参结论只需改 defaults.py
            或在构造 PickerConfig 时显式传值，无需再手工拼 DedupConfig。
    """

    model_name: str = "PhaseNet"
    pretrained: str = DEFAULT_PRETRAINED
    device: Optional[str] = None
    p_threshold: float = DEFAULT_P_THRESHOLD
    s_threshold: float = DEFAULT_S_THRESHOLD
    batch_size: int = 256
    overlap: float = 0.5
    local_weights_path: Optional[str] = None
    use_fp16: bool = False
    num_threads: Optional[int] = None
    compile_model: bool = False
    p_merge_window_s: Optional[float] = None
    s_merge_window_s: Optional[float] = None
    #: 亚采样到时精细化：annotate 拿概率曲线后对峰做三点抛物线插值。
    #: 挑峰与 classify 逐位同源（pick 集合不变），只精化到时；False 走原 classify。
    subsample_refine: bool = True
    #: 长记录 SNR 闸：波形时长 > long_snr_min_duration_s 时，丢弃
    #: snr_db < long_snr_threshold_db 的拾取。None = 关闭。
    #: 依据（fork 合成长记录 n=24 双侧定标 + 本仓三分布复验，2026-08-09）：
    #: 阈值 -1.0 dB 时 r2 +0.010 / 08 +0.013 / r1 恰好零触发（无长记录，零副作用）。
    #: snr_db = 拾取点后 long_snr_post_s 秒 RMS / 前 long_snr_pre_s 秒 RMS（dB）；
    #: 真震相到达时幅度跃升，比值应为正；深负值 = 拾在噪声里。
    long_snr_threshold_db: Optional[float] = None
    long_snr_min_duration_s: float = 300.0
    long_snr_pre_s: float = 2.0
    long_snr_post_s: float = 2.0
    #: 短文件强制成对兜底：时长 <= force_pair_max_duration_s 的波形，若某相位在
    #: 正常阈值上零触发，则用 force_pair_floor 低阈值再挑一次、只取最高峰补发。
    #: None = 关闭（与历史行为逐位一致）。
    #: 依据（2026-08-10 残差归因 + 文献）：
    #: - 评分规则下空输出 = 0 时分 + 数量罚（floor0 读法整文件 0 分），而错拾
    #:   一对只丢时分不罚数量 → 兜底期望恒非负；
    #: - 三分布真值统计 r1 1000 / r2 915 / 08 784 全部文件 >=1P+1S，无 0 相位
    #:   文件（唯一亏钱场景在去年数据不存在）；仍留概率地板对冲今年纯噪声文件；
    #: - arXiv:2511.06731：S 峰常被压在阈值下但位置信息仍在（幅度抑制=优化陷阱），
    #:   低阈值 argmax 兜底大概率落在时分容差窗内。
    force_pair_max_duration_s: Optional[float] = None
    force_pair_floor: float = 0.03
    #: 条件式兜底（2026-08-11 因应今年"含纯噪声条目"新规）：True 时仅当同台站
    #: 另一相位在正常阈值上有触发（=确有事件的强证据）才补发本相位；纯噪声
    #: 文件两相位都无触发 → 保持空输出，对噪声条目完全免疫。
    #: 实测（三分布 56 个兜底改善文件）：条件式保留 +36.7/43.2 分（85%），
    #: 放弃的 6.5 分换掉未知数量噪声条目 ×1.0 分/个的风险敞口。
    #: SNR 闸方案已证否：真实兜底拾取 SNR 中位 0.03dB，与平稳噪声不可分。
    force_pair_conditional: bool = True
    #: 长记录事件级去重：时长 > long_dedup_min_duration_s 的波形，在标准去重
    #: （合并窗 P 1s / S 3s）之后再按更宽的窗做一次簇合并（每簇留置信度最高者）。
    #: 动机（2026-08-10 残差归因）：长连续记录上滑窗推理把同一事件的余相/包络
    #: 反复触发，r2 两个 3600s 文件多拾 ~155 个（罚 63 分）、08 五个 4000s 文件
    #: 罚 119.5 分；短文件由 cap 限额处理，长文件此前无任何计数控制。
    #: 窗口取值原则：小于最小可信事件间隔（去年长文件事件间隔中位 ~114s），
    #: 大于重复触发的散布（数秒~数十秒）；None = 关闭。
    long_dedup_p_window_s: Optional[float] = None
    long_dedup_s_window_s: Optional[float] = None
    long_dedup_min_duration_s: float = 300.0
    #: 推理端 TTA：波形极性翻转（乘 -1）后再 annotate 一遍，概率曲线并入平均。
    #: 物理安全（P 初动极性本就双向），成本 = 推理时间 ×2。仅集成路径生效。
    #: 文献：LANL 滑窗预测不一致性缓解（多视角平均）；三分布同向不劣才准上生产。
    tta_polarity_flip: bool = False
    #: 长记录只用前 N 个集成成员（None=全部）。动机（2026-08-11 g7 验收）：
    #: GEOFON 域内微调成员在 60s 事件窗上训练，短文件三分布全升，但连续长
    #: 记录上破坏 5 成员的平衡（08 五个长文件 -10.4 分）。时长门控后三分布
    #: 7成员门控相对6成员均分继续同向：r1 +0.005450 / r2 +0.004912 / 08 +0.001864；
    #: 成员列表须把"长记录可信"成员放前面。
    ensemble_long_top_n: Optional[int] = None
    ensemble_long_max_duration_s: float = 300.0


class BasePicker(ABC):
    """拾取器抽象契约。任何实现只要吃 Waveform、吐 List[Pick] 即可接入系统。"""

    @abstractmethod
    def pick(self, wf: Waveform) -> List[Pick]:
        """对单台站波形做拾取，返回去重后的震相列表。"""
        raise NotImplementedError


def _resolve_device(requested: Optional[str]) -> str:
    """决定推理设备。请求 cuda 但不可用时安全退化到 cpu 并保持可用。"""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("推理需要 PyTorch，请先安装 torch") from exc
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class SeisBenchPicker(BasePicker):
    """基于 SeisBench 的默认拾取器。"""

    def __init__(self, model, cfg: PickerConfig, dedup_cfg: Optional[DedupConfig] = None):
        """通常不直接用构造函数，用 ``SeisBenchPicker.from_config(cfg)``。

        显式传入的 dedup_cfg 优先级最高；未传时由 cfg 的合并窗字段构造，
        字段为 None 再落到 defaults.py 的全局默认。
        """
        self._model = model
        self._cfg = cfg
        if dedup_cfg is None:
            dedup_cfg = DedupConfig(
                p_merge_window_s=(
                    cfg.p_merge_window_s
                    if cfg.p_merge_window_s is not None
                    else DEFAULT_P_MERGE_WINDOW_S
                ),
                s_merge_window_s=(
                    cfg.s_merge_window_s
                    if cfg.s_merge_window_s is not None
                    else DEFAULT_S_MERGE_WINDOW_S
                ),
            )
        self._dedup_cfg = dedup_cfg

    @classmethod
    def from_config(cls, cfg: PickerConfig, dedup_cfg: Optional[DedupConfig] = None) -> "SeisBenchPicker":
        """按配置加载模型权重（预训练或本地微调），移到目标设备并置 eval。"""
        import seisbench.models as sbm
        import torch

        device = _resolve_device(cfg.device)

        model_cls = getattr(sbm, cfg.model_name, None)
        if model_cls is None:
            raise ValueError(
                f"SeisBench 中不存在模型 {cfg.model_name!r}，"
                f"可选如 'PhaseNet' / 'EQTransformer'"
            )

        # ---- 运行时加速开关（不改变数值语义的部分默认开启）----
        if device == "cuda":
            # PhaseNet 输入长度固定（滑窗 3001 点），cudnn autotune 稳赚不赔
            torch.backends.cudnn.benchmark = True
        else:
            n_threads = cfg.num_threads
            if n_threads is None:
                import os as _os

                n_threads = _os.cpu_count() or 1
            try:
                torch.set_num_threads(int(n_threads))
            except Exception:  # noqa: BLE001 - 某些运行时禁止二次设置
                pass

        if cfg.local_weights_path:
            # 微调权重加载路径：先用同一个预训练配置实例化，确保标签顺序、
            # 归一化方式和网络结构与训练时完全一致，再灌入本地 state_dict。
            model = model_cls.from_pretrained(cfg.pretrained)
            try:
                checkpoint = torch.load(
                    cfg.local_weights_path,
                    map_location=device,
                    weights_only=False,
                )
            except TypeError:  # 兼容较老 PyTorch（没有 weights_only 参数）
                checkpoint = torch.load(cfg.local_weights_path, map_location=device)
            if isinstance(checkpoint, dict):
                # 本仓库两套训练代码分别使用 model / model_state_dict；同时兼容
                # 常见第三方 checkpoint 的 state_dict 键以及裸 state_dict。
                state = checkpoint.get("model_state_dict")
                if state is None:
                    state = checkpoint.get("model")
                if state is None:
                    state = checkpoint.get("state_dict")
                if state is None:
                    state = checkpoint
            else:
                state = checkpoint
            model.load_state_dict(state)
        else:
            model = model_cls.from_pretrained(cfg.pretrained)

        model.to(device)
        model.eval()

        if cfg.compile_model:
            try:
                model = torch.compile(model)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - 老 torch / 老显卡不支持则静默回退
                pass
        return cls(model, cfg, dedup_cfg)

    # ------------------------------------------------------------------
    # 推理上下文：inference_mode（比 no_grad 更快，关闭 view 追踪）
    # + 可选 CUDA autocast fp16。集中一处，pick / pick_batch 共用。
    # ------------------------------------------------------------------
    def _classify(self, stream):
        import contextlib

        import torch

        infer_ctx = getattr(torch, "inference_mode", torch.no_grad)
        amp_ctx = contextlib.nullcontext()
        if self._cfg.use_fp16 and self._device_type() == "cuda":
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        with infer_ctx(), amp_ctx:
            # 能力探测：模型缺 annotate/picks_from_annotations（如测试桩、
            # 非 SeisBench 模型）时自动回退 classify，不因精细化功能硬失败
            refinable = (
                self._cfg.subsample_refine
                and hasattr(self._model, "annotate")
                and hasattr(self._model, "picks_from_annotations")
            )
            if not refinable:
                return self._model.classify(
                    stream,
                    batch_size=self._cfg.batch_size,
                    overlap=self._cfg.overlap,
                    P_threshold=self._cfg.p_threshold,
                    S_threshold=self._cfg.s_threshold,
                )
            return self._classify_refined(stream)

    def _classify_refined(self, stream):
        """annotate → picks_from_annotations（复刻 classify_aggregate 逐位一致）
        → 峰值三点抛物线插值做亚采样到时精细化。

        为什么不直接用 classify：它只返回峰值采样点时刻，概率曲线被丢弃。
        模型输出按 sampling_rate(diting=50Hz) 网格量化，到时天然带 ±10ms
        量化误差；抛物线插值用峰及左右邻点恢复连续极值位置，P 满分容差
        0.1s 下这部分误差占比可观。挑峰逻辑（picks_from_annotations + 逐相位
        阈值）与 SeisBench classify 内部完全同源——pick 集合不变，只精化到时。
        """
        import seisbench.util as sbu

        ann = self._model.annotate(
            stream,
            batch_size=self._cfg.batch_size,
            overlap=self._cfg.overlap,
        )
        picks = sbu.PickList()
        thresholds = {"P": self._cfg.p_threshold, "S": self._cfg.s_threshold}
        prefix = self._model.__class__.__name__
        ann_by_phase, picks_by_phase = {}, {}
        for phase, th in thresholds.items():
            phase_ann = ann.select(channel=f"{prefix}_{phase}")
            phase_picks = self._model.picks_from_annotations(phase_ann, th, phase)
            for p in phase_picks:
                self._refine_peak_inplace(p, phase_ann)
            ann_by_phase[phase] = phase_ann
            picks_by_phase[phase] = phase_picks
        for phase in thresholds:
            picks += picks_by_phase[phase]
            picks += self._fallback_lowth_picks(phase, ann_by_phase, picks_by_phase, stream)
        return sbu.PickList(sorted(picks))

    def _fallback_lowth_picks(self, phase: str, ann_by_phase, picks_by_phase, stream) -> list:
        """短文件强制成对兜底（见 PickerConfig.force_pair_max_duration_s）。

        按台站粒度工作：合批推理时一个 stream 混编多个波形（临时台站码
        B00000...），必须逐台站判断"该相位是否零触发"，整 stream 粒度在
        合批路径下几乎永不触发（2026-08-10 首版 A/B 三分布逐位无变化的教训）。
        条件式（默认）：仅当该台站另一相位在正常阈值上有触发才补发——纯噪声
        文件两相位皆无触发 → 空输出，免疫今年新规的"纯噪声条目"。
        对零触发且时长 <= 上限的台站，用 force_pair_floor 低阈值重挑并只留
        最高峰；曲线最大值仍低于地板时放弃。返回的 picks 已做亚采样精化。
        """
        max_dur = self._cfg.force_pair_max_duration_s
        phase_ann = ann_by_phase.get(phase)
        if max_dur is None or phase_ann is None or len(phase_ann) == 0 or len(stream) == 0:
            return []
        dur_by_sta: dict = {}
        for tr in stream:
            sta = tr.stats.station
            d = float(tr.stats.endtime - tr.stats.starttime)
            dur_by_sta[sta] = max(d, dur_by_sta.get(sta, 0.0))

        def stations_of(picks) -> set:
            out = set()
            for p in picks:
                parts = str(getattr(p, "trace_id", "") or "").split(".")
                if len(parts) >= 2:
                    out.add(parts[1])
            return out

        have = stations_of(picks_by_phase.get(phase, []))
        other = "S" if phase == "P" else "P"
        other_have = stations_of(picks_by_phase.get(other, []))
        out = []
        for sta, dur in dur_by_sta.items():
            if sta in have or dur > max_dur:
                continue
            if self._cfg.force_pair_conditional and sta not in other_have:
                continue
            sel = phase_ann.select(station=sta)
            if len(sel) == 0:
                continue
            low = self._model.picks_from_annotations(
                sel, self._cfg.force_pair_floor, phase
            )
            if not low:
                continue
            best = max(low, key=lambda p: float(getattr(p, "peak_value", 0.0) or 0.0))
            self._refine_peak_inplace(best, sel)
            out.append(best)
        return out

    @staticmethod
    def _refine_peak_inplace(pick, phase_ann) -> None:
        """对单个 pick 的 peak_time 做三点抛物线亚采样插值（原位修改）。

        找到覆盖该峰的概率 trace，取峰及左右邻点 (y0,y1,y2)，顶点偏移
        delta = 0.5*(y0-y2)/(y0-2*y1+y2)，截断到 [-0.5, 0.5] 个采样间隔。
        边界峰 / trace 缺失 / 分母退化时不动原值——失败即保持 classify 语义。
        """
        pid = str(getattr(pick, "trace_id", "") or "")
        best = None
        for tr in phase_ann:
            if not tr.id.startswith(pid):
                continue
            if tr.stats.starttime <= pick.peak_time <= tr.stats.endtime:
                best = tr
                break
        if best is None:
            return
        dt = float(best.stats.delta)
        if dt <= 0:
            return
        idx = int(round(float(pick.peak_time - best.stats.starttime) / dt))
        data = best.data
        if idx <= 0 or idx >= len(data) - 1:
            return
        y0, y1, y2 = float(data[idx - 1]), float(data[idx]), float(data[idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) < 1e-12:
            return
        delta = 0.5 * (y0 - y2) / denom
        delta = max(-0.5, min(0.5, delta))
        pick.peak_time = best.stats.starttime + (idx + delta) * dt

    def _device_type(self) -> str:
        try:
            return next(self._model.parameters()).device.type
        except Exception:  # noqa: BLE001
            return "cpu"

    @staticmethod
    def _end_guard_utc(wf: Waveform) -> float:
        """该波形允许的 pick 到时上限：真实数据末尾 + 容差（epoch 秒）。"""
        return wf.starttime_utc + wf.n_samples / wf.sampling_rate + END_GUARD_TOLERANCE_S

    def pick(self, wf: Waveform) -> List[Pick]:
        """对单台站波形做拾取。

        流程：Waveform → ObsPy Stream → model.annotate（滑窗概率）→
        classify（阈值触发出 picks）→ 采样点/相对时间换算为绝对 UTC →
        构造 Pick → 去重合并。
        """
        stream = self._to_stream(wf)

        # classify 内部会 annotate 并按阈值挑峰，返回带绝对时间的 picks。
        # 用官方 API 而非自己挑峰，避免重复造轮子且行为与 SeisBench 一致。
        outputs = self._classify(stream)

        end_guard = self._end_guard_utc(wf)
        picks: List[Pick] = []
        for p in getattr(outputs, "picks", outputs):
            phase = self._normalize_phase(p.phase)
            if phase is None:
                continue
            # SeisBench pick 的 peak_time 是 ObsPy UTCDateTime（绝对时间）。
            # 直接取其 epoch 秒，与内部 time_utc 约定一致；不经过手动采样点换算，
            # 消除一次潜在的对齐误差来源。
            time_utc = float(p.peak_time.timestamp)
            if time_utc > end_guard:
                logger.warning(
                    "丢弃末端护栏外 pick [%s] phase=%s 超出真实数据末尾 %.3fs",
                    wf.station,
                    phase.value,
                    time_utc - (end_guard - END_GUARD_TOLERANCE_S),
                )
                continue
            picks.append(
                Pick(
                    phase=phase,
                    time_utc=time_utc,
                    confidence=float(getattr(p, "peak_value", 0.0)),
                    station=wf.station,
                )
            )

        return self._long_dedup(wf, self._filter_long_snr(wf, dedup_picks(picks, self._dedup_cfg)))

    def _long_dedup(self, wf: Waveform, picks: List[Pick]) -> List[Pick]:
        """长记录事件级去重（见 PickerConfig.long_dedup_p_window_s）。

        放在 SNR 闸之后：先滤掉拾在噪声里的，再对幸存者做宽窗簇合并，
        避免噪声 pick 抢走簇代表名额。短文件（<= long_dedup_min_duration_s）
        原样返回，与 cap 限额互补不重叠。
        """
        pw = self._cfg.long_dedup_p_window_s
        sw = self._cfg.long_dedup_s_window_s
        if (pw is None and sw is None) or not picks:
            return picks
        dur = wf.n_samples / wf.sampling_rate
        if dur <= self._cfg.long_dedup_min_duration_s:
            return picks
        windows = {}
        if pw is not None:
            windows[PhaseType.P] = pw
        if sw is not None:
            windows[PhaseType.S] = sw
        from ..postprocess.dedup import deduplicate

        return deduplicate(picks, merge_window_s=windows)

    def _filter_long_snr(self, wf: Waveform, picks: List[Pick]) -> List[Pick]:
        """长记录 SNR 过滤（见 PickerConfig.long_snr_threshold_db）。

        只在时长超过 long_snr_min_duration_s 时生效；短文件由限额机制
        （postprocess.cap）处理，两者互补不重叠。阈值 None 时原样返回（零开销）。
        逻辑与 fork dizheng-opus-5 逐位一致（本仓三分布复验通过）。
        """
        thr = self._cfg.long_snr_threshold_db
        if thr is None or not picks:
            return picks
        dur = wf.n_samples / wf.sampling_rate
        if dur <= self._cfg.long_snr_min_duration_s:
            return picks

        data = np.asarray(wf.data, dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]
        # 三分量合成幅度包络；单分量时退化为绝对值
        mag = np.sqrt(np.sum(data ** 2, axis=0))
        sr = float(wf.sampling_rate)
        n = mag.shape[0]
        n_pre = max(1, int(self._cfg.long_snr_pre_s * sr))
        n_post = max(1, int(self._cfg.long_snr_post_s * sr))

        kept: List[Pick] = []
        for p in picks:
            i = int(round((p.time_utc - wf.starttime_utc) * sr))
            a, b = max(0, i - n_pre), i
            c, d = i, min(n, i + n_post)
            # 窗口不足（贴边）时不判——宁可放过，也不误伤
            if b - a < n_pre // 2 or d - c < n_post // 2:
                kept.append(p)
                continue
            rms_pre = float(np.sqrt(np.mean(mag[a:b] ** 2)) + 1e-12)
            rms_post = float(np.sqrt(np.mean(mag[c:d] ** 2)) + 1e-12)
            snr_db = 20.0 * float(np.log10(rms_post / rms_pre))
            if snr_db >= thr:
                kept.append(p)
        return kept

    def pick_batch(self, wfs: "List[Waveform]") -> List[List[Pick]]:
        """批量拾取：多条波形合成一个 Stream，单次 classify 完成全部推理。

        为什么快：SeisBench 的 annotate 会把**所有台站**切出的滑窗混编进同一批
        张量喂模型。逐条调用时每条波形只有十几个窗口，凑不满 batch_size，
        GPU/CPU 大量时间耗在每次调用的固定开销上；合批后固定开销只付一次、
        批张量始终满载，这是把吞吐推到硬件上限的关键。

        正确性保障：每条波形分配唯一临时台站码（B00000, B00001, ...），
        用 SeisBench pick 的 trace_id 精确映射回原波形；到时取绝对 epoch 秒，
        与单条 pick() 完全同源同参，最后同样按波形分别去重。任何一步对不上
        （如异版本 trace_id 缺失）都抛异常，交由调用方回退到逐条路径。

        Returns:
            与输入等长的列表，第 i 项是第 i 条波形的去重后 Pick 列表。
        """
        if not wfs:
            return []
        from obspy import Stream

        stream = Stream()
        sta_to_idx = {}
        for i, wf in enumerate(wfs):
            tag = f"B{i:05d}"
            sta_to_idx[tag] = i
            stream += self._to_stream(wf, station_override=tag)

        outputs = self._classify(stream)

        # 末端护栏按 per-wf 各自的真实数据末尾判定（各波形长度不同）
        end_guards = [self._end_guard_utc(wf) for wf in wfs]
        per_wf: List[List[Pick]] = [[] for _ in wfs]
        for p in getattr(outputs, "picks", outputs):
            phase = self._normalize_phase(p.phase)
            if phase is None:
                continue
            trace_id = str(getattr(p, "trace_id", "") or "")
            parts = trace_id.split(".")
            tag = parts[1] if len(parts) >= 2 else ""
            idx = sta_to_idx.get(tag)
            if idx is None:
                raise RuntimeError(
                    f"pick_batch 无法把 trace_id={trace_id!r} 映射回输入波形；"
                    "请回退逐条 pick() 路径"
                )
            time_utc = float(p.peak_time.timestamp)
            if time_utc > end_guards[idx]:
                logger.warning(
                    "丢弃末端护栏外 pick [%s] phase=%s 超出真实数据末尾 %.3fs",
                    wfs[idx].station,
                    phase.value,
                    time_utc - (end_guards[idx] - END_GUARD_TOLERANCE_S),
                )
                continue
            per_wf[idx].append(
                Pick(
                    phase=phase,
                    time_utc=time_utc,
                    confidence=float(getattr(p, "peak_value", 0.0)),
                    station=wfs[idx].station,
                )
            )
        return [
            self._long_dedup(wf, self._filter_long_snr(wf, dedup_picks(group, self._dedup_cfg)))
            for wf, group in zip(wfs, per_wf)
        ]

    def _to_stream(self, wf: Waveform, station_override: Optional[str] = None):
        from obspy import Stream, Trace, UTCDateTime

        # seisbench annotate 对"短于模型一个输入窗"的流**静默输出空表**：
        # 模型窗长 = in_samples / model.sampling_rate 秒（diting 50Hz 下是
        # 60.02s，官方 51.5s 文件会全部空报）。不足时尾部边缘复制补齐——
        # 补尾部不动 starttime，到时换算不受影响；复制边缘值而非补零，
        # 免得原始计数的 DC 偏移在拼接处造成台阶假信号。多补 1s 余量，
        # 抵消重采样取整。
        data = wf.data
        try:
            need_s = float(self._model.in_samples) / float(self._model.sampling_rate)
        except (AttributeError, TypeError, ZeroDivisionError):
            need_s = 0.0
        if need_s > 0:
            need_n = int(np.ceil((need_s + 1.0) * wf.sampling_rate))
            if data.shape[-1] < need_n:
                pad_n = need_n - data.shape[-1]
                data = np.pad(data, ((0, 0), (0, pad_n)), mode="edge")

        st = Stream()
        starttime = UTCDateTime(wf.starttime_utc)
        station = station_override or (wf.station or "STA")
        # 台站码里可能带 NET.STA 的点号，会破坏 trace_id 结构，替换掉
        station = station.replace(".", "_")[:8]
        for i, comp in enumerate(CHANNEL_ORDER):
            tr = Trace(data=np.ascontiguousarray(data[i], dtype=np.float32))
            tr.stats.sampling_rate = wf.sampling_rate
            tr.stats.starttime = starttime
            tr.stats.channel = f"HH{comp}"  # SeisBench 按通道码识别分量
            tr.stats.station = station
            tr.stats.network = "XB"
            st.append(tr)
        return st

    @staticmethod
    def _normalize_phase(raw: str) -> Optional[PhaseType]:
        """把模型的相位标签统一到 PhaseType；非 P/S（如噪声/检测）返回 None。"""
        s = str(raw).upper()
        if s == "P":
            return PhaseType.P
        if s == "S":
            return PhaseType.S
        return None


class ProbEnsemblePicker(SeisBenchPicker):
    """多 PhaseNet 概率曲线逐点平均后统一挑峰的软集成拾取器.

    动机（2026-08-02，区域鲁棒性）：今年考题区域未知，单一区域权重存在错配
    风险（41 权重两轮实测最差-最好差 0.05~0.09 均分）。概率级平均能对冲单
    成员的系统性偏差——与此前证伪的"拾取级投票"不同（离散、稀释最强者），
    软集成在两轮上均**超过各自最优单模型**：
    r1 1.744（最优单 1.738）/ r2 1.723（最优单 1.717），成员=guangxi+jiangxi+shandong。

    行为契约：除 _classify_refined 外与单模型 picker 完全一致（同阈值、同去重、
    同亚采样精细化）；推理成本 = 成员数 × 单模型。成员输出 trace 必须逐条对齐
    （同 id/起点/长度），对不齐直接抛错——静默错位会产出坏拾取。
    """

    DEFAULT_MEMBERS = ("guangxi", "jiangxi", "shandong")

    @classmethod
    def from_member_names(
        cls,
        names: List[str],
        base_cfg: PickerConfig,
        weights_dir: str = "weights/ustc_pickers",
    ) -> "ProbEnsemblePicker":
        """按区域简名（或 .pt 路径，或 'diting'=纯预训练）构建集成。"""
        import dataclasses
        import os

        def cfg_for(name: str) -> PickerConfig:
            if name == "diting":
                path = None
            elif os.path.sep in name or name.endswith(".pt"):
                path = name
            else:
                path = os.path.join(weights_dir, f"{name}_sd.pt")
            # 集成必须走 annotate 路径（classify 拿不到概率曲线无从平均）
            return dataclasses.replace(
                base_cfg, local_weights_path=path, subsample_refine=True
            )

        host = cls.from_config(cfg_for(names[0]))
        members = [host._model]
        for n in names[1:]:
            members.append(SeisBenchPicker.from_config(cfg_for(n))._model)
        host._members = members
        return host

    def _classify_refined(self, stream):
        import seisbench.util as sbu

        streams = [stream]
        if self._cfg.tta_polarity_flip:
            flipped = stream.copy()
            for tr in flipped:
                tr.data = -tr.data
            streams.append(flipped)
        anns = [
            m.annotate(s, batch_size=self._cfg.batch_size, overlap=self._cfg.overlap)
            for m in self._members
            for s in streams
        ]
        # 长记录成员门控：>阈值时长的台站只平均前 top_n 个成员的曲线
        top_n = self._cfg.ensemble_long_top_n
        n_aug = len(streams)
        long_stas: set = set()
        if top_n is not None and 0 < top_n < len(self._members):
            for tr in stream:
                dur = float(tr.stats.endtime - tr.stats.starttime)
                if dur > self._cfg.ensemble_long_max_duration_s:
                    long_stas.add(tr.stats.station)
        base = anns[0]
        for tr in base:
            n_keep = len(anns)
            if tr.stats.station in long_stas:
                n_keep = top_n * n_aug
            stack = [tr.data.astype(np.float64)]
            for other in anns[1:n_keep]:
                match = [
                    t for t in other
                    if t.id == tr.id
                    and t.stats.starttime == tr.stats.starttime
                    and len(t.data) == len(tr.data)
                ]
                if len(match) != 1:
                    raise RuntimeError(
                        f"集成 trace 对不齐: {tr.id}@{tr.stats.starttime} 命中 {len(match)} 条"
                    )
                stack.append(match[0].data.astype(np.float64))
            tr.data = np.mean(stack, axis=0)

        picks = sbu.PickList()
        prefix = self._model.__class__.__name__
        ann_by_phase, picks_by_phase = {}, {}
        for phase, th in (("P", self._cfg.p_threshold), ("S", self._cfg.s_threshold)):
            phase_ann = base.select(channel=f"{prefix}_{phase}")
            phase_picks = self._model.picks_from_annotations(phase_ann, th, phase)
            for p in phase_picks:
                self._refine_peak_inplace(p, phase_ann)
            ann_by_phase[phase] = phase_ann
            picks_by_phase[phase] = phase_picks
        for phase in ("P", "S"):
            picks += picks_by_phase[phase]
            picks += self._fallback_lowth_picks(phase, ann_by_phase, picks_by_phase, stream)
        return sbu.PickList(sorted(picks))
