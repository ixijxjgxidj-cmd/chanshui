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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..types import Pick, PhaseType, Waveform, CHANNEL_ORDER
from ..utils.timing import sample_to_utc
from ..postprocess.dedup import dedup_picks, DedupConfig


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
    """

    model_name: str = "PhaseNet"
    pretrained: str = "original"
    device: Optional[str] = None
    p_threshold: float = 0.3
    s_threshold: float = 0.3
    batch_size: int = 256
    overlap: float = 0.5
    local_weights_path: Optional[str] = None
    use_fp16: bool = False
    num_threads: Optional[int] = None
    compile_model: bool = False


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
        """通常不直接用构造函数，用 ``SeisBenchPicker.from_config(cfg)``。"""
        self._model = model
        self._cfg = cfg
        self._dedup_cfg = dedup_cfg or DedupConfig()

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
            return self._model.classify(
                stream,
                batch_size=self._cfg.batch_size,
                overlap=self._cfg.overlap,
                P_threshold=self._cfg.p_threshold,
                S_threshold=self._cfg.s_threshold,
            )

    def _device_type(self) -> str:
        try:
            return next(self._model.parameters()).device.type
        except Exception:  # noqa: BLE001
            return "cpu"

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

        picks: List[Pick] = []
        for p in getattr(outputs, "picks", outputs):
            phase = self._normalize_phase(p.phase)
            if phase is None:
                continue
            # SeisBench pick 的 peak_time 是 ObsPy UTCDateTime（绝对时间）。
            # 直接取其 epoch 秒，与内部 time_utc 约定一致；不经过手动采样点换算，
            # 消除一次潜在的对齐误差来源。
            time_utc = float(p.peak_time.timestamp)
            picks.append(
                Pick(
                    phase=phase,
                    time_utc=time_utc,
                    confidence=float(getattr(p, "peak_value", 0.0)),
                    station=wf.station,
                )
            )

        return dedup_picks(picks, self._dedup_cfg)

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
            per_wf[idx].append(
                Pick(
                    phase=phase,
                    time_utc=float(p.peak_time.timestamp),
                    confidence=float(getattr(p, "peak_value", 0.0)),
                    station=wfs[idx].station,
                )
            )
        return [dedup_picks(group, self._dedup_cfg) for group in per_wf]

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
