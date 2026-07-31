#!/usr/bin/env python3
"""官方标准震相拾取 API 服务（2026 第二届"震智杯"提交格式）.

===== 官方要求（来自《比赛说明》PPT，逐条对齐）=====
- 提交方式：公网可访问的 HTTP API（阿里云/华为云等），比赛期间开放（约1天）。
- 请求：``resp = requests.post(url, files=files)`` —— multipart 上传 .mseed 波形
  文件（obspy.read(..., format="MSEED") 可解析）。
- 响应：(1) HTTP 状态码；(2) JSON 识别结果，最外层键为台站名，每台站下含
  P 波到时表（键 "P"）与 S 波到时表（键 "S"），到时为**绝对 UTC 时间**字符串：

      {
        "STA1": {
          "P": ["2025-06-07T12:34:56.789000Z", ...],
          "S": ["2025-06-07T12:36:55.000000Z"]
        },
        "STA2": {"P": [...], "S": [...]}
      }

- 评分：P 误差≤0.1s 得 1 分、0.1~1s 线性、>1s 不计分；S ≤0.2s 得 1 分、
  0.2~2s 线性；识别数量误差 >5% 每超 1 个扣 0.5 分；单项最低 0 分。
  输入为 100Hz 速度波形，不含位置信息，含噪声数据（可能没有任何震相）。

===== 设计要点 =====
- 模型**启动时加载一次**并预热（跑一条合成波形，触发 cudnn autotune / 懒初始化），
  正式请求不付冷启动代价。
- 单文件内多台站走 ``pick_batch`` 合批推理；推理段全局互斥（单卡/单机内存安全），
  上传解析与 JSON 组装并发。
- 宁可空表不可报错：波形质量问题（缺分量、采样率异常……）降级为该台站空 P/S，
  HTTP 仍 200 —— 评分只损失该台站，绝不因单文件把整轮请求打成 5xx。
  真正的协议错误（没传文件）才返回 4xx。
- 到时格式与 ``str(obspy.UTCDateTime)`` 一致：微秒 6 位 + "Z" 后缀。

===== 用法 =====
    pip install fastapi uvicorn python-multipart obspy seisbench torch
    python scripts/serve_api.py --weights weights/phasenet_ft.pt --port 8000
    # 自检（另开终端）：
    python scripts/check_api.py --url http://127.0.0.1:8000/pick --input <mseed目录或zip>
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.types import PhaseType, Pick, Waveform  # noqa: E402

# FastAPI 必须在模块级导入：本文件启用了 `from __future__ import annotations`，
# 处理函数的 `request: Request` 注解是字符串，FastAPI 用 get_type_hints 按模块
# globals 解析——若 Request 只在函数内导入，解析失败会被当成 query 参数 → 全 422。
try:  # pragma: no cover - 导入失败路径
    from fastapi import FastAPI, Request  # noqa: E402
    from fastapi.responses import JSONResponse  # noqa: E402
    from starlette.concurrency import run_in_threadpool  # noqa: E402

    _FASTAPI_IMPORT_ERROR = None
except ImportError as _exc:  # noqa: N816
    FastAPI = Request = JSONResponse = run_in_threadpool = None  # type: ignore[assignment]
    _FASTAPI_IMPORT_ERROR = _exc


# ---------------------------------------------------------------------------
# 时间格式：epoch 秒 → "2025-06-07T12:34:56.789000Z"（与官方示例逐字符一致）
# ---------------------------------------------------------------------------
def iso_utc(epoch_s: float) -> str:
    dt = datetime.fromtimestamp(round(float(epoch_s), 6), tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _station_key(full: str) -> str:
    """内部台站标识 NET.STA → 官方响应键（台站名）。"""
    s = (full or "STA").strip()
    return s.rsplit(".", 1)[-1] or s


def picks_to_official_json(
    waveforms: List[Waveform],
    picks_per_wf: List[List[Pick]],
) -> Dict[str, Dict[str, List[str]]]:
    """按官方格式组装 JSON：台站名 → {"P": [...], "S": [...]}（升序）。

    - 每个成功读出的台站都会出现在结果里（噪声数据 → 空表），
      这本身就是对"数量误差"项的正确表达。
    - 台站名冲突（不同 network 同名台站）时退回完整 NET.STA 作键，保证不覆盖。
    """
    keys: List[str] = []
    seen: Dict[str, int] = {}
    for wf in waveforms:
        k = _station_key(wf.station)
        seen[k] = seen.get(k, 0) + 1
        keys.append(k)
    for i, wf in enumerate(waveforms):
        if seen[keys[i]] > 1:
            keys[i] = (wf.station or keys[i]).strip()

    out: Dict[str, Dict[str, List[str]]] = {}
    for key, picks in zip(keys, picks_per_wf):
        slot = out.setdefault(key, {"P": [], "S": []})
        for p in picks:
            if p.phase == PhaseType.P:
                slot["P"].append(float(p.time_utc))
            elif p.phase == PhaseType.S:
                slot["S"].append(float(p.time_utc))
    for slot in out.values():
        slot["P"] = [iso_utc(t) for t in sorted(slot["P"])]
        slot["S"] = [iso_utc(t) for t in sorted(slot["S"])]
    return out


# ---------------------------------------------------------------------------
# 推理引擎：模型常驻 + 合批 + 互斥
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, picker):
        self._picker = picker
        self._lock = threading.Lock()  # 单模型串行推理；解析/组装在锁外并发

    def process_mseed_bytes(self, raw: bytes) -> Dict[str, Dict[str, List[str]]]:
        from phasepicker.io.mseed_reader import load_waveforms

        result = load_waveforms(raw)
        waveforms = result.waveforms
        if not waveforms:
            return {}

        with self._lock:
            picks_per_wf = self._pick_all(waveforms)
        return picks_to_official_json(waveforms, picks_per_wf)

    def _pick_all(self, waveforms: List[Waveform]) -> List[List[Pick]]:
        if hasattr(self._picker, "pick_batch"):
            try:
                return self._picker.pick_batch(waveforms)
            except Exception:  # noqa: BLE001 —— 合批失败回退逐条，绝不 5xx
                traceback.print_exc()  # 但必须留痕：无声吞掉会把真 bug 藏成"全空表"
        out: List[List[Pick]] = []
        for wf in waveforms:
            try:
                out.append(list(self._picker.pick(wf)))
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                out.append([])
        return out

    def warmup(self) -> float:
        """跑一条 60s 合成三分量，触发权重懒加载/cudnn autotune。返回耗时秒。"""
        import numpy as np

        rng = np.random.default_rng(0)
        wf = Waveform(
            data=rng.standard_normal((3, 6000)).astype("float32"),
            sampling_rate=100.0,
            starttime_utc=0.0,
            station="XB.WARM",
        )
        t0 = time.perf_counter()
        with self._lock:
            self._pick_all([wf])
        return time.perf_counter() - t0


def build_engine(args) -> Engine:
    from phasepicker.inference.picker import PickerConfig, SeisBenchPicker

    cfg = PickerConfig(
        device=None if args.device == "auto" else args.device,
        pretrained=args.pretrained,
        p_threshold=args.p_threshold,
        s_threshold=args.s_threshold,
        local_weights_path=args.weights,
        use_fp16=args.fp16,
        num_threads=args.threads,
        batch_size=args.batch_size,
        overlap=args.overlap,
        compile_model=args.compile,
    )
    picker = SeisBenchPicker.from_config(cfg)
    return Engine(picker)


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
def create_app(engine: Engine):
    if _FASTAPI_IMPORT_ERROR is not None:  # pragma: no cover
        raise SystemExit(
            f"API 服务需要 FastAPI：{_FASTAPI_IMPORT_ERROR!r}\n"
            "请先安装：pip install fastapi uvicorn python-multipart"
        )
    try:  # multipart 解析是硬依赖，缺了会在请求期 500，这里提前拦下
        import python_multipart  # noqa: F401
    except ImportError:
        try:
            import multipart  # noqa: F401  # 旧版包的导入名
        except ImportError:
            raise SystemExit(
                "缺少 multipart 解析库（官方 files= 上传必需）：\n"
                "pip install python-multipart"
            )

    app = FastAPI(title="震相拾取 API", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    async def _handle(request: Request):
        t0 = time.perf_counter()
        payloads: List[bytes] = []
        content_type = (request.headers.get("content-type") or "").lower()
        if "multipart/form-data" in content_type:
            form = await request.form()
            # multi_items()：同名字段（requests 传多个文件常这么编码）逐个取，
            # 用 values() 会丢同名重复项
            for _key, value in form.multi_items():
                read = getattr(value, "read", None)
                if read is None:
                    continue  # 非文件字段忽略
                data = await read()
                if data:
                    payloads.append(data)
        else:
            body = await request.body()
            if body:
                payloads.append(body)  # 兼容直接把 mseed 当 body POST

        if not payloads:
            return JSONResponse(
                status_code=400,
                content={"error": "未收到波形文件：请以 multipart files= 上传 .mseed"},
            )

        merged: Dict[str, Dict[str, List[str]]] = {}
        for raw in payloads:
            try:
                # 必须进线程池：seisbench 的 classify 在"已有运行中事件循环"的
                # 线程里无法 asyncio.run 自己的 classify_async（表现为
                # RuntimeWarning: coroutine was never awaited + 全空表）。
                # 线程池线程没有运行中的 loop，classify 正常；推理期间事件循环
                # 也不被长任务卡住，/health 始终秒回。
                one = await run_in_threadpool(engine.process_mseed_bytes, raw)
            except Exception:  # noqa: BLE001 —— 数据层问题降级为空，不 5xx
                traceback.print_exc()
                one = {}
            for sta, slot in one.items():
                tgt = merged.setdefault(sta, {"P": [], "S": []})
                tgt["P"] = sorted(tgt["P"] + slot["P"])
                tgt["S"] = sorted(tgt["S"] + slot["S"])

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return JSONResponse(
            content=merged,
            headers={"X-Process-Time-Ms": f"{elapsed_ms:.1f}"},
        )

    # 报名时登记哪个路径都行：/、/pick、/predict 三个入口等价
    app.post("/")(_handle)
    app.post("/pick")(_handle)
    app.post("/predict")(_handle)
    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="官方标准震相拾取 API 服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--weights", default=None, help="本地微调权重 (.pt)")
    ap.add_argument("--pretrained", default="diting",
                    help="SeisBench 基础权重名（去年真题 A/B 实测 diting 1.899 > "
                         "stead 1.496，见 deploy/README.md）")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--p-threshold", type=float, default=0.3)
    ap.add_argument("--s-threshold", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=None, help="CPU 推理线程数（默认满核）")
    ap.add_argument("--fp16", action="store_true", help="CUDA 半精度推理")
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="滑窗重叠：0~1 为窗长比例，>1 为采样点数（seisbench 语义）。"
                         "调小提速；须先用 ab_compare.py 验证同分")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile 加速（PyTorch 2.x；老显卡不支持会自动回退）")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args(argv)

    print("加载模型 …", flush=True)
    engine = build_engine(args)
    if not args.no_warmup:
        dt = engine.warmup()
        print(f"预热完成（{dt*1000:.0f}ms）")

    app = create_app(engine)
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("请先安装 uvicorn：pip install uvicorn")
    # 单 worker：模型只加载一份；并发由线程池 + 推理互斥锁治理
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
