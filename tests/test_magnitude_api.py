"""/magnitude 震级端点的单元测试（桩估计器，不需要真模型）.

覆盖：
- 输出形状 {台站: {"M": [一位小数...]}} 与多文件合并
- --mag-model off / 构建失败 → 501（/pick 不受影响）
- 与 /pick 同一套载荷契约：畸形 multipart→200{}，裸请求→400
- /mag 别名等价
- mags_to_official_json 的台站键冲突回退与取整

两种运行方式：
    pytest tests/test_magnitude_api.py
    python  tests/test_magnitude_api.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import serve_api  # noqa: E402
from phasepicker.types import Waveform  # noqa: E402

pytest.importorskip("fastapi")
from starlette.testclient import TestClient  # noqa: E402


class StubMagEngine:
    """绕开真推理：magnitude 路径直接回固定表；has_magnitude 可控。"""

    def __init__(self, result=None, has_mag=True, exc=None):
        self._result = {} if result is None else result
        self._exc = exc
        self.has_magnitude = has_mag
        self.calls = []

    def process_mseed_bytes(self, raw):
        return {}

    def process_mseed_bytes_magnitude(self, raw):
        self.calls.append(raw)
        if self._exc:
            raise self._exc
        return self._result


def _client(engine):
    return TestClient(serve_api.create_app(engine), raise_server_exceptions=False)


def test_magnitude_shape_and_merge():
    eng = StubMagEngine(result={"STA1": {"M": [4.3]}})
    c = _client(eng)
    r = c.post("/magnitude", files={"file": ("a.mseed", b"xx")})
    assert r.status_code == 200
    assert r.json() == {"STA1": {"M": [4.3]}}
    # 同请求多文件：同台站 M 列表拼接
    r = c.post("/magnitude", files=[("f", ("a.mseed", b"xx")), ("f", ("b.mseed", b"yy"))])
    assert r.status_code == 200
    assert r.json() == {"STA1": {"M": [4.3, 4.3]}}
    assert len(eng.calls) == 3


def test_mag_alias_equivalent():
    eng = StubMagEngine(result={"S": {"M": [5.0]}})
    c = _client(eng)
    assert c.post("/mag", files={"file": ("a.mseed", b"xx")}).json() == \
        c.post("/magnitude", files={"file": ("a.mseed", b"xx")}).json()


def test_disabled_returns_501_pick_unaffected():
    eng = StubMagEngine(has_mag=False)
    c = _client(eng)
    r = c.post("/magnitude", files={"file": ("a.mseed", b"xx")})
    assert r.status_code == 501
    assert "error" in r.json()
    # /pick 完全不受影响
    assert c.post("/pick", files={"file": ("a.mseed", b"xx")}).status_code == 200


def test_estimator_exception_degrades_200_empty():
    eng = StubMagEngine(exc=RuntimeError("模型炸了"))
    c = _client(eng)
    r = c.post("/magnitude", files={"file": ("a.mseed", b"xx")})
    assert r.status_code == 200
    assert r.json() == {}


def test_same_payload_contract_as_pick():
    eng = StubMagEngine(result={"S": {"M": [4.0]}})
    c = _client(eng)
    # 畸形 multipart → 200 {}
    r = c.post("/magnitude", content=b"junk",
               headers={"Content-Type": "multipart/form-data; boundary=xyz"})
    assert (r.status_code, r.json()) == (200, {})
    # 裸请求（无文件无 body）→ 400
    assert c.post("/magnitude").status_code == 400


def test_mags_to_official_json_rounding_and_collision():
    wfs = [
        Waveform(data=np.zeros((3, 10), dtype="float32"), sampling_rate=100.0,
                 starttime_utc=0.0, station="AA.STA"),
        Waveform(data=np.zeros((3, 10), dtype="float32"), sampling_rate=100.0,
                 starttime_utc=0.0, station="BB.STA"),
    ]
    out = serve_api.mags_to_official_json(wfs, [[4.4499], [5.0]])
    # 同名台站冲突 → 回退完整 NET.STA；数值一位小数
    assert out == {"AA.STA": {"M": [4.4]}, "BB.STA": {"M": [5.0]}}


def test_cli_defaults_magnitude_on():
    args = serve_api.make_arg_parser().parse_args([])
    assert args.mag_model == "seismicxm"  # 2026-08 起默认深度震级（r2 留出 MAE 0.621）
    assert args.mag_weights is None  # None → build_engine 内解析仓库默认路径


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
