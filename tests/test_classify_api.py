"""/classify 分类端点的单元测试（桩分类器）+ 真模型冒烟.

覆盖：输出形状 {台站: {"class": [int]}}、off/降级 501、与 /pick 同载荷契约、
/class 别名、CLI 默认值、真 T3 模型加载与整数类别输出。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import serve_api  # noqa: E402

pytest.importorskip("fastapi")
from starlette.testclient import TestClient  # noqa: E402


class StubClsEngine:
    has_magnitude = False
    has_classify = True

    def __init__(self, result=None, exc=None, has_cls=True):
        self._result = {} if result is None else result
        self._exc = exc
        self.has_classify = has_cls

    def process_mseed_bytes(self, raw):
        return {}

    def process_mseed_bytes_classify(self, raw):
        if self._exc:
            raise self._exc
        return self._result


def _client(engine):
    return TestClient(serve_api.create_app(engine), raise_server_exceptions=False)


def test_classify_shape_and_alias():
    eng = StubClsEngine(result={"STA1": {"class": [2]}})
    c = _client(eng)
    r = c.post("/classify", files={"file": ("a.mseed", b"xx")})
    assert (r.status_code, r.json()) == (200, {"STA1": {"class": [2]}})
    assert c.post("/class", files={"file": ("a.mseed", b"xx")}).json() == r.json()


def test_classify_disabled_501_pick_unaffected():
    c = _client(StubClsEngine(has_cls=False))
    assert c.post("/classify", files={"file": ("a.mseed", b"xx")}).status_code == 501
    assert c.post("/pick", files={"file": ("a.mseed", b"xx")}).status_code == 200


def test_classify_exception_degrades_200_empty():
    c = _client(StubClsEngine(exc=RuntimeError("炸")))
    r = c.post("/classify", files={"file": ("a.mseed", b"xx")})
    assert (r.status_code, r.json()) == (200, {})


def test_classify_payload_contract():
    c = _client(StubClsEngine(result={"S": {"class": [1]}}))
    r = c.post("/classify", content=b"junk",
               headers={"Content-Type": "multipart/form-data; boundary=xyz"})
    assert (r.status_code, r.json()) == (200, {})
    assert c.post("/classify").status_code == 400


def test_cli_defaults_classify_on():
    args = serve_api.make_arg_parser().parse_args([])
    assert args.cls_model == "baseline"
    assert args.cls_weights is None


def test_real_t3_model_loads_and_predicts_int():
    """真模型冒烟：joblib 可载、合成波形出 1..5 整数类别。"""
    pytest.importorskip("sklearn")
    np = pytest.importorskip("numpy")
    from phasepicker.classification import build_classifier
    from phasepicker.magnitude import MagnitudeInput
    from phasepicker.types import Waveform

    path = os.path.join(os.path.dirname(__file__), "..",
                        "weights", "official_r1_to_r2", "t3_event_baseline.joblib")
    if not os.path.exists(path):
        pytest.skip("T3 基线模型不在本机")
    est = build_classifier("baseline", path)
    wf = Waveform(data=np.random.default_rng(0).standard_normal((3, 6000)).astype("float32"),
                  sampling_rate=100.0, starttime_utc=0.0, station="XB.TST")
    out = est.estimate(MagnitudeInput(waveforms=[wf], picks_per_wf=[[]]))
    assert len(out) == 1 and len(out[0]) == 1 and out[0][0] in {1, 2, 3, 4, 5}


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
