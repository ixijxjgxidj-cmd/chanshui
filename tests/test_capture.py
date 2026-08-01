"""请求采集（--capture-dir）的单元测试：落盘正确、零污染主响应、默认关闭.

TestClient 会在响应返回后同步执行 BackgroundTask，正好可以断言落盘结果。
"""

import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import serve_api  # noqa: E402

pytest.importorskip("fastapi")
from starlette.testclient import TestClient  # noqa: E402


class StubEngine:
    has_magnitude = True

    def process_mseed_bytes(self, raw):
        return {"STA1": {"P": ["1970-01-01T00:00:01.000000Z"], "S": []}}

    def process_mseed_bytes_magnitude(self, raw):
        return {"STA1": {"M": [4.2]}}


def test_capture_writes_waveform_and_manifest(tmp_path):
    cap = str(tmp_path / "cap")
    c = TestClient(serve_api.create_app(StubEngine(), capture_dir=cap),
                   raise_server_exceptions=False)
    payload = b"fake-mseed-bytes"
    r = c.post("/pick", files={"file": ("T1.E001.mseed", payload)})
    assert r.status_code == 200
    # manifest 一行，指向的波形文件字节一致
    manifest = os.path.join(cap, "manifest.jsonl")
    assert os.path.exists(manifest)
    rec = json.loads(open(manifest, encoding="utf-8").readlines()[-1])
    assert rec["endpoint"] == "pick"
    assert rec["response"] == {"STA1": {"P": ["1970-01-01T00:00:01.000000Z"], "S": []}}
    item = rec["items"][0]
    assert item["orig"] == "T1.E001.mseed"
    saved = os.path.join(cap, item["file"])
    assert open(saved, "rb").read() == payload
    assert item["sha1"] == hashlib.sha1(payload).hexdigest()
    # 震级端点同样采集
    c.post("/magnitude", files={"file": ("T2.E001.mseed", b"zz")})
    lines = open(manifest, encoding="utf-8").readlines()
    assert len(lines) == 2 and json.loads(lines[-1])["endpoint"] == "magnitude"


def test_capture_disabled_by_default(tmp_path):
    c = TestClient(serve_api.create_app(StubEngine()), raise_server_exceptions=False)
    r = c.post("/pick", files={"file": ("a.mseed", b"xx")})
    assert r.status_code == 200  # 不落任何盘、行为与旧版完全一致


def test_capture_failure_never_breaks_response(tmp_path, monkeypatch):
    cap = str(tmp_path / "cap2")
    monkeypatch.setattr(serve_api, "capture_save",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("盘炸了")))
    c = TestClient(serve_api.create_app(StubEngine(), capture_dir=cap),
                   raise_server_exceptions=False)
    r = c.post("/pick", files={"file": ("a.mseed", b"xx")})
    # BackgroundTask 里的异常发生在响应之后；客户端已拿到 200 与正确 JSON
    assert r.status_code == 200
    assert r.json() == {"STA1": {"P": ["1970-01-01T00:00:01.000000Z"], "S": []}}


def test_cli_default_off():
    args = serve_api.make_arg_parser().parse_args([])
    assert args.capture_dir is None


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
