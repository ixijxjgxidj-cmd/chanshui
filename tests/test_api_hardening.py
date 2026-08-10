"""serve_api 请求层加固的单元测试（桩引擎，不需要 torch/obspy 真推理）.

对应 2026-08-01 评审发现：
- C8: 畸形 multipart（异常 boundary、截断头部……）→ 200 {}，绝不 5xx
- U21: 有文件字段但内容为空 → 200 {}；仅"既无文件字段又无 body"才 400
- U1: Content-Length 声明超上限 → 413；无头流/超大文件按上限截断读取
- defaults 接线: CLI 默认值与 src/phasepicker/defaults.py 单一真源一致
- 泄漏缓解: asyncio.to_thread 已被内联补丁替换（seisbench 每请求新线程各泄
  ~3.3MB 原生内存，见 serve_api 模块级注释与 outputs/leak_probe.py 实测）

两种运行方式：
    pytest tests/test_api_hardening.py
    python  tests/test_api_hardening.py
"""

import asyncio
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import serve_api  # noqa: E402
from phasepicker import defaults  # noqa: E402

pytest.importorskip("fastapi")
from starlette.testclient import TestClient  # noqa: E402


class StubEngine:
    """multipart/正文解析都发生在引擎调用之前，桩引擎不失真。"""

    def __init__(self, result=None, exc=None):
        self.calls = []
        self._result = {} if result is None else result
        self._exc = exc

    def process_mseed_bytes(self, raw):
        self.calls.append(raw)
        if self._exc is not None:
            raise self._exc
        return self._result


def _client(engine=None, extra_route=None):
    app = serve_api.create_app(engine or StubEngine(), extra_route=extra_route)
    # raise_server_exceptions=False 等价 uvicorn 的 ServerErrorMiddleware：
    # 未处理异常表现为 HTTP 500 响应，测试才能发现 5xx 回归
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# C8: 畸形 multipart 一律 200 {}
# ---------------------------------------------------------------------------
def test_malformed_multipart_returns_200_empty():
    client = _client()
    cases = [
        # 声称 boundary=xyz 但 body 完全不合规（原评审探针，修前 500）
        (b"this is not multipart at all",
         {"Content-Type": "multipart/form-data; boundary=xyz"}),
        # multipart 头但缺 boundary 参数
        (b"whatever", {"Content-Type": "multipart/form-data"}),
        # 有 boundary 开头但头部中途截断
        (b'--xyz\r\nContent-Disposition: form-data; name="f"; filename="a"\r\n\r\ndata',
         {"Content-Type": "multipart/form-data; boundary=xyz"}),
        # boundary 后跟非法字节
        (b"--xyz\r\n\x00\xff\x00\xff",
         {"Content-Type": "multipart/form-data; boundary=xyz"}),
        # 空 body + multipart 头
        (b"", {"Content-Type": "multipart/form-data; boundary=xyz"}),
    ]
    for content, headers in cases:
        r = client.post("/pick", content=content, headers=headers)
        assert r.status_code == 200, (content, headers, r.status_code, r.text)
        assert r.json() == {}


def test_service_alive_after_malformed_request():
    engine = StubEngine(result={"STA1": {"P": [], "S": []}})
    client = _client(engine)
    client.post("/pick", content=b"junk",
                headers={"Content-Type": "multipart/form-data; boundary=xyz"})
    r = client.post("/pick", files={"file": ("x.mseed", b"\x00\x01 bytes")})
    assert r.status_code == 200
    assert r.json() == {"STA1": {"P": [], "S": []}}
    assert client.get("/health").json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# U21: 空文件字段 → 200 {}；既无文件字段又无 body → 400
# ---------------------------------------------------------------------------
def test_empty_file_field_returns_200_empty():
    engine = StubEngine()
    client = _client(engine)
    r = client.post("/pick", files={"file": ("empty.mseed", b"")})
    assert r.status_code == 200
    assert r.json() == {}
    assert engine.calls == []  # 空内容不该进推理


def test_text_only_multipart_returns_200_empty():
    body = (b"--b1\r\n"
            b'Content-Disposition: form-data; name="note"\r\n\r\n'
            b"hello\r\n"
            b"--b1--\r\n")
    r = _client().post("/pick", content=body,
                       headers={"Content-Type": "multipart/form-data; boundary=b1"})
    assert r.status_code == 200
    assert r.json() == {}


def test_no_file_no_body_returns_400():
    r = _client().post("/pick")
    assert r.status_code == 400


def test_raw_body_fallback_still_works():
    engine = StubEngine(result={"NN01": {"P": ["2025-06-07T12:34:56.789000Z"], "S": []}})
    client = _client(engine)
    r = client.post("/pick", content=b"raw mseed bytes",
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200
    assert r.json() == {"NN01": {"P": ["2025-06-07T12:34:56.789000Z"], "S": []}}
    assert engine.calls == [b"raw mseed bytes"]


def test_engine_exception_degrades_to_200_empty():
    client = _client(StubEngine(exc=RuntimeError("boom")))
    r = client.post("/pick", files={"file": ("x.mseed", b"\x00\x01")})
    assert r.status_code == 200
    assert r.json() == {}


# ---------------------------------------------------------------------------
# U1: 正文上限
# ---------------------------------------------------------------------------
def test_declared_content_length_over_cap_413():
    client = _client()
    r = client.post("/pick", headers={"Content-Length": str(300 * 1024 * 1024)})
    assert r.status_code == 413
    # 上限内的声明不受影响
    r2 = client.post("/pick", content=b"ok", headers={"Content-Length": "2"})
    assert r2.status_code == 200


def test_streamed_body_truncated_at_cap(monkeypatch):
    monkeypatch.setattr(serve_api, "MAX_BODY_BYTES", 100)
    engine = StubEngine()
    client = _client(engine)

    def gen():  # 无 Content-Length 的 chunked 流，共 500 字节
        for _ in range(10):
            yield b"x" * 50

    r = client.post("/pick", content=gen(),
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200
    assert len(engine.calls) == 1 and len(engine.calls[0]) == 100


def test_multipart_file_read_truncated_at_cap(monkeypatch):
    monkeypatch.setattr(serve_api, "MAX_BODY_BYTES", 100)
    engine = StubEngine()
    client = _client(engine)
    # chunked（无 Content-Length）multipart，绕开 413 声明检查，专测文件读取截断
    body = (b"--bnd\r\n"
            b'Content-Disposition: form-data; name="file"; filename="big.mseed"\r\n'
            b"Content-Type: application/octet-stream\r\n\r\n"
            + b"y" * 500 +
            b"\r\n--bnd--\r\n")

    def gen():
        yield body

    r = client.post("/pick", content=gen(),
                    headers={"Content-Type": "multipart/form-data; boundary=bnd"})
    assert r.status_code == 200
    assert len(engine.calls) == 1 and len(engine.calls[0]) == 100


# ---------------------------------------------------------------------------
# --route 隐蔽入口（默认不开启）
# ---------------------------------------------------------------------------
def test_extra_route_registered_only_when_asked():
    payload = {"file": ("x.mseed", b"\x00\x01")}
    assert _client().post("/pick-secret", files=payload).status_code == 404
    hidden = _client(extra_route="/pick-secret")
    assert hidden.post("/pick-secret", files=payload).status_code == 200
    assert hidden.post("/pick", files=payload).status_code == 200  # 常规入口保留
    # 不带斜杠也接受
    assert _client(extra_route="pick-s2").post("/pick-s2", files=payload).status_code == 200


# ---------------------------------------------------------------------------
# defaults.py 单一真源接线
# ---------------------------------------------------------------------------
def test_cli_defaults_come_from_defaults_module():
    args = serve_api.make_arg_parser().parse_args([])
    assert args.pretrained == defaults.DEFAULT_PRETRAINED
    assert args.p_threshold == defaults.DEFAULT_P_THRESHOLD
    assert args.s_threshold == defaults.DEFAULT_S_THRESHOLD
    assert args.p_merge_window == defaults.DEFAULT_P_MERGE_WINDOW_S
    assert args.s_merge_window == defaults.DEFAULT_S_MERGE_WINDOW_S
    assert args.route is None  # 隐蔽入口默认关
    # 服务端默认必须就是 2026-08-11 七成员生产后处理；部署脚本漏参也不能跑偏。
    assert args.cap_short_s == 300.0
    assert args.cap_max_p == args.cap_max_s == 1
    assert args.long_snr_db == -1.0
    assert args.force_pair_mode == "conditional"
    assert args.long_dedup_s == 20.0
    assert args.ensemble_long_members == 5


# ---------------------------------------------------------------------------
# 泄漏缓解：asyncio.to_thread 内联补丁
# ---------------------------------------------------------------------------
def test_to_thread_patched_to_inline_execution():
    assert asyncio.to_thread is serve_api._to_thread_inline

    async def probe():
        return await asyncio.to_thread(threading.get_ident)

    # 内联后前向留在调用线程：不再每请求新建线程（每条泄 ~3.3MB 原生内存）
    assert asyncio.run(probe()) == threading.get_ident()


if __name__ == "__main__":
    import inspect

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            kwargs = {}
            if "monkeypatch" in inspect.signature(fn).parameters:
                mp = pytest.MonkeyPatch()
                kwargs["monkeypatch"] = mp
                try:
                    fn(**kwargs)
                finally:
                    mp.undo()
            else:
                fn()
            print(f"{name} ok")
    print("ALL OK")
