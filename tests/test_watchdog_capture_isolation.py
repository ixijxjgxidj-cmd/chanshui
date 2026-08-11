"""Security boundaries for authenticated loopback watchdog probes."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import check_api  # noqa: E402
import serve_api  # noqa: E402
from phasepicker.probe_auth import (  # noqa: E402
    PROBE_ACCEPTED_HEADER,
    PROBE_ACCEPTED_VALUE,
    PROBE_TOKEN_HEADER,
    is_loopback_host,
    is_trusted_probe,
    load_probe_token_file,
)


pytest.importorskip("fastapi")
from starlette.testclient import TestClient  # noqa: E402


TOKEN = "A" * 48


class StubEngine:
    has_magnitude = True
    has_classify = True

    def process_mseed_bytes(self, raw):
        return {"STA": {"P": ["1970-01-01T00:00:01.000000Z"], "S": []}}

    def process_mseed_bytes_magnitude(self, raw):
        return {"STA": 4.2}

    def process_mseed_bytes_classify(self, raw):
        return {"STA": 3}


def _client(cap: Path, *, peer: str, token: str | None = TOKEN) -> TestClient:
    return TestClient(
        serve_api.create_app(
            StubEngine(),
            capture_dir=str(cap),
            probe_token=token,
        ),
        client=(peer, 50000),
        raise_server_exceptions=False,
    )


def _post(client: TestClient, endpoint: str, *, token: str | None = TOKEN, headers=None):
    merged = dict(headers or {})
    if token is not None:
        merged[PROBE_TOKEN_HEADER] = token
    return client.post(
        endpoint,
        files={"file": ("probe.mseed", b"probe-bytes")},
        headers=merged,
    )


def _manifest(cap: Path) -> dict:
    return json.loads((cap / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[-1])


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.99.2.3", True),
        ("::1", True),
        ("[::1]", True),
        ("localhost", False),
        ("10.0.0.1", False),
        ("198.51.100.10", False),
        (None, False),
    ],
)
def test_loopback_detection_uses_numeric_direct_peer_only(host, expected):
    assert is_loopback_host(host) is expected


def test_trusted_probe_requires_both_loopback_and_exact_token():
    assert is_trusted_probe(
        peer_host="127.0.0.1", supplied_token=TOKEN, expected_token=TOKEN
    )
    assert not is_trusted_probe(
        peer_host="198.51.100.10", supplied_token=TOKEN, expected_token=TOKEN
    )
    assert not is_trusted_probe(
        peer_host="127.0.0.1", supplied_token="B" * 48, expected_token=TOKEN
    )
    assert not is_trusted_probe(
        peer_host="127.0.0.1", supplied_token=TOKEN, expected_token=None
    )


def test_probe_token_file_validation_is_fail_closed(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="missing or unreadable"):
        load_probe_token_file(missing, require_private=False)

    short = tmp_path / "short"
    short.write_text("too-short\n", encoding="ascii")
    with pytest.raises(ValueError, match="32-512"):
        load_probe_token_file(short, require_private=False)

    long = tmp_path / "long"
    long.write_text("A" * 513, encoding="ascii")
    with pytest.raises(ValueError, match="32-512"):
        load_probe_token_file(long, require_private=False)

    invalid = tmp_path / "invalid"
    invalid.write_text("A" * 40 + ":bad", encoding="ascii")
    with pytest.raises(ValueError, match="URL-safe"):
        load_probe_token_file(invalid, require_private=False)

    valid = tmp_path / "valid"
    valid.write_text(TOKEN + "\n", encoding="ascii")
    if os.name == "posix":
        valid.chmod(0o600)
    assert load_probe_token_file(valid, require_private=True) == TOKEN

    if os.name == "posix":
        valid.chmod(0o644)
        with pytest.raises(ValueError, match="permissions"):
            load_probe_token_file(valid, require_private=True)


def test_probe_token_file_rejects_symbolic_links_when_supported(tmp_path):
    target = tmp_path / "target"
    target.write_text(TOKEN, encoding="ascii")
    link = tmp_path / "link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable in this test environment")
    with pytest.raises(ValueError, match="symbolic link"):
        load_probe_token_file(link, require_private=False)


@pytest.mark.parametrize("peer", ["127.0.0.1", "::1"])
@pytest.mark.parametrize("endpoint", ["/pick", "/magnitude", "/classify"])
def test_authenticated_loopback_probe_runs_inference_without_capture(
    tmp_path, peer, endpoint
):
    cap = tmp_path / "cap"
    response = _post(_client(cap, peer=peer), endpoint)
    assert response.status_code == 200
    assert response.headers[PROBE_ACCEPTED_HEADER] == PROBE_ACCEPTED_VALUE
    assert not cap.exists()


@pytest.mark.parametrize("supplied", [None, "B" * 48])
def test_loopback_missing_or_wrong_token_is_captured(tmp_path, supplied):
    cap = tmp_path / "cap"
    response = _post(
        _client(cap, peer="127.0.0.1"),
        "/pick",
        token=supplied,
    )
    assert response.status_code == 200
    assert PROBE_ACCEPTED_HEADER not in response.headers
    assert _manifest(cap)["endpoint"] == "pick"


def test_public_peer_cannot_forge_bypass_with_token_and_proxy_headers(tmp_path):
    cap = tmp_path / "cap"
    response = _post(
        _client(cap, peer="198.51.100.10"),
        "/pick",
        headers={
            "X-Forwarded-For": "127.0.0.1",
            "X-Real-IP": "127.0.0.1",
        },
    )
    assert response.status_code == 200
    assert PROBE_ACCEPTED_HEADER not in response.headers
    record = _manifest(cap)
    assert record["endpoint"] == "pick"
    # Proxy headers remain capture metadata, but did not authorize bypass.
    assert record["client_ip"] == "127.0.0.1"


def test_probe_header_cannot_bypass_when_server_has_no_token_configured(tmp_path):
    cap = tmp_path / "cap"
    response = _post(
        _client(cap, peer="127.0.0.1", token=None),
        "/pick",
    )
    assert response.status_code == 200
    assert PROBE_ACCEPTED_HEADER not in response.headers
    assert _manifest(cap)["endpoint"] == "pick"


class _FakeResponse:
    status_code = 200

    def __init__(self, *, accepted: bool):
        self.headers = (
            {PROBE_ACCEPTED_HEADER: PROBE_ACCEPTED_VALUE} if accepted else {}
        )

    @staticmethod
    def json():
        return {}


def test_check_api_probe_mode_requires_server_acceptance(tmp_path, monkeypatch, capsys):
    sample = tmp_path / "probe.mseed"
    sample.write_bytes(b"not-parsed-by-client")
    token_file = tmp_path / "probe.token"
    token_file.write_text(TOKEN + "\n", encoding="ascii")
    observed = []

    def accepted_post(*_args, **kwargs):
        observed.append(kwargs)
        return _FakeResponse(accepted=True)

    import requests

    monkeypatch.setattr(requests, "post", accepted_post)
    args = [
        "--url",
        "http://127.0.0.1:8000/pick",
        "--input",
        str(sample),
        "--probe-token-file",
        str(token_file),
    ]
    assert check_api.main(args) == 0
    assert observed[0]["headers"] == {PROBE_TOKEN_HEADER: TOKEN}
    assert TOKEN not in capsys.readouterr().out

    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: _FakeResponse(accepted=False),
    )
    assert check_api.main(args) == 1
    output = capsys.readouterr()
    assert "未确认 authenticated loopback probe" in output.out
    assert TOKEN not in output.out + output.err


def test_check_api_never_sends_probe_token_to_non_loopback_url(tmp_path, monkeypatch):
    sample = tmp_path / "probe.mseed"
    sample.write_bytes(b"sample")
    token_file = tmp_path / "probe.token"
    token_file.write_text(TOKEN + "\n", encoding="ascii")

    import requests

    called = []
    monkeypatch.setattr(requests, "post", lambda *_a, **_k: called.append(True))
    with pytest.raises(SystemExit, match="loopback URL"):
        check_api.main(
            [
                "--url",
                "http://198.51.100.10:8000/pick",
                "--input",
                str(sample),
                "--probe-token-file",
                str(token_file),
            ]
        )
    assert called == []


def test_deploy_and_watchdog_scripts_fail_closed_and_never_embed_token_value():
    deploy = (ROOT / "deploy" / "deploy_api.sh").read_text(encoding="utf-8")
    watchdog = (ROOT / "deploy" / "watchdog.sh").read_text(encoding="utf-8")
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    verify_pos = deploy.index("verify_release_manifest.py")
    token_pos = deploy.index("secrets.token_urlsafe(48)")
    restart_pos = deploy.index("systemctl restart phasepick-api")
    assert verify_pos < token_pos < restart_pos
    assert "--probe-token-file $PROBE_TOKEN_FILE" in deploy
    assert '--probe-token-file "$PROBE_TOKEN_FILE"' in deploy
    assert 'PROBE_TOKEN_DIR="$REPO_ROOT/.runtime"' in deploy
    assert 'if [ -L "$PROBE_TOKEN_FILE" ]' in deploy
    assert "PROBE_TOKEN_FILE" in watchdog
    assert "缺失或不可读" in watchdog
    assert "--probe-token-file" in watchdog
    assert "is_loopback_host" in watchdog
    assert (
        'if ! LAST_LOG="$(mktemp "${TMPDIR:-/tmp}/phasepick_watchdog_last.XXXXXX")"'
        in watchdog
    )
    assert '> "$LAST_LOG" 2>&1' in watchdog
    assert 'tail -5 "$LAST_LOG"' in watchdog
    assert "trap cleanup_last_log EXIT" in watchdog
    assert "/tmp/phasepick_watchdog_last.log" not in watchdog
    assert ".runtime/" in ignore


def test_probe_cli_default_is_disabled_outside_deploy_script():
    args = serve_api.make_arg_parser().parse_args([])
    assert args.probe_token_file is None


def test_invalid_probe_token_fails_before_model_construction(tmp_path, monkeypatch):
    token_file = tmp_path / "short.token"
    token_file.write_text("short", encoding="ascii")
    constructed = []
    monkeypatch.setattr(
        serve_api,
        "build_engine",
        lambda _args: constructed.append(True),
    )
    with pytest.raises(SystemExit, match="probe token 配置无效"):
        serve_api.main(["--probe-token-file", str(token_file), "--no-warmup"])
    assert constructed == []
