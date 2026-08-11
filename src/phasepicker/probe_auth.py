"""Authentication helpers for loopback-only watchdog probes.

Forwarded client-IP headers are deliberately outside this module: they are
useful capture metadata but are never an authorization signal.
"""

from __future__ import annotations

import ipaddress
import os
import re
import secrets
import stat
from pathlib import Path


PROBE_TOKEN_HEADER = "X-PhasePicker-Probe-Token"
PROBE_ACCEPTED_HEADER = "X-PhasePicker-Probe"
PROBE_ACCEPTED_VALUE = "accepted"
PROBE_TOKEN_MIN_CHARS = 32
PROBE_TOKEN_MAX_CHARS = 512
_TOKEN_RE = re.compile(
    rf"^[A-Za-z0-9_-]{{{PROBE_TOKEN_MIN_CHARS},{PROBE_TOKEN_MAX_CHARS}}}$"
)


def load_probe_token_file(path: str | os.PathLike[str], *, require_private: bool) -> str:
    """Load and validate one URL-safe token without ever returning file content in errors."""

    token_path = Path(path)
    try:
        info = token_path.lstat()
    except OSError as exc:
        raise ValueError("probe token file is missing or unreadable") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ValueError("probe token file must not be a symbolic link")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("probe token path is not a regular file")
    if require_private and os.name == "posix" and stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("probe token file permissions must be 0600 or stricter")
    try:
        with token_path.open("r", encoding="ascii") as handle:
            raw = handle.read(PROBE_TOKEN_MAX_CHARS + 2)
    except (OSError, UnicodeError) as exc:
        raise ValueError("probe token file is missing or unreadable") from exc
    token = raw.strip()
    if not _TOKEN_RE.fullmatch(token):
        raise ValueError(
            "probe token must be 32-512 URL-safe ASCII characters"
        )
    return token


def is_loopback_host(host: object) -> bool:
    """Return true only for a numeric IPv4/IPv6 loopback address."""

    if not isinstance(host, str) or not host:
        return False
    normalized = host.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def is_trusted_probe(
    *,
    peer_host: object,
    supplied_token: object,
    expected_token: str | None,
) -> bool:
    """Authenticate a probe using the direct peer and a constant-time token check."""

    if (
        not expected_token
        or not isinstance(supplied_token, str)
        or not is_loopback_host(peer_host)
    ):
        return False
    return secrets.compare_digest(supplied_token, expected_token)
