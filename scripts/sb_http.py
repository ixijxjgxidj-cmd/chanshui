"""HTTP Range 读取器 + h5py fileobj 驱动：不下载全量即可读远端 SeisBench HDF5。

为什么需要：zzai 容器出网只有 ~1.6 MB/s，ETHZ waveforms.hdf5 = 23.2 GB，
全量下载约 4 小时；而本项目只需要其中的远场子集（S-P >= 阈值）。
DESY 存储已验证 Accept-Ranges: bytes，所以按需取块即可。

用法：
    from sb_http import HttpRangeFile
    import h5py
    with h5py.File(HttpRangeFile(url), "r", driver="fileobj") as f: ...
"""
from __future__ import annotations

import threading
import urllib.request


class HttpRangeFile:
    """只读、可 seek 的类文件对象，底层用 HTTP Range 取块并做 LRU 缓存。"""

    def __init__(self, url: str, block: int = 1 << 20, max_blocks: int = 512,
                 timeout: float = 120.0, retries: int = 4):
        self.url = url
        self.block = int(block)
        self.max_blocks = int(max_blocks)
        self.timeout = timeout
        self.retries = retries
        self._pos = 0
        self._cache: dict[int, bytes] = {}
        self._order: list[int] = []
        self._lock = threading.Lock()
        self.bytes_fetched = 0
        self.requests = 0
        self.size = self._head_size()

    # --- HTTP ---------------------------------------------------------
    def _head_size(self) -> int:
        req = urllib.request.Request(self.url, method="HEAD")
        last = None
        for _ in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    n = r.headers.get("Content-Length")
                    if n is not None:
                        return int(n)
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise IOError(f"HEAD failed for {self.url}: {last}")

    def _fetch(self, lo: int, hi: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
        last = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = r.read()
                self.requests += 1
                self.bytes_fetched += len(data)
                return data
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise IOError(f"range {lo}-{hi} failed: {last}")

    def _block_data(self, idx: int) -> bytes:
        with self._lock:
            hit = self._cache.get(idx)
            if hit is not None:
                return hit
        lo = idx * self.block
        hi = min(lo + self.block, self.size) - 1
        data = self._fetch(lo, hi)
        with self._lock:
            self._cache[idx] = data
            self._order.append(idx)
            while len(self._order) > self.max_blocks:
                self._cache.pop(self._order.pop(0), None)
        return data

    # --- file protocol ------------------------------------------------
    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self.size + offset
        else:
            raise ValueError(whence)
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self._pos
        n = max(0, min(n, self.size - self._pos))
        if n == 0:
            return b""
        out = bytearray()
        pos = self._pos
        end = pos + n
        while pos < end:
            idx = pos // self.block
            data = self._block_data(idx)
            off = pos - idx * self.block
            take = min(len(data) - off, end - pos)
            out += data[off:off + take]
            pos += take
        self._pos = end
        return bytes(out)

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
            self._order.clear()

    @property
    def stats(self) -> dict:
        return {"requests": self.requests,
                "mb_fetched": round(self.bytes_fetched / 2 ** 20, 2),
                "size_gb": round(self.size / 2 ** 30, 2)}