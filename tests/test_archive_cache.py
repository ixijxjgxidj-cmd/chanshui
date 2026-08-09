"""official_waveforms 归档缓存的正确性 / 等价性 / 并发安全测试——纯标准库.

缓存是性能优化，正确性标准只有一条：**与无缓存路径逐字节一致**。覆盖：
- 普通文件、单层 zip、嵌套 zip 三种 source_path 形态
- 缓存路径 vs _read_source_bytes_nocache 的字节等价
- 重复读取（命中缓存）仍等价
- 多线程并发读取等价且不崩
- clear_archive_cache 后仍能正常读取
- 找不到条目时抛 KeyError（与旧行为一致）

两种运行方式：
    pytest tests/test_archive_cache.py
    python  tests/test_archive_cache.py
"""

import io
import os
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.io.official_waveforms import (  # noqa: E402
    _read_source_bytes_nocache,
    clear_archive_cache,
    read_source_bytes,
)


def _build_fixture(tmp: str):
    """造一个 外层zip!内层zip!mseed 的迷你官方包，返回 (外层路径, 条目字节表)。"""
    payloads = {
        f"exam/T1-Q/T1.A.Q{i:04d}.mseed": (f"MSEED-DATA-{i}-".encode() * 50)
        for i in range(1, 21)
    }
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in payloads.items():
            zf.writestr(name, data)

    outer_path = os.path.join(tmp, "round2.zip")
    with zipfile.ZipFile(outer_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("exam-data.zip", inner.getvalue())
        zf.writestr("readme.txt", "hello")
    return outer_path, payloads


def test_plain_file_read():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "a.mseed")
        with open(p, "wb") as f:
            f.write(b"RAWBYTES")
        assert read_source_bytes(p) == b"RAWBYTES"


def test_cached_equals_nocache_all_entries():
    clear_archive_cache()
    with tempfile.TemporaryDirectory() as tmp:
        outer, payloads = _build_fixture(tmp)
        for name, expect in payloads.items():
            src = f"{outer}!exam-data.zip!{name}"
            assert read_source_bytes(src) == expect, name
            assert _read_source_bytes_nocache(src) == expect, name
        # 第二遍：全部命中缓存，仍等价
        for name, expect in payloads.items():
            src = f"{outer}!exam-data.zip!{name}"
            assert read_source_bytes(src) == expect, name
        clear_archive_cache()  # Windows: 先关句柄再让 TemporaryDirectory 删文件
    clear_archive_cache()


def test_single_layer_zip():
    clear_archive_cache()
    with tempfile.TemporaryDirectory() as tmp:
        outer = os.path.join(tmp, "round1.zip")
        with zipfile.ZipFile(outer, "w") as zf:
            zf.writestr("exam/TASK01/T1.A.Q0001.mseed", b"ONE-LAYER")
        src = f"{outer}!exam/TASK01/T1.A.Q0001.mseed"
        assert read_source_bytes(src) == b"ONE-LAYER"
        assert read_source_bytes(src) == _read_source_bytes_nocache(src)
        clear_archive_cache()  # Windows: 先关句柄再让 TemporaryDirectory 删文件
    clear_archive_cache()


def test_concurrent_reads():
    clear_archive_cache()
    with tempfile.TemporaryDirectory() as tmp:
        outer, payloads = _build_fixture(tmp)
        items = [(f"{outer}!exam-data.zip!{n}", d) for n, d in payloads.items()] * 8

        def work(item):
            src, expect = item
            return read_source_bytes(src) == expect

        with ThreadPoolExecutor(max_workers=8) as ex:
            assert all(ex.map(work, items))
    clear_archive_cache()


def test_clear_cache_then_read_again():
    with tempfile.TemporaryDirectory() as tmp:
        outer, payloads = _build_fixture(tmp)
        name = next(iter(payloads))
        src = f"{outer}!exam-data.zip!{name}"
        assert read_source_bytes(src) == payloads[name]
        clear_archive_cache()
        assert read_source_bytes(src) == payloads[name]
        clear_archive_cache()


def test_missing_entry_raises_keyerror():
    clear_archive_cache()
    with tempfile.TemporaryDirectory() as tmp:
        outer, _ = _build_fixture(tmp)
        src = f"{outer}!exam-data.zip!exam/T1-Q/NOT_EXIST.mseed"
        try:
            read_source_bytes(src)
        except KeyError:
            pass
        else:
            raise AssertionError("缺失条目应抛 KeyError")
        clear_archive_cache()  # Windows: 先关句柄再让 TemporaryDirectory 删文件
    clear_archive_cache()


if __name__ == "__main__":
    for fn in [
        test_plain_file_read,
        test_cached_equals_nocache_all_entries,
        test_single_layer_zip,
        test_concurrent_reads,
        test_clear_cache_then_read_again,
        test_missing_entry_raises_keyerror,
    ]:
        fn()
        print(f"{fn.__name__} ok")
    print("ALL OK")
