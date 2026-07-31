"""官方比赛包中的波形与答案读取。

支持三种来源：
1. 普通 ``.mseed`` 文件；
2. 外层 zip 里的 mseed；
3. 外层 zip → 内层 zip → mseed（第 2 轮真实布局）。

``source_path`` 使用 ``!`` 分隔归档层级，例如：
``round2.zip!exam-data07.zip!exam-data07/T2-Q/T2.Q0001.mseed``。
所有读取均为只读，不把官方大包解压到磁盘。
"""

from __future__ import annotations

import io
import os
import threading
import zipfile
from typing import Dict, Iterable, Optional, Union

from ..types import ExamTask, Task1Result, Task2Result, Task3Result
from .official_answers import (
    normalize_file_id,
    parse_task1_answer_lines,
    parse_task2_answer_lines,
    parse_task3_answer_lines,
)


def _decode_entry_name(info: zipfile.ZipInfo, metadata_encoding: str) -> str:
    name = info.filename
    if info.flag_bits & 0x800 == 0:
        try:
            return name.encode("cp437").decode(metadata_encoding)
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name


def _find_info(zf: zipfile.ZipFile, wanted: str, metadata_encoding: str) -> zipfile.ZipInfo:
    """按原始名或还原后的名字查找条目。"""
    try:
        return zf.getinfo(wanted)
    except KeyError:
        pass
    wanted_norm = wanted.replace("\\", "/")
    for info in zf.infolist():
        decoded = _decode_entry_name(info, metadata_encoding)
        if info.filename.replace("\\", "/") == wanted_norm:
            return info
        if decoded.replace("\\", "/") == wanted_norm:
            return info
    raise KeyError(f"zip 中找不到条目：{wanted}")


# =========================================================================
# 归档缓存（性能关键）
# =========================================================================
# 没有缓存时，每读一个样本都要：重新打开外层 zip → 把整个内层 zip（第2轮
# data zip 有 66MB+）完整解压进内存 → 再取出一个几十 KB 的 mseed。N 个样本
# 就是 N 次全量解压，代价 O(N × 内层zip体积)，是整条推理链路最大的隐形瓶颈。
#
# 缓存策略（全部只读、进程内）：
# - _NESTED_BYTES：嵌套归档链前缀（如 ``outer.zip!data.zip``）→ 该内层 zip 的
#   原始字节。只解压一次，之后所有样本共享；字节不可变，天然线程安全。
# - _NAME_INDEX：归档 key → {规范化条目名: 原始条目名}。把逐条 infolist 线性
#   扫描（O(条目数) / 次）换成 O(1) 字典查找；同时收录 GBK 还原名做兼容。
# - _TLS.handles：每线程各持一份 ZipFile 句柄（ZipFile.read 共享文件指针，
#   跨线程共用同一句柄不安全；句柄本身开销极小）。
#
# 由此 read_source_bytes 的均摊代价从"全量解压内层 zip"降到"解压单个成员"，
# 并且可以被多线程加载器安全并发调用。

_CACHE_LOCK = threading.Lock()
_NESTED_BYTES: Dict[str, bytes] = {}
_NAME_INDEX: Dict[str, Dict[str, str]] = {}
_TLS = threading.local()


def clear_archive_cache() -> None:
    """清空归档缓存（关闭当前线程句柄、丢弃内层 zip 字节与名称索引）。"""
    with _CACHE_LOCK:
        _NESTED_BYTES.clear()
        _NAME_INDEX.clear()
    handles = getattr(_TLS, "handles", None)
    if handles:
        for zf in handles.values():
            try:
                zf.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响正确性
                pass
        handles.clear()


def _get_handle(key: str) -> zipfile.ZipFile:
    """取当前线程持有的 ZipFile 句柄；没有则打开（磁盘路径或缓存的内层字节）。"""
    handles = getattr(_TLS, "handles", None)
    if handles is None:
        handles = {}
        _TLS.handles = handles
    zf = handles.get(key)
    if zf is not None:
        return zf
    if "!" in key:
        with _CACHE_LOCK:
            raw = _NESTED_BYTES.get(key)
        if raw is None:
            raise KeyError(f"内层归档字节未缓存：{key}")
        zf = zipfile.ZipFile(io.BytesIO(raw), "r")
    else:
        zf = zipfile.ZipFile(key, "r")
    handles[key] = zf
    return zf


def _get_name_index(key: str, zf: zipfile.ZipFile, metadata_encoding: str) -> Dict[str, str]:
    """构建/获取该归档的条目名索引：规范化名（原始 + GBK 还原）→ 原始名。"""
    with _CACHE_LOCK:
        index = _NAME_INDEX.get(key)
        if index is not None:
            return index
    index = {}
    for info in zf.infolist():
        raw_name = info.filename
        index.setdefault(raw_name.replace("\\", "/"), raw_name)
        decoded = _decode_entry_name(info, metadata_encoding)
        if decoded != raw_name:
            index.setdefault(decoded.replace("\\", "/"), raw_name)
    with _CACHE_LOCK:
        return _NAME_INDEX.setdefault(key, index)


def _resolve_member(key: str, zf: zipfile.ZipFile, wanted: str, metadata_encoding: str) -> str:
    """把请求的条目名解析成归档内的原始条目名（O(1) 索引查找）。"""
    try:
        zf.getinfo(wanted)
        return wanted
    except KeyError:
        pass
    index = _get_name_index(key, zf, metadata_encoding)
    resolved = index.get(str(wanted).replace("\\", "/"))
    if resolved is None:
        raise KeyError(f"zip 中找不到条目：{wanted}")
    return resolved


def _ensure_nested_bytes(parent_key: str, member: str, metadata_encoding: str) -> str:
    """确保 ``parent_key!member`` 的内层归档字节已缓存；返回子归档 key。"""
    zf = _get_handle(parent_key)
    raw_member = _resolve_member(parent_key, zf, member, metadata_encoding)
    child_key = f"{parent_key}!{raw_member}"
    with _CACHE_LOCK:
        if child_key in _NESTED_BYTES:
            return child_key
    data = zf.read(raw_member)  # 每线程独立句柄，读取无共享指针竞争
    with _CACHE_LOCK:
        _NESTED_BYTES.setdefault(child_key, data)
    return child_key


def _read_source_bytes_nocache(source_path: str, metadata_encoding: str = "gbk") -> bytes:
    """无缓存实现（原始逻辑）：每次全量重开归档链。保留作行为基准与兜底。"""
    parts = str(source_path).split("!")
    outer, chain = parts[0], parts[1:]
    with zipfile.ZipFile(outer, "r") as zf:
        current: Union[zipfile.ZipFile, None] = zf
        owned: list[zipfile.ZipFile] = []
        try:
            for index, name in enumerate(chain):
                assert current is not None
                info = _find_info(current, name, metadata_encoding)
                raw = current.read(info)
                if index == len(chain) - 1:
                    return raw
                inner = zipfile.ZipFile(io.BytesIO(raw), "r")
                owned.append(inner)
                current = inner
        finally:
            for inner in reversed(owned):
                inner.close()
    raise ValueError(f"无效 source_path：{source_path}")


def read_source_bytes(
    source_path: str,
    metadata_encoding: str = "gbk",
    use_cache: bool = True,
) -> bytes:
    """读取普通文件或 ``!`` 嵌套归档链指向的最终文件字节。

    默认走进程内归档缓存：外层 zip 句柄按线程复用、内层 zip 只解压一次、
    条目名 O(1) 索引。行为与无缓存路径完全一致（只读，不落盘），
    ``use_cache=False`` 可回退到逐次全量解压的原始实现。
    """
    parts = str(source_path).split("!")
    if len(parts) == 1:
        with open(source_path, "rb") as f:
            return f.read()
    if not use_cache:
        return _read_source_bytes_nocache(source_path, metadata_encoding)

    key = parts[0]
    for member in parts[1:-1]:
        key = _ensure_nested_bytes(key, member, metadata_encoding)
    zf = _get_handle(key)
    final = _resolve_member(key, zf, parts[-1], metadata_encoding)
    return zf.read(final)


def read_mseed_stream(source_path: str, metadata_encoding: str = "gbk"):
    """用 ObsPy 读取 source_path 指向的 mseed，返回 ``obspy.Stream``。

    ObsPy 延迟导入：纯格式解析/单元测试环境没有安装 ObsPy 时，其他模块仍能 import。
    """
    try:
        from obspy import read
    except ImportError as exc:
        raise RuntimeError("读取真实 mseed 需要安装 obspy：python -m pip install obspy") from exc

    if "!" not in str(source_path):
        return read(source_path, format="MSEED")
    return read(io.BytesIO(read_source_bytes(source_path, metadata_encoding)), format="MSEED")


_ANSWER_PREFIX = {
    ExamTask.T1: "t1.an",
    ExamTask.T2: "t2.an",
    ExamTask.T3: "t3.an",
}


def _iter_small_entries(
    zf: zipfile.ZipFile,
    metadata_encoding: str,
    depth: int,
) -> Iterable[tuple[str, bytes]]:
    """递归遍历答案等小文件；mseed 不读入内存。"""
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = _decode_entry_name(info, metadata_encoding)
        if info.filename.lower().endswith(".zip") and depth > 0:
            try:
                with zipfile.ZipFile(io.BytesIO(zf.read(info)), "r") as inner:
                    yield from _iter_small_entries(inner, metadata_encoding, depth - 1)
            except zipfile.BadZipFile:
                pass
            continue
        if not normalize_file_id(name).lower().endswith(".mseed"):
            yield name, zf.read(info)


def read_package_answers(
    zip_path: str,
    task: ExamTask,
    metadata_encoding: str = "gbk",
    max_depth: int = 3,
) -> Dict[str, Union[Task1Result, Task2Result, Task3Result]]:
    """从单层或嵌套官方 zip 中找到指定任务答案并解析。"""
    marker = _ANSWER_PREFIX[task]
    chosen: Optional[bytes] = None
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name, raw in _iter_small_entries(zf, metadata_encoding, max_depth):
            if normalize_file_id(name).lower().startswith(marker):
                chosen = raw
                break
    if chosen is None:
        raise KeyError(f"官方包中找不到 {task.value} 答案文件（{marker}[.txt]）")

    text: Optional[str] = None
    for enc in ("utf-8", metadata_encoding):
        try:
            text = chosen.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = chosen.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if task is ExamTask.T1:
        return parse_task1_answer_lines(lines)
    if task is ExamTask.T2:
        return parse_task2_answer_lines(lines)
    return parse_task3_answer_lines(lines)
