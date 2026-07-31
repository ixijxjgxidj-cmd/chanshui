#!/usr/bin/env python3
"""官方 API 提交前自检客户端（Standard-format API checker）.

对着**已经跑起来的** serve_api.py（本机或公网），模拟官方评测行为：
    resp = requests.post(url, files=files)
逐文件上传 mseed，校验：
  1. HTTP 状态码 == 200；
  2. 响应是 JSON 对象：{台站名: {"P": [到时...], "S": [到时...]}}；
  3. 每个到时都是官方示例风格的 UTC 字符串（``2025-06-07T12:34:56.789000Z``），
     可被解析且升序；
  4. 统计延迟（均值/中位/最大）与 P/S 总数。
全部通过打印 “API 自检通过”；任何一条不符打印明细并以非零码退出。

用法：
    python scripts/check_api.py --url http://127.0.0.1:8000/pick --input <目录或zip> [--limit 20]
"""

from __future__ import annotations

import argparse
import io
import os
import re
import statistics
import sys
import zipfile
from typing import Iterable, List, Tuple

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


def iter_mseed(path: str) -> Iterable[Tuple[str, bytes]]:
    """枚举目录或（嵌套）zip 里的所有 .mseed → (名字, 字节)。"""
    if os.path.isdir(path):
        for dirpath, _dirs, files in os.walk(path):
            for fn in sorted(files):
                if fn.lower().endswith(".mseed"):
                    full = os.path.join(dirpath, fn)
                    with open(full, "rb") as f:
                        yield fn, f.read()
        return
    if zipfile.is_zipfile(path):
        yield from _iter_zip(path)
        return
    if path.lower().endswith(".mseed"):
        with open(path, "rb") as f:
            yield os.path.basename(path), f.read()
        return
    raise SystemExit(f"--input 不是目录/zip/mseed：{path}")


def _iter_zip(zip_path: str, depth: int = 3) -> Iterable[Tuple[str, bytes]]:
    def walk(zf: zipfile.ZipFile, d: int) -> Iterable[Tuple[str, bytes]]:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name.lower().endswith(".zip") and d > 0:
                try:
                    with zipfile.ZipFile(io.BytesIO(zf.read(info)), "r") as inner:
                        yield from walk(inner, d - 1)
                except zipfile.BadZipFile:
                    pass
            elif name.lower().endswith(".mseed"):
                yield os.path.basename(name), zf.read(info)

    with zipfile.ZipFile(zip_path, "r") as zf:
        yield from walk(zf, depth)


def validate_payload(payload) -> List[str]:
    """校验一次响应体是否完全符合官方 JSON 规范；返回问题列表（空=通过）。"""
    problems: List[str] = []
    if not isinstance(payload, dict):
        return [f"最外层应是对象（台站名→结果），实际是 {type(payload).__name__}"]
    for sta, slot in payload.items():
        if not isinstance(sta, str) or not sta:
            problems.append(f"非法台站键：{sta!r}")
            continue
        if not isinstance(slot, dict) or set(slot.keys()) - {"P", "S"}:
            problems.append(f"[{sta}] 每台站应为 {{'P': [...], 'S': [...]}}，实际 {slot!r}")
            continue
        for phase in ("P", "S"):
            times = slot.get(phase, [])
            if not isinstance(times, list):
                problems.append(f"[{sta}].{phase} 应是列表")
                continue
            prev = None
            for t in times:
                if not isinstance(t, str) or not _ISO_RE.match(t):
                    problems.append(
                        f"[{sta}].{phase} 到时格式错误：{t!r}"
                        "（应如 2025-06-07T12:34:56.789000Z）"
                    )
                    break
                if prev is not None and t < prev:
                    problems.append(f"[{sta}].{phase} 到时未升序：{prev} → {t}")
                    break
                prev = t
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="官方 API 标准格式自检")
    ap.add_argument("--url", required=True, help="API 地址，如 http://127.0.0.1:8000/pick")
    ap.add_argument("--input", required=True, help="mseed 目录 / zip / 单文件")
    ap.add_argument("--limit", type=int, default=None, help="只测前 N 个文件")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args(argv)

    try:
        import requests
    except ImportError:
        raise SystemExit("请先安装 requests：pip install requests")

    import time

    n = 0
    n_fail = 0
    latencies: List[float] = []
    total_p = total_s = 0
    for name, raw in iter_mseed(args.input):
        if args.limit is not None and n >= args.limit:
            break
        n += 1
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                args.url, files={"file": (name, raw)}, timeout=args.timeout
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name}: 请求异常 {exc!r}")
            n_fail += 1
            continue
        dt = time.perf_counter() - t0
        latencies.append(dt)

        if resp.status_code != 200:
            print(f"[FAIL] {name}: HTTP {resp.status_code}（应为 200）")
            n_fail += 1
            continue
        try:
            payload = resp.json()
        except ValueError:
            print(f"[FAIL] {name}: 响应不是 JSON")
            n_fail += 1
            continue
        problems = validate_payload(payload)
        if problems:
            n_fail += 1
            print(f"[FAIL] {name}:")
            for p in problems[:5]:
                print(f"    - {p}")
            continue
        p_cnt = sum(len(v["P"]) for v in payload.values())
        s_cnt = sum(len(v["S"]) for v in payload.values())
        total_p += p_cnt
        total_s += s_cnt
        print(f"[ok] {name}: {len(payload)} 台站, P={p_cnt}, S={s_cnt}, {dt*1000:.0f}ms")

    if n == 0:
        print("没有找到任何 .mseed", file=sys.stderr)
        return 2
    print("-" * 60)
    if latencies:
        print(
            f"延迟: 均值 {statistics.mean(latencies)*1000:.0f}ms / "
            f"中位 {statistics.median(latencies)*1000:.0f}ms / "
            f"最大 {max(latencies)*1000:.0f}ms"
        )
    print(f"文件 {n} 个，失败 {n_fail}；P 到时共 {total_p}，S 到时共 {total_s}")
    if n_fail:
        print("API 自检未通过", file=sys.stderr)
        return 1
    print("API 自检通过：状态码 / JSON 结构 / 到时格式 / 升序 全部符合官方标准")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
