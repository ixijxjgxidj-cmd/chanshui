#!/usr/bin/env python3
"""远程 API 真题打分（Remote API end-to-end scoring）.

和 check_api.py 的区别：check_api 只验"格式对不对"，本脚本验"能拿几分" ——
完全模拟官方评测行为（``requests.post(url, files=files)``），把去年真题
逐文件打到**线上 API**，再用官方评分规则（scoring.scorer）对答案打分。

===== 关键：两套时间坐标的还原 =====
线上 API 按官方要求返回**绝对 UTC** 到时（台站名为键）；官方 T1 答案却是
**相对波形起点的秒**。要打分必须还原这次换算，而换算的锚点是"每台站自己的
起点"（见 tasks/task1_runner.py 的坐标系说明）：

    relative_s = iso_to_epoch(api_time) - waveform.starttime_utc

所以本脚本在发请求的同时，本地也用同一个 mseed_reader 读一遍波形，拿到每台站
的 starttime_utc，并复刻 serve_api._station_keys 的台站键规则（默认砍成台站名，
同名冲突退回完整 NET.STA），把响应键映回锚点。这样得到的 Task1Result 与
run_official_task1.py 的本地链路**同坐标系、可直接对比**。

用法：
    # 冒烟（先跑 30 个确认链路对齐）
    python scripts/score_remote_api.py --url https://HOST/pick \\
        --package "第1轮比赛试题与答案.zip" --limit 30

    # 正式全量 + 落盘
    python scripts/score_remote_api.py --url https://HOST/pick \\
        --package "第1轮比赛试题与答案.zip" --strict-missing \\
        --out outputs/remote_r1.an --json outputs/remote_r1.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.types import ExamTask, Task1Result  # noqa: E402
from phasepicker.scoring.scorer import DEFAULT_PENALTY_MODE, PENALTY_MODES  # noqa: E402
from phasepicker.io.official_exam import scan_exam_input  # noqa: E402
from phasepicker.io.official_waveforms import read_source_bytes, read_package_answers  # noqa: E402


def iso_to_epoch(s: str) -> float:
    """``2025-06-07T12:34:56.789000Z`` → epoch 秒。容忍无 Z / 有偏移的写法。"""
    t = s.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def station_keys(waveforms) -> List[str]:
    """复刻 serve_api._station_keys：NET.STA → 台站名，同名冲突退回完整 NET.STA。"""
    keys: List[str] = []
    seen: Dict[str, int] = {}
    for wf in waveforms:
        s = (wf.station or "STA").strip()
        k = s.rsplit(".", 1)[-1] or s
        seen[k] = seen.get(k, 0) + 1
        keys.append(k)
    for i, wf in enumerate(waveforms):
        if seen[keys[i]] > 1:
            keys[i] = (wf.station or keys[i]).strip()
    return keys


class Stats:
    """线程安全的跑批计数器。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latencies: List[float] = []
        self.http_fail = 0
        self.fmt_fail = 0
        self.unknown_station = 0
        self.negative_rel = 0
        self.done = 0


def process_one(
    sample,
    url: str,
    timeout: float,
    stats: Stats,
    total: int,
    t0: float,
) -> Tuple[str, Optional[Task1Result], Optional[str]]:
    """一个文件：读字节 → POST → 本地读锚点 → 换算相对秒。返回 (file_id, 结果, 错误)。"""
    import requests

    from phasepicker.io.mseed_reader import load_waveforms

    fid = sample.file_id
    try:
        raw = read_source_bytes(sample.source_path)
    except Exception as exc:  # noqa: BLE001
        return fid, None, f"读取失败 {exc!r}"

    # --- 1) 打线上 API（官方口径：requests.post(url, files=files)）---
    t_req = time.perf_counter()
    try:
        resp = requests.post(url, files={"file": (fid, raw)}, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        with stats.lock:
            stats.http_fail += 1
            stats.done += 1
        return fid, None, f"请求异常 {exc!r}"
    dt = time.perf_counter() - t_req

    if resp.status_code != 200:
        with stats.lock:
            stats.http_fail += 1
            stats.done += 1
        return fid, None, f"HTTP {resp.status_code}"
    try:
        payload = resp.json()
    except ValueError:
        with stats.lock:
            stats.fmt_fail += 1
            stats.done += 1
        return fid, None, "响应不是 JSON"
    if not isinstance(payload, dict):
        with stats.lock:
            stats.fmt_fail += 1
            stats.done += 1
        return fid, None, f"响应最外层不是对象：{type(payload).__name__}"

    # --- 2) 本地读同一份波形，取每台站起点作为换算锚点 ---
    anchors: Dict[str, float] = {}
    try:
        ing = load_waveforms(raw)
        for key, wf in zip(station_keys(ing.waveforms), ing.waveforms):
            anchors[key] = float(wf.starttime_utc)
    except Exception as exc:  # noqa: BLE001
        return fid, None, f"本地读波形失败（无法换算相对秒）{exc!r}"

    # --- 3) 绝对 UTC → 相对波形起点秒 ---
    p_times: List[float] = []
    s_times: List[float] = []
    n_unknown = 0
    n_neg = 0
    for sta, slot in payload.items():
        if not isinstance(slot, dict):
            continue
        anchor = anchors.get(sta)
        if anchor is None:
            # 台站键对不上本地解析结果：只能整站丢弃（无锚点无法换算）
            if any(slot.get(ph) for ph in ("P", "S")):
                n_unknown += 1
            continue
        for phase, bucket in (("P", p_times), ("S", s_times)):
            for t in slot.get(phase, []) or []:
                try:
                    rel = iso_to_epoch(t) - anchor
                except Exception:  # noqa: BLE001
                    n_unknown += 1
                    continue
                if rel < 0.0:
                    n_neg += 1
                    continue
                bucket.append(rel)
    p_times.sort()
    s_times.sort()

    with stats.lock:
        stats.latencies.append(dt)
        stats.unknown_station += n_unknown
        stats.negative_rel += n_neg
        stats.done += 1
        d = stats.done
    if d % 25 == 0 or d == total:
        rate = d / max(1e-9, time.perf_counter() - t0)
        eta = (total - d) / max(1e-9, rate)
        print(
            f"\r进度 {d}/{total}  {rate:.2f} 文件/秒  剩余约 {eta:.0f}s",
            end="" if d < total else "\n",
            flush=True,
        )

    return fid, Task1Result(file_id=fid, p_times_s=p_times, s_times_s=s_times), None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="用去年真题给线上 API 打分（官方评分规则）")
    ap.add_argument("--url", required=True, help="线上 API 地址，如 https://HOST/pick")
    ap.add_argument("--package", required=True, help="官方真题 zip（含答案）")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 个 T1 文件（冒烟用）")
    ap.add_argument("--concurrency", type=int, default=4, help="并发请求数")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--penalty-mode", default=DEFAULT_PENALTY_MODE, choices=PENALTY_MODES)
    ap.add_argument("--penalty-table", action="store_true", help="额外输出四种数量罚读法对照")
    ap.add_argument("--strict-missing", action="store_true",
                    help="答案有而预测无的文件按 0 分计入均分分母（官方口径）")
    ap.add_argument("--out", default=None, help="可选：把远程预测写成 T1.an 格式")
    ap.add_argument("--prefix", default="./T1-Q/", help="--out 的行前缀")
    ap.add_argument("--json", dest="json_out", default=None, help="可选：把汇总指标写成 JSON")
    args = ap.parse_args(argv)

    try:
        import requests  # noqa: F401
    except ImportError:
        raise SystemExit("请先安装 requests：pip install requests")

    samples = [s for s in scan_exam_input(args.package) if s.task == ExamTask.T1]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        print(f"未在 {args.package!r} 找到 T1 样本", file=sys.stderr)
        return 1
    print(f"真题包: {os.path.basename(args.package)}")
    print(f"T1 样本 {len(samples)} 个 → 打到 {args.url}（并发 {args.concurrency}）")

    stats = Stats()
    t0 = time.perf_counter()
    results: Dict[str, Task1Result] = {}
    errors: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futs = [
            pool.submit(process_one, s, args.url, args.timeout, stats, len(samples), t0)
            for s in samples
        ]
        for f in futs:
            fid, res, err = f.result()
            if err is not None:
                errors.append((fid, err))
                # 官方口径：请求失败也是"这个文件没交上答案" → 空预测参与评分
                results[fid] = Task1Result(file_id=fid)
            else:
                results[fid] = res
    wall = time.perf_counter() - t0

    print("-" * 70)
    if stats.latencies:
        print(
            f"延迟: 均值 {statistics.mean(stats.latencies)*1000:.0f}ms / "
            f"中位 {statistics.median(stats.latencies)*1000:.0f}ms / "
            f"最大 {max(stats.latencies)*1000:.0f}ms | "
            f"墙钟 {wall:.1f}s（{len(samples)/max(1e-9,wall):.2f} 文件/秒）"
        )
    print(
        f"请求失败 {stats.http_fail} | 格式异常 {stats.fmt_fail} | "
        f"无锚点台站 {stats.unknown_station} | 负相对到时丢弃 {stats.negative_rel}"
    )
    if errors:
        print(f"失败明细（前 10 / 共 {len(errors)}）:")
        for fid, err in errors[:10]:
            print(f"  - {fid}: {err}")

    n_p = sum(len(r.p_times_s) for r in results.values())
    n_s = sum(len(r.s_times_s) for r in results.values())
    print(f"远程返回震相合计: P={n_p}, S={n_s}")

    if args.out:
        from phasepicker.io.submission_writer import write_task1_results

        ordered = [results[s.file_id] for s in samples if s.file_id in results]
        write_task1_results(ordered, args.out, prefix=args.prefix)
        print(f"已写出 {len(ordered)} 行 → {args.out}")

    # ---- 打分 ----
    from phasepicker.eval.official_eval import (
        evaluate_task1,
        evaluate_task1_all_modes,
        penalty_modes_table,
    )

    answers = read_package_answers(args.package, ExamTask.T1)
    if args.limit is not None:
        # 冒烟模式只对跑过的文件打分，别让没跑的文件把均分打成 0
        keep = {s.file_id for s in samples}
        answers = {k: v for k, v in answers.items() if k in keep}
    report = evaluate_task1(
        results, answers, penalty_mode=args.penalty_mode, strict_missing=args.strict_missing
    )
    print("=" * 70)
    print(report.summary())

    # 逐相位细节：满分率与残差，便于定位是 P 还是 S 拖后腿
    from phasepicker.scoring.scorer import match_phases, phase_time_score

    detail = {}
    for phase, get_pred, get_true in (
        ("P", lambda r: r.p_times_s, lambda r: r.p_times_s),
        ("S", lambda r: r.s_times_s, lambda r: r.s_times_s),
    ):
        res_all: List[float] = []
        n_true = n_pred = 0
        score_sum = 0.0
        for fid in sorted(set(results) & set(answers)):
            pt = list(get_pred(results[fid]))
            tt = list(get_true(answers[fid]))
            n_pred += len(pt)
            n_true += len(tt)
            m = match_phases(pt, tt, phase)
            for _, _, r in m.matched:
                res_all.append(r)
                score_sum += phase_time_score(r, phase)
        full_thr = 0.1 if phase == "P" else 0.2
        n_full = sum(1 for r in res_all if r <= full_thr)
        detail[phase] = {
            "n_true": n_true,
            "n_pred": n_pred,
            "matched": len(res_all),
            "recall": (len(res_all) / n_true) if n_true else 0.0,
            "time_score": score_sum,
            "full_rate_of_matched": (n_full / len(res_all)) if res_all else 0.0,
            "full_rate_of_true": (n_full / n_true) if n_true else 0.0,
            "mean_residual_s": statistics.mean(res_all) if res_all else 0.0,
            "median_residual_s": statistics.median(res_all) if res_all else 0.0,
        }
        d = detail[phase]
        print(
            f"  {phase}: 真值 {n_true} 预测 {n_pred} 匹配 {len(res_all)}"
            f"（召回 {d['recall']*100:.1f}%）| 到时分 {score_sum:.1f} | "
            f"满分率(占真值) {d['full_rate_of_true']*100:.1f}% | "
            f"残差 中位 {d['median_residual_s']*1000:.0f}ms 均值 {d['mean_residual_s']*1000:.0f}ms"
        )

    if args.penalty_table:
        print(penalty_modes_table(
            evaluate_task1_all_modes(results, answers, strict_missing=args.strict_missing)
        ))

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "url": args.url,
                    "package": os.path.basename(args.package),
                    "n_files": report.n_files,
                    "mean_score": report.mean_score,
                    "total_score": report.total_score,
                    "penalty_mode": args.penalty_mode,
                    "strict_missing": args.strict_missing,
                    "missing": len(report.missing),
                    "extra": len(report.extra),
                    "http_fail": stats.http_fail,
                    "fmt_fail": stats.fmt_fail,
                    "unknown_station": stats.unknown_station,
                    "negative_rel_dropped": stats.negative_rel,
                    "latency_mean_ms": (statistics.mean(stats.latencies) * 1000)
                    if stats.latencies else None,
                    "latency_p50_ms": (statistics.median(stats.latencies) * 1000)
                    if stats.latencies else None,
                    "latency_max_ms": (max(stats.latencies) * 1000)
                    if stats.latencies else None,
                    "wall_s": wall,
                    "phase_detail": detail,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"指标已写出 → {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
