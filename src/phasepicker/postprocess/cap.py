"""短文件拾取限额（Cap）——对齐官方"按单文件计数量罚"的后处理.

===== 为什么存在（fork dizheng-opus-5 实测 + 本仓三分布复验，2026-08-09）=====
官方数量罚的 5% 容许带按**单个文件**算（scoring.scorer.DEFAULT_PENALTY_MODE =
merged_file_floor0）。三个已知数据集的短文件（<300s）真值 **100% 恰好 1P+1S**
（r1 1000/1000、r2 913/913、08 决赛 779/779，合计 2692/2692 零例外），
5% 容许带只有 0.1 个——多报任何 1 个就扣 0.5，而该文件满分才 2.0。

三分布消融（三成员集成基线）：
    r1 1.744→1.759 / r2 1.723→1.749 / 08 1.909→1.935，全部为正。

只对短文件生效：长连续记录（3600s 级）真值几十个 P/S，限成 1P+1S 会毁掉
它们。时长未知的文件一律不动——宁可不省 0.5，不能误伤长记录。
阈值 max_s 取 200~400s 等效（短文件全 ≤150s，长记录 ≥3600s）。

两个入口对应两条链路，逻辑同源：
- ``cap_short_waveform_picks``：Pick 对象级，serve_api 在线推理用；
- ``cap_short_file_results``：Task1Result 级，run_official_task1 离线评测用。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..types import PhaseType, Pick, Task1Result, Waveform


def cap_short_waveform_picks(
    waveforms: List[Waveform],
    picks_per_wf: List[List[Pick]],
    max_s: float,
    max_p: int = 1,
    max_s_picks: int = 1,
) -> List[List[Pick]]:
    """短波形按置信度限额：时长 <= max_s 的波形最多留 max_p 个 P、max_s_picks 个 S。

    Args:
        waveforms: 与 picks_per_wf 下标对齐的波形列表（提供 duration）。
        picks_per_wf: 每个波形的 Pick 列表。
        max_s: 时长阈值（秒）。<=0 表示不限额，原样返回。
        max_p: 短波形最多保留几个 P。
        max_s_picks: 短波形最多保留几个 S。

    Returns:
        新的 picks_per_wf（长波形与未超额的波形原样引用，不复制）。
    """
    if max_s <= 0:
        return picks_per_wf

    def _top(picks: List[Pick], k: int) -> List[Pick]:
        if len(picks) <= k:
            return picks
        # 按置信度降序取前 k，再按时间升序（与 fork 原实现逐位一致）
        best = sorted(picks, key=lambda p: -float(getattr(p, "confidence", 0.0) or 0.0))[:k]
        best.sort(key=lambda p: float(p.time_utc))
        return best

    out: List[List[Pick]] = []
    for wf, picks in zip(waveforms, picks_per_wf):
        dur = float(getattr(wf, "duration", 0.0) or 0.0)
        if dur <= 0.0 or dur > max_s:
            out.append(picks)  # 时长未知或长记录 → 一律不动
            continue
        p_list = [p for p in picks if p.phase == PhaseType.P]
        s_list = [p for p in picks if p.phase == PhaseType.S]
        if len(p_list) <= max_p and len(s_list) <= max_s_picks:
            out.append(picks)
            continue
        out.append(_top(p_list, max_p) + _top(s_list, max_s_picks))
    return out


def cap_short_file_results(
    results_map: Dict[str, Task1Result],
    durations: Dict[str, float],
    max_s: float,
    max_p: int = 1,
    max_s_picks: int = 1,
) -> Tuple[Dict[str, Task1Result], int, int]:
    """Task1Result 级限额（离线评测链路）。

    与 cap_short_waveform_picks 同一逻辑；置信度列表不全（无从择优）的文件
    跳过不动，交由上层保持原样。

    Returns:
        (新的 results_map, 被改动的文件数, 丢掉的到时数)
    """
    if max_s <= 0:
        return results_map, 0, 0

    def _top(times: List[float], confs: List[float], k: int):
        if len(times) <= k:
            return list(times), list(confs)
        if len(confs) < len(times):  # 置信度不全 → 标记跳过
            return None, None
        idx = sorted(range(len(times)), key=lambda i: -confs[i])[:k]
        idx.sort(key=lambda i: times[i])
        return [times[i] for i in idx], [confs[i] for i in idx]

    out = dict(results_map)
    n_changed = 0
    n_dropped = 0
    for fid, res in results_map.items():
        dur = durations.get(fid)
        if dur is None or dur > max_s:
            continue
        p_t, p_c = _top(res.p_times_s, res.p_confidences, max_p)
        s_t, s_c = _top(res.s_times_s, res.s_confidences, max_s_picks)
        if p_t is None or s_t is None:
            continue
        dropped = (len(res.p_times_s) - len(p_t)) + (len(res.s_times_s) - len(s_t))
        if dropped <= 0:
            continue
        out[fid] = Task1Result(
            file_id=res.file_id,
            p_times_s=p_t, s_times_s=s_t,
            p_confidences=p_c, s_confidences=s_c,
        )
        n_changed += 1
        n_dropped += dropped
    return out, n_changed, n_dropped
