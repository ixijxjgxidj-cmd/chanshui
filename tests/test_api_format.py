"""官方 API JSON 输出格式的单元测试——纯标准库（不需要 fastapi/torch/obspy）.

对齐《比赛说明》PPT 的输出规范：
    { "STA1": {"P": ["2025-06-07T12:34:56.789000Z", ...], "S": [...]}, ... }
覆盖：
- iso_utc 的字符串格式与官方示例逐字符一致（微秒 6 位 + Z）
- picks_to_official_json：台站键取台站名、P/S 分流、升序、空表保留
- 台站名冲突（不同 network 同名）退回完整 NET.STA
- check_api.validate_payload 能接受合法样例、拒绝各类非法结构
- serve_api 输出 ↔ check_api 校验 的闭环自洽

两种运行方式：
    pytest tests/test_api_format.py
    python  tests/test_api_format.py
"""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from phasepicker.types import PhaseType, Pick, Waveform  # noqa: E402

from serve_api import iso_utc, picks_to_official_json  # noqa: E402
from check_api import validate_payload  # noqa: E402


class _FakeArray:
    def __init__(self, n):
        self.shape = (3, n)


def _wf(station, start=0.0):
    return Waveform(data=_FakeArray(1000), sampling_rate=100.0, starttime_utc=start, station=station)


def test_iso_utc_matches_official_example():
    # 官方示例：2025-06-07T12:34:56.789000Z
    t = datetime(2025, 6, 7, 12, 34, 56, 789000, tzinfo=timezone.utc).timestamp()
    assert iso_utc(t) == "2025-06-07T12:34:56.789000Z"
    # 整秒也必须保留 6 位微秒
    t2 = datetime(2025, 6, 7, 12, 36, 55, 0, tzinfo=timezone.utc).timestamp()
    assert iso_utc(t2) == "2025-06-07T12:36:55.000000Z"


def test_station_key_and_phase_split():
    wfs = [_wf("GX.NN01"), _wf("GX.NN02")]
    picks = [
        [
            Pick(phase=PhaseType.S, time_utc=30.0, confidence=0.7, station="GX.NN01"),
            Pick(phase=PhaseType.P, time_utc=10.0, confidence=0.9, station="GX.NN01"),
            Pick(phase=PhaseType.P, time_utc=5.0, confidence=0.8, station="GX.NN01"),
        ],
        [],  # 噪声台站：空表也必须保留
    ]
    out = picks_to_official_json(wfs, picks)
    assert set(out.keys()) == {"NN01", "NN02"}
    assert out["NN01"]["P"] == [iso_utc(5.0), iso_utc(10.0)]  # 升序
    assert out["NN01"]["S"] == [iso_utc(30.0)]
    assert out["NN02"] == {"P": [], "S": []}


def test_station_collision_uses_full_id():
    wfs = [_wf("GX.NN01"), _wf("YN.NN01")]
    out = picks_to_official_json(wfs, [[], []])
    assert set(out.keys()) == {"GX.NN01", "YN.NN01"}


def test_validate_payload_accepts_legal():
    payload = {
        "STA1": {"P": ["2025-06-07T12:34:56.789000Z"], "S": ["2025-06-07T12:36:55.000000Z"]},
        "STA2": {"P": [], "S": []},
    }
    assert validate_payload(payload) == []


def test_validate_payload_rejects_illegal():
    assert validate_payload([]) != []                                   # 非对象
    assert validate_payload({"S1": {"P": "notalist", "S": []}}) != []   # P 非列表
    assert validate_payload({"S1": {"P": ["2025-06-07 12:00:00"], "S": []}}) != []  # 格式错
    assert validate_payload({"S1": {"P": [], "S": [], "X": []}}) != []  # 多余键
    assert validate_payload(
        {"S1": {"P": ["2025-06-07T12:00:01.000000Z", "2025-06-07T12:00:00.000000Z"], "S": []}}
    ) != []                                                             # 未升序


def test_roundtrip_serve_to_check():
    wfs = [_wf("GX.NN01"), _wf("GX.NN02"), _wf("YN.KM01")]
    picks = [
        [Pick(phase=PhaseType.P, time_utc=1.25, confidence=0.9, station="GX.NN01")],
        [
            Pick(phase=PhaseType.P, time_utc=2.0, confidence=0.9, station="GX.NN02"),
            Pick(phase=PhaseType.S, time_utc=4.5, confidence=0.6, station="GX.NN02"),
        ],
        [],
    ]
    out = picks_to_official_json(wfs, picks)
    assert validate_payload(out) == [], "serve_api 的输出必须通过 check_api 校验"


if __name__ == "__main__":
    for fn in [
        test_iso_utc_matches_official_example,
        test_station_key_and_phase_split,
        test_station_collision_uses_full_id,
        test_validate_payload_accepts_legal,
        test_validate_payload_rejects_illegal,
        test_roundtrip_serve_to_check,
    ]:
        fn()
        print(f"{fn.__name__} ok")
    print("ALL OK")
