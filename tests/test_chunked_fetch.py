"""chunked_fetch / split_train_holdout 纯逻辑测试——numpy + 标准库.

覆盖取数器里真正决定训练质量的三段纯函数：
- cut_window：锚定/随机/补零三分支，P/S 窗内换算与 -1 语义，边界夹取
- wave_ok：形状/NaN/死道闸门
- stable_hash01 + split_keys：稳定性、事件级切分不泄漏

两种运行方式：
    pytest tests/test_chunked_fetch.py
    python  tests/test_chunked_fetch.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from chunked_fetch import cut_window, event_key, stable_hash01, wave_ok  # noqa: E402
from split_train_holdout import split_keys  # noqa: E402

WIN = 3001


def _wave(n=9000, seed=0):
    rng = np.random.RandomState(seed)
    return rng.standard_normal((3, n)).astype(np.float32)


def test_cut_window_p_anchor_in_range():
    rng = np.random.RandomState(1)
    w = _wave()
    for _ in range(50):
        seg, p_rel, s_rel = cut_window(w, p=4500, s=5200, win=WIN, rng=rng)
        assert seg.shape == (3, WIN)
        assert 0 <= p_rel < WIN, "P 必须落在窗内"
        # P/S 双有效且放得下 → 联合可行区间采样：两相都离窗边 >= 盲区 margin(250)
        assert p_rel >= 250 and s_rel <= WIN - 250
        assert s_rel >= 0, "放得下时 S 必须同窗"
        assert abs((s_rel - p_rel) - 700) < 1e-6, "窗内 S-P 间距必须保持"


def test_cut_window_content_matches_source():
    rng = np.random.RandomState(2)
    w = _wave()
    seg, p_rel, _ = cut_window(w, p=4500, s=-1, win=WIN, rng=rng)
    start = int(4500 - p_rel)
    assert np.array_equal(seg, w[:, start:start + WIN]), "窗口内容必须与原波形逐点一致"


def test_cut_window_near_edges_clamped():
    rng = np.random.RandomState(3)
    w = _wave()
    seg, p_rel, _ = cut_window(w, p=50, s=-1, win=WIN, rng=rng)  # 起点附近
    assert seg.shape == (3, WIN) and p_rel == 50, "夹到 0 起点后 P 下标应原样保留"
    seg, p_rel, _ = cut_window(w, p=8990, s=-1, win=WIN, rng=rng)  # 终点附近
    assert seg.shape == (3, WIN) and p_rel == 8990 - (9000 - WIN)


def test_cut_window_noise_and_short():
    rng = np.random.RandomState(4)
    seg, p_rel, s_rel = cut_window(_wave(), p=-1, s=-1, win=WIN, rng=rng)
    assert seg.shape == (3, WIN) and p_rel == -1.0 and s_rel == -1.0
    short = _wave(n=1200)
    seg, p_rel, s_rel = cut_window(short, p=800, s=1100, win=WIN, rng=rng)
    assert seg.shape == (3, WIN)
    assert p_rel == 800 and s_rel == 1100
    assert np.array_equal(seg[:, :1200], short) and float(np.abs(seg[:, 1200:]).max()) == 0.0


def test_cut_window_s_outside_marked_missing():
    rng = np.random.RandomState(5)
    w = _wave(n=20000)
    seg, p_rel, s_rel = cut_window(w, p=1000, s=19000, win=WIN, rng=rng)
    assert 0 <= p_rel < WIN and s_rel == -1.0


def test_wave_ok_gates():
    assert wave_ok(_wave())
    assert not wave_ok(np.zeros((3, 3000), dtype=np.float32))          # 死道
    bad = _wave(); bad[1, 5] = np.nan
    assert not wave_ok(bad)                                            # NaN
    assert not wave_ok(_wave()[:2])                                    # 缺分量
    assert not wave_ok(_wave(n=50))                                    # 过短


def test_stable_hash_deterministic_uniform():
    vals = [stable_hash01(f"ev{i}") for i in range(2000)]
    assert vals == [stable_hash01(f"ev{i}") for i in range(2000)], "必须跨调用稳定"
    assert all(0.0 <= v < 1.0 for v in vals)
    frac = sum(v < 0.1 for v in vals) / len(vals)
    assert 0.06 < frac < 0.14, f"10% 切分比例应大致成立，实际 {frac:.3f}"


def test_split_keys_event_level_no_leak():
    # 同一事件多窗口必须落在同一侧
    keys_events = [(f"k{i}", f"ev{i % 300}") for i in range(3000)]
    train, hold = split_keys(keys_events, holdout_frac=0.1)
    assert len(train) + len(hold) == 3000
    ev_side = {}
    for k, ev in keys_events:
        side = "h" if k in set(hold) else "t"
        assert ev_side.setdefault(ev, side) == side, f"事件 {ev} 被切到两侧（泄漏）"


def test_event_key_fallback():
    assert event_key({"source_id": "abc"}, "fb") == "abc"
    assert event_key({"source_id": ""}, "fb") == "fb"
    assert event_key({}, "fb") == "fb"


if __name__ == "__main__":
    for fn in [
        test_cut_window_p_anchor_in_range,
        test_cut_window_content_matches_source,
        test_cut_window_near_edges_clamped,
        test_cut_window_noise_and_short,
        test_cut_window_s_outside_marked_missing,
        test_wave_ok_gates,
        test_stable_hash_deterministic_uniform,
        test_split_keys_event_level_no_leak,
        test_event_key_fallback,
    ]:
        fn()
        print(f"{fn.__name__} ok")
    print("ALL OK")
