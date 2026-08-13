"""2026-08-01 评审修复的回归测试（fix:train——C3/C4/C5/U8/U9/U10/U11/U18）.

覆盖：
- U9  cut_window / _fit_window：P/S 联合可行区间随机采样 + 放不下回落锚定
- C3  软标签 sigma 按秒定义（50/100Hz 物理宽度一致）、load_hdf5_dataset 采样率校验
- C5  eval_holdout 分层抽样（标注/噪声分开、固定 seed 可复现）
- C4  基线零 pick 防呆
- U8  checkpoint 含 np 标量时 load 不崩（torch>=2.6 weights_only 问题）
- U10 单例事件占比 / fallback 告警的纯逻辑

两种运行方式：
    pytest tests/test_finetune_fixes.py
    python  tests/test_finetune_fixes.py
"""

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import finetune_phasenet as ftp  # noqa: E402
from chunked_fetch import cut_window, fallback_alarm  # noqa: E402
from split_train_holdout import singleton_fraction  # noqa: E402

WIN = 3001
MARGIN = 250


def _wave(n=9000, seed=0):
    rng = np.random.RandomState(seed)
    return rng.standard_normal((3, n)).astype(np.float32)


# ----------------------------- U9: cut_window 联合采样 -----------------------------

def test_cut_window_joint_keeps_both_phases_with_margin():
    # 旧版只锚 P：s-p=2200 时 S 常被切出窗外。新版联合区间必须保住两相且避开盲区。
    rng = np.random.RandomState(11)
    w = _wave()
    p_positions = set()
    for _ in range(100):
        seg, p_rel, s_rel = cut_window(w, p=1000, s=3200, win=WIN, rng=rng)
        assert seg.shape == (3, WIN)
        assert p_rel >= MARGIN, f"P 距窗头必须 >= 盲区 {MARGIN}，实际 {p_rel}"
        assert 0 <= s_rel <= WIN - MARGIN, f"S 必须在窗内且距窗尾 >= {MARGIN}，实际 {s_rel}"
        assert abs((s_rel - p_rel) - 2200) < 1e-6, "S-P 间距必须保持"
        p_positions.add(int(p_rel))
    assert len(p_positions) > 5, "起点必须是随机的（免费增广），不能退化成常数"


def test_cut_window_joint_infeasible_falls_back_p_anchor():
    # s-p=2800 > win-2*margin=2501：放不下，回落 P 锚（S 必然出窗记 -1）
    rng = np.random.RandomState(12)
    w = _wave()
    for _ in range(50):
        seg, p_rel, s_rel = cut_window(w, p=1000, s=3800, win=WIN, rng=rng)
        assert WIN * 0.1 - 1 <= p_rel <= WIN * 0.6 + 1, "回落后应服从旧版 P 锚范围"
        assert s_rel == -1.0, "放不下的 S 必须诚实记 -1"


# ----------------------------- U9: _fit_window 联合采样+随机偏移 -----------------------------

def test_fit_window_joint_margins_and_randomness():
    w = _wave(n=9000, seed=3)
    rng = np.random.RandomState(7)
    starts = set()
    for _ in range(50):
        seg, p, s = ftp._fit_window(w, 4000, 5800, WIN, rng=rng)
        assert seg.shape == (3, WIN)
        assert p >= MARGIN and s <= WIN - MARGIN and s >= 0
        assert (s - p) == 1800, "S-P 间距必须保持"
        starts.add(p)
    assert len(starts) > 5, "裁窗起点必须随机（旧版固定 win//3 处零增广）"
    # 固定 seed 可复现：留出集窗口跨运行必须一致，monitor 才可比
    a = ftp._fit_window(w, 4000, 5800, WIN, rng=np.random.RandomState(9))
    b = ftp._fit_window(w, 4000, 5800, WIN, rng=np.random.RandomState(9))
    assert a[1] == b[1] and a[2] == b[2] and np.array_equal(a[0], b[0])


def test_fit_window_single_phase_random_offset_and_pad():
    w = _wave(n=9000, seed=4)
    rng = np.random.RandomState(8)
    p_positions = set()
    for _ in range(30):
        seg, p, s = ftp._fit_window(w, 4000, -1, WIN, rng=rng)
        assert s == -1
        assert int(WIN * 0.1) <= p <= int(WIN * 0.6), "单相回落锚定的随机偏移范围"
        start = 4000 - p
        assert np.array_equal(seg, w[:, start:start + WIN]), "窗口内容必须与原波形逐点一致"
        p_positions.add(p)
    assert len(p_positions) > 3
    # n < win：右侧补零，P/S 原样保留
    short = _wave(n=1200, seed=5)
    seg, p, s = ftp._fit_window(short, 800, 1100, WIN, rng=rng)
    assert seg.shape == (3, WIN) and p == 800 and s == 1100
    assert np.array_equal(seg[:, :1200], short) and float(np.abs(seg[:, 1200:]).max()) == 0.0


# ----------------------------- C3: sigma 按秒 -----------------------------

def test_soft_label_sigma_defined_in_seconds():
    order = ["P", "S", "N"]
    y50 = ftp.make_soft_label(3001, 1500, 2000, order, sr=50.0, sigma_p_s=0.2, sigma_s_s=0.3)
    y100 = ftp.make_soft_label(3001, 1500, 2000, order, sr=100.0, sigma_p_s=0.2, sigma_s_s=0.3)
    # 0.2s * 50Hz = 10 个采样点：中心 ±10 处应为 exp(-0.5)
    assert np.isclose(y50[0, 1510], math.exp(-0.5), atol=1e-3)
    # 0.2s * 100Hz = 20 个采样点：±10 处更宽（exp(-0.125)），±20 处才到 exp(-0.5)
    assert np.isclose(y100[0, 1520], math.exp(-0.5), atol=1e-3)
    assert y100[0, 1510] > y50[0, 1510] + 0.2
    # S 通道同理：0.3s*50=15 点
    assert np.isclose(y50[1, 2015], math.exp(-0.5), atol=1e-3)
    # 概率归一约束不变
    assert np.allclose(y50.sum(axis=0), 1.0, atol=1e-5)


# ----------------------------- C3: 数据采样率校验 -----------------------------

def _make_pool(path, sr_attr=100.0, with_sr=True):
    import h5py
    with h5py.File(path, "w") as f:
        g = f.create_group("data")
        d = g.create_dataset("CWA_00000001",
                             data=np.random.RandomState(0).randn(3, WIN).astype("float32"))
        d.attrs["p_sample_100hz"] = 1200.0
        d.attrs["s_sample_100hz"] = -1.0
        if with_sr:
            d.attrs["sampling_rate"] = float(sr_attr)


def test_load_hdf5_rejects_mismatched_sampling_rate():
    with tempfile.TemporaryDirectory() as tmp:
        pool = os.path.join(tmp, "pool.hdf5")
        _make_pool(pool, sr_attr=100.0)
        try:
            ftp.load_hdf5_dataset(pool, WIN, expect_sr=50.0)
            assert False, "100Hz 数据 + --sr 50 必须报错退出"
        except SystemExit as exc:
            assert "50" in str(exc) and "sampling_rate" in str(exc)
        # 一致时正常读出
        items = ftp.load_hdf5_dataset(pool, WIN, expect_sr=100.0)
        assert len(items) == 1 and items[0][1] == 1200
        # 旧池缺属性：只警告不拦（无法校验）
        pool2 = os.path.join(tmp, "pool2.hdf5")
        _make_pool(pool2, with_sr=False)
        assert len(ftp.load_hdf5_dataset(pool2, WIN, expect_sr=50.0)) == 1


# ----------------------------- C5: 分层抽样 -----------------------------

def _fake_items():
    w = np.zeros((3, 4), dtype="float32")
    items = []
    for i in range(200):
        items.append((w, float(i), -1.0))       # 标注窗
    for i in range(40):
        items.append((w, -1.0, -1.0))           # 噪声窗
    return items


def test_stratified_sample_quota_and_determinism():
    items = _fake_items()
    lab, noi = ftp.stratified_sample(items, 60, np.random.RandomState(1))
    assert len(noi) == 10 and len(lab) == 50, "300→250+50 同比例：60→50+10"
    assert all(it[1] >= 0 for it in lab) and all(it[1] < 0 and it[2] < 0 for it in noi)
    lab2, noi2 = ftp.stratified_sample(items, 60, np.random.RandomState(1))
    assert [it[1] for it in lab] == [it[1] for it in lab2], "固定 seed 必须抽到同一批标注窗"
    assert len(noi2) == 10
    # max_files=0：全量但仍分层
    lab_all, noi_all = ftp.stratified_sample(items, 0, np.random.RandomState(1))
    assert len(lab_all) == 200 and len(noi_all) == 40


def test_stratified_sample_noise_only_pool_gives_zero_labeled():
    w = np.zeros((3, 4), dtype="float32")
    items = [(w, -1.0, -1.0)] * 30
    lab, noi = ftp.stratified_sample(items, 20, np.random.RandomState(2))
    assert len(lab) == 0 and 0 < len(noi) <= 20  # 主流程据此拒绝开训


# ----------------------------- C4: 基线零 pick 防呆 -----------------------------

def test_baseline_guard_blocks_dead_monitor():
    try:
        ftp.ensure_baseline_has_picks("留出集", {"n_pred": 0, "mean_score": 0.0})
        assert False, "零 pick 基线必须 SystemExit"
    except SystemExit as exc:
        assert "50" in str(exc), "报错信息应提示 diting 用 --sr 50"
    ftp.ensure_baseline_has_picks("合成", {"n_pred": 9, "mean_score": 0.5})  # 不应抛


# ----------------------------- U8: checkpoint 含 np 标量可加载 -----------------------------

def test_load_checkpoint_state_with_np_scalars():
    import torch
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "last.pt")
        payload = {
            "model": {}, "opt": None, "epoch": 3, "best_score": 1.25,
            "score": {"p_hit": np.float64(0.72), "s_hit": np.float64(0.64)},
        }
        torch.save(payload, path)
        state = ftp.load_checkpoint_state(path, "cpu")
        assert state["epoch"] == 3 and abs(float(state["score"]["p_hit"]) - 0.72) < 1e-9


# ----------------------------- U10: 退化切分探测 -----------------------------

def test_singleton_fraction_detects_degenerate_split():
    assert singleton_fraction([]) == 0.0
    mixed = [("k1", "e1"), ("k2", "e1"), ("k3", "e2")]
    assert abs(singleton_fraction(mixed) - 0.5) < 1e-9
    degenerate = [(f"k{i}", f"k{i}") for i in range(100)]  # fallback=窗 key
    assert singleton_fraction(degenerate) == 1.0


def test_fallback_alarm_threshold():
    assert fallback_alarm(51, 100)
    assert not fallback_alarm(50, 100)
    assert not fallback_alarm(0, 0)


def test_configure_trainable_scope_out_only():
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Conv1d(3, 4, 3)
            self.out = torch.nn.Conv1d(4, 3, 1)

    model = Tiny()
    names = ftp.configure_trainable_scope(model, "out")
    assert names == ["out.weight", "out.bias"]
    assert not model.body.weight.requires_grad
    assert model.out.weight.requires_grad
    assert set(ftp.configure_trainable_scope(model, "all")) == {
        name for name, _ in model.named_parameters()
    }


def test_checkpoint_selection_path():
    assert ftp.checkpoint_selection_path("last", "last.pt", "best.pt") == "last.pt"
    assert ftp.checkpoint_selection_path("best", "last.pt", "best.pt") == "best.pt"
    try:
        ftp.checkpoint_selection_path("invalid", "last.pt", "best.pt")
        assert False, "invalid selection must fail"
    except ValueError:
        pass


if __name__ == "__main__":
    for fn in [
        test_cut_window_joint_keeps_both_phases_with_margin,
        test_cut_window_joint_infeasible_falls_back_p_anchor,
        test_fit_window_joint_margins_and_randomness,
        test_fit_window_single_phase_random_offset_and_pad,
        test_soft_label_sigma_defined_in_seconds,
        test_load_hdf5_rejects_mismatched_sampling_rate,
        test_stratified_sample_quota_and_determinism,
        test_stratified_sample_noise_only_pool_gives_zero_labeled,
        test_baseline_guard_blocks_dead_monitor,
        test_load_checkpoint_state_with_np_scalars,
        test_singleton_fraction_detects_degenerate_split,
        test_fallback_alarm_threshold,
        test_configure_trainable_scope_out_only,
        test_checkpoint_selection_path,
    ]:
        fn()
        print(f"{fn.__name__} ok")
    print("ALL OK")
