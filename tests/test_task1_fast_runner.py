"""run_task1_samples_fast 的等价性与容错测试——纯标准库.

高速 runner 的正确性标准：**与顺序版 run_task1_samples 结果逐字段一致**。覆盖：
- 合批路径（picker 带 pick_batch）与顺序版等价
- 无 pick_batch 的普通 picker 也能走（逐条回退）且等价
- pick_batch 整体抛错 → 自动回退逐条，结果仍等价
- 读波形失败 / 空波形 → 空结果（与顺序版一致）
- file_batch=1 / 大于样本数 两个边界
- progress_cb 收到单调递增、终值=总数

两种运行方式：
    pytest tests/test_task1_fast_runner.py
    python  tests/test_task1_fast_runner.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.types import ExamSample, ExamTask, PhaseType, Pick, Waveform  # noqa: E402
from phasepicker.tasks.task1_runner import (  # noqa: E402
    run_task1_samples,
    run_task1_samples_fast,
)


class _FakeArray:
    """最小 (3, n) 数组替身，避免测试依赖 numpy。"""

    def __init__(self, n):
        self.shape = (3, n)


def _mk_samples(n):
    return [
        ExamSample(file_id=f"T1.A.Q{i:04d}.mseed", task=ExamTask.T1, source_path=f"/in/{i}.mseed")
        for i in range(n)
    ]


def _mk_waveform(start, station="GX.NN01", n=3000):
    return Waveform(data=_FakeArray(n), sampling_rate=100.0, starttime_utc=start, station=station)


def _deterministic_picks(wf: Waveform):
    """由波形起点确定性生成 picks：起点+1.5s 一个 P，起点+3.25s 一个 S。"""
    return [
        Pick(phase=PhaseType.P, time_utc=wf.starttime_utc + 1.5, confidence=0.9, station=wf.station),
        Pick(phase=PhaseType.S, time_utc=wf.starttime_utc + 3.25, confidence=0.8, station=wf.station),
    ]


class SeqPicker:
    """只有 pick() 的普通 picker。"""

    def pick(self, wf):
        return _deterministic_picks(wf)


class BatchPicker(SeqPicker):
    """带 pick_batch 的 picker；记录是否真的走了批路径。"""

    def __init__(self):
        self.batch_calls = 0

    def pick_batch(self, wfs):
        self.batch_calls += 1
        return [_deterministic_picks(wf) for wf in wfs]


class BrokenBatchPicker(SeqPicker):
    """pick_batch 永远抛错 → 应回退逐条且结果不变。"""

    def pick_batch(self, wfs):
        raise RuntimeError("batch inference broken")


def _loader_ok(sample: ExamSample):
    idx = int(sample.file_id[6:10])
    if idx % 3 == 0:
        # 每第 3 个文件是双台站
        return [
            _mk_waveform(1000.0 + idx, "GX.NN01"),
            _mk_waveform(1000.0 + idx + 0.5, "GX.NN02"),
        ]
    return [_mk_waveform(1000.0 + idx)]


def _loader_flaky(sample: ExamSample):
    idx = int(sample.file_id[6:10])
    if idx == 2:
        raise IOError("disk error")
    if idx == 5:
        return []
    return _loader_ok(sample)


def _assert_same(a, b):
    assert set(a.keys()) == set(b.keys())
    for k in a:
        assert a[k].file_id == b[k].file_id
        assert a[k].p_times_s == b[k].p_times_s, k
        assert a[k].s_times_s == b[k].s_times_s, k


def test_batch_path_equals_sequential():
    samples = _mk_samples(10)
    base = run_task1_samples(samples, _loader_ok, SeqPicker())
    picker = BatchPicker()
    fast = run_task1_samples_fast(samples, _loader_ok, picker, io_workers=4, file_batch=4)
    _assert_same(base, fast)
    assert picker.batch_calls >= 1, "应走合批路径"


def test_plain_picker_fast_equals_sequential():
    samples = _mk_samples(7)
    base = run_task1_samples(samples, _loader_ok, SeqPicker())
    fast = run_task1_samples_fast(samples, _loader_ok, SeqPicker(), io_workers=3, file_batch=3)
    _assert_same(base, fast)


def test_broken_batch_falls_back():
    samples = _mk_samples(6)
    base = run_task1_samples(samples, _loader_ok, SeqPicker())
    fast = run_task1_samples_fast(samples, _loader_ok, BrokenBatchPicker(), io_workers=2, file_batch=4)
    _assert_same(base, fast)


def test_load_failures_isolated():
    samples = _mk_samples(8)
    base = run_task1_samples(samples, _loader_flaky, SeqPicker())
    fast = run_task1_samples_fast(samples, _loader_flaky, BatchPicker(), io_workers=4, file_batch=3)
    _assert_same(base, fast)
    assert fast["T1.A.Q0002.mseed"].p_times_s == []
    assert fast["T1.A.Q0005.mseed"].s_times_s == []


def test_batch_size_edges():
    samples = _mk_samples(5)
    base = run_task1_samples(samples, _loader_ok, SeqPicker())
    for fb in (1, 5, 99):
        fast = run_task1_samples_fast(samples, _loader_ok, BatchPicker(), io_workers=2, file_batch=fb)
        _assert_same(base, fast)


def test_progress_callback():
    samples = _mk_samples(9)
    seen = []
    run_task1_samples_fast(
        samples,
        _loader_ok,
        BatchPicker(),
        io_workers=2,
        file_batch=4,
        progress_cb=lambda done, total: seen.append((done, total)),
    )
    assert seen, "应有进度回调"
    dones = [d for d, _ in seen]
    assert dones == sorted(dones), "进度应单调递增"
    assert seen[-1] == (9, 9)


if __name__ == "__main__":
    for fn in [
        test_batch_path_equals_sequential,
        test_plain_picker_fast_equals_sequential,
        test_broken_batch_falls_back,
        test_load_failures_isolated,
        test_batch_size_edges,
        test_progress_callback,
    ]:
        fn()
        print(f"{fn.__name__} ok")
    print("ALL OK")
