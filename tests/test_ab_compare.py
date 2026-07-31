"""ab_compare.diff_results 的纯逻辑测试——纯标准库.

A/B 闸门的判定逻辑必须绝对可靠：数量差异、超容差、等价三种结论各覆盖。

两种运行方式：
    pytest tests/test_ab_compare.py
    python  tests/test_ab_compare.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from phasepicker.types import Task1Result  # noqa: E402
from ab_compare import diff_results  # noqa: E402


def _r(fid, p=(), s=()):
    return Task1Result(file_id=fid, p_times_s=list(p), s_times_s=list(s))


def test_identical_is_same():
    a = {"f1": _r("f1", [1.0, 2.0], [3.0]), "f2": _r("f2")}
    b = {"f1": _r("f1", [1.0, 2.0], [3.0]), "f2": _r("f2")}
    d = diff_results(a, b, tol=0.01)
    assert d["same"] and d["max_dt"] == 0.0 and not d["count_mismatch"]


def test_within_tol_is_same():
    a = {"f1": _r("f1", [1.000], [3.000])}
    b = {"f1": _r("f1", [1.004], [2.996])}
    d = diff_results(a, b, tol=0.01)
    assert d["same"]
    assert 0.003 < d["max_dt"] < 0.005


def test_over_tol_flagged():
    a = {"f1": _r("f1", [1.00])}
    b = {"f1": _r("f1", [1.05])}
    d = diff_results(a, b, tol=0.01)
    assert not d["same"]
    assert d["over_tol"] and d["over_tol"][0][0] == "f1"
    assert abs(d["over_tol"][0][2] - 0.05) < 1e-9


def test_count_mismatch_flagged():
    a = {"f1": _r("f1", [1.0, 2.0])}
    b = {"f1": _r("f1", [1.0])}
    d = diff_results(a, b, tol=0.01)
    assert not d["same"]
    assert d["count_mismatch"] == [("f1", "P", 2, 1)]


def test_missing_file_treated_as_empty():
    a = {"f1": _r("f1", [1.0])}
    b = {}
    d = diff_results(a, b, tol=0.01)
    assert not d["same"]
    assert d["count_mismatch"] == [("f1", "P", 1, 0)]


def test_unsorted_input_aligned_by_sort():
    a = {"f1": _r("f1", [2.0, 1.0])}
    b = {"f1": _r("f1", [1.0, 2.0])}
    d = diff_results(a, b, tol=0.001)
    assert d["same"]


if __name__ == "__main__":
    for fn in [
        test_identical_is_same,
        test_within_tol_is_same,
        test_over_tol_flagged,
        test_count_mismatch_flagged,
        test_missing_file_treated_as_empty,
        test_unsorted_input_aligned_by_sort,
    ]:
        fn()
        print(f"{fn.__name__} ok")
    print("ALL OK")
