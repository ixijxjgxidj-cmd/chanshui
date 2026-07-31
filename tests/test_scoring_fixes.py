"""数量罚读法(C2)/缺失文件口径(U23)/默认值同源(U6)/ab_compare 修饰 的守门测试.

关键数字全部来自 outputs/评审报告_20260801.md C2 验证记录的手算复现：
真值 1P+1S、P 拾取完美 S 漏检（当前模型最常见失误模式）时，
"多报一个假 P"在 合并读法 vs 分相位读法 下净效应 +0.5 vs -0.5（策略排序翻转）。

两种运行方式：
    PYTHONUTF8=1 python -m pytest tests/test_scoring_fixes.py -q
    PYTHONUTF8=1 python tests/test_scoring_fixes.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from phasepicker.defaults import (  # noqa: E402
    DEFAULT_P_THRESHOLD,
    DEFAULT_PRETRAINED,
    DEFAULT_S_THRESHOLD,
)
from phasepicker.scoring.scorer import (  # noqa: E402
    DEFAULT_PENALTY_MODE,
    PENALTY_MODES,
    exam_total_score,
    score_file,
)
from phasepicker.eval.official_eval import (  # noqa: E402
    evaluate_task1,
    evaluate_task1_all_modes,
    penalty_modes_table,
)
from phasepicker.types import Task1Result  # noqa: E402
from ab_compare import diff_results, format_max_dt_line  # noqa: E402


EPS = 1e-9

# C2 验证记录的标准场景：真值 1P@100 + 1S@105，P 拾取完美、S 漏检
TRUTH = [("P", 100.0), ("S", 105.0)]
PRED_1P = [("P", 100.0)]                       # 只报 1P
PRED_1P_FAKE_P = [("P", 100.0), ("P", 130.0)]  # 加一个 30s 外假 P
PRED_1P_WILD_S = [("P", 100.0), ("S", 300.0)]  # 加一个乱猜 S（远超 2s 容差）


# ---------------- C2: 四种读法的手算数字（文件级） ----------------

def test_c2_merged_file_floor0_matches_report():
    # 只报1P=0.5；加假P=1.0（假P抵消计数差）；加乱猜S=1.0
    assert abs(score_file(PRED_1P, TRUTH, "merged_file_floor0").total_score - 0.5) < EPS
    assert abs(score_file(PRED_1P_FAKE_P, TRUTH, "merged_file_floor0").total_score - 1.0) < EPS
    assert abs(score_file(PRED_1P_WILD_S, TRUTH, "merged_file_floor0").total_score - 1.0) < EPS


def test_c2_per_phase_floor0_matches_report():
    # 分相位读法：只报1P=1.0；加假P=0.5；加乱猜S=1.0
    assert abs(score_file(PRED_1P, TRUTH, "per_phase_floor0").total_score - 1.0) < EPS
    assert abs(score_file(PRED_1P_FAKE_P, TRUTH, "per_phase_floor0").total_score - 0.5) < EPS
    assert abs(score_file(PRED_1P_WILD_S, TRUTH, "per_phase_floor0").total_score - 1.0) < EPS


def test_c2_fake_pick_flips_strategy_ranking():
    # C2 的核心：同一个"多报假P"动作，合并读法 +0.5、分相位读法 -0.5，方向相反
    merged_delta = (
        score_file(PRED_1P_FAKE_P, TRUTH, "merged_file_floor0").total_score
        - score_file(PRED_1P, TRUTH, "merged_file_floor0").total_score
    )
    per_phase_delta = (
        score_file(PRED_1P_FAKE_P, TRUTH, "per_phase_floor0").total_score
        - score_file(PRED_1P, TRUTH, "per_phase_floor0").total_score
    )
    assert abs(merged_delta - 0.5) < EPS
    assert abs(per_phase_delta - (-0.5)) < EPS


def test_c2_exam_modes_file_level_no_penalty():
    # 全卷读法下文件层不扣罚：total=到时分之和，count_penalty=0
    for mode in ("merged_exam", "per_phase_exam"):
        rep = score_file(PRED_1P_FAKE_P, TRUTH, mode)
        assert abs(rep.total_score - 1.0) < EPS
        assert rep.count_penalty == 0.0


def _eval_single(pred_pairs, mode):
    """单文件走 evaluate_task1（含卷级聚合），返回报告。"""
    pred = {"f.mseed": Task1Result(
        "f.mseed",
        [t for p, t in pred_pairs if p == "P"],
        [t for p, t in pred_pairs if p == "S"],
    )}
    ans = {"f.mseed": Task1Result("f.mseed", [100.0], [105.0])}
    return evaluate_task1(pred, ans, penalty_mode=mode)


def test_c2_exam_modes_single_file_aggregate():
    # merged_exam: 只报1P → 1.0 - 罚(1,2)=0.5 → 0.5；加假P → 罚(2,2)=0 → 1.0
    r = _eval_single(PRED_1P, "merged_exam")
    assert abs(r.total_score - 0.5) < EPS and abs(r.exam_count_penalty - 0.5) < EPS
    r = _eval_single(PRED_1P_FAKE_P, "merged_exam")
    assert abs(r.total_score - 1.0) < EPS and r.exam_count_penalty == 0.0
    # per_phase_exam: 只报1P → P满分1.0 + S截0 → 1.0；加假P → P罚0.5 → 0.5
    r = _eval_single(PRED_1P, "per_phase_exam")
    assert abs(r.total_score - 1.0) < EPS and abs(r.exam_count_penalty - 0.5) < EPS
    r = _eval_single(PRED_1P_FAKE_P, "per_phase_exam")
    assert abs(r.total_score - 0.5) < EPS and abs(r.exam_count_penalty - 1.0) < EPS


def test_c2_multi_file_exam_vs_file_divergence():
    # f1 少报一个S、f2 多报一个假P：全卷合并读法下计数互相抵消（罚0），
    # 文件级读法各自挨罚——四种读法给出四组不同总分
    ans = {
        "f1": Task1Result("f1", [100.0], [105.0]),
        "f2": Task1Result("f2", [200.0], [205.0]),
    }
    pred = {
        "f1": Task1Result("f1", [100.0], []),
        "f2": Task1Result("f2", [200.0, 250.0], [205.0]),
    }
    expect = {
        "merged_file_floor0": 2.0,   # 0.5 + 1.5
        "per_phase_floor0": 2.5,     # 1.0 + 1.5
        "merged_exam": 3.0,          # 到时和3.0，全卷 pred=4=true → 罚0
        "per_phase_exam": 2.0,       # P: 2.0-罚(3,2)0.5=1.5; S: 1.0-罚(1,2)0.5=0.5
    }
    for mode, want in expect.items():
        rep = evaluate_task1(pred, ans, penalty_mode=mode)
        assert abs(rep.total_score - want) < EPS, (mode, rep.total_score, want)


def test_default_mode_identical_to_legacy_call():
    # 不传参数 == 显式 merged_file_floor0，且复刻既有测试的历史数字
    truth = [("P", 100.0), ("S", 105.0)]
    pred = [("P", 100.55), ("S", 105.0), ("P", 999.0)]
    legacy = score_file(pred, truth)
    explicit = score_file(pred, truth, DEFAULT_PENALTY_MODE)
    # 两条调用路径必须逐位一致；数值本身按既有测试口径带容差
    assert legacy.total_score == explicit.total_score
    assert legacy.count_penalty == explicit.count_penalty == 0.5
    assert abs(legacy.total_score - 1.0) < EPS


def test_invalid_penalty_mode_raises():
    try:
        score_file(PRED_1P, TRUTH, "merged_paper")
        raise AssertionError("应对未知读法抛 ValueError")
    except ValueError:
        pass
    try:
        exam_total_score([], "whole_paper")
        raise AssertionError("应对未知读法抛 ValueError")
    except ValueError:
        pass


def test_all_modes_interface_and_table():
    ans = {"f1": Task1Result("f1", [100.0], [105.0])}
    pred = {"f1": Task1Result("f1", [100.0], [])}
    reports = evaluate_task1_all_modes(pred, ans)
    assert tuple(reports) == PENALTY_MODES
    assert abs(reports["merged_file_floor0"].total_score - 0.5) < EPS
    assert abs(reports["per_phase_floor0"].total_score - 1.0) < EPS
    table = penalty_modes_table(reports)
    for mode in PENALTY_MODES:
        assert mode in table


# ---------------- U23: 答案侧缺失文件的口径 ----------------

def _missing_fixture():
    ans = {
        "f1": Task1Result("f1", [10.0], [15.0]),
        "f2": Task1Result("f2", [20.0], [25.0]),
    }
    pred = {"f1": Task1Result("f1", [10.0], [15.0])}  # f2 缺失
    return pred, ans


def test_u23_default_excludes_missing_but_warns():
    pred, ans = _missing_fixture()
    rep = evaluate_task1(pred, ans)
    # 默认口径不变：只对共有文件计分
    assert rep.n_files == 1
    assert abs(rep.mean_score - 2.0) < EPS
    assert rep.missing == ["f2"]
    assert "f2" not in rep.per_file
    # 但 summary 必须醒目告警且含数量
    s = rep.summary()
    assert "警告" in s and "1 个文件" in s


def test_u23_strict_counts_missing_as_zero():
    pred, ans = _missing_fixture()
    rep = evaluate_task1(pred, ans, strict_missing=True)
    assert rep.n_files == 2
    assert abs(rep.total_score - 2.0) < EPS
    assert abs(rep.mean_score - 1.0) < EPS
    assert rep.per_file["f2"] == 0.0
    assert "strict" in rep.summary()


def test_u23_strict_missing_enters_exam_penalty():
    # strict + merged_exam：缺失文件的真值数计入全卷计数罚
    pred, ans = _missing_fixture()
    rep = evaluate_task1(pred, ans, penalty_mode="merged_exam", strict_missing=True)
    # 到时和=2.0；全卷 pred=2 true=4 → 差2、容许0.2 → 罚 2*0.5=1.0
    assert abs(rep.exam_count_penalty - 1.0) < EPS
    assert abs(rep.total_score - 1.0) < EPS
    assert abs(rep.mean_score - 0.5) < EPS
    # 非 strict 下缺失文件整体排除，不进卷级计数
    rep2 = evaluate_task1(pred, ans, penalty_mode="merged_exam", strict_missing=False)
    assert rep2.exam_count_penalty == 0.0
    assert abs(rep2.total_score - 2.0) < EPS


def test_u23_clean_run_summary_unchanged():
    # 无缺失 + 默认读法：summary 保持历史单行格式，不多任何告警
    ans = {"f1": Task1Result("f1", [10.0], [15.0])}
    rep = evaluate_task1(dict(ans), ans)
    assert rep.summary() == "T1: 均分=2.000 总分=2.000 文件数=1 缺失=0 多余=0"


# ---------------- U6: CLI 默认值与 defaults.py 同源 ----------------

def test_u6_run_official_task1_defaults_from_defaults_py():
    from run_official_task1 import build_arg_parser

    args = build_arg_parser().parse_args(["--input", "x", "--output", "y"])
    assert args.pretrained == DEFAULT_PRETRAINED
    assert args.p_threshold == DEFAULT_P_THRESHOLD
    assert args.s_threshold == DEFAULT_S_THRESHOLD
    # U6 顺手补的旋钮：--overlap/--compile 必须存在且与 ab_compare 同义
    assert args.overlap == 0.5
    assert args.compile is False
    assert args.penalty_mode == DEFAULT_PENALTY_MODE
    assert args.strict_missing is False


def test_u6_ab_compare_defaults_from_defaults_py():
    from ab_compare import _parse_variant

    v = _parse_variant("")
    assert v.pretrained == DEFAULT_PRETRAINED
    assert v.p_threshold == DEFAULT_P_THRESHOLD
    assert v.s_threshold == DEFAULT_S_THRESHOLD


# ---------------- ab_compare: 无可对齐样本时 max|Δt| 显示 "—" ----------------

def _r(fid, p=(), s=()):
    return Task1Result(file_id=fid, p_times_s=list(p), s_times_s=list(s))


def test_ab_all_count_mismatch_shows_dash():
    a = {"f1": _r("f1", [1.0]), "f2": _r("f2", [2.0, 3.0])}
    b = {"f1": _r("f1"), "f2": _r("f2", [2.0])}
    d = diff_results(a, b, tol=0.01)
    assert d["n_aligned"] == 0
    assert not d["same"]
    line = format_max_dt_line(d, tol=0.01)
    assert "—" in line and "ms @" not in line


def test_ab_no_picks_both_sides_shows_dash():
    a = {"f1": _r("f1")}
    b = {"f1": _r("f1")}
    d = diff_results(a, b, tol=0.01)
    assert d["same"] and d["n_aligned"] == 0
    assert "—" in format_max_dt_line(d, tol=0.01)


def test_ab_aligned_picks_show_ms():
    a = {"f1": _r("f1", [1.0, 2.0], [3.0]), "f2": _r("f2", [5.0])}
    b = {"f1": _r("f1", [1.0, 2.004], [3.0]), "f2": _r("f2")}
    d = diff_results(a, b, tol=0.01)
    # f1 的 2P+1S 可对齐，f2 数量不一致不计入
    assert d["n_aligned"] == 3
    assert 0.003 < d["max_dt"] < 0.005
    line = format_max_dt_line(d, tol=0.01)
    assert "ms" in line and "—" not in line


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {exc!r}")
            continue
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"SUMMARY {passed}/{len(fns)}")
    return 0 if passed == len(fns) else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
