"""Hard experiment-data boundaries shared by data preparation and training tools."""

from __future__ import annotations

from pathlib import Path


_SEALED_08_MARKERS = (
    "08-exam",
    "08_exam",
    "08-an",
    "08_an",
    "final08",
    "final_08",
)

# AGENTS.md 第 2 条：第 1、2 轮官方比赛数据永久禁止参与训练、蒸馏、微调，也不得用于
# 阈值/成员/超参选择；只允许作为评估与不回归检查。历史上生产成员 4/5 由 r2train.h5
# 微调而来（实测 1001 窗、915 个 R2 source_file），T2/T3 默认 joblib 也是 r1+r2 合训，
# 因此这里把 R1/R2 派生物一并列为训练禁用标记，避免同类违规再次静默发生。
_SEALED_ROUND12_MARKERS = (
    "r1train",
    "r2train",
    "round1_train",
    "round2_train",
    "第1轮",
    "第2轮",
    "round1-exam",
    "round2-exam",
    "_r1r2",
    "r1_to_r2",
)


def assert_experiment_path_allowed(path: str | Path, label: str = "data") -> None:
    """训练/池构建入口的硬闸：拒绝任何封存测试包派生路径。

    封存范围（与 AGENTS.md 第 1、2 条一致）：
    - 08 决赛包（`08-exam` / `08-an` 及衍生物）：永久封存，任何用途都不得进入实验；
    - 第 1、2 轮官方比赛包：永久训练禁用，只可用于评估与不回归检查。

    只做路径级拦截，因此它是"防手滑"而非"防绕过"；真正的合规仍依赖预注册与评审。
    """
    if not path:
        return
    normalized = str(Path(path)).replace("\\", "/").lower()
    marker = next((item for item in _SEALED_08_MARKERS if item in normalized), None)
    if marker is not None:
        raise ValueError(
            f"{label} path contains sealed 08 final-test marker {marker!r}: {path}"
        )
    marker = next((item for item in _SEALED_ROUND12_MARKERS if item in normalized), None)
    if marker is not None:
        raise ValueError(
            f"{label} path contains train-forbidden round1/round2 marker {marker!r}: "
            f"{path} (AGENTS.md rule 2: evaluation only, never training)"
        )
