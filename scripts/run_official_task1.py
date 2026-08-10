#!/usr/bin/env python3
"""官方 T1 端到端 CLI —— 从官方 .mseed 输入产出 Task1Result/T1.an（相对秒）.

用法：
    python scripts/run_official_task1.py --input exam2025/ --output T1.an
    python scripts/run_official_task1.py --input round1.zip --output T1.an --answer answers.txt
    python scripts/run_official_task1.py --input in/ --output T1.an \\
        --weights ckpts/phasenet_ft.pt --device cuda --p-threshold 0.4 --s-threshold 0.3

流程（复用已测通的各层，不重复造轮子）：
    scan_exam_input → 过滤 T1 → run_task1_samples(load_waveforms, picker) →
    write_task1_results → （可选）evaluate_task1 打分。

依赖说明：
- 扫描 / 换算 / 写出 / 评估都是纯标准库或纯 numpy，随时可跑。
- **真实推理**需要 ObsPy（读 mseed）+ SeisBench + PyTorch（PhaseNet）。缺任一
  都会在构建 picker 或读波形时给出清晰的中文报错，指明缺哪个包，而不是隐晦栈回溯。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence

# 让脚本无需安装即可 import 到包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phasepicker.types import ExamSample, ExamTask, Waveform  # noqa: E402
from phasepicker.defaults import (  # noqa: E402
    DEFAULT_P_THRESHOLD,
    DEFAULT_PRETRAINED,
    DEFAULT_S_THRESHOLD,
)
from phasepicker.scoring.scorer import DEFAULT_PENALTY_MODE, PENALTY_MODES  # noqa: E402
from phasepicker.io.official_exam import scan_exam_input  # noqa: E402
from phasepicker.io.submission_writer import write_task1_results  # noqa: E402
from phasepicker.tasks.task1_runner import (  # noqa: E402
    run_task1_samples,
    run_task1_samples_fast,
)


def _read_sample_bytes(sample: ExamSample) -> bytes:
    """把 ExamSample.source_path 读成原始 mseed 字节，兼容普通文件与 zip 内条目。

    official_exam 对 zip 内条目记 source_path 为 ``<zip_path>!<entry_name>``；
    这里据此分流：含 ``!`` 且前段是 zip → 从 zip 读该条目（不解压落盘），
    否则按普通文件路径读。
    """
    from phasepicker.io.official_waveforms import read_source_bytes

    return read_source_bytes(sample.source_path)


def _make_load_waveforms_fn(duration_sink: dict | None = None):
    """构造 load_waveforms_fn；ObsPy 缺失时给清晰报错。

    返回一个吃 ExamSample、吐 List[Waveform] 的闭包。内部用 mseed_reader
    （依赖 ObsPy）把字节解成校验过的多台站波形。
    """
    try:
        from phasepicker.io.mseed_reader import load_waveforms
    except Exception as exc:  # pragma: no cover - 环境相关
        raise SystemExit(
            f"读取 mseed 需要 ObsPy，导入失败：{exc!r}\n"
            "请先安装：pip install obspy"
        )

    def _load(sample: ExamSample) -> List[Waveform]:
        raw = _read_sample_bytes(sample)
        result = load_waveforms(raw)
        for w in result.warnings:
            print(f"[warn] {sample.file_id} [{w.station}] {w.reason}: {w.detail}", file=sys.stderr)
        if duration_sink is not None and result.waveforms:
            # 供 --cap-short-s 判断长短文件；单键赋值在 GIL 下原子，多线程安全
            duration_sink[sample.file_id] = max(w.duration for w in result.waveforms)
        return result.waveforms

    return _load


def _make_picker(
    weights: str | None,
    device: str,
    p_threshold: float,
    s_threshold: float,
    pretrained: str,
    use_fp16: bool = False,
    num_threads: int | None = None,
    batch_size: int = 256,
    overlap: float = 0.5,
    compile_model: bool = False,
    long_snr_db: float | None = None,
    long_snr_min_s: float = 300.0,
    force_pair_short_s: float = 0.0,
    force_pair_floor: float = 0.03,
    force_pair_mode: str = "conditional",
    long_dedup_s: float = 0.0,
    tta_flip: bool = False,
    ensemble_long_members: int = 0,
):
    """按参数构建 picker；torch/seisbench 缺失时给清晰报错。

    --weights 三种形态（与 serve_api 同义，"评测用什么就上线什么"）：
    空=纯预训练 / 单路径=单模型 / 逗号分隔=概率软集成。
    """
    try:
        from phasepicker.inference.picker import PickerConfig, SeisBenchPicker
    except Exception as exc:  # pragma: no cover - 环境相关
        raise SystemExit(f"构建 picker 失败（import 阶段）：{exc!r}")

    # 集成时 local_weights_path 必须留空：整串逗号会被当成单个文件名
    is_ensemble = bool(weights) and "," in weights
    cfg = PickerConfig(
        device=None if device == "auto" else device,
        pretrained=pretrained,
        p_threshold=p_threshold,
        s_threshold=s_threshold,
        local_weights_path=None if is_ensemble else weights,
        use_fp16=use_fp16,
        num_threads=num_threads,
        batch_size=batch_size,
        overlap=overlap,
        compile_model=compile_model,
        long_snr_threshold_db=long_snr_db,
        long_snr_min_duration_s=long_snr_min_s,
        force_pair_max_duration_s=(force_pair_short_s if force_pair_short_s > 0 else None),
        force_pair_floor=force_pair_floor,
        force_pair_conditional=(force_pair_mode == "conditional"),
        long_dedup_p_window_s=(long_dedup_s if long_dedup_s > 0 else None),
        long_dedup_s_window_s=(long_dedup_s if long_dedup_s > 0 else None),
        tta_polarity_flip=tta_flip,
        ensemble_long_top_n=(ensemble_long_members if ensemble_long_members > 0 else None),
    )
    if is_ensemble:
        from phasepicker.inference.picker import ProbEnsemblePicker

        names = [x.strip() for x in weights.split(",") if x.strip()]
        try:
            picker = ProbEnsemblePicker.from_member_names(names, cfg)
        except Exception as exc:  # pragma: no cover - 环境相关
            raise SystemExit(f"构建概率集成失败：{exc!r}")
        print(f"拾取器: 概率集成 × {len(names)} 成员 {names}")
        return picker
    try:
        return SeisBenchPicker.from_config(cfg)
    except ImportError as exc:
        raise SystemExit(
            f"真实推理需要 SeisBench + PyTorch，加载失败：{exc!r}\n"
            "请先安装：pip install seisbench torch"
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"加载模型失败：{exc!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI 参数定义单独成函数，便于单测校验默认值与 defaults.py 同源。"""
    ap = argparse.ArgumentParser(description="官方 T1 端到端拾取（输出相对秒 T1.an）")
    ap.add_argument("--input", required=True, help="官方输入目录或 zip")
    ap.add_argument("--output", required=True, help="输出提交文件（T1.an）")
    ap.add_argument("--weights", default=None, help="可选：本地微调权重 (.pt) 路径")
    ap.add_argument("--pretrained", default=DEFAULT_PRETRAINED,
                    help="SeisBench 基础权重名；默认取 phasepicker.defaults（与部署 API 同源）")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="推理设备；默认 auto（有 CUDA 用 CUDA，否则 CPU）")
    ap.add_argument("--p-threshold", type=float, default=DEFAULT_P_THRESHOLD,
                    help="P 波触发概率阈值；默认取 phasepicker.defaults")
    ap.add_argument("--s-threshold", type=float, default=DEFAULT_S_THRESHOLD,
                    help="S 波触发概率阈值；默认取 phasepicker.defaults")
    ap.add_argument("--prefix", default="./T1-Q/",
                    help="写出行的路径前缀；默认第2轮官方风格 ./T1-Q/（第1轮传 exam2025/TASK01/）")
    ap.add_argument("--answer", default=None, help="可选：官方答案文件，提供则跑 official_eval 打分")
    ap.add_argument("--answer-package", default=None,
                    help="可选：直接从官方 zip（含嵌套 zip）读取 T1 答案并打分")
    ap.add_argument("--penalty-mode", default=DEFAULT_PENALTY_MODE, choices=PENALTY_MODES,
                    help="数量罚读法（官方规则口径存在歧义）；默认保持历史口径 merged_file_floor0")
    ap.add_argument("--penalty-table", action="store_true",
                    help="打分时额外输出全部数量罚读法的对照表（读法敏感度一览）")
    ap.add_argument("--strict-missing", action="store_true",
                    help="答案有而预测无的文件按 0 分计入均分分母（官方口径）；默认排除以便 --limit 调试")
    ap.add_argument("--limit", type=int, default=None,
                    help="仅调试：只跑前 N 个 T1 文件；正式提交不要设置")
    # ---- 性能相关 ----
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="读 mseed 的 I/O 线程数（与推理重叠）")
    ap.add_argument("--file-batch", type=int, default=16,
                    help="合批推理的文件窗口大小；1=逐文件推理")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="SeisBench 滑窗 batch 大小（显存富余可加大）")
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="SeisBench 滑窗重叠比例（0~1）；与 ab_compare 同名同义")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile 加速（先用 ab_compare 验证同分再用于正式提交）")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU 推理线程数；默认自动取满核")
    ap.add_argument("--fp16", action="store_true",
                    help="CUDA 半精度推理（提速；先用本地评分验证同分再用于正式提交）")
    ap.add_argument("--no-fast", action="store_true",
                    help="禁用流水线/合批优化，回退原始顺序实现（排查对比用）")
    ap.add_argument("--no-validate", action="store_true",
                    help="跳过写出后的提交文件回读自检")
    # ---- 短文件限额 + 长记录 SNR 闸（fork 验证、本仓三分布复验，2026-08-09）----
    ap.add_argument("--cap-short-s", type=float, default=0.0,
                    help="短文件按置信度限额的时长阈值（秒）：<=该值的文件最多保留 "
                         "--cap-max-p 个 P、--cap-max-s 个 S。0=关闭（与历史行为逐位一致）。"
                         "生产配置 300（三分布消融 r1+0.015/r2+0.026/08+0.026）")
    ap.add_argument("--cap-max-p", type=int, default=1,
                    help="短文件最多保留几个 P（配合 --cap-short-s）")
    ap.add_argument("--cap-max-s", type=int, default=1,
                    help="短文件最多保留几个 S（配合 --cap-short-s）")
    ap.add_argument("--long-snr-db", type=float, default=None,
                    help="长记录 SNR 闸阈值(dB)：时长>--long-snr-min-s 的波形丢弃 "
                         "snr_db<该值的拾取；snr_db=拾取点后2s RMS/前2s RMS。"
                         "缺省=关闭。生产配置 -1.0（r2+0.010/08+0.013/r1 零触发零副作用）")
    ap.add_argument("--long-snr-min-s", type=float, default=300.0,
                    help="多长的波形才算长记录（秒），配合 --long-snr-db")
    ap.add_argument("--force-pair-short-s", type=float, default=0.0,
                    help="短文件强制成对兜底的时长阈值（秒）：<=该值的波形若某相位"
                         "阈值上零触发，用 --force-pair-floor 低阈值补发最高峰。"
                         "0=关闭（与历史行为逐位一致）。依据：三分布真值全为>=1P+1S，"
                         "空输出必吃数量罚；arXiv:2511.06731 阈值下峰位置信息仍在")
    ap.add_argument("--force-pair-floor", type=float, default=0.03,
                    help="兜底的概率地板：曲线最大值低于该值仍放弃补发"
                         "（对冲真值无该相位的纯噪声文件）")
    ap.add_argument("--force-pair-mode", choices=["conditional", "always"],
                    default="conditional",
                    help="conditional=另一相位阈值上有触发才补（纯噪声条目免疫，"
                         "保留 36.7/43.2 分收益）；always=零触发就补（去年数据"
                         "+6.5 分但噪声条目每个 -1.0）")
    ap.add_argument("--long-dedup-s", type=float, default=0.0,
                    help="长记录事件级去重合并窗（秒）：>300s 波形在标准去重后按此"
                         "宽窗再簇合并一次（每簇留置信度最高者）。0=关闭")
    ap.add_argument("--tta-flip", action="store_true",
                    help="推理端 TTA：极性翻转副本并入概率平均（仅集成路径，耗时×2）")
    ap.add_argument("--ensemble-long-members", type=int, default=0,
                    help=">300s 长记录只用集成前 N 个成员（0=全部）。事件窗训练的"
                         "成员（如 GEOFON 微调）放列表尾部，长记录自动排除")
    return ap


def main(argv=None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    # 1) 扫描输入并过滤出 T1 样本
    samples = [s for s in scan_exam_input(args.input) if s.task == ExamTask.T1]
    if args.limit is not None:
        if args.limit <= 0:
            ap.error("--limit 必须是正整数")
        samples = samples[: args.limit]
    if not samples:
        print(f"未在 {args.input!r} 找到任何 T1 样本（.mseed）", file=sys.stderr)
        return 1
    print(f"扫描到 {len(samples)} 个 T1 样本")
    # 启动横幅：把实际生效的基座/阈值大声打出来，防止"以为在评 diting 实际评的 stead"
    print(
        f"配置: 基座={args.pretrained} 权重={args.weights or '无(纯预训练)'} "
        f"P阈值={args.p_threshold} S阈值={args.s_threshold} "
        f"overlap={args.overlap} device={args.device}"
        + (" compile" if args.compile else "")
    )

    # 2) 构建依赖（缺 obspy/seisbench/torch 会在此给出清晰报错）
    durations: dict = {}
    load_waveforms_fn = _make_load_waveforms_fn(
        duration_sink=durations if args.cap_short_s > 0 else None
    )
    picker = _make_picker(
        args.weights,
        args.device,
        args.p_threshold,
        args.s_threshold,
        args.pretrained,
        use_fp16=args.fp16,
        num_threads=args.threads,
        batch_size=args.batch_size,
        overlap=args.overlap,
        compile_model=args.compile,
        long_snr_db=args.long_snr_db,
        long_snr_min_s=args.long_snr_min_s,
        force_pair_short_s=args.force_pair_short_s,
        force_pair_floor=args.force_pair_floor,
        force_pair_mode=args.force_pair_mode,
        long_dedup_s=args.long_dedup_s,
        tta_flip=args.tta_flip,
        ensemble_long_members=args.ensemble_long_members,
    )

    # 3) 端到端推理 → 相对秒 Task1Result
    import time

    t0 = time.perf_counter()
    if args.no_fast:
        results_map = run_task1_samples(samples, load_waveforms_fn, picker)
    else:
        last_print = [0.0]

        def _progress(done: int, total: int) -> None:
            now = time.perf_counter()
            if done == total or now - last_print[0] >= 2.0:
                last_print[0] = now
                rate = done / max(1e-9, now - t0)
                eta = (total - done) / max(1e-9, rate)
                print(
                    f"\r进度 {done}/{total}  {rate:.1f} 文件/秒  剩余约 {eta:.0f}s",
                    end="" if done < total else "\n",
                    flush=True,
                )

        results_map = run_task1_samples_fast(
            samples,
            load_waveforms_fn,
            picker,
            io_workers=args.workers,
            file_batch=args.file_batch,
            progress_cb=_progress,
        )
    elapsed = time.perf_counter() - t0
    print(f"推理完成：{len(samples)} 文件，{elapsed:.1f}s（{len(samples)/max(1e-9,elapsed):.2f} 文件/秒）")

    # 3.5) 短文件限额（--cap-short-s > 0 时启用；见 postprocess/cap.py）
    if args.cap_short_s > 0:
        from phasepicker.postprocess.cap import cap_short_file_results

        results_map, n_capped, n_dropped = cap_short_file_results(
            results_map, durations, args.cap_short_s, args.cap_max_p, args.cap_max_s
        )
        print(f"短文件限额: 改动 {n_capped} 个文件，丢弃 {n_dropped} 个拾取"
              f"（阈值 {args.cap_short_s:.0f}s，P<={args.cap_max_p} S<={args.cap_max_s}）")

    # 4) 写出（保持输入扫描顺序，便于复现）
    ordered = [results_map[s.file_id] for s in samples if s.file_id in results_map]
    write_task1_results(ordered, args.output, prefix=args.prefix)
    n_p = sum(len(r.p_times_s) for r in ordered)
    n_s = sum(len(r.s_times_s) for r in ordered)
    print(f"已写出 {len(ordered)} 行到 {args.output}（P 到时 {n_p} 个，S 到时 {n_s} 个）")

    # 4.5) 提交文件回读自检：写出的每一行都必须能被官方格式 parser 原样读回
    if not args.no_validate:
        from phasepicker.io.official_answers import parse_task1_answer_lines

        with open(args.output, "r", encoding="utf-8") as f:
            reparsed = parse_task1_answer_lines(f.read().splitlines())
        rp = sum(len(r.p_times_s) for r in reparsed.values())
        rs = sum(len(r.s_times_s) for r in reparsed.values())
        if len(reparsed) != len(ordered) or rp != n_p or rs != n_s:
            print(
                f"[自检失败] 回读 {len(reparsed)} 行/P{rp}/S{rs}，"
                f"期望 {len(ordered)} 行/P{n_p}/S{n_s} —— 请检查输出格式！",
                file=sys.stderr,
            )
            return 3
        print("输出自检通过：行数与 P/S 数量回读一致，格式符合官方标准")

    # 5) 可选打分
    if args.answer and args.answer_package:
        ap.error("--answer 与 --answer-package 只能选一个")
    if args.answer or args.answer_package:
        from phasepicker.io.official_answers import parse_task1_answer_lines
        from phasepicker.eval.official_eval import (
            evaluate_task1,
            evaluate_task1_all_modes,
            penalty_modes_table,
        )

        if args.answer_package:
            from phasepicker.io.official_waveforms import read_package_answers

            answers = read_package_answers(args.answer_package, ExamTask.T1)
        else:
            with open(args.answer, "r", encoding="utf-8", errors="replace") as f:
                answers = parse_task1_answer_lines(f.read().splitlines())
        report = evaluate_task1(
            results_map,
            answers,
            penalty_mode=args.penalty_mode,
            strict_missing=args.strict_missing,
        )
        print(report.summary())
        if args.penalty_table:
            all_reports = evaluate_task1_all_modes(
                results_map, answers, strict_missing=args.strict_missing
            )
            print(penalty_modes_table(all_reports))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
