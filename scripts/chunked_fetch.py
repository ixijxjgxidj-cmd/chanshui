#!/usr/bin/env python3
"""超大 SeisBench 数据集的"分块下载→抽窗→删块"循环取数器（90GB 盘也能吃 300GB 数据集）.

===== 为什么这样设计 =====
DiTing/CWA/CEED 这类数据集动辄 100~300GB，Colab 盘只有 ~90GB，整包下载必死。
但训练根本不需要原始 180s 全长波形——PhaseNet 只吃 3001 点窗口。
所以按块循环：

    下载 1 个数据块 → 每条波形切出紧凑训练窗(3,3001)写入输出 HDF5 → 删掉该块 → 下一块

峰值磁盘占用 = 单块大小 + 紧凑输出（每 10 万窗 ≈ 3.5GB），与数据集总大小无关。
断点续传按（块，块内下标）双层记录：重跑同一命令自动跳过已完成的块。

===== 用法（Colab / 任何 Linux）=====
    # 台湾 CWA（100Hz、含 P/S 标注、按年分块——域上最接近华南的开放大集）
    python scripts/chunked_fetch.py --dataset CWA --out /content/train_pool.hdf5 \\
        --cache /content/sb_cache --max-total 200000 --mirror /content/drive/MyDrive/dizheng

    # 纯噪声块（降误报，赛题含纯噪声条目）
    python scripts/chunked_fetch.py --dataset CWANoise --out /content/train_pool.hdf5 \\
        --cache /content/sb_cache --noise --max-total 20000 --mirror ...

    # 小集冒烟（不分块，几分钟）
    python scripts/chunked_fetch.py --dataset Iquique --out /tmp/pool.hdf5 --cache /tmp/cache

输出格式与 finetune_phasenet.py --data 完全对齐：
    group "data"，每条 (3, win) float32，attrs: p_sample_100hz / s_sample_100hz /
    sampling_rate=100.0 / event_id（供按事件防泄漏切分）。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np

WIN_DEFAULT = 3001


# ---------------------------------------------------------------------------
# 纯逻辑（单元测试覆盖，不碰 seisbench）
# ---------------------------------------------------------------------------
def cut_window(
    wave: np.ndarray,
    p: float,
    s: float,
    win: int,
    rng: np.random.RandomState,
) -> tuple:
    """从 (3, n) 波形切一个训练窗，返回 (window(3,win), p_rel, s_rel)。

    规则：
    - 有 P：窗口起点 = P - U[win*0.1, win*0.6]（随机相位位置=免费增广），
      再夹进 [0, n-win]；P/S 换算窗内下标，落窗外记 -1。
    - 无 P 有 S：以 S 同理锚定。
    - 全无（噪声）：均匀随机起点。
    - n < win：右侧补零，P/S 下标不变（超界记 -1）。
    """
    c, n = wave.shape
    p = float(p) if p is not None and np.isfinite(p) and p >= 0 else -1.0
    s = float(s) if s is not None and np.isfinite(s) and s >= 0 else -1.0

    if n < win:
        out = np.zeros((c, win), dtype=np.float32)
        out[:, :n] = wave.astype(np.float32, copy=False)
        p_rel = p if 0 <= p < win else -1.0
        s_rel = s if 0 <= s < win else -1.0
        return out, p_rel, s_rel

    anchor = p if p >= 0 else s
    if anchor >= 0:
        lo, hi = int(win * 0.1), int(win * 0.6)
        start = int(anchor) - int(rng.randint(lo, hi + 1))
    else:
        start = int(rng.randint(0, max(1, n - win + 1)))
    start = int(np.clip(start, 0, n - win))

    seg = wave[:, start : start + win].astype(np.float32, copy=False)
    p_rel = p - start if p >= 0 else -1.0
    s_rel = s - start if s >= 0 else -1.0
    if not (0 <= p_rel < win):
        p_rel = -1.0
    if not (0 <= s_rel < win):
        s_rel = -1.0
    return seg, float(p_rel), float(s_rel)


def wave_ok(wave: np.ndarray, need_std: float = 1e-8) -> bool:
    """基本质量闸：形状 (3,n)、无非有限值、非死道。"""
    if wave.ndim != 2 or wave.shape[0] != 3 or wave.shape[1] < 100:
        return False
    if not np.isfinite(wave).all():
        return False
    if float(np.abs(wave).max()) <= 0 or float(wave.std()) < need_std:
        return False
    return True


def event_key(meta_row: dict, fallback: str) -> str:
    """从元数据行取事件标识（按事件切分防泄漏的键）。"""
    for k in ("source_id", "source_origin_time", "source_event_id", "event_id"):
        v = meta_row.get(k)
        if v is not None and str(v) not in ("", "nan", "None"):
            return str(v)
    return fallback


def stable_hash01(text: str) -> float:
    """字符串 → [0,1) 的稳定散列（跨进程/跨机器一致，用于切分）。"""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


# ---------------------------------------------------------------------------
# 进度与磁盘
# ---------------------------------------------------------------------------
def load_progress(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"chunks_done": [], "current": None, "written": 0}


def save_progress(path: str, prog: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prog, f)
    os.replace(tmp, path)


def free_gb(path: str) -> float:
    usage = shutil.disk_usage(os.path.dirname(os.path.abspath(path)) or ".")
    return usage.free / 1e9


def delete_chunk_cache(cache_root: str, dataset: str, chunk: str) -> int:
    """删掉某块在 seisbench 缓存里的文件（waveforms/metadata/残留压缩包）。"""
    if not chunk:
        return 0
    base = os.path.join(cache_root, "datasets", dataset.lower())
    n = 0
    for pat in (f"waveforms{chunk}*", f"metadata{chunk}*", "*.tar.gz", "*.partial"):
        for f in glob.glob(os.path.join(base, pat)):
            try:
                os.remove(f)
                n += 1
            except OSError:
                pass
    return n


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="分块下载→抽窗→删块 的大集取数器")
    ap.add_argument("--dataset", required=True,
                    help="seisbench.data 里的类名，如 CWA / CWANoise / CEED / MLAAPDE / ETHZ / Iquique")
    ap.add_argument("--out", required=True, help="紧凑训练池 HDF5 输出路径")
    ap.add_argument("--cache", required=True, help="SEISBENCH_CACHE_ROOT（临时盘，如 /content/sb_cache）")
    ap.add_argument("--chunks", default="auto",
                    help="'auto'=全部可用块；或逗号分隔块名（如 _2019,_2020）；无块数据集自动单趟")
    ap.add_argument("--win", type=int, default=WIN_DEFAULT)
    ap.add_argument("--sr", type=float, default=100.0, help="统一重采样目标（seisbench 会同步换算到时下标）")
    ap.add_argument("--noise", action="store_true", help="纯噪声数据集：不要求 P/S，窗口随机")
    ap.add_argument("--max-per-chunk", type=int, default=0, help="每块最多抽多少窗（0=不限）")
    ap.add_argument("--max-total", type=int, default=200000, help="总窗数上限")
    ap.add_argument("--min-free-gb", type=float, default=8.0, help="磁盘剩余低于此值即优雅停止")
    ap.add_argument("--mirror", default="", help="每完成一块后把输出+进度快照到此目录（如 Drive）")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ["SEISBENCH_CACHE_ROOT"] = args.cache
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    import h5py
    import seisbench.data as sbd

    cls = getattr(sbd, args.dataset, None)
    if cls is None:
        print(f"seisbench.data 里没有 {args.dataset!r}；可用示例：CWA, CWANoise, CEED, "
              f"MLAAPDE, ETHZ, Iquique, STEAD, TXED, LenDB", file=sys.stderr)
        return 2

    # ---- 块清单 ----
    if args.chunks != "auto":
        chunks = [c for c in args.chunks.split(",") if c]
    elif hasattr(cls, "available_chunks"):
        try:
            chunks = list(cls.available_chunks())
        except Exception as exc:  # noqa: BLE001
            print(f"查询块列表失败（{exc!r}），按无块数据集处理")
            chunks = [""]
    else:
        chunks = [""]
    if not chunks:
        chunks = [""]
    print(f"数据集 {args.dataset}：{len(chunks)} 个块 → {chunks[:8]}{' ...' if len(chunks) > 8 else ''}")

    prog_path = args.out + ".progress.json"
    prog = load_progress(prog_path)
    written = int(prog.get("written", 0))
    rng = np.random.RandomState(args.seed)

    h5 = h5py.File(args.out, "a")
    grp = h5.require_group("data")

    def mirror_snapshot() -> None:
        if not args.mirror:
            return
        os.makedirs(args.mirror, exist_ok=True)
        h5.flush()
        try:
            shutil.copy2(args.out, os.path.join(args.mirror, os.path.basename(args.out)))
            shutil.copy2(prog_path, os.path.join(args.mirror, os.path.basename(prog_path)))
            print(f"  [mirror] 已快照到 {args.mirror}")
        except OSError as exc:
            print(f"  [mirror] 快照失败（不影响本地继续）：{exc}")

    try:
        for chunk in chunks:
            if written >= args.max_total:
                print("已达 --max-total，停止。")
                break
            if chunk in prog["chunks_done"]:
                print(f"[skip] 块 {chunk or '<single>'} 已完成")
                continue
            if free_gb(args.cache) < args.min_free_gb:
                print(f"磁盘剩余不足 {args.min_free_gb}GB，优雅停止（重跑可续）。")
                break

            print(f"==== 块 {chunk or '<single>'}：下载/打开 ====", flush=True)
            kwargs = dict(sampling_rate=args.sr, component_order="ZNE", cache=None)
            ds = cls(chunks=[chunk], **kwargs) if chunk else cls(**kwargs)
            meta = ds.metadata
            total = len(meta)

            start_idx = 0
            cur = prog.get("current") or {}
            if cur.get("chunk") == chunk:
                start_idx = int(cur.get("idx", 0))
            print(f"    共 {total} 条，从 {start_idx} 继续；累计已写 {written}")

            taken = 0
            for i in range(start_idx, total):
                if written >= args.max_total:
                    break
                if args.max_per_chunk and taken >= args.max_per_chunk:
                    break
                if i % 2000 == 0 and free_gb(args.out) < args.min_free_gb:
                    print("    输出盘空间告急，停止本块。")
                    break
                try:
                    wave, mrow = ds.get_sample(i)
                except Exception:  # noqa: BLE001 —— 单条坏数据直接跳过
                    continue
                wave = np.asarray(wave, dtype=np.float32)
                if wave.ndim == 2 and wave.shape[0] > wave.shape[1]:
                    wave = wave.T
                if not wave_ok(wave):
                    continue

                mrow = dict(mrow) if not isinstance(mrow, dict) else mrow
                p = mrow.get("trace_p_arrival_sample", mrow.get("trace_P_arrival_sample"))
                s = mrow.get("trace_s_arrival_sample", mrow.get("trace_S_arrival_sample"))
                if not args.noise:
                    has_p = p is not None and np.isfinite(float(p)) and float(p) >= 0
                    has_s = s is not None and np.isfinite(float(s)) and float(s) >= 0
                    if not has_p and not has_s:
                        continue  # 拾取训练需要至少一个标注震相

                seg, p_rel, s_rel = cut_window(
                    wave,
                    -1.0 if args.noise else (float(p) if p is not None and np.isfinite(float(p)) else -1.0),
                    -1.0 if args.noise else (float(s) if s is not None and np.isfinite(float(s)) else -1.0),
                    args.win,
                    rng,
                )

                key = f"{args.dataset}{chunk}_{i:08d}"
                if key in grp:
                    continue
                d = grp.create_dataset(key, data=seg, compression="gzip", compression_opts=4)
                d.attrs["p_sample_100hz"] = float(p_rel)
                d.attrs["s_sample_100hz"] = float(s_rel)
                d.attrs["sampling_rate"] = float(args.sr)
                d.attrs["event_id"] = event_key(mrow, fallback=key)
                written += 1
                taken += 1

                if written % args.save_every == 0:
                    h5.flush()
                    prog["current"] = {"chunk": chunk, "idx": i + 1}
                    prog["written"] = written
                    save_progress(prog_path, prog)
                    print(f"    进度 块内 {i+1}/{total}  本块 {taken}  累计 {written}", flush=True)

            # ---- 本块完成：记账 + 删缓存 + 快照 ----
            h5.flush()
            prog["chunks_done"].append(chunk)
            prog["current"] = None
            prog["written"] = written
            save_progress(prog_path, prog)
            del ds
            n_rm = delete_chunk_cache(args.cache, args.dataset, chunk)
            print(f"    块 {chunk or '<single>'} 完成：+{taken} 窗；清理缓存文件 {n_rm} 个；"
                  f"盘剩 {free_gb(args.cache):.1f}GB")
            mirror_snapshot()

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 保存断点后退出…")
        prog["written"] = written
        save_progress(prog_path, prog)
    finally:
        h5.flush()
        h5.close()
        mirror_ok = False
        try:
            size_gb = os.path.getsize(args.out) / 1e9
            print(f"\n训练池：{args.out}  共 {written} 窗，{size_gb:.2f}GB")
            mirror_ok = True
        except OSError:
            pass
        if args.mirror and mirror_ok:
            try:
                shutil.copy2(args.out, os.path.join(args.mirror, os.path.basename(args.out)))
                shutil.copy2(prog_path, os.path.join(args.mirror, os.path.basename(prog_path)))
            except OSError:
                pass
        print("重跑同一命令即可断点续传。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("\n[出错]", repr(exc), file=sys.stderr)
        sys.exit(1)
