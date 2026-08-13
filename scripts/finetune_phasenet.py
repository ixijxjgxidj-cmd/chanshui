#!/usr/bin/env python3
"""PhaseNet 微调训练脚本（自包含，可断点续训，微调前后自动对比打分）.

===== 这个脚本做什么（写给非 AI 背景的队友）=====
1. 加载 SeisBench 的 PhaseNet 预训练权重（默认 diting，见 defaults.py）作为起点；
2. 在训练数据上继续训练（微调），让它适应我们的数据；
3. 微调【前】先打一次分，微调【后】再打一次分，直接看 P/S 精度有没有提升；
4. 每个 epoch 存 checkpoint，机器关机/断网后重跑同一条命令即可【断点续训】。

===== 数据来源可切换（关键设计）=====
- 默认用【合成数据】：现在就能跑通整条训练逻辑，不必等 DiTing/官方数据；
- DiTing / 官方数据到位后：把 --data 换成 hdf5 路径，脚本自动读真实数据，
  训练与评分逻辑【一行不用改】。DiTing 子集格式 = 我们 diting_seisbench.py 的输出
  （group "data" 下每条 dataset，attrs 含 p_sample_100hz / s_sample_100hz——
  历史名，实为取数 --sr 采样率下的窗内下标；attrs['sampling_rate'] 会与 --sr 校验）。

===== 为什么评分逻辑内嵌 =====
GPU 机器上没有 phasepicker 包，所以把已测过 20/20 的评分规则原样抄进来，
保证本地分和官方口径一致（P<=0.1s 满分,1.0s 零分; S<=0.2s 满分,2.0s 零分;
数量误差>5% 每个扣 0.5）。
"""
from __future__ import annotations
import argparse, json, os, sys, math, time
import numpy as np

# 拾取阈值单一真源（src/phasepicker/defaults.py）。Colab/云机都是整库 clone，此相对路径必在。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from phasepicker.defaults import (  # noqa: E402
    DEFAULT_P_THRESHOLD,
    DEFAULT_PRETRAINED,
    DEFAULT_S_THRESHOLD,
)
from phasepicker.training.data_policy import assert_experiment_path_allowed  # noqa: E402

# ============ 内嵌评分（与已测 scorer 一字不差）============
_PHASE = {"P": (0.1, 1.0), "S": (0.2, 2.0)}

def phase_time_score(residual_s, phase_type):
    full, zero = _PHASE[phase_type]
    r = abs(residual_s)
    if r <= full: return 1.0
    if r >= zero: return 0.0
    return (zero - r) / (zero - full)

def match_phases(pred_times, true_times, phase_type):
    _, zero = _PHASE[phase_type]
    cand = []
    for i, pt in enumerate(pred_times):
        for j, tt in enumerate(true_times):
            r = abs(pt - tt)
            if r < zero: cand.append((r, i, j))
    cand.sort(key=lambda x: x[0])
    up, ut, matched = set(), set(), []
    for r, i, j in cand:
        if i in up or j in ut: continue
        up.add(i); ut.add(j); matched.append((i, j, r))
    return matched

def count_error_penalty(n_pred, n_true):
    if n_true == 0: return 0.5 * n_pred
    allowed = 0.05 * n_true
    diff = abs(n_pred - n_true)
    if diff <= allowed: return 0.0
    return 0.5 * int(math.ceil(diff - allowed))

def score_file(pred, truth):
    def sp(items, t): return [tt for pt, tt in items if pt == t]
    pp, ps = sp(pred, "P"), sp(pred, "S")
    tp, ts = sp(truth, "P"), sp(truth, "S")
    mp = match_phases(pp, tp, "P"); ms = match_phases(ps, ts, "S")
    p_sc = sum(phase_time_score(r, "P") for _, _, r in mp)
    s_sc = sum(phase_time_score(r, "S") for _, _, r in ms)
    pen = count_error_penalty(len(pp)+len(ps), len(tp)+len(ts))
    total = max(0.0, p_sc + s_sc - pen)
    return dict(total=total, p_sc=p_sc, s_sc=s_sc, pen=pen,
                pres=[r for _,_,r in mp], sres=[r for _,_,r in ms])

# ============ 合成数据（训练/评分用，与 closed_loop 同套路）============
def synth_window(n, sr, p_sample, s_sample, seed):
    """合成一条 (3, n) 窗。子波时长/衰减按【秒】定义再乘 sr——
    50Hz 与 100Hz 下物理形态一致（100Hz 时与旧版逐点相同）。"""
    rng = np.random.RandomState(seed)
    z = rng.normal(0, 0.02, n); ns = rng.normal(0, 0.02, n); e = rng.normal(0, 0.02, n)
    tp = np.arange(0, int(round(4.0 * sr)))              # P 子波 4s
    pw = np.exp(-tp/(1.2*sr))*np.sin(2*np.pi*8.0*tp/sr)  # 衰减 1.2s, 8Hz
    if p_sample >= 0:
        z[p_sample:p_sample+len(tp)] += 1.0*pw
        ns[p_sample:p_sample+len(tp)] += 0.3*pw
        e[p_sample:p_sample+len(tp)] += 0.3*pw
    ts = np.arange(0, int(round(6.0 * sr)))              # S 子波 6s
    sw = np.exp(-ts/(2.0*sr))*np.sin(2*np.pi*3.5*ts/sr)  # 衰减 2.0s, 3.5Hz
    if s_sample >= 0:
        ns[s_sample:s_sample+len(ts)] += 1.6*sw
        e[s_sample:s_sample+len(ts)] += 1.6*sw
        z[s_sample:s_sample+len(ts)] += 0.4*sw
    return np.vstack([z, ns, e]).astype("float32")

def normalize(x):
    x = x - x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + 1e-6
    return (x / s).astype("float32")

def gaussian(n, center, sigma):
    t = np.arange(n)
    return np.exp(-0.5*((t-center)/sigma)**2).astype("float32")

def make_soft_label(n, p_sample, s_sample, label_order, sr=100.0, sigma_p_s=0.2, sigma_s_s=0.3):
    """按模型 label 顺序生成软标签 (C, n)。

    sigma 按【秒】定义（内部换算 sigma*sr 个采样点）：50Hz 与 100Hz 下物理宽度一致。
    默认 0.2s/0.3s = 旧版 100Hz 采样点常数 20/30 的等效值。

    PhaseNet 的 forward 在当前 SeisBench 版本里已经输出 softmax 概率，因此训练目标
    也必须是每个采样点三通道和为 1 的概率分布。旧写法在 P/S 有重叠尾巴时可能让
    P+S+N > 1；这里统一做一次逐点归一化，保证 loss 口径稳定。
    """
    sigma_p = float(sigma_p_s) * float(sr)
    sigma_s = float(sigma_s_s) * float(sr)
    P = gaussian(n, p_sample, sigma_p) if p_sample >= 0 else np.zeros(n, "float32")
    S = gaussian(n, s_sample, sigma_s) if s_sample >= 0 else np.zeros(n, "float32")
    N = np.clip(1.0 - P - S, 0, 1).astype("float32")
    chans = []
    for lab in label_order:
        u = str(lab).upper()
        chans.append(P if u.startswith("P") else S if u.startswith("S") else N)
    y = np.vstack(chans).astype("float32")
    y /= np.maximum(y.sum(axis=0, keepdims=True), 1e-6)
    return y

def phasenet_log_probs(out):
    """把 PhaseNet forward 输出统一转成 log-probabilities。

    SeisBench PhaseNet(stead) 的 forward 通常已经是 softmax 概率；少数版本或未来模型
    可能返回 logits。这里用通道和是否接近 1 来判断，避免对概率再 softmax 一次。
    """
    import torch

    with torch.no_grad():
        o = out.detach()
        channel_sum = o.sum(dim=1)
        is_prob = (
            torch.isfinite(o).all()
            and float(o.min()) >= -1e-5
            and float(o.max()) <= 1.0 + 1e-5
            and torch.allclose(
                channel_sum,
                torch.ones_like(channel_sum),
                rtol=1e-3,
                atol=1e-3,
            )
        )
    if is_prob:
        return torch.log(out.clamp_min(1e-7))
    return torch.log_softmax(out, dim=1)

def set_safe_finetune_mode(model, update_bn=False):
    """小样本微调时冻结 BatchNorm/Dropout 的训练态。

    这次崩溃最像 BN running stats 被 40 条高度相似的合成数据冲坏：训练 loss 看似正常，
    但 eval/classify 使用被污染的 running stats 后 P/S 峰消失。卷积参数仍然会训练；
    只是 BN 用预训练统计量、Dropout 关闭，让训练前向和推理前向保持一致。
    """
    import torch

    model.train()
    if update_bn:
        return

    bn_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    for module in model.modules():
        if isinstance(module, bn_types):
            module.eval()
            for p in module.parameters(recurse=False):
                p.requires_grad = False
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.eval()


def configure_trainable_scope(model, scope="all"):
    """Freeze model parameters according to a preregistered fine-tuning scope."""
    if scope not in {"all", "out"}:
        raise ValueError(f"unsupported trainable scope: {scope}")
    for parameter in model.parameters():
        parameter.requires_grad = scope == "all"
    if scope == "out":
        for parameter in model.out.parameters():
            parameter.requires_grad = True
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names:
        raise ValueError(f"trainable scope {scope!r} selected no parameters")
    return names


def checkpoint_selection_path(selection, last_path, best_path):
    if selection == "last":
        return last_path
    if selection == "best":
        return best_path
    raise ValueError(f"unsupported checkpoint selection: {selection}")

def save_checkpoint(path, model, opt, epoch, loss, best_score, args, extra=None):
    import torch

    payload = {
        "model": model.state_dict(),
        "opt": opt.state_dict() if opt is not None else None,
        "epoch": epoch,
        "loss": loss,
        "best_score": best_score,
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)

def load_checkpoint_state(path, device):
    """加载自家 checkpoint。torch>=2.6 默认 weights_only=True，会拒载 extras 里
    np.float64 之类的对象——自家产物可信，必须显式 weights_only=False。"""
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # 老 torch 无 weights_only 参数
        return torch.load(path, map_location=device)

def ensure_baseline_has_picks(tag, sc):
    """微调前基线一个 pick 都没有 → monitor 恒 0，best 守门与前后对比全瞎，禁止开训。
    最常见根因：采样率错位（diting 原生 50Hz，数据/评分却按 100Hz 声明）或阈值过高。"""
    if int(sc.get("n_pred", 0)) == 0:
        raise SystemExit(
            "[%s] 微调前基线评分一个 pick 都没有——monitor 已死，拒绝开训。\n"
            "请检查：1) 采样率是否与基座一致（diting 用 --sr 50，数据也须按 50Hz 取）；\n"
            "        2) 阈值是否过高（当前 P=%.2f S=%.2f）；3) 留出集构成是否全是噪声窗。" % (
                tag, DEFAULT_P_THRESHOLD, DEFAULT_S_THRESHOLD))

# ============ 数据集构造 ============
def build_synth_dataset(n_samples, win, sr, seed0=0):
    """随机造一批训练窗口。返回 [(wave(3,win), p, s), ...]。"""
    rng = np.random.RandomState(seed0)
    items = []
    for k in range(n_samples):
        p = int(rng.randint(int(win*0.15), int(win*0.45)))
        s = int(p + rng.randint(int(win*0.15), int(win*0.35)))
        s = min(s, win - int(round(6.0*sr)) - 50)  # 给 6s 的 S 子波留满长度+护边
        wave = synth_window(win, sr, p, s, seed=seed0+k+1)
        items.append((normalize(wave), p, s))
    return items

def load_hdf5_dataset(path, win, expect_sr=None, rng=None):
    """读 diting_seisbench.py / chunked_fetch.py 产出的子集。

    expect_sr: 校验每条 attrs['sampling_rate'] 与之一致——训练前向不走 seisbench
    重采样，采样率错位不会报错只会静默学错时间尺度，必须在入口拦死。
    attr 名 p_sample_100hz 是历史名，实为【取数时 --sr】采样率下的窗内下标。
    """
    import h5py
    if rng is None:
        rng = np.random.RandomState(1234)  # 固定 seed：留出集窗口跨 epoch/跨运行可比
    items = []
    warned_no_sr = False
    with h5py.File(path, "r") as f:
        grp = f["data"]
        for key in grp:
            d = grp[key]
            if expect_sr is not None:
                ds_sr = d.attrs.get("sampling_rate")
                if ds_sr is None:
                    if not warned_no_sr:
                        print("[警告] %s 的窗缺 sampling_rate 属性（旧池），无法校验与 --sr=%g 是否一致"
                              % (path, expect_sr))
                        warned_no_sr = True
                elif abs(float(ds_sr) - float(expect_sr)) > 1e-6:
                    raise SystemExit(
                        "数据采样率不一致：%s 里 %s 的 sampling_rate=%g，而本次训练 --sr=%g。\n"
                        "请用 chunked_fetch.py --sr %g 重新取数（diting 路线用 --sr 50）。" % (
                            path, key, float(ds_sr), float(expect_sr), float(expect_sr)))
            wave = np.asarray(d, dtype="float32")
            if wave.shape[0] > wave.shape[1]:
                wave = wave.T
            p = int(d.attrs.get("p_sample_100hz", -1))
            s = int(d.attrs.get("s_sample_100hz", -1))
            # 裁/补到统一窗口 win,并同步平移到时
            wave, p, s = _fit_window(wave, p, s, win, rng=rng)
            items.append((normalize(wave), p, s))
    return items

def _fit_window(wave, p, s, win, rng=None, margin=250):
    """裁/补到 win。裁窗时优先在 P/S 联合可行区间随机采样（margin=diting 盲区 250 点），
    保证 P、S 同窗且避开首尾盲区；放不下回落到锚相+随机偏移（不再固定 1/3 处）。"""
    c, n = wave.shape
    if n == win:
        return wave, p, s
    if n > win:
        if rng is None:
            rng = np.random.RandomState(0)
        start = None
        if p >= 0 and s >= 0 and (s - p) <= win - 2 * margin:
            lo = max(0, int(s) - win + margin)
            hi = min(int(p) - margin, n - win)
            if lo <= hi:  # hi<lo：P 贴波形头或 S 贴波形尾,联合区间被夹没,回落锚定
                start = int(rng.randint(lo, hi + 1))
        if start is None:
            center = p if p >= 0 else (s if s >= 0 else n // 2)
            off = int(rng.randint(int(win * 0.1), int(win * 0.6) + 1))
            start = int(np.clip(center - off, 0, max(0, n - win)))
        wave = wave[:, start:start+win]
        p = p-start if p >= 0 else -1
        s = s-start if s >= 0 else -1
        if p < 0 or p >= win: p = -1
        if s < 0 or s >= win: s = -1
        return wave, p, s
    out = np.zeros((c, win), dtype="float32"); out[:, :n] = wave
    return out, p, s

# ============ 评分(用 model.classify,前后对比同一套测试集) ============
def _predict_window(model, wave, sr, station="SYN"):
    """单窗 classify → [(相型, 秒), ...]。阈值统一走 phasepicker.defaults（与部署一致）。"""
    from obspy import Stream, Trace, UTCDateTime
    t0 = UTCDateTime(0)
    st = Stream()
    for ch, name in zip(wave, ["Z", "N", "E"]):
        tr = Trace(data=np.asarray(ch, dtype="float32"))
        tr.stats.sampling_rate = sr
        tr.stats.starttime = t0
        tr.stats.channel = "HH" + name
        tr.stats.station = station
        st.append(tr)
    out = model.classify(st, P_threshold=DEFAULT_P_THRESHOLD, S_threshold=DEFAULT_S_THRESHOLD)
    picks = getattr(out, "picks", out)
    pred = []
    for pk_ in list(picks):
        pk = getattr(pk_, "peak_time", None)
        sec = float(pk - t0) if pk is not None else float("nan")
        ptype = str(getattr(pk_, "phase", "?")).upper()
        if ptype in ("P", "S") and not math.isnan(sec):
            pred.append((ptype, sec))
    return pred

def _aggregate(reports, n_pred_total, extra=None):
    """汇总打分报告。数值全部 float() 化——np 标量进 checkpoint 会触发
    torch>=2.6 weights_only 拒载，进 json 也不稳。n_pred=0 是"评分已死"的信号。"""
    n = len(reports)
    tot = sum(r["total"] for r in reports)
    allp = [x for r in reports for x in r["pres"]]
    alls = [x for r in reports for x in r["sres"]]
    out = dict(
        mean_score=float(tot / n) if n else float("nan"),
        p_res=float(np.mean(allp)) if allp else float("nan"),
        s_res=float(np.mean(alls)) if alls else float("nan"),
        p_hit=float(np.mean([1.0 if x <= 0.1 else 0.0 for x in allp])) if allp else 0.0,
        s_hit=float(np.mean([1.0 if x <= 0.2 else 0.0 for x in alls])) if alls else 0.0,
        n=n, n_pred=int(n_pred_total),
    )
    if extra:
        out.update(extra)
    return out

def eval_score(model, sr, device):
    cases = [(1500,2800,101),(2000,3500,102),(1000,2200,103),(2500,4200,104),(1800,3100,105)]
    reports = []
    n_pred_total = 0
    model.eval()
    for (ps_, ss_, sd) in cases:
        arr = synth_window(6000, sr, ps_, ss_, seed=sd)
        pred = _predict_window(model, arr, sr, station="SYN")
        n_pred_total += len(pred)
        truth = [("P", ps_/sr), ("S", ss_/sr)]
        reports.append(score_file(pred, truth))
    return _aggregate(reports, n_pred_total)

def print_score(tag, sc):
    n = sc.get("n", 0)
    print("[%s] 平均分=%.4f/2.0 | P残差=%.3fs(满分率%.0f%%) | S残差=%.3fs(满分率%.0f%%) | 样本=%d" % (
        tag, sc["mean_score"], sc["p_res"], sc["p_hit"]*100, sc["s_res"], sc["s_hit"]*100, n))
    if sc.get("n_noise"):
        print("[%s] 噪声窗=%d | 误报=%.2f picks/窗 (对应赛制数量罚,越低越好;不计入 monitor)" % (
            tag, sc["n_noise"], sc["noise_fp_per_win"]))

# ============ 真实留出集评分（用 model.classify，与合成评分同一套打分口径）============
_HOLDOUT_SAMPLE_SEED = 20260801  # 固定 seed：跨 epoch/跨微调前后抽到同一批窗，monitor 才可比
_HOLDOUT_CACHE = {}  # (path, win, max_files, sr) -> (labeled, noise)；每 epoch 重读+解压太浪费

def stratified_sample(items, max_files, rng):
    """分层抽样：标注窗与噪声窗分开配额（噪声约 1/6，如 300 → 250+50）。

    旧版 items[:max_files] 按 h5py 字母序截断，CWANoise_* 排在 CWA_* 前面，
    抽出来全是噪声窗 → monitor 恒 0。噪声窗官方计分上限就是 0 分（只会被扣），
    不能混进 monitor 均值，只单独观测"误报/窗"。
    """
    labeled = [it for it in items if it[1] >= 0 or it[2] >= 0]
    noise = [it for it in items if it[1] < 0 and it[2] < 0]
    if max_files and max_files > 0:
        n_noise = min(len(noise), max_files // 6)
        n_labeled = min(len(labeled), max_files - n_noise)
        if len(labeled) > n_labeled:
            idx = rng.choice(len(labeled), size=n_labeled, replace=False)
            idx.sort()
            labeled = [labeled[i] for i in idx]
        if len(noise) > n_noise:
            idx = rng.choice(len(noise), size=n_noise, replace=False)
            idx.sort()
            noise = [noise[i] for i in idx]
    return labeled, noise

def eval_holdout(model, holdout_path, sr, device, win, max_files=0):
    """在真实留出集上评分。留出集是标准 HDF5（group 'data'，attrs p/s_sample_100hz）。

    为什么要单独有它：eval_score 用的是合成 case，预训练模型本就近满分，
    证明不了模型在真实数据上的泛化。真实留出集用【训练没见过】的真波形，
    才能回答“这次微调到底有没有用”。打分规则与合成完全一致，可直接对比。
    monitor = 标注窗官方分均值；噪声窗单独输出"误报 picks/窗"。
    """
    cache_key = (os.path.abspath(holdout_path), int(win), int(max_files), float(sr))
    if cache_key not in _HOLDOUT_CACHE:
        items = load_hdf5_dataset(holdout_path, win, expect_sr=sr)
        _HOLDOUT_CACHE[cache_key] = stratified_sample(
            items, max_files, np.random.RandomState(_HOLDOUT_SAMPLE_SEED))
    labeled, noise = _HOLDOUT_CACHE[cache_key]
    model.eval()
    reports = []
    n_pred_total = 0
    for wave, p, s in labeled:
        pred = _predict_window(model, wave, sr, station="HLD")
        n_pred_total += len(pred)
        truth = []
        if p >= 0:
            truth.append(("P", p / sr))
        if s >= 0:
            truth.append(("S", s / sr))
        reports.append(score_file(pred, truth))
    noise_fp = 0
    for wave, p, s in noise:
        pred = _predict_window(model, wave, sr, station="HLD")
        noise_fp += len(pred)  # 噪声窗 picks 只进误报指标,不进 n_pred——
        # 否则"标注窗全零、噪声窗有误报"的极端坏况会骗过基线零 pick 守门
    extra = dict(
        n_noise=len(noise),
        noise_fp_per_win=float(noise_fp) / len(noise) if noise else float("nan"),
    )
    return _aggregate(reports, n_pred_total, extra=extra)

# ============ 主流程 ============
def main():
    ap = argparse.ArgumentParser(description="PhaseNet 微调(可断点续训,前后对比)")
    ap.add_argument("--data", default="synth", help="'synth' 或 训练集 hdf5 路径(DiTing/GeoNet子集)")
    ap.add_argument("--holdout", default="", help="真实留出集 hdf5(训练没见过的真波形);留空则只做合成评分")
    ap.add_argument("--holdout-max", type=int, default=200, help="留出集最多评多少条(0=全部);classify 慢,默认封顶")
    ap.add_argument("--out", default="/data/coding/dizheng/runs/ft1", help="产物目录(checkpoint等)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16, help="8G显存建议8~16,炸显存就调小")
    ap.add_argument("--lr", type=float, default=3e-5, help="微调用小学习率,别破坏预训练特征")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="小样本 sanity check 默认不做权重衰减")
    ap.add_argument("--phase-weight", type=float, default=5.0, help="P/S loss 权重；原 30 对小样本过猛")
    ap.add_argument("--p-weight", type=float, default=None,
                    help="单独指定 P 的 loss 权重；不给则回落到 --phase-weight")
    ap.add_argument("--s-weight", type=float, default=None,
                    help="单独指定 S 的 loss 权重；不给则回落到 --phase-weight。"
                         "想把 S 满分率拉起来就调大它（如 8~10）")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值；<=0 表示关闭")
    ap.add_argument("--update-bn", action="store_true", help="允许更新 BatchNorm running stats（默认冻结，防小样本冲坏）")
    ap.add_argument("--score-every", type=int, default=1, help="每多少 epoch 跑一次合成评分并更新 best；<=0 关闭")
    ap.add_argument("--checkpoint-selection", choices=("best", "last"), default="best",
                    help="best=按 monitor 选轮次；last=固定末轮，留出集只做最终报告")
    ap.add_argument("--trainable-scope", choices=("all", "out"), default="all",
                    help="all=全模型微调；out=只训练最终输出卷积，降低域外遗忘风险")
    ap.add_argument("--n_synth", type=int, default=400, help="合成模式:造多少训练窗口")
    ap.add_argument("--win", type=int, default=3001, help="训练窗口长度(PhaseNet默认3001)")
    ap.add_argument("--sr", type=float, default=None,
                    help="采样率。留空=自动取基座模型原生采样率(diting=50, stead=100);"
                         "显式给定但与模型不符会报错退出(逃生门 --force-sr)")
    ap.add_argument("--force-sr", action="store_true",
                    help="跳过 --sr 与模型原生采样率的一致性校验(明知故犯的逃生门;"
                         "训练前向不走 seisbench 重采样,错位=学错时间尺度)")
    ap.add_argument("--sigma-s", type=float, nargs=2, default=(0.2, 0.3),
                    metavar=("P_SEC", "S_SEC"),
                    help="高斯软标签宽度(秒,先 P 后 S)。按秒定义与采样率解耦;"
                         "默认 0.2/0.3s=旧版 100Hz 下 20/30 采样点的等效值")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="从上次 checkpoint 断点续训")
    ap.add_argument("--pretrained", default=DEFAULT_PRETRAINED,
                    help="SeisBench 预训练权重名(默认取 defaults.py 单一真源=diting,"
                         "去年真题 A/B 实测完胜 stead;复现旧 stead 实验需显式传 --pretrained stead)")
    ap.add_argument("--init-weights", default="",
                    help="可选：本地权重(.pt)作为微调起点（如 USTC-Pickers 的省级 picker），"
                         "在 --pretrained 骨架上覆盖加载")
    args = ap.parse_args()

    for label, path in (
        ("training data", args.data),
        ("holdout", args.holdout),
        ("output", args.out),
        ("initial weights", args.init_weights),
    ):
        if path and path != "synth":
            try:
                assert_experiment_path_allowed(path, label)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc

    import torch
    import seisbench.models as sbm
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==== 加载预训练 PhaseNet(%s) ====" % args.pretrained)
    model = sbm.PhaseNet.from_pretrained(args.pretrained).to(device)
    if args.init_weights:
        print("==== 覆盖加载本地起点权重: %s ====" % args.init_weights)
        try:
            ckpt = torch.load(args.init_weights, map_location=device, weights_only=False)
        except TypeError:  # 老 torch 无 weights_only
            ckpt = torch.load(args.init_weights, map_location=device)
        state = ckpt
        if isinstance(ckpt, dict):
            for k in ("model", "model_state_dict", "state_dict"):
                if k in ckpt:
                    state = ckpt[k]
                    break
        model.load_state_dict(state)
    label_order = list(getattr(model, "labels", ["P","S","N"]))
    trainable_names = configure_trainable_scope(model, args.trainable_scope)
    print("设备=%s | 模型输出通道顺序=%s" % (device, label_order))
    print("可训练范围=%s | 参数张量=%d | %s" % (
        args.trainable_scope, len(trainable_names), ",".join(trainable_names)))

    # ---- 采样率唯一真源 = 模型原生采样率（训练前向喂裸张量,不走 seisbench 重采样,错位即学错时间尺度）----
    sr_model = getattr(model, "sampling_rate", None)
    if sr_model is not None:
        sr_model = float(sr_model)
        if args.sr is None:
            args.sr = sr_model
            print("采样率: 自动取 %s 原生 %gHz（数据必须按该采样率取,chunked_fetch --sr %g）" % (
                args.pretrained, sr_model, sr_model))
        elif abs(args.sr - sr_model) > 1e-6:
            if args.force_sr:
                print("[警告] --sr %g 与 %s 原生 %gHz 不一致,--force-sr 已跳过校验——"
                      "训练与推理的时间尺度将错位,后果自负" % (args.sr, args.pretrained, sr_model))
            else:
                raise SystemExit(
                    "--sr %g 与 %s 原生采样率 %gHz 不一致。\n"
                    "请改用 --sr %g（数据也必须按该采样率取: chunked_fetch.py --sr %g），"
                    "或加 --force-sr 强行继续。" % (
                        args.sr, args.pretrained, sr_model, sr_model, sr_model))
    elif args.sr is None:
        args.sr = 100.0
        print("[警告] 模型无 sampling_rate 属性，--sr 回退默认 100Hz")
    print("评分阈值: P=%.2f S=%.2f（phasepicker.defaults 单一真源,与部署一致）" % (
        DEFAULT_P_THRESHOLD, DEFAULT_S_THRESHOLD))

    # ---- 微调前基线分 ----
    print("\n==== 微调【前】基线评分 ====")
    before = eval_score(model, args.sr, device)
    print_score("微调前(合成)", before)
    ensure_baseline_has_picks("合成", before)
    before_ho = None
    if args.holdout and args.checkpoint_selection == "best":
        print("\n==== 微调【前】真实留出集评分 ====")
        before_ho = eval_holdout(model, args.holdout, args.sr, device, args.win, args.holdout_max)
        print_score("微调前(真实)", before_ho)
        if before_ho["n"] == 0:
            raise SystemExit("留出集抽样后没有标注窗（为空或全是噪声窗）——monitor 无法工作，检查 holdout 构成")
        ensure_baseline_has_picks("留出集", before_ho)
    elif args.holdout:
        print("\n[固定末轮] 训练前不打开真实留出集；它只在训练结束后做一次终检。")

    # ---- 训练数据 ----
    print("\n==== 构造训练数据 (%s) ====" % args.data)
    if args.data == "synth":
        raw = build_synth_dataset(args.n_synth, args.win, args.sr, seed0=args.seed)
    else:
        raw = load_hdf5_dataset(args.data, args.win, expect_sr=args.sr)
    print("训练样本数: %d" % len(raw))
    # 8 万窗时单份数据 ~3GB：先出 X 就释放 raw、torch.from_numpy 零拷贝，
    # 避免 raw/X_np/Y_np/X_torch 四份并存把标准 Colab 12.7GB 撑爆。
    n_items = len(raw)
    ps_pairs = [(p, s) for _, p, s in raw]
    X = torch.from_numpy(np.stack([w for w, _, _ in raw]))
    del raw
    Y = torch.from_numpy(np.stack([
        make_soft_label(args.win, p, s, label_order, args.sr, args.sigma_s[0], args.sigma_s[1])
        for p, s in ps_pairs]))
    del ps_pairs

    # 小样本微调的第一原则：别让 BN/Dropout 的训练态和 classify/eval 推理态错位。
    # requires_grad 会在这里前设置好，因此 optimizer 不会更新被冻结的 BN affine 参数。
    set_safe_finetune_mode(model, update_bn=args.update_bn)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---- 断点续训 ----
    ckpt_last = os.path.join(args.out, "last.pt")
    ckpt_best = os.path.join(args.out, "best.pt")
    start_epoch = 0
    best_score = float("-inf")
    resumed = False
    if args.resume and os.path.exists(ckpt_last):
        state = load_checkpoint_state(ckpt_last, device)
        model.load_state_dict(state["model"])
        if state.get("opt") is not None:
            try:
                opt.load_state_dict(state["opt"])
            except ValueError as exc:
                print("[断点续训] optimizer 状态与当前冻结策略不兼容，已只加载模型权重：%r" % exc)
        start_epoch = state["epoch"]
        best_score = float(state.get("best_score", best_score))
        resumed = True
        print("[断点续训] 从 epoch %d 继续，历史 best=%.4f" % (start_epoch, best_score))

    # baseline 作为 best 守门员：训练若始终打不过基线，best 就停在预训练权重（诚实的"没提升"）。
    # 守门指标：有 --holdout 时用【真实留出集】分数；否则退回合成分（纯 sanity check 场景）。
    # 关键修复：旧版一律用合成分挑 best，但预训练在合成上本就 2.0，微调只会让它降，
    # 于是 best 永远停在微调前权重——真实数据上学到的东西被丢弃。必须用留出集挑 best。
    # 续训时不写：此刻模型已是 last 权重，而 before 分数是预训练权重的，写进去会张冠李戴，
    # 还可能把上一轮真 best.pt 覆盖掉。
    baseline_monitor = before_ho["mean_score"] if before_ho is not None else before["mean_score"]
    if args.checkpoint_selection == "best" and not resumed and baseline_monitor >= best_score:
        best_score = baseline_monitor
        save_checkpoint(
            ckpt_best, model, opt, start_epoch, loss=None, best_score=best_score,
            args=args, extra={"score": before, "holdout": before_ho, "tag": "baseline_or_resume"},
        )

    # ---- 训练循环 ----
    print("\n==== 开始微调: epochs=%d batch=%d lr=%g ====" % (args.epochs, args.batch, args.lr))
    print("     BN/Dropout: %s | phase_weight=%.2f | weight_decay=%g | grad_clip=%g" % (
        "允许更新" if args.update_bn else "冻结为推理态",
        args.phase_weight,
        args.weight_decay,
        args.grad_clip,
    ))
    # 类别权重: N(噪声)恒为 1.0；P/S 稀疏需加权。
    # 默认 P/S 都用 --phase-weight；也可用 --p-weight / --s-weight 单独覆盖，
    # 想把某一相(如 S)的满分率拉起来就单独调大它。原先 30 倍在小样本上过猛，默认 5 倍更稳。
    p_w = args.p_weight if args.p_weight is not None else args.phase_weight
    s_w = args.s_weight if args.s_weight is not None else args.phase_weight
    w = []
    for lab in label_order:
        u = str(lab).upper()
        if u.startswith("P"):
            w.append(p_w)
        elif u.startswith("S"):
            w.append(s_w)
        else:  # N / 噪声
            w.append(1.0)
    print("     类别权重: P=%.2f S=%.2f N=1.00 (通道顺序 %s)" % (p_w, s_w, label_order))
    class_w = torch.tensor(w, dtype=torch.float32, device=device).view(1, -1, 1)  # (1,C,1)
    nB = int(math.ceil(n_items / args.batch))
    for ep in range(start_epoch, args.epochs):
        set_safe_finetune_mode(model, update_bn=args.update_bn)
        perm = torch.randperm(n_items)
        ep_loss = 0.0
        for b in range(nB):
            idx = perm[b*args.batch:(b+1)*args.batch]
            xb = X[idx].to(device); yb = Y[idx].to(device)
            out = model(xb)
            logp = phasenet_log_probs(out)
            loss = -(class_w * yb * logp).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            ep_loss += float(loss.detach())
        ep_loss /= nB

        score = None
        holdout_score = None
        should_score = (
            args.checkpoint_selection == "best"
            and args.score_every > 0
            and ((ep + 1) % args.score_every == 0 or ep + 1 == args.epochs)
        )
        if should_score:
            score = eval_score(model, args.sr, device)
            # 挑 best 的守门指标：有 --holdout 用真实留出集，否则退回合成分。
            # 合成分只作 sanity（看有没有崩），best 的真正依据是留出集泛化。
            if args.holdout:
                holdout_score = eval_holdout(
                    model, args.holdout, args.sr, device, args.win, args.holdout_max)
                monitor = holdout_score["mean_score"]
                msg = " | synth=%.4f real=%.4f/2.0" % (score["mean_score"], monitor)
            else:
                monitor = score["mean_score"]
                msg = " | score=%.4f/2.0" % monitor
            # 严格大于才刷新：monitor 恒定（如评分死掉恒 0）时不许反复覆盖 best.pt
            if monitor > best_score:
                best_score = monitor
                save_checkpoint(
                    ckpt_best, model, opt, ep + 1, ep_loss, best_score, args,
                    extra={"score": score, "holdout": holdout_score, "tag": "best"},
                )
                msg += " (刷新 best)"
        else:
            msg = ""

        save_checkpoint(ckpt_last, model, opt, ep + 1, ep_loss, best_score, args,
                        extra={"score": score, "holdout": holdout_score})
        with open(os.path.join(args.out, "progress.json"), "w") as f:
            json.dump({"epoch": ep+1, "loss": ep_loss, "best_score": best_score,
                       "score": score, "holdout": holdout_score}, f)
        print("  epoch %2d/%d  loss=%.5f%s  (已存 last.pt/best.pt)" % (
            ep+1, args.epochs, ep_loss, msg
        ), flush=True)

    # ---- 微调后评分 + 对比 ----
    selected_checkpoint = checkpoint_selection_path(
        args.checkpoint_selection, ckpt_last, ckpt_best)
    if os.path.exists(selected_checkpoint):
        state = load_checkpoint_state(selected_checkpoint, device)
        model.load_state_dict(state["model"])
        print("\n==== 微调【后】评分（使用 %s checkpoint）====" % args.checkpoint_selection)
    else:
        print("\n==== 微调【后】评分 ====")
    after = eval_score(model, args.sr, device)
    print_score("微调后(合成)", after)
    after_ho = None
    if args.holdout:
        print("\n==== 微调【后】真实留出集评分 ====")
        after_ho = eval_holdout(model, args.holdout, args.sr, device, args.win, args.holdout_max)
        print_score("微调后(真实)", after_ho)

    print("\n==== 对比 ====")
    print("[合成] 关键看'微调后不崩'(预训练本就近满分):")
    print_score("微调前(合成)", before)
    print_score("微调后(合成)", after)
    d = after["mean_score"] - before["mean_score"]
    print("  合成平均分变化: %+.4f  %s" % (d, "(提升↑)" if d > 0 else "(未提升)"))
    if before_ho is not None and after_ho is not None:
        print("[真实] 这才是泛化能力的真实证据:")
        print_score("微调前(真实)", before_ho)
        print_score("微调后(真实)", after_ho)
        dh = after_ho["mean_score"] - before_ho["mean_score"]
        print("  真实平均分变化: %+.4f  %s" % (dh, "(提升↑)" if dh > 0 else "(未提升/退化)"))
    print("\n权重已存: %s  (selection=%s)" % (selected_checkpoint, args.checkpoint_selection))
    print("提示: 合成数据上预训练模型本就近满分, 关键看'微调后不崩';")
    print("      真正的提升要在真实【留出集】上才看得出来(用 --holdout 指定)。")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback; traceback.print_exc()
        print("\n[出错]", repr(exc), file=sys.stderr); sys.exit(1)
