"""轮39A：波形 JEPA 自监督预训练对 T2 跨区域震级泛化的作用（仅公开 STEAD）。

合规（最高优先级）：
  只读 outputs/t2_cache_station27（公开 STEAD，P-5s..P+5s，100Hz，3分量）。
  不读 08-exam/08-an/第1轮/第2轮 及其任何衍生物；脚本启动时断言比赛路径不可见。

协议（沿用轮38，保证与已有基线可比）：
  训练区域 A -> 标定区域 B -> 评估区域 C，三者台网互斥；事件级(source_id)隔离。
  B 只用于选 lam，C 只做一次独立评估。评分为官方 T2 hinge：200*mean(max(0,1-|p-y|))。

三条对照臂（同一 A/B/C 划分、同一随机种子、同一评估代码）：
  ARM_CNN     : 轮38 的 CNN Member，从零训练（历史基线，用于锚定）
  ARM_SCRATCH : 本轮 Tokenizer+Transformer 架构，从零训练（消融，隔离"架构"因素）
  ARM_JEPA_A  : 同架构，先在 A 的训练事件上做 JEPA 自监督预训练，再监督微调
  ARM_JEPA_MR : 同架构，先在"除 B、C 以外所有区域 + 排除 A 的留出事件"上做 JEPA 预训练，再在 A 上微调

预训练泄漏控制：JEPA 预训练池严格排除 B、C 全部记录与 A 的留出事件，
因此 ARM_JEPA_* 在标定域和评估域上都是真正的零样本域外。
"""
import os, sys, json, time, math, hashlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr

ROOT = '/root/5.6+chanshui1'
CACHE = f'{ROOT}/outputs/t2_cache_station27'
OUT = f'{ROOT}/outputs/t2_round39a_jepa'
os.makedirs(OUT, exist_ok=True)

# ---------- 合规守卫（运行时硬阻断，非目录名启发式） ----------
sys.path.insert(0, ROOT)
import compliance_guard as _cg
_cg.allow(CACHE, OUT)
_cg.install()
assert os.path.isdir(CACHE), CACHE
print('COMPLIANCE_GUARD_ON 数据白名单:', CACHE, flush=True)
print(_cg.selftest(f'{ROOT}/t2data/T2.A.Q0001.mseed'), flush=True)
print(_cg.selftest(f'{ROOT}/outputs/r1_t2_meta.csv'), flush=True)

def sha256(p, cap=None):
    h = hashlib.sha256()
    n = 0
    with open(p, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
            if cap and n >= cap:
                break
    return h.hexdigest()

X = np.load(f'{CACHE}/X.npy', mmap_mode='r')
Y = np.load(f'{CACHE}/y.npy')
NET = np.load(f'{CACHE}/net.npy').astype(str)
STA = np.load(f'{CACHE}/sta.npy').astype(str)
SRC = np.load(f'{CACHE}/src.npy').astype(str)

REGIONS = {'ALASKA': ['AK', 'AT', 'AV'], 'CALIF': ['CI', 'BK', 'AZ', 'NN'],
           'GREECE': ['HL', 'HP', 'HT', 'HA'], 'CHILE': ['C', 'C1', 'PB'],
           'NZ': ['NZ'], 'OTHER': ['TA', 'OK', 'GS', 'PR', 'SV', 'IU', 'MN', 'CN', 'KR', 'GO']}
NET2REG = {n: r for r, ns in REGIONS.items() for n in ns}

band = np.where((Y >= 3.8) & (Y <= 6.5))[0]
REG_all = np.array([NET2REG.get(n, 'DROP') for n in NET[band]])
keep = REG_all != 'DROP'
idx = band[keep]
REG = REG_all[keep]
Ym = Y[idx]
SRCm = SRC[idx]
print('records', len(idx), {r: int((REG == r).sum()) for r in REGIONS}, flush=True)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', dev, torch.__version__, flush=True)

# ---------- 共享前端与特征（与轮38 完全一致，隔离"特征"因素） ----------
class RmsNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt((x * x).mean(2, keepdim=True) + 1e-12)

class LogAmp(nn.Module):
    def forward(self, x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

BANDS = [(.05, .2), (.2, .5), (.5, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 45)]

def spec(x):
    xw = x - x.mean(2, keepdim=True)
    Fx = torch.fft.rfft(xw * torch.hann_window(xw.shape[2], device=x.device), dim=2)
    P = Fx.real ** 2 + Fx.imag ** 2
    fr = torch.fft.rfftfreq(xw.shape[2], d=.01).to(x.device)
    P = P / (P.sum(2, keepdim=True) + 1e-20)
    return torch.cat([torch.log10(P[:, :, (fr >= a) & (fr < b)].sum(2) + 1e-12) for a, b in BANDS], 1)

def snr(x):
    h = x.shape[2] // 2
    return 20 * torch.log10(torch.sqrt((x[:, :, h:] ** 2).mean((1, 2)) + 1e-12) /
                            torch.sqrt((x[:, :, :h] ** 2).mean((1, 2)) + 1e-12))

class Member(nn.Module):
    """轮38 CNN 基线成员。"""
    def __init__(s_):
        super().__init__()
        def bl(a, b):
            return nn.Sequential(nn.Conv1d(a, b, 7, 2, 3), nn.BatchNorm1d(b), nn.ReLU())
        s_.pre = nn.Sequential(RmsNorm(), LogAmp())
        s_.cnn = nn.Sequential(bl(3, 32), bl(32, 64), bl(64, 128), bl(128, 128), bl(128, 256),
                               nn.AdaptiveAvgPool1d(1), nn.Flatten())
        s_.head = nn.Sequential(nn.Linear(256 + 25, 128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, 1))
    def forward(s_, x, sn):
        z = torch.cat([s_.cnn(s_.pre(x)), sn[:, None] / 40, spec(x)], 1)
        return s_.head(z)[:, 0]

# ---------- JEPA 架构 ----------
D_MODEL = 128
N_TOK = 125          # 1000 / 8
PATCH = 8

class Tokenizer(nn.Module):
    """3x1000 -> 125 x D_MODEL，stride 8。"""
    def __init__(s_):
        super().__init__()
        s_.pre = nn.Sequential(RmsNorm(), LogAmp())
        s_.conv = nn.Sequential(
            nn.Conv1d(3, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, 5, 2, 2), nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, D_MODEL, 5, 2, 2), nn.BatchNorm1d(D_MODEL), nn.GELU())
    def forward(s_, x):
        return s_.conv(s_.pre(x)).transpose(1, 2)   # B,T,D

class Encoder(nn.Module):
    def __init__(s_, depth=4, heads=4):
        super().__init__()
        s_.tok = Tokenizer()
        s_.pos = nn.Parameter(torch.zeros(1, N_TOK, D_MODEL))
        nn.init.trunc_normal_(s_.pos, std=.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, heads, D_MODEL * 4, dropout=.1,
                                           batch_first=True, norm_first=True, activation='gelu')
        s_.tr = nn.TransformerEncoder(layer, depth)
        s_.norm = nn.LayerNorm(D_MODEL)
    def forward(s_, x, keep=None):
        t = s_.tok(x) + s_.pos
        if keep is not None:
            B = t.shape[0]
            t = torch.gather(t, 1, keep[:, :, None].expand(B, keep.shape[1], D_MODEL))
        return s_.norm(s_.tr(t))

class Predictor(nn.Module):
    """由可见 token 预测被遮蔽位置的目标潜表示。"""
    def __init__(s_, depth=2, heads=4):
        super().__init__()
        s_.mask_tok = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        nn.init.trunc_normal_(s_.mask_tok, std=.02)
        s_.pos = nn.Parameter(torch.zeros(1, N_TOK, D_MODEL))
        nn.init.trunc_normal_(s_.pos, std=.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, heads, D_MODEL * 4, dropout=.1,
                                           batch_first=True, norm_first=True, activation='gelu')
        s_.tr = nn.TransformerEncoder(layer, depth)
        s_.out = nn.Linear(D_MODEL, D_MODEL)
    def forward(s_, ctx, ctx_ix, tgt_ix):
        B, Nc, _ = ctx.shape
        Nt = tgt_ix.shape[1]
        ctx = ctx + torch.gather(s_.pos.expand(B, N_TOK, D_MODEL), 1,
                                 ctx_ix[:, :, None].expand(B, Nc, D_MODEL))
        q = s_.mask_tok.expand(B, Nt, D_MODEL) + torch.gather(
            s_.pos.expand(B, N_TOK, D_MODEL), 1, tgt_ix[:, :, None].expand(B, Nt, D_MODEL))
        h = s_.tr(torch.cat([ctx, q], 1))
        return s_.out(h[:, Nc:])

class MagHead(nn.Module):
    def __init__(s_):
        super().__init__()
        s_.head = nn.Sequential(nn.Linear(D_MODEL * 2 + 25, 128), nn.GELU(),
                                nn.Dropout(.1), nn.Linear(128, 1))
    def forward(s_, tokens, x, sn):
        z = torch.cat([tokens.mean(1), tokens.max(1).values, sn[:, None] / 40, spec(x)], 1)
        return s_.head(z)[:, 0]

def load_batch(rows):
    order = np.argsort(np.argsort(rows))
    xb = torch.from_numpy(np.asarray(X[np.sort(rows)], np.float32))
    return xb[order]

def sample_masks(B, rng, n_blocks=4, blk=10, ctx_keep=0.55):
    """多块目标遮蔽 + 互补上下文；返回等长 index 张量。"""
    n_tgt = n_blocks * blk
    n_ctx = int(N_TOK * ctx_keep)
    tgt = np.zeros((B, n_tgt), np.int64)
    ctx = np.zeros((B, n_ctx), np.int64)
    for b in range(B):
        chosen = set()
        while len(chosen) < n_tgt:
            s = rng.randint(0, N_TOK - blk)
            chosen.update(range(s, s + blk))
        t = np.array(sorted(chosen))[:n_tgt]
        rest = np.setdiff1d(np.arange(N_TOK), t)
        if len(rest) >= n_ctx:
            c = rng.choice(rest, n_ctx, replace=False)
        else:
            c = np.concatenate([rest, rng.choice(rest, n_ctx - len(rest), replace=True)])
        tgt[b] = t
        ctx[b] = np.sort(c)
    return torch.from_numpy(ctx), torch.from_numpy(tgt)

def jepa_pretrain(pool_rows, seed, epochs, bs=192, tag=''):
    """自监督：EMA 目标编码器 + 潜空间 smooth-L1，无标签参与。"""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    enc = Encoder().to(dev)
    tgt_enc = Encoder().to(dev)
    tgt_enc.load_state_dict(enc.state_dict())
    for p in tgt_enc.parameters():
        p.requires_grad_(False)
    pred = Predictor().to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), 1.5e-3, weight_decay=.05)
    steps = max(1, (len(pool_rows) // bs)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1.5e-3, total_steps=steps, pct_start=.15)
    hist = []
    done = 0
    for ep in range(epochs):
        perm = rng.permutation(len(pool_rows))
        tot, nb = 0.0, 0
        for st in range(0, len(perm) - bs + 1, bs):
            rows = pool_rows[perm[st:st + bs]]
            xb = load_batch(rows).to(dev)
            g = torch.from_numpy((10 ** rng.uniform(-1.5, 1.5, (len(rows), 1, 1))).astype(np.float32)).to(dev)
            xb = xb * g
            ctx_ix, tgt_ix = sample_masks(len(rows), rng)
            ctx_ix, tgt_ix = ctx_ix.to(dev), tgt_ix.to(dev)
            with torch.no_grad():
                full = tgt_enc(xb)
                z_tgt = torch.gather(full, 1, tgt_ix[:, :, None].expand(len(rows), tgt_ix.shape[1], D_MODEL))
                z_tgt = F.layer_norm(z_tgt, (D_MODEL,))
            z_ctx = enc(xb, ctx_ix)
            z_hat = pred(z_ctx, ctx_ix, tgt_ix)
            loss = F.smooth_l1_loss(z_hat, z_tgt, beta=1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), 1.0)
            opt.step()
            if done + 1 < steps:
                sched.step()
            done += 1
            m = 0.996 + 0.003 * min(1.0, done / max(1, steps))
            with torch.no_grad():
                for a, b_ in zip(tgt_enc.parameters(), enc.parameters()):
                    a.mul_(m).add_(b_.detach(), alpha=1 - m)
                for a, b_ in zip(tgt_enc.buffers(), enc.buffers()):
                    a.copy_(b_)
            tot += float(loss)
            nb += 1
        hist.append(tot / max(nb, 1))
        print('    [jepa%s] ep%d loss=%.5f' % (tag, ep, hist[-1]), flush=True)
    return enc, hist

def finetune(enc, tr_rows, seed, epochs=8, bs=128, lr_enc=2e-4, lr_head=1e-3):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed + 1)
    head = MagHead().to(dev)
    prior = float(np.median(Ym[tr_rows]))
    opt = torch.optim.AdamW([{'params': enc.parameters(), 'lr': lr_enc},
                             {'params': head.parameters(), 'lr': lr_head}], weight_decay=.01)
    ytr = torch.from_numpy(Ym[tr_rows].astype(np.float32))
    for ep in range(epochs):
        perm = rng.permutation(len(tr_rows))
        tot, nb = 0.0, 0
        for st in range(0, len(perm), bs):
            sl = perm[st:st + bs]
            xb = load_batch(idx[tr_rows[sl]]).to(dev)
            g = torch.from_numpy((10 ** rng.uniform(-1.5, 1.5, (len(sl), 1, 1))).astype(np.float32)).to(dev)
            xb = xb * g
            p = head(enc(xb), xb, snr(xb)) + prior
            loss = F.l1_loss(p, ytr[sl].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        print('    ep%d l1=%.4f' % (ep, tot / max(nb, 1)), flush=True)
    return (enc, head, prior)

@torch.no_grad()
def predict_jepa(model, rows, bs=384):
    enc, head, prior = model
    enc.eval(); head.eval()
    out = []
    for st in range(0, len(rows), bs):
        xb = load_batch(idx[rows[st:st + bs]]).to(dev)
        out.append((head(enc(xb), xb, snr(xb)) + prior).cpu().numpy())
    enc.train(); head.train()
    return np.concatenate(out)

def train_cnn(tr_rows, seed, epochs=8, bs=128):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    m = Member().to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    ytr = torch.from_numpy(Ym[tr_rows].astype(np.float32))
    prior = float(np.median(Ym[tr_rows]))
    for ep in range(epochs):
        perm = rng.permutation(len(tr_rows))
        tot, nb = 0.0, 0
        for st in range(0, len(perm), bs):
            sl = perm[st:st + bs]
            xb = load_batch(idx[tr_rows[sl]]).to(dev)
            g = torch.from_numpy((10 ** rng.uniform(-1.5, 1.5, (len(sl), 1, 1))).astype(np.float32)).to(dev)
            xb = xb * g
            p = m(xb, snr(xb)) + prior
            loss = F.l1_loss(p, ytr[sl].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        print('    ep%d l1=%.4f' % (ep, tot / max(nb, 1)), flush=True)
    return (m, prior)

@torch.no_grad()
def predict_cnn(model, rows, bs=512):
    m, prior = model
    m.eval()
    out = []
    for st in range(0, len(rows), bs):
        xb = load_batch(idx[rows[st:st + bs]]).to(dev)
        out.append((m(xb, snr(xb)) + prior).cpu().numpy())
    m.train()
    return np.concatenate(out)

# ---------- 评估（与轮38 逐字一致） ----------
LAMS = np.round(np.arange(0.0, 1.61, 0.05), 3)

def event_split(mask, frac, seed):
    ev = np.unique(SRCm[mask])
    rs = np.random.RandomState(seed)
    rs.shuffle(ev)
    a = set(ev[:int(len(ev) * frac)])
    sel = np.where(mask)[0]
    inA = np.array([SRCm[i] in a for i in sel])
    return sel[inA], sel[~inA]

def batch_eval(pred, y, rng, nb=200, lo=50, hi=200):
    outs = []
    for _ in range(nb):
        a = rng.uniform(3.8, 5.4)
        b = a + rng.uniform(.6, 2.2)
        c = np.where((y >= a) & (y <= b))[0]
        if len(c) < lo:
            continue
        ii = rng.choice(c, min(len(c), rng.randint(lo, hi + 1)), replace=False)
        z = pred[ii] - np.median(pred[ii])
        outs.append((z, y[ii], float(np.median(y[ii]))))
    return outs

def lam_curve(bs):
    tot = np.zeros(len(LAMS)); n = 0
    for z, y, c in bs:
        for k, l in enumerate(LAMS):
            tot[k] += np.maximum(0, 1 - np.abs(np.clip(c + l * z, 0, 9.9) - y)).sum()
        n += len(y)
    return 200.0 * tot / max(n, 1)

SETUPS = [('ALASKA', 'CALIF', 'GREECE'), ('CALIF', 'GREECE', 'ALASKA'), ('GREECE', 'CHILE', 'CALIF'),
          ('CHILE', 'ALASKA', 'NZ'), ('OTHER', 'NZ', 'CHILE')]
PRE_EPOCHS = int(os.environ.get('PRE_EPOCHS', '10'))
PRE_CAP = int(os.environ.get('PRE_CAP', '30000'))
ARMS = os.environ.get('ARMS', 'cnn,scratch,jepa_a,jepa_mr').split(',')

def evaluate(pred_in, y_in, pred_b, y_b, pred_c, y_c):
    rng = np.random.RandomState(7)
    cv_in = lam_curve(batch_eval(pred_in, y_in, rng))
    cv_b = lam_curve(batch_eval(pred_b, y_b, rng))
    cv_c = lam_curve(batch_eval(pred_c, y_c, rng))
    lam_in = float(LAMS[int(cv_in.argmax())])
    lam_b = float(LAMS[int(cv_b.argmax())])
    lam_c = float(LAMS[int(cv_c.argmax())])
    i040 = int(np.argmin(np.abs(LAMS - 0.40)))
    r_in = float(pearsonr(pred_in, y_in)[0])
    r_b = float(pearsonr(pred_b, y_b)[0])
    r_c = float(pearsonr(pred_c, y_c)[0])
    return dict(rho_in=r_in, rho_calib=r_b, rho_eval=r_c,
                decay_eval=(r_c / r_in if r_in > 0 else None),
                lam_in=lam_in, lam_calib=lam_b, lam_eval_oracle=lam_c,
                score_eval_lam040=float(cv_c[i040]),
                score_eval_lam_calib=float(cv_c[int(np.argmin(np.abs(LAMS - lam_b)))]),
                score_eval_oracle=float(cv_c.max()))

rep = {a: [] for a in ARMS}
pre_hist = {}
for (A, B, Cr) in SETUPS:
    key = f'{A}->{B}->{Cr}'
    mA = REG == A; mB = REG == B; mC = REG == Cr
    tr, ho = event_split(mA, 0.85, seed=hash(A) % 9999)
    print('=== %s train=%d(%d) calib=%d eval=%d' % (key, len(tr), mA.sum(), mB.sum(), mC.sum()), flush=True)
    yi, yb, yc = Ym[ho], Ym[mB], Ym[mC]
    rows_b, rows_c = np.where(mB)[0], np.where(mC)[0]
    seed = 1234 + len(tr) % 97

    # 预训练池（严格排除 B、C 全部记录与 A 的留出事件）
    ho_set = set(ho.tolist())
    pool_a = np.array([i for i in tr])
    pool_mr = np.array([i for i in range(len(idx))
                        if (not mB[i]) and (not mC[i]) and (i not in ho_set)])
    rs = np.random.RandomState(seed)
    if len(pool_mr) > PRE_CAP:
        pool_mr = np.sort(rs.choice(pool_mr, PRE_CAP, replace=False))
    assert not mB[pool_a].any() and not mC[pool_a].any()
    assert not mB[pool_mr].any() and not mC[pool_mr].any()
    assert len(set(pool_mr.tolist()) & ho_set) == 0
    print('  pretrain pool: A=%d MR=%d' % (len(pool_a), len(pool_mr)), flush=True)

    for arm in ARMS:
        t0 = time.time()
        print('  -- arm %s' % arm, flush=True)
        if arm == 'cnn':
            mdl = train_cnn(tr, seed)
            pi, pb, pc = predict_cnn(mdl, ho), predict_cnn(mdl, rows_b), predict_cnn(mdl, rows_c)
        else:
            if arm == 'scratch':
                enc = Encoder().to(dev)
            elif arm == 'jepa_a':
                enc, h = jepa_pretrain(idx[pool_a], seed, PRE_EPOCHS, tag='_A')
                pre_hist[key + '|A'] = h
            elif arm == 'jepa_mr':
                enc, h = jepa_pretrain(idx[pool_mr], seed, PRE_EPOCHS, tag='_MR')
                pre_hist[key + '|MR'] = h
            else:
                raise SystemExit('unknown arm ' + arm)
            mdl = finetune(enc, tr, seed)
            pi, pb, pc = predict_jepa(mdl, ho), predict_jepa(mdl, rows_b), predict_jepa(mdl, rows_c)
        d = evaluate(pi, yi, pb, yb, pc, yc)
        d.update(train=A, calib=B, eval=Cr, n_train=int(len(tr)), n_in=int(len(ho)),
                 n_b=int(mB.sum()), n_c=int(mC.sum()), secs=time.time() - t0)
        rep[arm].append(d)
        print('     rho in=%.3f eval=%.3f decay=%.3f | lam_calib=%.2f | score040=%.3f score_calib=%.3f oracle=%.3f (%.0fs)'
              % (d['rho_in'], d['rho_eval'], d['decay_eval'] or float('nan'), d['lam_calib'],
                 d['score_eval_lam040'], d['score_eval_lam_calib'], d['score_eval_oracle'], d['secs']), flush=True)

def agg_of(rs):
    return dict(mean_rho_in=float(np.mean([r['rho_in'] for r in rs])),
                mean_rho_eval=float(np.mean([r['rho_eval'] for r in rs])),
                mean_decay_eval=float(np.mean([r['decay_eval'] for r in rs if r['decay_eval']])),
                mean_lam_calib=float(np.mean([r['lam_calib'] for r in rs])),
                mean_score_040=float(np.mean([r['score_eval_lam040'] for r in rs])),
                mean_score_lam_calib=float(np.mean([r['score_eval_lam_calib'] for r in rs])),
                worst_score_040=float(np.min([r['score_eval_lam040'] for r in rs])),
                worst_score_lam_calib=float(np.min([r['score_eval_lam_calib'] for r in rs])))

agg = {a: agg_of(rep[a]) for a in ARMS if rep[a]}
dec = {}
if 'scratch' in agg:
    for a in ['jepa_a', 'jepa_mr']:
        if a in agg:
            dec[f'{a}_beats_scratch_mean_and_worst'] = bool(
                agg[a]['mean_score_lam_calib'] > agg['scratch']['mean_score_lam_calib'] and
                agg[a]['worst_score_lam_calib'] >= agg['scratch']['worst_score_lam_calib'])
if 'cnn' in agg:
    for a in ['jepa_a', 'jepa_mr', 'scratch']:
        if a in agg:
            dec[f'{a}_beats_cnn_mean_and_worst'] = bool(
                agg[a]['mean_score_lam_calib'] > agg['cnn']['mean_score_lam_calib'] and
                agg[a]['worst_score_lam_calib'] >= agg['cnn']['worst_score_lam_calib'])

res = dict(protocol='public STEAD only; A->B->C mutually exclusive networks; event-level isolation; '
                    'JEPA pretrain pool excludes all B/C records and A holdout events; official T2 hinge score; '
                    'competition packages (08/R1/R2) never read',
           cache=CACHE, cache_meta=json.load(open(f'{CACHE}/meta.json')),
           cache_y_sha256=sha256(f'{CACHE}/y.npy'),
           cache_X_sha256_first256MB=sha256(f'{CACHE}/X.npy', cap=256 << 20),
           torch=torch.__version__, hip=torch.version.hip, device=dev,
           pre_epochs=PRE_EPOCHS, pre_cap=PRE_CAP, arms=ARMS,
           regions=REGIONS, setups=[list(s) for s in SETUPS],
           per_setup=rep, pretrain_loss=pre_hist, aggregate=agg, decision=dec)
json.dump(res, open(f'{OUT}/jepa_cross_region.json', 'w'), indent=2)
print('AGG', json.dumps(agg, indent=1), flush=True)
print('DECISION', json.dumps(dec, indent=1), flush=True)