"""轮39C：JEPA 关键设计消融（仅公开 STEAD）。

文献动机（轮39A 精读结论）：
  I-JEPA 指出目标块尺度与上下文分散度是潜表示预测成败的关键；
  MAE 指出高冗余信号需要较高遮蔽率；
  PatchTST 指出 patch 尺度决定局部语义与序列长度的权衡。
这三点在地震波形上未被验证，本轮做预注册网格。

协议与轮38/39A 完全一致：A->B->C 台网互斥、事件级隔离、B 选 lam、C 一次评估。
所有臂共享同一监督微调配方；唯一变化的是自监督阶段的遮蔽率/块长/patch。

合规：只读公开 STEAD 缓存；运行时白名单守卫；比赛包与 R1/R2 衍生物硬阻断。
预训练池 = 除 B、C 全部记录与 A 留出事件之外的公开记录（与轮39A 的 JEPA-MR 相同）。
"""
import os, sys, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr

ROOT = '/root/5.6+chanshui1'
OUT = f'{ROOT}/outputs/t2_round39c_ablation'
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)
import public_data as pdata

pdata.enable_guard()
_cg = sys.modules['compliance_guard']
print(_cg.selftest(f'{ROOT}/t2data/T2.A.Q0001.mseed'), flush=True)
print(_cg.selftest(f'{ROOT}/outputs/r1_t2_meta.csv'), flush=True)
_cg.allow(OUT)

ds = pdata.load_stead()
prov = ds.provenance()
print('DATA', prov, flush=True)

band = np.where((ds.y >= 3.8) & (ds.y <= 6.5) & (ds.region != 'DROP'))[0]
X = ds.X
Ym = ds.y[band]
SRCm = ds.event_id[band]
REG = ds.region[band]
idx = band
print('records', len(idx), {r: int((REG == r).sum()) for r in np.unique(REG)}, flush=True)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
D_MODEL = 128

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

class Tokenizer(nn.Module):
    """patch 长度可配置：8 -> 125 token，16 -> 63 token，32 -> 32 token。"""
    def __init__(s_, patch):
        super().__init__()
        s_.pre = nn.Sequential(RmsNorm(), LogAmp())
        chans = [(3, 64), (64, 96), (96, D_MODEL)]
        strides = {8: [2, 2, 2], 16: [2, 4, 2], 32: [4, 4, 2]}[patch]
        layers = []
        for (a, b), st in zip(chans, strides):
            k = max(3, st * 2 + 1)
            layers += [nn.Conv1d(a, b, k, st, k // 2), nn.BatchNorm1d(b), nn.GELU()]
        s_.conv = nn.Sequential(*layers)
    def forward(s_, x):
        return s_.conv(s_.pre(x)).transpose(1, 2)

class Encoder(nn.Module):
    def __init__(s_, patch, n_tok, depth=4, heads=4):
        super().__init__()
        s_.n_tok = n_tok
        s_.tok = Tokenizer(patch)
        s_.pos = nn.Parameter(torch.zeros(1, n_tok, D_MODEL))
        nn.init.trunc_normal_(s_.pos, std=.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, heads, D_MODEL * 4, dropout=.1,
                                           batch_first=True, norm_first=True, activation='gelu')
        s_.tr = nn.TransformerEncoder(layer, depth)
        s_.norm = nn.LayerNorm(D_MODEL)
    def forward(s_, x, keep=None):
        t = s_.tok(x)
        t = t[:, :s_.n_tok] + s_.pos[:, :t.shape[1]][:, :s_.n_tok]
        if keep is not None:
            t = torch.gather(t, 1, keep[:, :, None].expand(t.shape[0], keep.shape[1], D_MODEL))
        return s_.norm(s_.tr(t))

class Predictor(nn.Module):
    def __init__(s_, n_tok, depth=2, heads=4):
        super().__init__()
        s_.mask_tok = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        nn.init.trunc_normal_(s_.mask_tok, std=.02)
        s_.pos = nn.Parameter(torch.zeros(1, n_tok, D_MODEL))
        nn.init.trunc_normal_(s_.pos, std=.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, heads, D_MODEL * 4, dropout=.1,
                                           batch_first=True, norm_first=True, activation='gelu')
        s_.tr = nn.TransformerEncoder(layer, depth)
        s_.out = nn.Linear(D_MODEL, D_MODEL)
    def forward(s_, ctx, ctx_ix, tgt_ix):
        B, Nc, _ = ctx.shape
        Nt = tgt_ix.shape[1]
        P = s_.pos.expand(B, s_.pos.shape[1], D_MODEL)
        ctx = ctx + torch.gather(P, 1, ctx_ix[:, :, None].expand(B, Nc, D_MODEL))
        q = s_.mask_tok.expand(B, Nt, D_MODEL) + torch.gather(P, 1, tgt_ix[:, :, None].expand(B, Nt, D_MODEL))
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

def sample_masks(B, rng, n_tok, mask_ratio, blk, ctx_keep):
    n_tgt = max(blk, int(round(n_tok * mask_ratio / blk) * blk))
    n_tgt = min(n_tgt, n_tok - blk)
    n_ctx = max(blk, int(n_tok * ctx_keep))
    tgt = np.zeros((B, n_tgt), np.int64)
    ctx = np.zeros((B, n_ctx), np.int64)
    for b in range(B):
        chosen = set()
        guard = 0
        while len(chosen) < n_tgt and guard < 10000:
            s = rng.randint(0, max(1, n_tok - blk))
            chosen.update(range(s, min(s + blk, n_tok)))
            guard += 1
        t = np.array(sorted(chosen))[:n_tgt]
        if len(t) < n_tgt:
            t = np.resize(t, n_tgt)
        rest = np.setdiff1d(np.arange(n_tok), t)
        if len(rest) >= n_ctx:
            c = rng.choice(rest, n_ctx, replace=False)
        elif len(rest) > 0:
            c = np.concatenate([rest, rng.choice(rest, n_ctx - len(rest), replace=True)])
        else:
            c = rng.choice(np.arange(n_tok), n_ctx, replace=True)
        tgt[b] = t
        ctx[b] = np.sort(c)
    return torch.from_numpy(ctx), torch.from_numpy(tgt)

def jepa_pretrain(pool_rows, seed, epochs, patch, n_tok, mask_ratio, blk, ctx_keep, bs=192):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    enc = Encoder(patch, n_tok).to(dev)
    tgt_enc = Encoder(patch, n_tok).to(dev)
    tgt_enc.load_state_dict(enc.state_dict())
    for p in tgt_enc.parameters():
        p.requires_grad_(False)
    pred = Predictor(n_tok).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), 1.5e-3, weight_decay=.05)
    steps = max(1, len(pool_rows) // bs) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1.5e-3, total_steps=steps, pct_start=.15)
    done = 0
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(len(pool_rows))
        tot, nb = 0.0, 0
        for st in range(0, len(perm) - bs + 1, bs):
            rows = pool_rows[perm[st:st + bs]]
            xb = load_batch(rows).to(dev)
            g = torch.from_numpy((10 ** rng.uniform(-1.5, 1.5, (len(rows), 1, 1))).astype(np.float32)).to(dev)
            xb = xb * g
            ctx_ix, tgt_ix = sample_masks(len(rows), rng, n_tok, mask_ratio, blk, ctx_keep)
            ctx_ix, tgt_ix = ctx_ix.to(dev), tgt_ix.to(dev)
            with torch.no_grad():
                full = tgt_enc(xb)
                z_t = torch.gather(full, 1, tgt_ix[:, :, None].expand(len(rows), tgt_ix.shape[1], D_MODEL))
                z_t = F.layer_norm(z_t, (D_MODEL,))
            z_hat = pred(enc(xb, ctx_ix), ctx_ix, tgt_ix)
            loss = F.smooth_l1_loss(z_hat, z_t)
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
        print('      pre ep%d loss=%.5f' % (ep, hist[-1]), flush=True)
    return enc, hist

def finetune(enc, tr_rows, seed, epochs=8, bs=128):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed + 1)
    head = MagHead().to(dev)
    prior = float(np.median(Ym[tr_rows]))
    opt = torch.optim.AdamW([{'params': enc.parameters(), 'lr': 2e-4},
                             {'params': head.parameters(), 'lr': 1e-3}], weight_decay=.01)
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
    return enc, head, prior

@torch.no_grad()
def predict(model, rows, bs=384):
    enc, head, prior = model
    enc.eval(); head.eval()
    out = []
    for st in range(0, len(rows), bs):
        xb = load_batch(idx[rows[st:st + bs]]).to(dev)
        out.append((head(enc(xb), xb, snr(xb)) + prior).cpu().numpy())
    enc.train(); head.train()
    return np.concatenate(out)

LAMS = np.round(np.arange(0.0, 1.61, 0.05), 3)

def batch_eval(pred_, y, rng, nb=200, lo=50, hi=200):
    outs = []
    for _ in range(nb):
        a = rng.uniform(3.8, 5.4)
        b = a + rng.uniform(.6, 2.2)
        c = np.where((y >= a) & (y <= b))[0]
        if len(c) < lo:
            continue
        ii = rng.choice(c, min(len(c), rng.randint(lo, hi + 1)), replace=False)
        outs.append((pred_[ii] - np.median(pred_[ii]), y[ii], float(np.median(y[ii]))))
    return outs

def lam_curve(bs):
    tot = np.zeros(len(LAMS)); n = 0
    for z, y, c in bs:
        for k, l in enumerate(LAMS):
            tot[k] += np.maximum(0, 1 - np.abs(np.clip(c + l * z, 0, 9.9) - y)).sum()
        n += len(y)
    return 200.0 * tot / max(n, 1)

def evaluate(pi, yi, pb, yb, pc, yc):
    rng = np.random.RandomState(7)
    cv_in, cv_b, cv_c = (lam_curve(batch_eval(p, y, rng)) for p, y in ((pi, yi), (pb, yb), (pc, yc)))
    lam_b = float(LAMS[int(cv_b.argmax())])
    i040 = int(np.argmin(np.abs(LAMS - 0.40)))
    return dict(rho_in=float(pearsonr(pi, yi)[0]), rho_eval=float(pearsonr(pc, yc)[0]),
                lam_calib=lam_b, score_eval_lam040=float(cv_c[i040]),
                score_eval_lam_calib=float(cv_c[int(np.argmin(np.abs(LAMS - lam_b)))]),
                score_eval_oracle=float(cv_c.max()))

SETUPS = [('ALASKA', 'CALIF', 'GREECE'), ('CALIF', 'GREECE', 'ALASKA'), ('GREECE', 'CHILE', 'CALIF'),
          ('CHILE', 'ALASKA', 'NZ'), ('OTHER', 'NZ', 'CHILE')]
N_TOK = {8: 125, 16: 63, 32: 32}
# 预注册网格：patch x 遮蔽率 x 块长；上下文比例与遮蔽率互补。
GRID = [dict(tag='p8_m32_b10', patch=8, mask_ratio=.32, blk=10, ctx_keep=.60),
        dict(tag='p8_m48_b10', patch=8, mask_ratio=.48, blk=10, ctx_keep=.45),
        dict(tag='p8_m64_b10', patch=8, mask_ratio=.64, blk=10, ctx_keep=.30),
        dict(tag='p8_m48_b25', patch=8, mask_ratio=.48, blk=25, ctx_keep=.45),
        dict(tag='p16_m48_b6', patch=16, mask_ratio=.48, blk=6, ctx_keep=.45),
        dict(tag='p32_m48_b4', patch=32, mask_ratio=.48, blk=4, ctx_keep=.45)]
PRE_EPOCHS = int(os.environ.get('PRE_EPOCHS', '10'))
PRE_CAP = int(os.environ.get('PRE_CAP', '30000'))

rep = {g['tag']: [] for g in GRID}
losses = {}
for (A, B, Cr) in SETUPS:
    key = f'{A}->{B}->{Cr}'
    mA, mB, mC = REG == A, REG == B, REG == Cr
    tr, ho = pdata.event_split(np.where(mA)[0], SRCm, 0.85, seed=hash(A) % 9999)
    seed = 1234 + len(tr) % 97
    ho_set = set(ho.tolist())
    pool = np.array([i for i in range(len(idx)) if (not mB[i]) and (not mC[i]) and (i not in ho_set)])
    rs = np.random.RandomState(seed)
    if len(pool) > PRE_CAP:
        pool = np.sort(rs.choice(pool, PRE_CAP, replace=False))
    assert not mB[pool].any() and not mC[pool].any() and not (set(pool.tolist()) & ho_set)
    rows_b, rows_c = np.where(mB)[0], np.where(mC)[0]
    yi, yb, yc = Ym[ho], Ym[mB], Ym[mC]
    print('=== %s train=%d pool=%d calib=%d eval=%d' % (key, len(tr), len(pool), mB.sum(), mC.sum()), flush=True)
    for g in GRID:
        t0 = time.time()
        n_tok = N_TOK[g['patch']]
        enc, hist = jepa_pretrain(idx[pool], seed, PRE_EPOCHS, g['patch'], n_tok,
                                  g['mask_ratio'], g['blk'], g['ctx_keep'])
        losses[key + '|' + g['tag']] = hist
        mdl = finetune(enc, tr, seed)
        d = evaluate(predict(mdl, ho), yi, predict(mdl, rows_b), yb, predict(mdl, rows_c), yc)
        d.update(setup=key, secs=time.time() - t0, **g)
        rep[g['tag']].append(d)
        print('   %-12s rho_eval=%.3f lam=%.2f score040=%.3f score_calib=%.3f (%.0fs)' %
              (g['tag'], d['rho_eval'], d['lam_calib'], d['score_eval_lam040'],
               d['score_eval_lam_calib'], d['secs']), flush=True)

def agg(rs):
    return dict(mean_rho_eval=float(np.mean([r['rho_eval'] for r in rs])),
                mean_score_lam_calib=float(np.mean([r['score_eval_lam_calib'] for r in rs])),
                worst_score_lam_calib=float(np.min([r['score_eval_lam_calib'] for r in rs])),
                mean_score_040=float(np.mean([r['score_eval_lam040'] for r in rs])),
                mean_lam_calib=float(np.mean([r['lam_calib'] for r in rs])))
A = {t: agg(rep[t]) for t in rep if rep[t]}
base = 'p8_m32_b10'
best = max(A, key=lambda t: (A[t]['mean_score_lam_calib'], A[t]['worst_score_lam_calib']))
res = dict(protocol='public STEAD only; A->B->C exclusive networks; event isolation; pretrain pool excludes B/C and A holdout; '
                    'official T2 hinge; competition packages never read',
           data_provenance=prov, torch=torch.__version__, hip=torch.version.hip, device=dev,
           pre_epochs=PRE_EPOCHS, pre_cap=PRE_CAP, grid=GRID, setups=[list(s) for s in SETUPS],
           per_setup=rep, pretrain_loss=losses, aggregate=A, best_tag=best,
           reference_39a_jepa_mr=dict(mean_score_lam_calib=158.52251823213723,
                                      worst_score_lam_calib=155.83363665031968,
                                      mean_rho_eval=0.3762227773666382))
json.dump(res, open(f'{OUT}/ablation.json', 'w'), indent=2)
print('AGG', json.dumps(A, indent=1), flush=True)
print('BEST', best, flush=True)