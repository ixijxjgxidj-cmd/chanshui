"""T2 J1：P±5 秒 JEPA 公开预训练 + 监督震级微调（token 长度动态推断修正版）。

目的：提升目标域中唯一已验证可迁移的量——事件大小排序。

严格证伪设计：
- 自监督预训练只用 STEAD P±5s 公开缓存（outputs/t2_cache_p10），不使用 R1/R2、不使用任何标签。
- JEPA：context encoder 见遮挡后序列，predictor 预测 EMA target encoder 的被遮挡 token latent；
  loss = 归一化 latent MSE，避免重构绝对振幅（与台站增益解耦）。
- 增广：±2 log10 全局增益 + 时移 + 通道扰动；encoder 前置逐道 RMS 归一化。
- 微调：STEAD 事件级 GroupShuffleSplit，输出 (mu, logsigma)。
- 准入门槛：公开 M4-6.1 子区间需超过本轮 G 系列最优；否则终止该路线。
- R1/R2 只在预注册 train 上做一次外部参照，不用于早停/选参；两 holdout 不读取。
"""
import json, os, sys, time, copy
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from t2_features import score200

CACHE = "outputs/t2_cache_p10"; OUT = "outputs/t2_jepa"; os.makedirs(OUT, exist_ok=True)
PRE_EPOCHS = int(os.environ.get("PRE_EPOCHS", "30"))
FT_EPOCHS = int(os.environ.get("FT_EPOCHS", "24"))
BATCH = 256; MAGMIN = 3.8
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(20260816); rng = np.random.RandomState(20260816)

X = np.load(f"{CACHE}/X.npy", mmap_mode="r"); y = np.load(f"{CACHE}/y.npy"); g = np.load(f"{CACHE}/g.npy").astype(str)
keep = np.where(y >= MAGMIN)[0]
tr0, va0 = next(GroupShuffleSplit(1, test_size=.15, random_state=42).split(np.zeros(len(keep)), y[keep], groups=g[keep]))
TR, VA = keep[tr0], keep[va0]
assert not (set(g[TR]) & set(g[VA])), "事件级分组泄漏"
ytr, yva = y[TR], y[VA]; mk61 = (yva >= 4.0) & (yva <= 6.1)
CONST61 = score200(np.full(int(mk61.sum()), np.median(yva[mk61])), yva[mk61])
print("dev=%s train=%d val=%d M4-6.1=%d const=%.2f" % (dev, len(TR), len(VA), int(mk61.sum()), CONST61), flush=True)


class RmsNorm(nn.Module):
    def forward(self, x): return x / torch.sqrt((x * x).mean(dim=2, keepdim=True) + 1e-12)


class BlockEncoder(nn.Module):
    """3×1000 波形 -> (B, L, D) token 序列；L 由卷积步长决定，运行期推断。"""
    def __init__(self, d=128):
        super().__init__(); self.pre = RmsNorm()
        def b(a, c): return nn.Sequential(nn.Conv1d(a, c, 7, stride=2, padding=3), nn.BatchNorm1d(c), nn.GELU())
        self.net = nn.Sequential(b(3, 32), b(32, 64), b(64, 96), b(96, d))
    def forward(self, x): return self.net(self.pre(x)).transpose(1, 2)


class Predictor(nn.Module):
    def __init__(self, d=128, L=64):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=4 * d,
                                           batch_first=True, dropout=0.1, activation="gelu")
        self.tx = nn.TransformerEncoder(layer, 2)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
    def forward(self, z, mask):
        keep = (~mask).unsqueeze(-1)
        h = z * keep + self.mask_token * (~keep) + self.pos[:, :z.shape[1]]
        return self.out(self.tx(h))


def aug(x):
    B = x.shape[0]
    x = x * (10.0 ** torch.empty(B, 1, 1, device=x.device).uniform_(-2.0, 2.0))
    sh = torch.randint(-25, 26, (B,), device=x.device)
    x = torch.stack([torch.roll(x[i], int(s), dims=1) for i, s in enumerate(sh.tolist())])
    return x * (1.0 + torch.empty(B, 3, 1, device=x.device).uniform_(-0.1, 0.1))


def mask_blocks(B, L, ratio=0.55, span=6):
    m = torch.zeros(B, L, dtype=torch.bool, device=dev)
    need = int(L * ratio)
    for i in range(B):
        guard = 0
        while int(m[i].sum()) < need and guard < 64:
            st = int(torch.randint(0, max(1, L - span), (1,)))
            m[i, st:st + span] = True; guard += 1
    return m


with torch.no_grad():
    L_TOK = BlockEncoder().to(dev)(torch.zeros(2, 3, X.shape[2], device=dev)).shape[1]
print("token length L=%d" % L_TOK, flush=True)

enc = BlockEncoder().to(dev); tgt = copy.deepcopy(enc).to(dev)
for p in tgt.parameters(): p.requires_grad = False
pred = Predictor(L=L_TOK).to(dev)
params = list(enc.parameters()) + list(pred.parameters())
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.04)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=PRE_EPOCHS * max(1, len(TR) // BATCH))
pre_hist = []
for ep in range(1, PRE_EPOCHS + 1):
    enc.train(); pred.train(); tot = 0.0; nb = 0; t0 = time.time()
    perm = rng.permutation(TR)
    for i in range(0, len(perm), BATCH):
        sel = np.sort(perm[i:i + BATCH])
        xb = torch.from_numpy(np.asarray(X[sel], np.float32)).to(dev)
        z = enc(aug(xb))
        m = mask_blocks(len(xb), z.shape[1])
        p = pred(z, m)
        with torch.no_grad(): t = tgt(aug(xb))
        loss = F.mse_loss(F.normalize(p[m], dim=-1), F.normalize(t[m], dim=-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        try: sch.step()
        except Exception: pass
        with torch.no_grad():
            for a, b in zip(tgt.parameters(), enc.parameters()):
                a.data.mul_(0.996).add_(b.data, alpha=0.004)
        tot += float(loss.detach()); nb += 1
    pre_hist.append(dict(epoch=ep, loss=tot / max(nb, 1), sec=time.time() - t0))
    print("pre ep%02d loss=%.5f %.0fs" % (ep, pre_hist[-1]["loss"], pre_hist[-1]["sec"]), flush=True)
torch.save(dict(encoder=enc.state_dict(), target=tgt.state_dict(), predictor=pred.state_dict(),
                L=L_TOK, pre_hist=pre_hist), f"{OUT}/jepa_pretrain.pt")


class Reg(nn.Module):
    def __init__(self, encoder):
        super().__init__(); self.enc = encoder
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(128, 2))
    def forward(self, x): return self.head(self.enc(x).mean(1))


reg = Reg(enc).to(dev)
opt = torch.optim.AdamW(reg.parameters(), lr=6e-4, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.2e-3, total_steps=FT_EPOCHS * max(1, len(TR) // BATCH))

def infer(idx):
    reg.eval(); P=[]; S=[]
    with torch.no_grad():
        for i in range(0, len(idx), 512):
            sel = np.sort(idx[i:i + 512])
            o = reg(torch.from_numpy(np.asarray(X[sel], np.float32)).to(dev))
            P.append(o[:, 0].cpu().numpy()); S.append(o[:, 1].cpu().numpy())
    return np.concatenate(P), np.concatenate(S)

best=None; hist=[]
for ep in range(1, FT_EPOCHS + 1):
    reg.train(); perm = rng.permutation(TR); tot=0.0; nb=0; t0=time.time()
    for i in range(0, len(perm), BATCH):
        sel = np.sort(perm[i:i + BATCH])
        xb = torch.from_numpy(np.asarray(X[sel], np.float32)).to(dev)
        tt = torch.from_numpy(y[sel]).to(dev)
        o = reg(aug(xb)); mu, ls = o[:, 0], o[:, 1].clamp(-3, 3)
        loss = (torch.abs(mu - tt) * torch.exp(-ls) + ls).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(reg.parameters(), 5.0); opt.step()
        try: sch.step()
        except Exception: pass
        tot += float(loss.detach()); nb += 1
    P, S = infer(VA); pc = np.clip(P, 0, 9.9)
    d = dict(epoch=ep, loss=tot/max(nb,1), mae=float(np.abs(pc-yva).mean()), score=score200(pc,yva),
             r=float(np.corrcoef(P,yva)[0,1]), score61=score200(pc[mk61],yva[mk61]),
             mae61=float(np.abs(pc[mk61]-yva[mk61]).mean()),
             center61=score200(np.clip(pc[mk61]-np.median(pc[mk61])+np.median(yva[mk61]),0,9.9), yva[mk61]),
             sec=time.time()-t0)
    hist.append(d)
    print("ft ep%02d loss=%.4f all=%.2f r=%+.3f | M4-6.1 score=%.2f 中心化=%.2f const=%.2f (%.0fs)"
          % (ep, d["loss"], d["score"], d["r"], d["score61"], d["center61"], CONST61, d["sec"]), flush=True)
    key = max(d["score61"], d["center61"])
    if best is None or key > best[0]:
        best = (key, d, ep)
        torch.save(dict(arch="J1_JEPA", encoder=reg.enc.state_dict(), head=reg.head.state_dict(),
                        metrics=d, epoch=ep, L=L_TOK), f"{OUT}/J1_best.pt")
json.dump(dict(pre_epochs=PRE_EPOCHS, ft_epochs=FT_EPOCHS, const61=CONST61,
               best=best[1], best_epoch=best[2], best_key=best[0], history=hist, pre_hist=pre_hist),
          open(f"{OUT}/J1_result.json","w"), indent=2)
print("BEST key=%.2f (ep%d)" % (best[0], best[2]), flush=True)
print(json.dumps(best[1]), flush=True)
