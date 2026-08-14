"""J1(JEPA) 外部参照：在 R1/R2 预注册 train 上评估排序相关性与中心化后分数。

用途：判定 JEPA 表征是否比 G 系列（轮23 最优 r=+0.421/+0.305）提供更强的可迁移排序。
严格性：
- 只读 R1/R2 的预注册 train 清单；两个 holdout 断言不读取；不读 08。
- 本脚本只报告，不做任何超参/成员选择（选择已在公开 STEAD 上完成并冻结）。
"""
import json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn
from obspy import read
from scipy import signal as ss
from scipy.stats import pearsonr, spearmanr
from t2_io import to_enz_from_obspy
from t2_features import score200

class RmsNorm(nn.Module):
    def forward(self, x): return x / torch.sqrt((x*x).mean(dim=2, keepdim=True) + 1e-12)
class BlockEncoder(nn.Module):
    def __init__(self, d=128):
        super().__init__(); self.pre = RmsNorm()
        def b(a,c): return nn.Sequential(nn.Conv1d(a,c,7,stride=2,padding=3), nn.BatchNorm1d(c), nn.GELU())
        self.net = nn.Sequential(b(3,32), b(32,64), b(64,96), b(96,d))
    def forward(self, x): return self.net(self.pre(x)).transpose(1,2)
class Reg(nn.Module):
    def __init__(self):
        super().__init__(); self.enc = BlockEncoder()
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128,128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128,2))
    def forward(self, x): return self.head(self.enc(x).mean(1))

def aic_onset(x):
    n=len(x); k=np.arange(1,n-1)
    v1=np.array([x[:i].var() for i in k]); v2=np.array([x[i:].var() for i in k])
    with np.errstate(divide="ignore", invalid="ignore"):
        a=k*np.log(np.maximum(v1,1e-30))+(n-k-1)*np.log(np.maximum(v2,1e-30))
    a[~np.isfinite(a)]=np.inf
    return int(k[np.argmin(a)])
def window(dirp, f, pre=5.0, post=5.0):
    w, sr = to_enz_from_obspy(read(os.path.join(dirp, f)))
    w = w - w.mean(axis=1, keepdims=True)
    env = np.abs(ss.hilbert(w, axis=1)).max(axis=0)
    sm = np.convolve(env, np.ones(25)/25, mode="same"); pk=int(sm.argmax())
    lo=max(0,pk-int(45*sr)); on = lo+aic_onset(w[2,lo:pk+1]) if pk-lo>200 else pk
    s0,e0=int(on-pre*sr),int(on+post*sr)
    seg = np.pad(w[:, :max(1,e0)], ((0,0),(-s0,0)), mode="reflect") if s0<0 else w[:, s0:e0]
    need=int((pre+post)*sr)
    if seg.shape[1]<need: seg=np.pad(seg,((0,0),(0,need-seg.shape[1])),mode="reflect")
    return (seg[:,:need]-seg[:,:need].mean(axis=1,keepdims=True)).astype(np.float32)

sp = json.load(open("outputs/t2_prereg_split.json"))["splits"]
def load_set(key, dirp, ansfile, strip=False):
    tr=sp[key]["train"]; ho=set(sp[key]["holdout"]); assert not (set(tr)&ho), "train/holdout 交集"
    ans={}
    for ln in open(ansfile):
        if not ln.strip(): continue
        p=ln.split(); k=os.path.basename(p[0]) if strip else p[0]; ans[k]=float(p[1])
    return np.stack([window(dirp,f) for f in tr]), np.asarray([ans[f] for f in tr], np.float32)

dev="cuda" if torch.cuda.is_available() else "cpu"
ck=torch.load("outputs/t2_jepa/J1_best.pt", map_location="cpu", weights_only=False)
reg=Reg(); reg.enc.load_state_dict(ck["encoder"]); reg.head.load_state_dict(ck["head"]); reg.eval().to(dev)
print("J1 checkpoint ep%d | STEAD M4-6.1 score=%.2f 中心化=%.2f r=%+.3f"
      % (ck["epoch"], ck["metrics"]["score61"], ck["metrics"]["center61"], ck["metrics"]["r"]))

out={}
for key, dirp, ansfile, strip in [("R1","r1_t2","r1_t2/answers.txt",False), ("R2","r2_t2","r2_t2/T2.an",True)]:
    X,y = load_set(key,dirp,ansfile,strip)
    with torch.no_grad():
        o = reg(torch.from_numpy(X).to(dev)).cpu().numpy()
    mu = o[:,0]
    z = mu - np.median(mu)
    r,p = pearsonr(mu,y); rho,pr = spearmanr(mu,y)
    cen = np.median(y)                                  # 与轮23 同口径的"中心化后上界"参照
    best=None
    for s in np.arange(0,1.51,0.05):
        sc=score200(np.clip(cen+s*z,0,9.9),y)
        if best is None or sc>best[0]: best=(sc,float(s))
    out[key]=dict(n=len(y), r=float(r), p=float(p), rho=float(rho), p_rho=float(pr),
                  raw_score=score200(np.clip(mu,0,9.9),y),
                  centered_score=score200(np.clip(cen+0.40*z,0,9.9),y),
                  best_slope_score=best[0], best_slope=best[1],
                  const_self=score200(np.full_like(y,cen),y))
    d=out[key]
    print("  %s-train n=%d: r=%+.3f (p=%.2g) rho=%+.3f | 原始 %.2f | 中心化@0.40 %.2f | 最优斜率 %.2f@%.2f | 常数 %.2f"
          % (key,d["n"],d["r"],d["p"],d["rho"],d["raw_score"],d["centered_score"],d["best_slope_score"],d["best_slope"],d["const_self"]))
json.dump(dict(checkpoint_metrics=ck["metrics"], external_trainonly=out),
          open("outputs/t2_jepa/J1_external_trainonly.json","w"), indent=2)
print("saved outputs/t2_jepa/J1_external_trainonly.json")
