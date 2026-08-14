"""T2 生产推理管线（定型）+ 冻结留出一次性盲评。

管线（predict_t2）：
 1. 读 mseed -> AIC 起跳 -> [起跳-5s, 起跳+5s] 3通道窗口
 2. 7 成员集成，各成员减自身批内中位数 -> 排序分数 z
 3. 批级特征 -> center_estimator_v2 估计该批震级中心 c 与分位 q10/q25/q75/q90
 4. 输出 pred = clip(c + slope*(z - median(z)), 0, 9.9)
    slope 由 R1/R2-train 联合标定（不使用 holdout）；另给分位映射版本作对照

盲评：R1-holdout(60) 与 R2-holdout(60) 各评估一次。
本脚本是**首次且唯一**读取两个 holdout 的地方；参数在此之前已全部冻结。
合规：不读 08；不因 holdout 结果回头改参数（若不达标则记录失败并另开新一轮训练）。
"""
import json, os, sys, pickle, hashlib
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn
from obspy import read
from scipy import signal as ss
from scipy.stats import pearsonr
from t2_io import to_enz_from_obspy
from t2_features import score200

BANDS = [(0.05,0.2),(0.2,0.5),(0.5,1),(1,2),(2,5),(5,10),(10,20),(20,45)]
QS = [10,25,50,75,90]
FEATK = ["mu_abs","z","sigma","logR","snr","b1","b2","b5","b6"]
def spec_feats_t(x, sr=100.0):
    xw = x - x.mean(dim=2, keepdim=True)
    win = torch.hann_window(xw.shape[2], device=x.device)
    F = torch.fft.rfft(xw*win, dim=2); P = F.real**2 + F.imag**2
    fr = torch.fft.rfftfreq(xw.shape[2], d=1.0/sr).to(x.device)
    Pn = P/(P.sum(dim=2, keepdim=True)+1e-20)
    return torch.cat([torch.log10(Pn[:, :, (fr>=a)&(fr<b)].sum(dim=2)+1e-12) for a,b in BANDS], dim=1)
class RmsNorm(nn.Module):
    def forward(self, x): return x/torch.sqrt((x**2).mean(dim=2, keepdim=True)+1e-12)
class LogAmp(nn.Module):
    def forward(self, x): return torch.sign(x)*torch.log1p(torch.abs(x))
class Member(nn.Module):
    def __init__(self, use_snr=False, use_spec=False, multitask=False):
        super().__init__()
        self.use_snr, self.use_spec, self.multitask = use_snr, use_spec, multitask
        def blk(ci,co,k=7,s=2):
            return nn.Sequential(nn.Conv1d(ci,co,k,stride=s,padding=k//2), nn.BatchNorm1d(co), nn.ReLU(inplace=True))
        self.pre = nn.Sequential(RmsNorm(), LogAmp())
        self.cnn = nn.Sequential(blk(3,32),blk(32,64),blk(64,128),blk(128,128),blk(128,256),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten())
        extra = (1 if use_snr else 0) + (3*len(BANDS) if use_spec else 0)
        self.head = nn.Sequential(nn.Linear(256+extra,128), nn.ReLU(inplace=True), nn.Dropout(0.1),
                                  nn.Linear(128, 3 if multitask else 2))
    def forward(self, x, snr):
        h = self.cnn(self.pre(x)); parts=[h]
        if self.use_snr: parts.append(snr.unsqueeze(1)/40.0)
        if self.use_spec: parts.append(spec_feats_t(x))
        o = self.head(torch.cat(parts,dim=1))
        return o[:,0], o[:,1].clamp(-3,3), (o[:,2] if self.multitask else None)

def aic_onset(x):
    n=len(x); k=np.arange(1,n-1)
    v1=np.array([x[:i].var() for i in k]); v2=np.array([x[i:].var() for i in k])
    with np.errstate(divide="ignore", invalid="ignore"):
        a=k*np.log(np.maximum(v1,1e-30))+(n-k-1)*np.log(np.maximum(v2,1e-30))
    a[~np.isfinite(a)]=np.inf
    return int(k[np.argmin(a)])
def window(path, pre=5.0, post=5.0):
    w, sr = to_enz_from_obspy(read(path))
    w = w - w.mean(axis=1, keepdims=True)
    env = np.abs(ss.hilbert(w, axis=1)).max(axis=0)
    sm = np.convolve(env, np.ones(25)/25, mode="same"); pk = int(sm.argmax())
    lo = max(0, pk-int(45*sr))
    on = lo + aic_onset(w[2, lo:pk+1]) if pk-lo > 200 else pk
    s0, e0 = int(on-pre*sr), int(on+post*sr)
    seg = np.pad(w[:, :max(1,e0)], ((0,0),(-s0,0)), mode="reflect") if s0 < 0 else w[:, s0:e0]
    need = int((pre+post)*sr)
    if seg.shape[1] < need: seg = np.pad(seg, ((0,0),(0,need-seg.shape[1])), mode="reflect")
    return (seg[:, :need]-seg[:, :need].mean(axis=1, keepdims=True)).astype(np.float32)

dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT="outputs/t2_ens"
MEM=[]
for nm in ["M1","M2","M3","M4","M5","M6","M7"]:
    ck = torch.load(f"{OUT}/{nm}.pt", map_location="cpu", weights_only=False)
    s = ck["spec"]; net = Member(s["snr"], s["spec"], s["mt"]); net.load_state_dict(ck["state_dict"])
    net.eval().to(dev); MEM.append((nm, net, s))
EST = pickle.load(open(f"{OUT}/center_estimator_v2.pkl","rb"))
def snr_t(x):
    half = x.shape[2]//2
    n = torch.sqrt((x[:,:,:half]**2).mean(dim=(1,2)))+1e-12
    s = torch.sqrt((x[:,:,half:]**2).mean(dim=(1,2)))+1e-12
    return 20.0*torch.log10(s/n)

def per_record(Xnp, bs=512):
    keys=["mu_abs","sigma","logR","snr","b1","b2","b5","b6"]
    acc={k:[] for k in keys}; MUs=[]
    for i in range(0,len(Xnp),bs):
        xb = torch.from_numpy(np.asarray(Xnp[i:i+bs], np.float32)).to(dev)
        sn = snr_t(xb); mus=[]; lss=[]; drs=None
        for nm, net, s in MEM:
            xb2 = torch.roll(xb, s["shift"], dims=2) if s["shift"] else xb
            with torch.no_grad(): mu, ls, d = net(xb2, snr_t(xb2))
            mus.append(mu); lss.append(ls)
            if d is not None: drs = d
        MU = torch.stack(mus); MUs.append(MU.cpu().numpy())
        acc["mu_abs"].append(MU.mean(0).cpu().numpy())
        acc["sigma"].append(torch.stack(lss).mean(0).cpu().numpy())
        acc["logR"].append((drs if drs is not None else torch.zeros_like(sn)).cpu().numpy())
        acc["snr"].append(sn.cpu().numpy())
        sf = spec_feats_t(xb)
        for bi,k in [(1,"b1"),(2,"b2"),(5,"b5"),(6,"b6")]:
            acc[k].append(sf[:, bi*3:(bi+1)*3].mean(1).cpu().numpy())
    out={k:np.concatenate(v) for k,v in acc.items()}
    MU=np.concatenate(MUs, axis=1)
    out["z"]=np.mean(MU - np.median(MU,axis=1,keepdims=True), axis=0)
    return out

def batch_feats(pr, n):
    f=[]
    for k in FEATK:
        v=pr[k]; f += list(np.percentile(v,QS)) + [float(v.std())]
    f += [float(n)/200.0]
    return np.asarray(f, np.float32).reshape(1,-1)

SLOPE = float(os.environ.get("SLOPE","0.40"))
def predict_t2(paths):
    X = np.stack([window(p) for p in paths])
    pr = per_record(X)
    fb = (batch_feats(pr, len(paths)) - EST["mu"]) / EST["sd"]
    c = float(EST["center"].predict(fb)[0])
    qs = [float(m.predict(fb)[0]) for m in EST["quantiles"]]   # q10,q25,q75,q90
    z = pr["z"]
    pred = np.clip(c + SLOPE*(z - np.median(z)), 0, 9.9)
    # 分位映射版本：把 z 的秩映射到估计分布
    rk = (np.argsort(np.argsort(z))+0.5)/len(z)*100.0
    qx = np.array([10,25,50,75,90]); qy = np.array([qs[0],qs[1],c,qs[2],qs[3]])
    qy = np.maximum.accumulate(qy)
    pred_q = np.clip(np.interp(rk, qx, qy), 0, 9.9)
    return pred, pred_q, dict(center=c, q=qs, z=z)

sp = json.load(open("outputs/t2_prereg_split.json"))["splits"]
print("SLOPE=%.2f (冻结) | 中心估计器 n_synth=%d" % (SLOPE, EST["n_synth"]))
report={}
for key, dirp, ansfile, strip in [("R1","r1_t2","r1_t2/answers.txt",False),
                                  ("R2","r2_t2","r2_t2/T2.an",True)]:
    hold = sp[key]["holdout"]; tr = set(sp[key]["train"])
    assert not (set(hold) & tr), "holdout 与 train 交集"
    sha = sp[key]["sha256"]
    ans={}
    for ln in open(ansfile):
        if not ln.strip(): continue
        p=ln.split(); k=os.path.basename(p[0]) if strip else p[0]; ans[k]=float(p[1])
    paths=[os.path.join(dirp,f) for f in hold]
    y = np.asarray([ans[f] for f in hold], np.float32)
    pred, pred_q, info = predict_t2(paths)
    d = dict(n=len(y), split_sha256=sha, est_center=info["center"], true_median=float(np.median(y)),
             center_err=float(info["center"]-np.median(y)),
             score_linear=score200(pred,y), mae_linear=float(np.abs(pred-y).mean()),
             score_quantile=score200(pred_q,y), mae_quantile=float(np.abs(pred_q-y).mean()),
             score_const_est_center=score200(np.full_like(y,info["center"]),y),
             score_const_true_center=score200(np.full_like(y,np.median(y)),y),
             score_const_merged480=score200(np.full_like(y,4.80),y),
             r=float(pearsonr(pred,y)[0]))
    report[key]=d
    print("\n=== %s-holdout (n=%d, sha=%s) 首次且唯一评估 ===" % (key, len(y), sha[:16]))
    print("  真中位数 %.2f | 估计中心 %.2f (误差 %+.2f)" % (d["true_median"], d["est_center"], d["center_err"]))
    print("  线性版   score=%.2f  MAE=%.4f  r=%+.3f" % (d["score_linear"], d["mae_linear"], d["r"]))
    print("  分位映射 score=%.2f  MAE=%.4f" % (d["score_quantile"], d["mae_quantile"]))
    print("  对照: 常数@估计中心 %.2f | 常数@真中心(不可得) %.2f | 常数@合并4.80 %.2f"
          % (d["score_const_est_center"], d["score_const_true_center"], d["score_const_merged480"]))
json.dump(report, open("outputs/t2_holdout_eval.json","w"), indent=2)
print("\nsaved outputs/t2_holdout_eval.json")
