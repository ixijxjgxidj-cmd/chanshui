"""8 成员（7 监督 + 1 JEPA）纳入判定：只用 R1/R2 预注册 train + STEAD 公开验证。

判定口径（预先声明，避免结果导向）：
 主指标 = R1-train 与 R2-train 的排序相关性 r 的最小值（最差轮次表现）
 次指标 = 两轮"中心化@真中心 + 斜率0.40"分数之和
仅当 8 成员在主指标上不劣于 7 成员，且次指标更高时，才纳入 J1。
合规：两个 holdout 断言不读取；不读 08；不使用 holdout 做任何选择。
"""
import json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn
from obspy import read
from scipy import signal as ss
from scipy.stats import pearsonr
from t2_io import to_enz_from_obspy
from t2_features import score200

BANDS=[(0.05,0.2),(0.2,0.5),(0.5,1),(1,2),(2,5),(5,10),(10,20),(20,45)]
def spec_feats_t(x, sr=100.0):
    xw=x-x.mean(dim=2,keepdim=True); win=torch.hann_window(xw.shape[2],device=x.device)
    F=torch.fft.rfft(xw*win,dim=2); P=F.real**2+F.imag**2
    fr=torch.fft.rfftfreq(xw.shape[2],d=1.0/sr).to(x.device)
    Pn=P/(P.sum(dim=2,keepdim=True)+1e-20)
    return torch.cat([torch.log10(Pn[:,:,(fr>=a)&(fr<b)].sum(dim=2)+1e-12) for a,b in BANDS],dim=1)
class RmsNorm(nn.Module):
    def forward(self,x): return x/torch.sqrt((x*x).mean(dim=2,keepdim=True)+1e-12)
class LogAmp(nn.Module):
    def forward(self,x): return torch.sign(x)*torch.log1p(torch.abs(x))
class Member(nn.Module):
    def __init__(self,use_snr=False,use_spec=False,multitask=False):
        super().__init__(); self.use_snr,self.use_spec,self.multitask=use_snr,use_spec,multitask
        def blk(ci,co,k=7,s=2): return nn.Sequential(nn.Conv1d(ci,co,k,stride=s,padding=k//2),nn.BatchNorm1d(co),nn.ReLU(inplace=True))
        self.pre=nn.Sequential(RmsNorm(),LogAmp())
        self.cnn=nn.Sequential(blk(3,32),blk(32,64),blk(64,128),blk(128,128),blk(128,256),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        extra=(1 if use_snr else 0)+(3*len(BANDS) if use_spec else 0)
        self.head=nn.Sequential(nn.Linear(256+extra,128),nn.ReLU(inplace=True),nn.Dropout(0.1),nn.Linear(128,3 if multitask else 2))
    def forward(self,x,snr):
        h=self.cnn(self.pre(x)); parts=[h]
        if self.use_snr: parts.append(snr.unsqueeze(1)/40.0)
        if self.use_spec: parts.append(spec_feats_t(x))
        o=self.head(torch.cat(parts,dim=1)); return o[:,0]
class BlockEncoder(nn.Module):
    def __init__(self,d=128):
        super().__init__(); self.pre=RmsNorm()
        def b(a,c): return nn.Sequential(nn.Conv1d(a,c,7,stride=2,padding=3),nn.BatchNorm1d(c),nn.GELU())
        self.net=nn.Sequential(b(3,32),b(32,64),b(64,96),b(96,d))
    def forward(self,x): return self.net(self.pre(x)).transpose(1,2)
class JepaReg(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=BlockEncoder()
        self.head=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,2))
    def forward(self,x): return self.head(self.enc(x).mean(1))[:,0]

def aic_onset(x):
    n=len(x); k=np.arange(1,n-1)
    v1=np.array([x[:i].var() for i in k]); v2=np.array([x[i:].var() for i in k])
    with np.errstate(divide="ignore",invalid="ignore"):
        a=k*np.log(np.maximum(v1,1e-30))+(n-k-1)*np.log(np.maximum(v2,1e-30))
    a[~np.isfinite(a)]=np.inf
    return int(k[np.argmin(a)])
def window(dirp,f,pre=5.0,post=5.0):
    w,sr=to_enz_from_obspy(read(os.path.join(dirp,f))); w=w-w.mean(axis=1,keepdims=True)
    env=np.abs(ss.hilbert(w,axis=1)).max(axis=0); sm=np.convolve(env,np.ones(25)/25,mode="same"); pk=int(sm.argmax())
    lo=max(0,pk-int(45*sr)); on=lo+aic_onset(w[2,lo:pk+1]) if pk-lo>200 else pk
    s0,e0=int(on-pre*sr),int(on+post*sr)
    seg=np.pad(w[:,:max(1,e0)],((0,0),(-s0,0)),mode="reflect") if s0<0 else w[:,s0:e0]
    need=int((pre+post)*sr)
    if seg.shape[1]<need: seg=np.pad(seg,((0,0),(0,need-seg.shape[1])),mode="reflect")
    return (seg[:,:need]-seg[:,:need].mean(axis=1,keepdims=True)).astype(np.float32)
def snr_np(xb):
    half=xb.shape[2]//2
    n=np.sqrt((xb[:,:,:half]**2).mean(axis=(1,2)))+1e-12
    s=np.sqrt((xb[:,:,half:]**2).mean(axis=(1,2)))+1e-12
    return (20.0*np.log10(s/n)).astype(np.float32)

dev="cuda" if torch.cuda.is_available() else "cpu"
MEM=[]
for nm in ["M1","M2","M3","M4","M5","M6","M7"]:
    ck=torch.load(f"outputs/t2_ens/{nm}.pt",map_location="cpu",weights_only=False)
    s=ck["spec"]; net=Member(s["snr"],s["spec"],s["mt"]); net.load_state_dict(ck["state_dict"]); net.eval().to(dev)
    MEM.append((nm,net,s))
jck=torch.load("outputs/t2_jepa/J1_best.pt",map_location="cpu",weights_only=False)
J=JepaReg(); J.enc.load_state_dict(jck["encoder"]); J.head.load_state_dict(jck["head"]); J.eval().to(dev)

sp=json.load(open("outputs/t2_prereg_split.json"))["splits"]
def load_set(key,dirp,ansfile,strip=False):
    tr=sp[key]["train"]; ho=set(sp[key]["holdout"]); assert not (set(tr)&ho)
    ans={}
    for ln in open(ansfile):
        if not ln.strip(): continue
        p=ln.split(); k=os.path.basename(p[0]) if strip else p[0]; ans[k]=float(p[1])
    return np.stack([window(dirp,f) for f in tr]), np.asarray([ans[f] for f in tr],np.float32)
D={"R1":load_set("R1","r1_t2","r1_t2/answers.txt"),
   "R2":load_set("R2","r2_t2","r2_t2/T2.an",True)}

def z_scores(X, include_jepa):
    Z=[]
    xb=torch.from_numpy(X).to(dev); sn=torch.from_numpy(snr_np(X)).to(dev)
    for nm,net,s in MEM:
        x2=torch.roll(xb,s["shift"],dims=2) if s["shift"] else xb
        with torch.no_grad(): mu=net(x2, sn).cpu().numpy()
        Z.append(mu-np.median(mu))
    if include_jepa:
        with torch.no_grad(): mj=J(xb).cpu().numpy()
        Z.append(mj-np.median(mj))
    return np.mean(Z,axis=0)

rep={}
for tag, inc in [("ens7",False), ("ens8_with_J1",True)]:
    d={}
    for key,(X,y) in D.items():
        z=z_scores(X,inc); cen=float(np.median(y))
        d[key]=dict(r=float(pearsonr(z,y)[0]), p=float(pearsonr(z,y)[1]),
                    centered040=score200(np.clip(cen+0.40*z,0,9.9),y),
                    const=score200(np.full_like(y,cen),y))
    d["min_r"]=min(d["R1"]["r"], d["R2"]["r"])
    d["sum_centered"]=d["R1"]["centered040"]+d["R2"]["centered040"]
    rep[tag]=d
    print("%s: R1 r=%+.3f score=%.2f | R2 r=%+.3f score=%.2f | min_r=%+.3f sum=%.2f"
          % (tag,d["R1"]["r"],d["R1"]["centered040"],d["R2"]["r"],d["R2"]["centered040"],d["min_r"],d["sum_centered"]))
a,b=rep["ens7"],rep["ens8_with_J1"]
include = (b["min_r"] >= a["min_r"] - 1e-9) and (b["sum_centered"] > a["sum_centered"])
print("\n判定：%s J1（min_r %+.3f->%+.3f, sum %.2f->%.2f）"
      % ("纳入" if include else "不纳入", a["min_r"], b["min_r"], a["sum_centered"], b["sum_centered"]))
json.dump(dict(criteria="min_r not worse AND sum_centered higher", decision_include_J1=bool(include), report=rep),
          open("outputs/t2_ens8_decision.json","w"), indent=2)
print("saved outputs/t2_ens8_decision.json")
