"""轮28公开真实台站 LOSO：无标签风险选择与常数回退。
仅公开 STEAD station27 缓存；不读取比赛或08。
"""
import os,json,time
import numpy as np,torch,torch.nn as nn
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
ROOT='/root/5.6+chanshui1';C=f'{ROOT}/outputs/t2_cache_station27';OUT=f'{ROOT}/outputs/t2_round28';os.makedirs(OUT,exist_ok=True)
X=np.load(f'{C}/X.npy',mmap_mode='r');Y=np.load(f'{C}/y.npy');N=np.load(f'{C}/net.npy').astype(str);S=np.load(f'{C}/sta.npy').astype(str);SID=np.char.add(np.char.add(N,'.'),S)
sel=np.where((Y>=3.8)&(Y<=6.5))[0]; Ym=Y[sel]; Gm=SID[sel]
class RmsNorm(nn.Module):
 def forward(self,x):return x/torch.sqrt((x*x).mean(2,keepdim=True)+1e-12)
class LogAmp(nn.Module):
 def forward(self,x):return torch.sign(x)*torch.log1p(torch.abs(x))
B=[(.05,.2),(.2,.5),(.5,1),(1,2),(2,5),(5,10),(10,20),(20,45)]
def spec(x):
 x=x-x.mean(2,keepdim=True);F=torch.fft.rfft(x*torch.hann_window(x.shape[2],device=x.device),dim=2);P=F.real**2+F.imag**2;fr=torch.fft.rfftfreq(x.shape[2],d=.01).to(x.device);P=P/(P.sum(2,keepdim=True)+1e-20);return torch.cat([torch.log10(P[:,:,(fr>=a)&(fr<b)].sum(2)+1e-12) for a,b in B],1)
class Net(nn.Module):
 def __init__(s_,s):
  super().__init__();s_.s=s
  def bl(a,b):return nn.Sequential(nn.Conv1d(a,b,7,2,3),nn.BatchNorm1d(b),nn.ReLU())
  s_.pre=nn.Sequential(RmsNorm(),LogAmp());s_.cnn=nn.Sequential(bl(3,32),bl(32,64),bl(64,128),bl(128,128),bl(128,256),nn.AdaptiveAvgPool1d(1),nn.Flatten());ex=(1 if s['snr'] else 0)+(24 if s['spec'] else 0);s_.head=nn.Sequential(nn.Linear(256+ex,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,3 if s['mt'] else 2))
 def forward(s_,x,sn):
  z=[s_.cnn(s_.pre(x))]
  if s_.s['snr']:z.append(sn[:,None]/40)
  if s_.s['spec']:z.append(spec(x))
  return s_.head(torch.cat(z,1))
def snr(x):
 h=x.shape[2]//2;return 20*torch.log10(torch.sqrt((x[:,:,h:]**2).mean((1,2))+1e-12)/torch.sqrt((x[:,:,:h]**2).mean((1,2))+1e-12))
dev='cuda' if torch.cuda.is_available() else 'cpu';M=[]
for nm in ['M1','M2','M3','M4','M5','M6','M7']:
 ck=torch.load(f'{ROOT}/outputs/t2_ens/{nm}.pt',map_location='cpu',weights_only=False);n=Net(ck['spec']);n.load_state_dict(ck['state_dict']);n.eval().to(dev);M.append((n,ck['spec']))
P=[];SN=[];SP=[]
for st in range(0,len(sel),512):
 xb=torch.from_numpy(np.asarray(X[sel[st:st+512]],np.float32)).to(dev);pp=[]
 with torch.no_grad():
  for n,s in M:
   xx=torch.roll(xb,s['shift'],2) if s['shift'] else xb;pp.append(n(xx,snr(xx))[:,0])
  SN.append(snr(xb).cpu().numpy());SP.append(spec(xb).mean(1).cpu().numpy())
 P.append(torch.stack(pp).cpu().numpy())
P=np.concatenate(P,1);SN=np.concatenate(SN);SP=np.concatenate(SP)
Q=[10,25,50,75,90]
def batch(ix):
 z=np.mean(P[:,ix]-np.median(P[:,ix],axis=1,keepdims=True),0);f=[]
 for v in [np.mean(P[:,ix],0),z,np.std(P[:,ix],0),SN[ix]]:f+=list(np.percentile(v,Q))+[float(v.std())]
 f+=list(np.percentile(SP[ix],Q))+[len(ix)/200]
 return np.asarray(f,np.float32),float(np.median(z)),float(np.std(P[:,ix]))
rng=np.random.RandomState(20260818);stations=[g for g,c in __import__('collections').Counter(Gm.tolist()).most_common() if c>=250][:10];F=[];T=[];Z=[];SD=[];GG=[]
for g in stations:
 ids=np.where(Gm==g)[0]
 for _ in range(260):
  lo=rng.uniform(3.8,5.4);hi=lo+rng.uniform(.6,2.2);ca=ids[(Ym[ids]>=lo)&(Ym[ids]<=hi)]
  if len(ca)<50:continue
  ix=rng.choice(ca,min(len(ca),rng.randint(50,201)),False);f,z,sd=batch(ix);F.append(f);T.append(float(np.median(Ym[ix])));Z.append(z);SD.append(sd);GG.append(g)
F=np.asarray(F);T=np.asarray(T);Z=np.asarray(Z);SD=np.asarray(SD);GG=np.asarray(GG)
# score definition
def score(p,y):return float(np.abs(np.clip(p,0,9.9)-y).mean())
rep={}
for g in stations:
 a=GG!=g;b=GG==g;mu=F[a].mean(0);ss=F[a].std(0)+1e-6;mdl=GradientBoostingRegressor(n_estimators=350,max_depth=3,learning_rate=.05,random_state=0);mdl.fit((F[a]-mu)/ss,T[a]);c=mdl.predict((F[b]-mu)/ss); y=T[b]; z=Z[b];
 lin=c+.4*z; const=c
 # train selector on training groups: whether constant is better than linear; risk feature = spread + |z| + center uncertainty residual proxy
 risk=np.c_[SD[b],np.abs(z),np.abs(c-np.median(T[a])),np.std(F[b],1)]
 at=GG[a];ct=mdl.predict((F[a]-mu)/ss);yt=T[a];zt=Z[a]; rt=np.c_[SD[a],np.abs(zt),np.abs(ct-np.median(yt)),np.std(F[a],1)]; label=(np.abs(ct+.4*zt-yt)>np.abs(ct-yt)).astype(int)
 clf=GradientBoostingClassifier(n_estimators=100,max_depth=2,learning_rate=.05,random_state=0);clf.fit(rt,label);prob=clf.predict_proba(risk)[:,1];selc=prob>=.5
 preds=np.where(selc,const,lin)
 rep[g]=dict(n=int(b.sum()),base_mae=score(lin,y),const_mae=score(const,y),selective_mae=score(preds,y),coverage=float((~selc).mean()),highrisk=int(selc.sum()),risk_r=float(pearsonr(prob,np.abs(lin-y))[0]) if len(y)>2 else 0)
 print(g,rep[g],flush=True)
# aggregate
allbase=[];allsel=[];allconst=[]
for g in stations:
 # stored not predictions; recompute aggregate metrics from per groups only weighted mean
 pass
json.dump(dict(protocol='public STEAD true station lineage; LOSO risk selector; no competition data',n_records=int(len(sel)),stations=stations,n_batches=int(len(T)),per_station=rep),open(f'{OUT}/selective_loso.json','w'),indent=2)
print('DONE')
