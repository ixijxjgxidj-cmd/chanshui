"""轮25公开 STEAD 留一伪域中心估计器实验。
只读取远程公开缓存 outputs/t2_cache_p10；不读取 R1/R2/08。
"""
import os, json, time, pickle, hashlib
import numpy as np, torch, torch.nn as nn
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor, Ridge

ROOT='/root/5.6+chanshui1'; CACHE=f'{ROOT}/outputs/t2_cache_p10'; OUT=f'{ROOT}/outputs/t2_round25'; os.makedirs(OUT,exist_ok=True)
X=np.load(f'{CACHE}/X.npy',mmap_mode='r'); Y=np.load(f'{CACHE}/y.npy'); G=np.load(f'{CACHE}/g.npy').astype(str)
sel=np.where((Y>=3.8)&(Y<=6.5))[0]
# pseudo-environment: stable hash of public source/event ID, never competition files
D=np.asarray([int(hashlib.md5(s.encode()).hexdigest()[:8],16)%4 for s in G[sel]],np.int64)
class RmsNorm(nn.Module):
 def forward(self,x): return x/torch.sqrt((x*x).mean(dim=2,keepdim=True)+1e-12)
class LogAmp(nn.Module):
 def forward(self,x): return torch.sign(x)*torch.log1p(torch.abs(x))
BANDS=[(.05,.2),(.2,.5),(.5,1),(1,2),(2,5),(5,10),(10,20),(20,45)]
def specf(x):
 x=x-x.mean(2,keepdim=True); F=torch.fft.rfft(x*torch.hann_window(x.shape[2],device=x.device),dim=2); P=F.real**2+F.imag**2; fr=torch.fft.rfftfreq(x.shape[2],d=.01).to(x.device); P=P/(P.sum(2,keepdim=True)+1e-20)
 return torch.cat([torch.log10(P[:,:,(fr>=a)&(fr<b)].sum(2)+1e-12) for a,b in BANDS],1)
class Net(nn.Module):
 def __init__(self,s):
  super().__init__(); self.s=s
  def b(a,b): return nn.Sequential(nn.Conv1d(a,b,7,2,3),nn.BatchNorm1d(b),nn.ReLU())
  self.pre=nn.Sequential(RmsNorm(),LogAmp()); self.cnn=nn.Sequential(b(3,32),b(32,64),b(64,128),b(128,128),b(128,256),nn.AdaptiveAvgPool1d(1),nn.Flatten()); ex=(1 if s['snr'] else 0)+(24 if s['spec'] else 0); self.head=nn.Sequential(nn.Linear(256+ex,128),nn.ReLU(),nn.Dropout(0.1),nn.Linear(128,3 if s['mt'] else 2))
 def forward(self,x,snr):
  z=[self.cnn(self.pre(x))]
  if self.s['snr']: z.append(snr[:,None]/40)
  if self.s['spec']: z.append(specf(x))
  o=self.head(torch.cat(z,1)); return o[:,0]
def snr(x):
 h=x.shape[2]//2; return 20*torch.log10(torch.sqrt((x[:,:,:h]**2).mean((1,2))+1e-12)/torch.sqrt((x[:,:,h:]**2).mean((1,2))+1e-12))
MEM=[]; dev='cuda' if torch.cuda.is_available() else 'cpu'
for nm in ['M1','M2','M3','M4','M5','M6','M7']:
 ck=torch.load(f'{ROOT}/outputs/t2_ens/{nm}.pt',map_location='cpu',weights_only=False); s=ck['spec']; n=Net(s); n.load_state_dict(ck['state_dict']); n.eval().to(dev); MEM.append((n,s))
print('loaded',len(sel),'public records on',dev,flush=True)
# infer compact predictions in single public gain; RMS normalization makes gain nuisance largely irrelevant
P=[]
for st in range(0,len(sel),512):
 xb=torch.from_numpy(np.asarray(X[sel[st:st+512]],np.float32)).to(dev); ss=snr(xb); pp=[]
 with torch.no_grad():
  for n,s in MEM:
   xx=torch.roll(xb,s['shift'],2) if s['shift'] else xb; pp.append(n(xx,snr(xx)))
 P.append(torch.stack(pp).cpu().numpy())
P=np.concatenate(P,1); print('pred',P.shape,flush=True)
# batch features from member predictions and simple signal stats
def feat(idx):
 q=[10,25,50,75,90]; z=np.mean(P[:,idx]-np.median(P[:,idx],axis=1,keepdims=True),0); vals=[np.mean(P[:,idx],0),z,np.std(P[:,idx],0)]
 out=[]
 for v in vals: out += list(np.percentile(v,q))+[float(v.std())]
 out += [float(len(idx))/200]
 return np.asarray(out,np.float32)
rng=np.random.RandomState(20260815); F=[]; T=[]; E=[]
for e in range(4):
 ids=np.where(D==e)[0]
 for k in range(800):
  lo=rng.uniform(3.8,5.5); hi=lo+rng.uniform(.5,2.2); cand=ids[(Y[sel[ids]]>=lo)&(Y[sel[ids]]<=hi)]
  if len(cand)<50: continue
  ix=rng.choice(cand,min(len(cand),rng.randint(50,201)),replace=False); F.append(feat(ix)); T.append(np.median(Y[sel[ix]])); E.append(e)
F=np.asarray(F); T=np.asarray(T); E=np.asarray(E); print('batches',F.shape,'domains',np.bincount(E),flush=True)
mu=F[E!=3].mean(0); sd=F[E!=3].std(0)+1e-6; a=E!=3; b=E==3
mods={'gbm':GradientBoostingRegressor(n_estimators=350,max_depth=3,learning_rate=.05,random_state=0),'extra':ExtraTreesRegressor(n_estimators=300,min_samples_leaf=3,random_state=0,n_jobs=-1),'huber':HuberRegressor(epsilon=1.35,alpha=1e-4,max_iter=500)}
rep={}
for name,m in mods.items():
 m.fit((F[a]-mu)/sd,T[a]); pred=m.predict((F[b]-mu)/sd); rep[name]=dict(test_domain=3,mae=float(np.abs(pred-T[b]).mean()),r=float(pearsonr(pred,T[b])[0]),train_mae=float(np.abs(m.predict((F[a]-mu)/sd)-T[a]).mean()))
 print(name,rep[name],flush=True)
# standard 5-fold batch CV on all domains
for name,mk in [('gbm',lambda:GradientBoostingRegressor(n_estimators=350,max_depth=3,learning_rate=.05,random_state=0)),('extra',lambda:ExtraTreesRegressor(n_estimators=300,min_samples_leaf=3,random_state=0,n_jobs=-1)),('huber',lambda:HuberRegressor(epsilon=1.35,alpha=1e-4,max_iter=500))]:
 pp=np.zeros(len(T))
 for e in range(4):
  aa=E!=e; bb=E==e; m=mk(); m.fit((F[aa]-F[aa].mean(0))/(F[aa].std(0)+1e-6),T[aa]); pp[bb]=m.predict((F[bb]-F[aa].mean(0))/(F[aa].std(0)+1e-6))
 rep[name]['lodo_mae']=float(np.abs(pp-T).mean()); rep[name]['lodo_r']=float(pearsonr(pp,T)[0])
json.dump(dict(protocol='public STEAD only; source_id hash pseudo-domains; domain3 held out',n_records=int(len(sel)),n_batches=int(len(T)),report=rep),open(f'{OUT}/public_lodo.json','w'),indent=2)
# freeze best by LODO MAE, tie r
best=min(rep,key=lambda k:(rep[k]['lodo_mae'],-rep[k]['lodo_r'])); m=mods[best]; m.fit((F-mu)/sd,T); pickle.dump(dict(model=m,mu=mu,sd=sd,best=best),open(f'{OUT}/center_lodo.pkl','wb'))
print('BEST',best,rep[best],flush=True)
