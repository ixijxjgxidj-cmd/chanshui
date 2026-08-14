"""轮26：真实台站/网络留出的批级中心估计器比较（只用公开 STEAD）。

预注册准则：
 主指标 = 留一网络(LONO) 的批中心 MAE，取所有留出网络的加权平均
 次指标 = 留一台站(LOSO) 的批中心 MAE
 仅当候选在主指标优于 v2 GBM 复现基线，且次指标不退化，才冻结。
合规：不读 R1/R2/08 任何文件。
"""
import os,json,pickle,time,collections
import numpy as np, torch, torch.nn as nn
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
ROOT='/root/5.6+chanshui1'; C=f'{ROOT}/outputs/t2_cache_p10'; OUT=f'{ROOT}/outputs/t2_round26'; os.makedirs(OUT,exist_ok=True)
X=np.load(f'{C}/X.npy',mmap_mode='r'); Y=np.load(f'{C}/y.npy'); AUX=np.load(f'{C}/aux.npy')
NET=np.load(f'{C}/net.npy').astype(str); STA=np.load(f'{C}/sta.npy').astype(str)
m=np.where((Y>=3.8)&(Y<=6.5))[0]
class RmsNorm(nn.Module):
 def forward(self,x): return x/torch.sqrt((x*x).mean(2,keepdim=True)+1e-12)
class LogAmp(nn.Module):
 def forward(self,x): return torch.sign(x)*torch.log1p(torch.abs(x))
B=[(.05,.2),(.2,.5),(.5,1),(1,2),(2,5),(5,10),(10,20),(20,45)]
def spec(x):
 x=x-x.mean(2,keepdim=True); F=torch.fft.rfft(x*torch.hann_window(x.shape[2],device=x.device),dim=2); P=F.real**2+F.imag**2
 fr=torch.fft.rfftfreq(x.shape[2],d=.01).to(x.device); P=P/(P.sum(2,keepdim=True)+1e-20)
 return torch.cat([torch.log10(P[:,:,(fr>=a)&(fr<b)].sum(2)+1e-12) for a,b in B],1)
class Net(nn.Module):
 def __init__(s_,s):
  super().__init__(); s_.s=s
  def bl(a,b): return nn.Sequential(nn.Conv1d(a,b,7,2,3),nn.BatchNorm1d(b),nn.ReLU())
  s_.pre=nn.Sequential(RmsNorm(),LogAmp()); s_.cnn=nn.Sequential(bl(3,32),bl(32,64),bl(64,128),bl(128,128),bl(128,256),nn.AdaptiveAvgPool1d(1),nn.Flatten())
  ex=(1 if s['snr'] else 0)+(24 if s['spec'] else 0); s_.head=nn.Sequential(nn.Linear(256+ex,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,3 if s['mt'] else 2))
 def forward(s_,x,sn):
  z=[s_.cnn(s_.pre(x))]
  if s_.s['snr']: z.append(sn[:,None]/40)
  if s_.s['spec']: z.append(spec(x))
  return s_.head(torch.cat(z,1))
def snr(x):
 h=x.shape[2]//2; return 20*torch.log10(torch.sqrt((x[:,:,h:]**2).mean((1,2))+1e-12)/torch.sqrt((x[:,:,:h]**2).mean((1,2))+1e-12))
dev='cuda' if torch.cuda.is_available() else 'cpu'; MEM=[]
for nm in ['M1','M2','M3','M4','M5','M6','M7']:
 ck=torch.load(f'{ROOT}/outputs/t2_ens/{nm}.pt',map_location='cpu',weights_only=False); n=Net(ck['spec']); n.load_state_dict(ck['state_dict']); n.eval().to(dev); MEM.append((n,ck['spec']))
P=[]; SNR=[]; SP=[]
t0=time.time()
for st in range(0,len(m),512):
 xb=torch.from_numpy(np.asarray(X[m[st:st+512]],np.float32)).to(dev); pp=[]
 with torch.no_grad():
  for n,s in MEM:
   xx=torch.roll(xb,s['shift'],2) if s['shift'] else xb; pp.append(n(xx,snr(xx))[:,0])
  SP.append(spec(xb).mean(1).cpu().numpy()); SNR.append(snr(xb).cpu().numpy())
 P.append(torch.stack(pp).cpu().numpy())
P=np.concatenate(P,1); SNR=np.concatenate(SNR); SP=np.concatenate(SP,0)
print('inference done',P.shape,SP.shape,'%.0fs'%(time.time()-t0),flush=True)
Ym=Y[m]; NETm=NET[m]; STAm=STA[m]
Q=[10,25,50,75,90]
def feats(ix):
 z=np.mean(P[:,ix]-np.median(P[:,ix],axis=1,keepdims=True),0)
 f=[]
 for v in [np.mean(P[:,ix],0), z, np.std(P[:,ix],0), SNR[ix]]:
  f+=list(np.percentile(v,Q))+[float(v.std())]
 f+=list(np.percentile(SP[ix],Q))
 f+=[len(ix)/200.0]
 return np.asarray(f,np.float32)
def build(groups, gname, nb_per_group, rng):
 F=[];T=[];GG=[]
 for g in gname:
  ids=np.where(groups==g)[0]
  if len(ids)<120: continue
  for _ in range(nb_per_group):
   lo=rng.uniform(3.8,5.4); hi=lo+rng.uniform(.6,2.2)
   cand=ids[(Ym[ids]>=lo)&(Ym[ids]<=hi)]
   if len(cand)<50: continue
   ix=rng.choice(cand,min(len(cand),rng.randint(50,201)),replace=False)
   F.append(feats(ix)); T.append(float(np.median(Ym[ix]))); GG.append(g)
 return np.asarray(F),np.asarray(T),np.asarray(GG)
rng=np.random.RandomState(20260816)
netc=collections.Counter(NETm); stac=collections.Counter(STAm)
top_net=[k for k,v in netc.most_common() if v>=400][:10]
top_sta=[k for k,v in stac.most_common() if v>=300][:10]
Fn,Tn,Gn=build(NETm,top_net,300,rng); Fs,Ts,Gs=build(STAm,top_sta,300,rng)
print('network batches',Fn.shape,len(set(Gn.tolist())),'station batches',Fs.shape,len(set(Gs.tolist())),flush=True)
MK={'gbm_v2ref':lambda:GradientBoostingRegressor(random_state=0,n_estimators=500,max_depth=3,learning_rate=.05,subsample=.8),
    'extra':lambda:ExtraTreesRegressor(n_estimators=400,min_samples_leaf=3,random_state=0,n_jobs=-1),
    'hgb':lambda:HistGradientBoostingRegressor(random_state=0,max_iter=400,learning_rate=.06),
    'ridge':lambda:RidgeCV(alphas=np.logspace(-3,3,25))}
def logo(F,T,G):
 out={}
 for name,mk in MK.items():
  pred=np.zeros(len(T)); 
  for g in sorted(set(G.tolist())):
   a=G!=g; b=G==g
   mu=F[a].mean(0); sd=F[a].std(0)+1e-6; mdl=mk(); mdl.fit((F[a]-mu)/sd,T[a]); pred[b]=mdl.predict((F[b]-mu)/sd)
  out[name]=dict(mae=float(np.abs(pred-T).mean()),r=float(pearsonr(pred,T)[0]),
                 worst_group_mae=float(max(np.abs(pred[G==g]-T[G==g]).mean() for g in set(G.tolist()))))
  print(' ',name,out[name],flush=True)
 return out
print('== leave-one-network-out =='); rn=logo(Fn,Tn,Gn)
print('== leave-one-station-out =='); rs=logo(Fs,Ts,Gs)
rep=dict(protocol='public STEAD only; real network/station groups; competition data untouched',
         n_records=int(len(m)),n_networks_used=top_net,n_stations_used=top_sta,
         lono=rn,loso=rs)
base=rn['gbm_v2ref']['mae']; cand={k:v for k,v in rn.items() if k!='gbm_v2ref'}
best=min(cand,key=lambda k:cand[k]['mae'])
accept=bool(cand[best]['mae']<base and rs[best]['mae']<=rs['gbm_v2ref']['mae'])
rep['decision']=dict(baseline_lono_mae=base,best_candidate=best,best_lono_mae=cand[best]['mae'],
                     candidate_loso_mae=rs[best]['mae'],baseline_loso_mae=rs['gbm_v2ref']['mae'],accept=accept)
json.dump(rep,open(f'{OUT}/station_domain.json','w'),indent=2)
print('DECISION',rep['decision'],flush=True)
if accept:
 mu=Fn.mean(0); sd=Fn.std(0)+1e-6; mdl=MK[best](); mdl.fit((Fn-mu)/sd,Tn)
 pickle.dump(dict(model=mdl,mu=mu,sd=sd,best=best,feature_layout='mean/z/std/snr percentiles + spec percentiles + size'),open(f'{OUT}/center_station.pkl','wb'))
 print('frozen candidate saved',best,flush=True)

