"""轮27：真实台站留出下的分层/收缩批中心估计。
数据仅来自公开缓存 outputs/t2_cache_station27（含逐样本 network.station 血缘）。
预注册准则：
 主门槛 = 留一台站(LOSO) 平均 MAE 必须优于全局 GBM 基线
 次门槛 = LOSO 最差台站 MAE 不得恶化
两者同时满足才冻结候选；否则保留现有 GBM。
合规：不读取 R1/R2/08。
"""
import os,json,time,pickle,collections
import numpy as np,torch,torch.nn as nn
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV,HuberRegressor
ROOT='/root/5.6+chanshui1';C=f'{ROOT}/outputs/t2_cache_station27';OUT=f'{ROOT}/outputs/t2_round27';os.makedirs(OUT,exist_ok=True)
X=np.load(f'{C}/X.npy',mmap_mode='r');Y=np.load(f'{C}/y.npy');NET=np.load(f'{C}/net.npy').astype(str);STA=np.load(f'{C}/sta.npy').astype(str)
SID=np.char.add(np.char.add(NET,'.'),STA)
sel=np.where((Y>=3.8)&(Y<=6.5))[0];print('records',len(sel),'stations',len(set(SID[sel].tolist())),flush=True)
class RmsNorm(nn.Module):
 def forward(self,x):return x/torch.sqrt((x*x).mean(2,keepdim=True)+1e-12)
class LogAmp(nn.Module):
 def forward(self,x):return torch.sign(x)*torch.log1p(torch.abs(x))
B=[(.05,.2),(.2,.5),(.5,1),(1,2),(2,5),(5,10),(10,20),(20,45)]
def spec(x):
 x=x-x.mean(2,keepdim=True);F=torch.fft.rfft(x*torch.hann_window(x.shape[2],device=x.device),dim=2);P=F.real**2+F.imag**2
 fr=torch.fft.rfftfreq(x.shape[2],d=.01).to(x.device);P=P/(P.sum(2,keepdim=True)+1e-20)
 return torch.cat([torch.log10(P[:,:,(fr>=a)&(fr<b)].sum(2)+1e-12) for a,b in B],1)
class Net(nn.Module):
 def __init__(s_,s):
  super().__init__();s_.s=s
  def bl(a,b):return nn.Sequential(nn.Conv1d(a,b,7,2,3),nn.BatchNorm1d(b),nn.ReLU())
  s_.pre=nn.Sequential(RmsNorm(),LogAmp());s_.cnn=nn.Sequential(bl(3,32),bl(32,64),bl(64,128),bl(128,128),bl(128,256),nn.AdaptiveAvgPool1d(1),nn.Flatten())
  ex=(1 if s['snr'] else 0)+(24 if s['spec'] else 0);s_.head=nn.Sequential(nn.Linear(256+ex,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,3 if s['mt'] else 2))
 def forward(s_,x,sn):
  z=[s_.cnn(s_.pre(x))]
  if s_.s['snr']:z.append(sn[:,None]/40)
  if s_.s['spec']:z.append(spec(x))
  return s_.head(torch.cat(z,1))
def snr(x):
 h=x.shape[2]//2;return 20*torch.log10(torch.sqrt((x[:,:,h:]**2).mean((1,2))+1e-12)/torch.sqrt((x[:,:,:h]**2).mean((1,2))+1e-12))
dev='cuda' if torch.cuda.is_available() else 'cpu';MEM=[]
for nm in ['M1','M2','M3','M4','M5','M6','M7']:
 ck=torch.load(f'{ROOT}/outputs/t2_ens/{nm}.pt',map_location='cpu',weights_only=False);n=Net(ck['spec']);n.load_state_dict(ck['state_dict']);n.eval().to(dev);MEM.append((n,ck['spec']))
P=[];SN=[];SPm=[];t0=time.time()
for st in range(0,len(sel),512):
 xb=torch.from_numpy(np.asarray(X[sel[st:st+512]],np.float32)).to(dev);pp=[]
 with torch.no_grad():
  for n,s in MEM:
   xx=torch.roll(xb,s['shift'],2) if s['shift'] else xb;pp.append(n(xx,snr(xx))[:,0])
  SPm.append(spec(xb).mean(1).cpu().numpy());SN.append(snr(xb).cpu().numpy())
 P.append(torch.stack(pp).cpu().numpy())
P=np.concatenate(P,1);SN=np.concatenate(SN);SPm=np.concatenate(SPm)
print('inference',P.shape,'%.0fs'%(time.time()-t0),flush=True)
Ym=Y[sel];SIDm=SID[sel]
Q=[10,25,50,75,90]
def feats(ix):
 z=np.mean(P[:,ix]-np.median(P[:,ix],axis=1,keepdims=True),0);f=[]
 for v in [np.mean(P[:,ix],0),z,np.std(P[:,ix],0),SN[ix]]:f+=list(np.percentile(v,Q))+[float(v.std())]
 f+=list(np.percentile(SPm[ix],Q));f+=[len(ix)/200.0]
 return np.asarray(f,np.float32)
cnt=collections.Counter(SIDm.tolist())
stations=[k for k,v in cnt.most_common() if v>=250][:14]
print('stations used',len(stations),stations,flush=True)
rng=np.random.RandomState(20260817);F=[];T=[];G=[]
for s in stations:
 ids=np.where(SIDm==s)[0]
 for _ in range(320):
  lo=rng.uniform(3.8,5.4);hi=lo+rng.uniform(.6,2.2);cand=ids[(Ym[ids]>=lo)&(Ym[ids]<=hi)]
  if len(cand)<50:continue
  ix=rng.choice(cand,min(len(cand),rng.randint(50,201)),replace=False)
  F.append(feats(ix));T.append(float(np.median(Ym[ix])));G.append(s)
F=np.asarray(F);T=np.asarray(T);G=np.asarray(G)
print('batches',F.shape,'groups',len(set(G.tolist())),flush=True)
def loso(fit_predict):
 pred=np.zeros(len(T))
 for g in sorted(set(G.tolist())):
  a=G!=g;b=G==g;pred[b]=fit_predict(F[a],T[a],G[a],F[b])
 per={g:float(np.abs(pred[G==g]-T[G==g]).mean()) for g in sorted(set(G.tolist()))}
 return dict(mae=float(np.abs(pred-T).mean()),r=float(pearsonr(pred,T)[0]),worst=float(max(per.values())),per_station=per)
def gbm(Fa,Ta,Ga,Fb):
 mu=Fa.mean(0);sd=Fa.std(0)+1e-6;m=GradientBoostingRegressor(random_state=0,n_estimators=500,max_depth=3,learning_rate=.05,subsample=.8);m.fit((Fa-mu)/sd,Ta);return m.predict((Fb-mu)/sd)
def gbm_shrunk(Fa,Ta,Ga,Fb):
 """全局 GBM + 台站随机效应经验贝叶斯收缩：对未见台站，效应收缩到 0。"""
 mu=Fa.mean(0);sd=Fa.std(0)+1e-6;m=GradientBoostingRegressor(random_state=0,n_estimators=500,max_depth=3,learning_rate=.05,subsample=.8);m.fit((Fa-mu)/sd,Ta)
 res=Ta-m.predict((Fa-mu)/sd)
 gs=sorted(set(Ga.tolist()));bg=np.array([res[Ga==g].mean() for g in gs]);ng=np.array([(Ga==g).sum() for g in gs])
 within=np.mean([res[Ga==g].var(ddof=1) for g in gs]);between=max(bg.var(ddof=1)-within/np.mean(ng),0.0)
 # 未见台站的后验均值 = 总体均值（0），故只做全局截距校正
 corr=float(np.average(bg,weights=ng)*(between/(between+within/np.mean(ng)+1e-12)))
 return m.predict((Fb-mu)/sd)+corr
def huber(Fa,Ta,Ga,Fb):
 mu=Fa.mean(0);sd=Fa.std(0)+1e-6;m=HuberRegressor(epsilon=1.35,alpha=1e-4,max_iter=800);m.fit((Fa-mu)/sd,Ta);return m.predict((Fb-mu)/sd)
def ridge_grp(Fa,Ta,Ga,Fb):
 mu=Fa.mean(0);sd=Fa.std(0)+1e-6;m=RidgeCV(alphas=np.logspace(-3,3,25));m.fit((Fa-mu)/sd,Ta);return m.predict((Fb-mu)/sd)
def gbm_grpmean(Fa,Ta,Ga,Fb):
 """分层两阶段：先用组均值去中心再拟合，预测时加回总体均值。"""
 mu=Fa.mean(0);sd=Fa.std(0)+1e-6;gs=sorted(set(Ga.tolist()));gm={g:Ta[Ga==g].mean() for g in gs};overall=float(np.mean(list(gm.values())))
 Tc=Ta-np.array([gm[g] for g in Ga]);m=GradientBoostingRegressor(random_state=0,n_estimators=500,max_depth=3,learning_rate=.05,subsample=.8);m.fit((Fa-mu)/sd,Tc)
 return m.predict((Fb-mu)/sd)+overall
rep={}
for name,fn in [('gbm_global',gbm),('gbm_shrunk',gbm_shrunk),('gbm_group_centered',gbm_grpmean),('huber',huber),('ridge',ridge_grp)]:
 t=time.time();rep[name]=loso(fn);print(name,'mae=%.5f r=%.4f worst=%.5f (%.0fs)'%(rep[name]['mae'],rep[name]['r'],rep[name]['worst'],time.time()-t),flush=True)
base=rep['gbm_global'];cands={k:v for k,v in rep.items() if k!='gbm_global'}
best=min(cands,key=lambda k:cands[k]['mae'])
accept=bool(cands[best]['mae']<base['mae'] and cands[best]['worst']<=base['worst'])
out=dict(protocol='public STEAD with true per-record station lineage; LOSO over >=250-record stations; competition data untouched',
         n_records=int(len(sel)),stations=stations,n_batches=int(len(T)),report=rep,
         decision=dict(baseline_mae=base['mae'],baseline_worst=base['worst'],best=best,best_mae=cands[best]['mae'],best_worst=cands[best]['worst'],accept=accept))
json.dump(out,open(f'{OUT}/station_shrinkage.json','w'),indent=2)
print('DECISION',out['decision'],flush=True)
