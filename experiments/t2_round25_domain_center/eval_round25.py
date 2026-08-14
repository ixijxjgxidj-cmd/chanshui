"""轮25冻结后的R1/R2 train-only 外部报告。禁止读取holdout与08。"""
import os,json,pickle,sys,numpy as np,torch,torch.nn as nn
from scipy import signal as ss
from scipy.stats import pearsonr
sys.path.insert(0,'/root/5.6+chanshui1'); from obspy import read
from t2_io import to_enz_from_obspy
from t2_features import score200
ROOT='/root/5.6+chanshui1'; OUT=f'{ROOT}/outputs/t2_round25'; dev='cuda' if torch.cuda.is_available() else 'cpu'
class RmsNorm(nn.Module):
 def forward(self,x): return x/torch.sqrt((x*x).mean(2,keepdim=True)+1e-12)
class LogAmp(nn.Module):
 def forward(self,x): return torch.sign(x)*torch.log1p(torch.abs(x))
BANDS=[(.05,.2),(.2,.5),(.5,1),(1,2),(2,5),(5,10),(10,20),(20,45)]
def specf(x):
 x=x-x.mean(2,keepdim=True); F=torch.fft.rfft(x*torch.hann_window(x.shape[2],device=x.device),dim=2); P=F.real**2+F.imag**2; fr=torch.fft.rfftfreq(x.shape[2],d=.01).to(x.device); P=P/(P.sum(2,keepdim=True)+1e-20); return torch.cat([torch.log10(P[:,:,(fr>=a)&(fr<b)].sum(2)+1e-12) for a,b in BANDS],1)
class Net(nn.Module):
 def __init__(self,s):
  super().__init__(); self.s=s
  def b(a,b):return nn.Sequential(nn.Conv1d(a,b,7,2,3),nn.BatchNorm1d(b),nn.ReLU())
  self.pre=nn.Sequential(RmsNorm(),LogAmp());self.cnn=nn.Sequential(b(3,32),b(32,64),b(64,128),b(128,128),b(128,256),nn.AdaptiveAvgPool1d(1),nn.Flatten()); ex=(1 if s['snr'] else 0)+(24 if s['spec'] else 0);self.head=nn.Sequential(nn.Linear(256+ex,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,3 if s['mt'] else 2))
 def forward(self,x,sn):
  z=[self.cnn(self.pre(x))]
  if self.s['snr']:z.append(sn[:,None]/40)
  if self.s['spec']:z.append(specf(x))
  return self.head(torch.cat(z,1))[:,0]
def snr(x):
 h=x.shape[2]//2;return 20*torch.log10(torch.sqrt((x[:,:,:h]**2).mean((1,2))+1e-12)/torch.sqrt((x[:,:,h:]**2).mean((1,2))+1e-12))
def aic(x):
 k=np.arange(1,len(x)-1);v1=np.array([x[:i].var() for i in k]);v2=np.array([x[i:].var() for i in k]);v=k*np.log(np.maximum(v1,1e-30))+(len(x)-k-1)*np.log(np.maximum(v2,1e-30));return int(k[np.nanargmin(v)])
def window(d,f):
 w,sr=to_enz_from_obspy(read(os.path.join(ROOT,d,f)));w-=w.mean(1,keepdims=True); env=np.abs(ss.hilbert(w,axis=1)).max(0);pk=int(np.convolve(env,np.ones(25)/25,'same').argmax());lo=max(0,pk-int(45*sr));on=lo+aic(w[2,lo:pk+1]) if pk-lo>200 else pk;s0,e0=int(on-5*sr),int(on+5*sr);seg=np.pad(w[:,:max(1,e0)],((0,0),(-s0,0)),mode='reflect') if s0<0 else w[:,s0:e0];seg=np.pad(seg,((0,0),(0,max(0,int(10*sr)-seg.shape[1]))),mode='reflect');return (seg[:,:int(10*sr)]-seg[:,:int(10*sr)].mean(1,keepdims=True)).astype('float32')
M=[]
for nm in ['M1','M2','M3','M4','M5','M6','M7']:
 ck=torch.load(f'{ROOT}/outputs/t2_ens/{nm}.pt',map_location='cpu',weights_only=False);n=Net(ck['spec']);n.load_state_dict(ck['state_dict']);n.eval().to(dev);M.append((n,ck['spec']))
obj=pickle.load(open(f'{OUT}/center_lodo.pkl','rb'))
sp=json.load(open(f'{ROOT}/outputs/t2_prereg_split.json'))['splits']
def run(key,d,ans,strip=False):
 tr=sp[key]['train'];ho=set(sp[key]['holdout']);assert not(set(tr)&ho), 'holdout intersection'; assert d in ('r1_t2','r2_t2') and ans in ('r1_t2/answers.txt','r2_t2/T2.an'), 'unapproved input path'
 aa={}
 for ln in open(f'{ROOT}/{ans}'):
  z=ln.split();
  if z:aa[os.path.basename(z[0]) if strip else z[0]]=float(z[1])
 X=np.stack([window(d,f) for f in tr]);y=np.asarray([aa[f] for f in tr],np.float32);P=[]
 for st in range(0,len(X),64):
  xb=torch.from_numpy(X[st:st+64]).to(dev);pp=[]
  with torch.no_grad():
   for n,s in M:
    xx=torch.roll(xb,s['shift'],2) if s['shift'] else xb;pp.append(n(xx,snr(xx)))
  P.append(torch.stack(pp).cpu().numpy())
 P=np.concatenate(P,1);q=[10,25,50,75,90];z=np.mean(P-np.median(P,axis=1,keepdims=True),0);F=[]
 for v in [np.mean(P,0),z,np.std(P,0)]:F+=list(np.percentile(v,q))+[float(v.std())]
 F+=[len(y)/200]; c=float(obj['model'].predict((np.asarray(F)[None,:]-obj['mu'])/obj['sd'])[0]); pr=np.clip(c+.4*z,0,9.9)
 return dict(n=len(y),holdout_count=len(ho),truth_center=float(np.median(y)),estimated_center=c,center_error=c-float(np.median(y)),const_score=score200(np.full_like(y,c),y),rank_r=float(pearsonr(z,y)[0]),linear040_score=score200(pr,y))
r={'protocol':'frozen public LODO ExtraTrees; R1/R2 pre-registered train only; holdouts and 08 not read','R1':run('R1','r1_t2','r1_t2/answers.txt'),'R2':run('R2','r2_t2','r2_t2/T2.an',True)}
json.dump(r,open(f'{OUT}/external_trainonly.json','w'),indent=2);print(json.dumps(r,indent=2))
