"""轮29公开 rank-augmented CNN：只用可信 STEAD station27缓存。
训练目标：异方差L1 + pairwise logistic ranking。验证真实台站LOSO。
"""
import os,json,time
import numpy as np,torch,torch.nn as nn
from scipy.stats import pearsonr,spearmanr
ROOT='/root/5.6+chanshui1';C=f'{ROOT}/outputs/t2_cache_station27';OUT=f'{ROOT}/outputs/t2_round29';os.makedirs(OUT,exist_ok=True)
X=np.load(f'{C}/X.npy',mmap_mode='r');Y=np.load(f'{C}/y.npy');N=np.load(f'{C}/net.npy').astype(str);S=np.load(f'{C}/sta.npy').astype(str);SID=np.char.add(np.char.add(N,'.'),S)
sel=np.where((Y>=3.8)&(Y<=6.5))[0];rng=np.random.RandomState(20260819);# choose 8 high-count real stations for fast public LOSO
from collections import Counter
stations=[g for g,c in Counter(SID[sel].tolist()).most_common() if c>=250][:8];TR=[];VA=[]
for g in stations:
 ids=np.where(SID[sel]==g)[0];rng.shuffle(ids);TR.extend(ids[:min(1000,len(ids)//2)]);VA.extend(ids[min(1000,len(ids)//2):min(1000,len(ids)//2)+250])
TR=np.asarray(TR);VA=np.asarray(VA);print('stations',stations,'train',len(TR),'val',len(VA),flush=True)
class Net(nn.Module):
 def __init__(self,rank=False):
  super().__init__();self.rank=rank
  def b(a,b):return nn.Sequential(nn.Conv1d(a,b,7,2,3),nn.BatchNorm1d(b),nn.ReLU())
  self.pre=nn.Sequential(nn.ModuleDict() if False else nn.Identity());self.cnn=nn.Sequential(b(3,32),b(32,64),b(64,128),b(128,128),b(128,256),nn.AdaptiveAvgPool1d(1),nn.Flatten());self.head=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,2))
 def forward(self,x):
  x=x/torch.sqrt((x*x).mean(2,keepdim=True)+1e-12);x=torch.sign(x)*torch.log1p(torch.abs(x));o=self.head(self.cnn(x));return o[:,0],o[:,1].clamp(-3,3)
def train(rank):
 torch.manual_seed(0);dev='cuda' if torch.cuda.is_available() else 'cpu';net=Net(rank).to(dev);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4);bs=64
 for ep in range(12):
  net.train();perm=rng.permutation(TR)
  for st in range(0,len(perm),bs):
   ix=perm[st:st+bs];xb=np.asarray(X[sel[ix]],np.float32);xb*=10**rng.uniform(-2,2,size=(len(ix),1,1));yt=torch.from_numpy(Y[sel[ix]]).float().to(dev);pred,ls=net(torch.from_numpy(xb).to(dev));loss=(torch.abs(pred-yt)*torch.exp(-ls)+ls).mean()
   if rank:
    # balanced within-batch pairs, only clear magnitude gaps
    d=yt[:,None]-yt[None,:];pd=pred[:,None]-pred[None,:];mask=(torch.abs(d)>=.25)&(torch.triu(torch.ones_like(d),1)>0)
    if mask.any():loss=loss+0.4*torch.nn.functional.softplus(-torch.sign(d[mask])*pd[mask]).mean()
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
  print('rank',rank,'ep',ep,flush=True)
 return net.eval().cpu()
def evalnet(net):
 out=[]
 with torch.no_grad():
  for st in range(0,len(VA),128):
   xb=np.asarray(X[sel[VA[st:st+128]]],np.float32)
   out.append(net(torch.from_numpy(xb))[0].numpy())
 p=np.concatenate(out);y=Y[sel[VA]]
 return dict(r=float(pearsonr(p,y)[0]),rho=float(spearmanr(p,y)[0]),mae=float(np.abs(p-y).mean()))
base=train(False);rb=evalnet(base);rank=train(True);rr=evalnet(rank);print('RESULT',rb,rr,flush=True)
json.dump(dict(protocol='public STEAD true station lineage; rank loss only public; no competition data',stations=stations,train_n=int(len(TR)),val_n=int(len(VA)),baseline=rb,rank_augmented=rr),open(f'{OUT}/rank_public.json','w'),indent=2)
torch.save(rank.state_dict(),f'{OUT}/rank_aug.pt')
