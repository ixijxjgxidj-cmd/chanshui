#!/usr/bin/env python3
"""T3 SeismicXM 特征上的平衡 MLP + mixup 重复CV实验。"""
import argparse,json,random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class TinyMLP(nn.Module):
    def __init__(self,d,n=5):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,256),nn.GELU(),nn.Dropout(.25),nn.Linear(256,n))
    def forward(self,x): return self.net(x)

def mixup_batch(x,y,n,alpha,rng):
    lam=float(rng.beta(alpha,alpha)); idx=rng.permutation(len(x)); one=np.eye(n,dtype=np.float32)[y]
    return lam*x+(1-lam)*x[idx],lam*one+(1-lam)*one[idx]

def train_predict(Xtr,ytr,Xev,seed,epochs=80):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);dev='cuda' if torch.cuda.is_available() else 'cpu';mean=Xtr.mean(0);sd=Xtr.std(0)+1e-6;xt=((Xtr-mean)/sd).astype(np.float32);xe=((Xev-mean)/sd).astype(np.float32)
    m=TinyMLP(xt.shape[1]).to(dev); counts=np.bincount(ytr,minlength=5); w=torch.tensor(len(ytr)/(5*np.maximum(counts,1)),dtype=torch.float32,device=dev);opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=2e-3);rng=np.random.default_rng(seed)
    for _ in range(epochs):
        xx,tt=mixup_batch(xt,ytr,5,.4,rng);order=rng.permutation(len(xx));m.train()
        for j in range(0,len(order),32):
            ix=order[j:j+32];logits=m(torch.from_numpy(xx[ix]).to(dev));target=torch.from_numpy(tt[ix]).to(dev);loss=-(target*F.log_softmax(logits,1)*w.unsqueeze(0)).sum(1).mean();opt.zero_grad();loss.backward();opt.step()
    m.eval()
    with torch.no_grad():return m(torch.from_numpy(xe).to(dev)).argmax(1).cpu().numpy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--output',required=True);ap.add_argument('--epochs',type=int,default=80);a=ap.parse_args();z=np.load(a.features);X=np.vstack([z['X1t3'],z['X2t3']]).astype(np.float32);y=np.r_[z['y1t3'],z['y2t3']].astype(int)-1;cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=20260816);mp=[];lp=[];ys=[]
    for fold,(tr,te) in enumerate(cv.split(X,y)):
        mp.append(train_predict(X[tr],y[tr],X[te],20260816+fold,a.epochs));cw=np.bincount(y[tr],minlength=5);sw=len(tr)/(5*np.maximum(cw[y[tr]],1));lm=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'));lm.fit(X[tr],y[tr],logisticregression__sample_weight=sw);lp.append(lm.predict(X[te]));ys.append(y[te])
    yy=np.r_[*ys];pp=np.r_[*mp];base=np.r_[*lp];out={'experiment':'t3-balanced-mlp-20260816','protocol':'5x5 repeated stratified CV','n':int(len(yy)),'accuracy':float(accuracy_score(yy,pp)),'balanced_accuracy':float(balanced_accuracy_score(yy,pp)),'weighted_logreg_baseline':{'accuracy':float(accuracy_score(yy,base)),'balanced_accuracy':float(balanced_accuracy_score(yy,base))},'device':'cuda' if torch.cuda.is_available() else 'cpu','epochs':a.epochs};json.dump(out,open(a.output,'w'),indent=2);print(json.dumps(out,indent=2))
if __name__=='__main__':main()