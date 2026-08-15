#!/usr/bin/env python3
"""T3 嵌套概率融合：加权LogReg + 平衡MLP，外层严格评估。"""
import argparse,json,random
import numpy as np,torch
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold,StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score

def fuse_predictions(a,b,alpha):
 p=(1-alpha)*np.asarray(a,float)+alpha*np.asarray(b,float);return p/np.maximum(p.sum(1,keepdims=True),1e-12)
class Tiny(nn.Module):
 def __init__(self,d): super().__init__();self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,5))
 def forward(self,x):return self.net(x)
def mlp_fit_predict(Xtr,ytr,Xev,seed,epochs=50,return_prob=True):
 torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);dev='cuda' if torch.cuda.is_available() else 'cpu';mu=Xtr.mean(0);sd=Xtr.std(0)+1e-6;tr=((Xtr-mu)/sd).astype('float32');ev=((Xev-mu)/sd).astype('float32');m=Tiny(tr.shape[1]).to(dev);c=np.bincount(ytr,minlength=5);w=torch.tensor(len(ytr)/(5*np.maximum(c,1)),dtype=torch.float32,device=dev);opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=2e-3);x=torch.from_numpy(tr).to(dev);y=torch.from_numpy(ytr).to(dev)
 for _ in range(epochs):
  m.train();o=m(x);loss=nn.functional.cross_entropy(o,y,weight=w);opt.zero_grad();loss.backward();opt.step()
 m.eval();
 with torch.no_grad():p=torch.softmax(m(torch.from_numpy(ev).to(dev)),1).cpu().numpy()
 return p
def logreg_probs(Xtr,ytr,Xev):
 c=np.bincount(ytr,minlength=5);sw=len(ytr)/(5*np.maximum(c[ytr],1));m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'));m.fit(Xtr,ytr,logisticregression__sample_weight=sw);return m.predict_proba(Xev)
def met(y,p):return {'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--output',required=True);ap.add_argument('--epochs',type=int,default=50);a=ap.parse_args();z=np.load(a.features);X=np.vstack([z['X1t3'],z['X2t3']]).astype('float32');y=np.r_[z['y1t3'],z['y2t3']].astype(int)-1;outer=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=20260816); rows=[]
 for fi,(tr,te) in enumerate(outer.split(X,y)):
  inner=StratifiedKFold(3,shuffle=True,random_state=700+fi); a1=[];b1=[];yi=[]
  for it,iv in inner.split(X[tr],y[tr]):
   a1.append(logreg_probs(X[tr][it],y[tr][it],X[tr][iv]));b1.append(mlp_fit_predict(X[tr][it],y[tr][it],X[tr][iv],9000+fi));yi.append(y[tr][iv])
  pa=np.vstack(a1);pb=np.vstack(b1);yy=np.concatenate(yi);best=max(({'alpha':al,'score':accuracy_score(yy,fuse_predictions(pa,pb,al).argmax(1))} for al in np.linspace(0,1,11)),key=lambda q:q['score']);pa=logreg_probs(X[tr],y[tr],X[te]);pb=mlp_fit_predict(X[tr],y[tr],X[te],12000+fi,a.epochs);pf=fuse_predictions(pa,pb,best['alpha']);rows.append({'alpha':best['alpha'],'logreg':met(y[te],pa.argmax(1)),'mlp':met(y[te],pb.argmax(1)),'fusion':met(y[te],pf.argmax(1))})
 out={'experiment':'t3-nested-fusion-20260816','protocol':'3x5 outer CV, 3-fold inner alpha selection','folds':rows};json.dump(out,open(a.output,'w'),indent=2);print(json.dumps(out,indent=2))
if __name__=='__main__':main()