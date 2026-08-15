#!/usr/bin/env python3
"""T3 联合 R1/R2 轮次+类别加权对照（仅训练特征）。"""
import argparse,json,time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score

def met(y,p): return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),"n":len(y)}
def fit_predict(kind,Xtr,ytr,Xev,weights=None):
 if kind=='logreg':
  m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
 else: m=make_pipeline(Normalizer(),KNeighborsClassifier(5,metric='cosine',weights='distance'))
 # sample weights pass through pipeline only for final estimator
 if weights is None: m.fit(Xtr,ytr)
 else: m.fit(Xtr,ytr,logisticregression__sample_weight=weights)
 return m.predict(Xev)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--output',required=True); a=ap.parse_args(); z=np.load(a.features); X1=z['X1t3'].astype(float);y1=z['y1t3'].astype(int);X2=z['X2t3'].astype(float);y2=z['y2t3'].astype(int);X=np.vstack([X1,X2]);y=np.r_[y1,y2]; dom=np.r_[np.zeros(len(y1),int),np.ones(len(y2),int)]
 # equalize round mass and inverse class frequency within each round
 counts={(d,c):max(1,int(np.sum((dom==d)&(y==c)))) for d in [0,1] for c in np.unique(y)}
 w=np.array([1/(2*counts[(d,c)]) for d,c in zip(dom,y)],float); w*=len(w)/w.sum()
 out={'experiment':'t3-weighted-joint-20260816','shapes':{'r1':list(X1.shape),'r2':list(X2.shape)},'weights':{f'{d}_{c}':v for (d,c),v in counts.items()}}
 for kind in ['logreg','knn']:
  out[kind]={}
  for name,Xtr,ytr,Xev,yev in [('joint_to_r1',X,y,X1,y1),('joint_to_r2',X,y,X2,y2)]:
   p=fit_predict(kind,Xtr,ytr,Xev,w if kind=='logreg' else None);out[kind][name]=met(yev,p)
  sk=StratifiedKFold(5,shuffle=True,random_state=20260816); ys=[];ps=[]
  for tr,te in sk.split(X,y):
   p=fit_predict(kind,X[tr],y[tr],X[te],w[tr] if kind=='logreg' else None);ys.append(y[te]);ps.append(p)
  out[kind]['joint_cv']=met(np.r_[*ys],np.r_[*ps])
 with open(a.output,'w') as f:json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()