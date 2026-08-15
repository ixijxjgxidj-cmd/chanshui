#!/usr/bin/env python3
"""T3 训练折内少数类特征扰动扩增，严格重复分层CV。"""
import argparse,json
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score

def augment_minority(x,y,per_class,noise_scale,rng):
    xs=[x];ys=[y]; global_std=x.std(axis=0,keepdims=True)+1e-8
    for c in np.unique(y):
        idx=np.flatnonzero(y==c); need=max(0,per_class-len(idx))
        if need:
            chosen=rng.choice(idx,need,replace=True); xs.append(x[chosen]+rng.normal(0,noise_scale,size=(need,x.shape[1]))*global_std);ys.append(np.full(need,c,dtype=y.dtype))
    return np.vstack(xs),np.concatenate(ys)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seed',type=int,default=20260816);a=ap.parse_args();z=np.load(a.features);X=np.vstack([z['X1t3'],z['X2t3']]).astype(float);y=np.r_[z['y1t3'],z['y2t3']].astype(int)-1;cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=a.seed);out={'experiment':'t3-feature-augmentation-20260816','protocol':'augmentation occurs only inside each training fold','grid':{}}
    for pc in [20,30,40]:
      for ns in [.01,.03,.05]:
        ps=[];ys=[]
        for fi,(tr,te) in enumerate(cv.split(X,y)):
          xa,ya=augment_minority(X[tr],y[tr],pc,ns,np.random.default_rng(a.seed+fi));m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'));m.fit(xa,ya);ps.append(m.predict(X[te]));ys.append(y[te])
        yy=np.r_[*ys];pp=np.r_[*ps];out['grid'][f'pc{pc}_ns{ns}']={'accuracy':float(accuracy_score(yy,pp)),'balanced_accuracy':float(balanced_accuracy_score(yy,pp)),'n':int(len(yy))}
    json.dump(out,open(a.output,'w'),indent=2);print(json.dumps(out,indent=2))
if __name__=='__main__':main()