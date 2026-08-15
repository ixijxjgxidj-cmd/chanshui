#!/usr/bin/env python3
"""T3 训练折内 CORAL/whitening 对齐对照。"""
import argparse,json
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score

class CORALAligner:
    def __init__(self,reg=1e-2): self.reg=reg
    def fit(self,x):
        x=np.asarray(x,float);self.mean_=x.mean(0);v=x.var(0)+1e-6;self.scale_=np.sqrt((1-self.reg)*v+self.reg*np.median(v));return self
    def transform(self,x):
        x=np.asarray(x,float)
        if x.shape[1]!=len(self.mean_):raise ValueError('dimension mismatch')
        return (x-self.mean_)/self.scale_

def met(y,p):return {'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();z=np.load(a.features);X=np.vstack([z['X1t3'],z['X2t3']]).astype(float);y=np.r_[z['y1t3'],z['y2t3']].astype(int)-1;cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=20260816);out={'experiment':'t3-coral-20260816','protocol':'fit alignment inside training fold only','grid':{}}
 for reg in [0.0,.01,.1,.5,1.0]:
  ps=[];ys=[]
  for tr,te in cv.split(X,y):
   al=CORALAligner(reg).fit(X[tr]);xa=al.transform(X[tr]);xe=al.transform(X[te]);cw=np.bincount(y[tr],minlength=5);sw=len(tr)/(5*np.maximum(cw[y[tr]],1));m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'));m.fit(xa,y[tr],logisticregression__sample_weight=sw);ps.append(m.predict(xe));ys.append(y[te])
  yy=np.r_[*ys];pp=np.r_[*ps];out['grid'][str(reg)]={**met(yy,pp),'n':len(yy)}
 json.dump(out,open(a.output,'w'),indent=2);print(json.dumps(out,indent=2))
if __name__=='__main__':main()