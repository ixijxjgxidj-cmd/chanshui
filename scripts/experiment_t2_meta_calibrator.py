#!/usr/bin/env python3
"""T2 训练侧元校准器：成员统计到震级的稳健映射，严格5折。"""
import argparse,json
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import Ridge,HuberRegressor

def score(y,p): return float(200*np.mean(np.exp(-np.abs(np.asarray(p)-y))))
def feats(M):
    return np.column_stack([M.mean(0),np.median(M,0),M.std(0),M.min(0),M.max(0),np.percentile(M,25,axis=0),np.percentile(M,75,axis=0)])
def run(M,y,seed=20260816):
    X=feats(M); rows=[]; kf=KFold(5,shuffle=True,random_state=seed)
    for tr,te in kf.split(X):
        for name,model in [('ridge',make_pipeline(StandardScaler(),Ridge(alpha=1.0))),('ridge10',make_pipeline(StandardScaler(),Ridge(alpha=10.0))),('huber',make_pipeline(StandardScaler(),HuberRegressor(epsilon=1.35,alpha=0.001,max_iter=2000)))]:
            model.fit(X[tr],y[tr]); p=np.clip(model.predict(X[te]),0,9.9); rows.append((name,score(y[te],p),float(np.mean(np.abs(p-y[te])))))
    out={}
    for name in ['ridge','ridge10','huber']:
        a=[r for r in rows if r[0]==name]; out[name]={'score_mean':float(np.mean([r[1] for r in a])),'mae_mean':float(np.mean([r[2] for r in a]))}
    # non-calibrated member mean baseline with identical folds
    base=[]
    for _,te in kf.split(X): base += [score(y[te],M.mean(0)[te])]
    out['baseline_mean']={'score_mean':float(np.mean(base)),'mae_mean':float(np.mean(np.abs(M.mean(0)-y)))}
    return out

def cross(Mtr,ytr,Mev,yev):
    Xtr=feats(Mtr); Xev=feats(Mev); out={}
    for name,model in [('ridge',make_pipeline(StandardScaler(),Ridge(alpha=1.0))),('ridge10',make_pipeline(StandardScaler(),Ridge(alpha=10.0))),('huber',make_pipeline(StandardScaler(),HuberRegressor(epsilon=1.35,alpha=0.001,max_iter=2000)))]:
        model.fit(Xtr,ytr); p=np.clip(model.predict(Xev),0,9.9); out[name]={'score':score(yev,p),'mae':float(np.mean(np.abs(p-yev)))}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--npz',required=True); ap.add_argument('--labels1',required=True); ap.add_argument('--labels2',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    z=np.load(a.npz); y1=np.load(a.labels1)['y']; y2=np.load(a.labels2)['y']
    out={'experiment':'t2-meta-calibrator-20260816','protocol':'5-fold train-side OOF only','within_R1':run(z['m1'],y1),'within_R2':run(z['m2'],y2),'cross_R1_to_R2':cross(z['m1'],y1,z['m2'],y2),'cross_R2_to_R1':cross(z['m2'],y2,z['m1'],y1)}
    json.dump(out,open(a.output,'w'),indent=2); print(json.dumps(out,indent=2))
if __name__=='__main__': main()