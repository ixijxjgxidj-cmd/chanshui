#!/usr/bin/env python3
"""T2 训练侧：按7成员分歧自适应收缩，严格5折、无holdout。"""
import argparse,json
import numpy as np
from sklearn.model_selection import KFold

def score(y,p): return float(200*np.mean(np.exp(-np.abs(np.asarray(p)-y))))
def eval_round(M,y,base_slope=.85,alpha=.25,seed=20260816):
    n=M.shape[1]; rows=[]; kf=KFold(5,shuffle=True,random_state=seed)
    for fi,(tr,te) in enumerate(kf.split(np.arange(n))):
        train_center=float(np.median(y[tr]))
        P=M[:,te].mean(0); med=float(np.median(P)); D=M[:,te].std(0)
        scale=float(np.median(D))
        u=D/(scale+1e-8)
        for gamma in [0.0,.15,.30,.45,.60,.75]:
            s=np.clip(base_slope*(1-gamma*np.clip(u-1,0,2)),.10,base_slope)
            c=(1-alpha)*med+alpha*train_center
            pred=np.clip(c+s*(P-med),0,9.9)
            rows.append({'fold':fi,'gamma':gamma,'score':score(y[te],pred),'mae':float(np.mean(np.abs(pred-y[te]))),'mean_slope':float(np.mean(s))})
    agg={}
    for g in [0.0,.15,.30,.45,.60,.75]:
        a=[r for r in rows if r['gamma']==g]
        agg[str(g)]={'score_mean':float(np.mean([x['score'] for x in a])),'mae_mean':float(np.mean([x['mae'] for x in a])),'mean_slope':float(np.mean([x['mean_slope'] for x in a]))}
    return agg

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--r1',required=True); ap.add_argument('--r2',required=True); ap.add_argument('--y1',required=True); ap.add_argument('--y2',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    M1=np.load(a.r1)['m1']; M2=np.load(a.r2)['m2']; y1=np.load(a.y1)['y']; y2=np.load(a.y2)['y']
    out={'experiment':'t2-disagreement-shrink-20260816','protocol':'5-fold train-side OOF only','R1':{},'R2':{}}
    for name,M,y,sl in [('R1',M1,y1,.85),('R2',M2,y2,.50)]:
        for alpha in [.25,.50,.75]: out[name][str(alpha)]=eval_round(M,y,sl,alpha)
    json.dump(out,open(a.output,'w'),indent=2); print(json.dumps(out,indent=2))
if __name__=='__main__': main()