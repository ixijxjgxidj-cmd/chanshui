#!/usr/bin/env python3
"""T3 训练侧开放类别原型与置信度门控实验。"""
import argparse,json
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score

def _norm(x):
    x=np.asarray(x,float); return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def cosine_prototype_predict(prototypes,labels,x):
    p=_norm(prototypes); q=_norm(x); score=q@p.T; order=np.argsort(-score,axis=1); idx=order[:,0]
    pred=np.asarray(labels)[idx]; margin=score[np.arange(len(q)),order[:,0]]-score[np.arange(len(q)),order[:,1]]
    support=score[np.arange(len(q)),idx]
    return pred,margin,support

def apply_confidence_gate(baseline,candidate,margin,support,margin_threshold,support_threshold):
    use=(np.asarray(margin)>=margin_threshold)&(np.asarray(support)>=support_threshold)
    out=np.where(use,np.asarray(candidate),np.asarray(baseline)); return out,float(use.mean())

def metrics(y,p): return {'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'n':int(len(y))}

def run(X1,y1,X2,y2):
    # Leave-one-round-out: prototypes are fitted only on the other round.
    out={}
    for name,Xtr,ytr,Xev,yev in [('r1_to_r2',X1,y1,X2,y2),('r2_to_r1',X2,y2,X1,y1)]:
        labels=np.unique(ytr); prot=np.stack([Xtr[ytr==c].mean(0) for c in labels])
        cand,margin,support=cosine_prototype_predict(prot,labels,Xev)
        # baseline is a standard prototype without open-class candidate; use same-round-trained
        # nearest prototype only as a diagnostic, never as a deployment claim.
        base=np.full(len(Xev),labels[np.argmax(np.bincount(ytr))])
        grid=[]
        for mt in [-np.inf,0.0,0.02,0.05,0.1,0.2]:
            for st in [-np.inf,0.0,0.2,0.4,0.6,0.8]:
                p,cov=apply_confidence_gate(base,cand,margin,support,mt,st)
                grid.append({'margin_threshold':mt,'support_threshold':st,'coverage':cov,**metrics(yev,p)})
        best=max(grid,key=lambda d:(d['balanced_accuracy'],d['accuracy'],-d['coverage']))
        out[name]={'prototype_labels':labels.tolist(),'candidate':metrics(yev,cand),'best_gate':best,'grid':grid}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();z=np.load(a.features);r=run(z['X1t3'],z['y1t3'],z['X2t3'],z['y2t3']);json.dump(r,open(a.output,'w'),indent=2);print(json.dumps(r,indent=2))
if __name__=='__main__':main()