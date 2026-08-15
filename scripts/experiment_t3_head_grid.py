#!/usr/bin/env python3
import argparse,json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler,Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.base import BaseEstimator,ClassifierMixin

def candidate_specs():
    out=[]
    for scale in ['std','robust','norm']:
        for metric in ['cosine','euclidean']:
            for k in [1,3,5,7,9]: out.append({'name':f'{scale}_{metric}_knn{k}','scale':scale,'metric':metric,'k':k,'kind':'knn'})
    for C in [.1,1.,10.]: out.append({'name':f'std_logreg_C{C:g}','scale':'std','kind':'logreg','C':C})
    return out

class SafeKNN(BaseEstimator,ClassifierMixin):
    def __init__(self,k=5,metric='cosine'): self.k=k; self.metric=metric
    def fit(self,X,y): self.model_=KNeighborsClassifier(min(self.k,len(X)),metric=self.metric,weights='distance').fit(X,y); return self
    def predict(self,X): return self.model_.predict(X)

def build_head(spec):
    scale={'std':StandardScaler(),'robust':RobustScaler(),'norm':Normalizer()}[spec['scale']]
    if spec['kind']=='knn': return make_pipeline(scale,SafeKNN(spec['k'],spec['metric']))
    return make_pipeline(scale,LogisticRegression(C=spec['C'],max_iter=2000,class_weight='balanced'))

def met(y,p): return {'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p))}
def select_source(X,y,seed=20260816):
    sk=StratifiedKFold(5,shuffle=True,random_state=seed); scores=[]
    for spec in candidate_specs():
        vals=[]
        for tr,te in sk.split(X,y):
            m=build_head(spec).fit(X[tr],y[tr]); vals.append(met(y[te],m.predict(X[te])))
        scores.append({**spec,'cv_accuracy':float(np.mean([v['accuracy'] for v in vals])),'cv_balanced_accuracy':float(np.mean([v['balanced_accuracy'] for v in vals]))})
    return max(scores,key=lambda x:(x['cv_balanced_accuracy'],x['cv_accuracy'],x['name'])),scores

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();z=np.load(a.features); X1=z['X1t3'].astype(float);y1=z['y1t3'].astype(int);X2=z['X2t3'].astype(float);y2=z['y2t3'].astype(int); out={'experiment':'t3-head-grid-20260816','protocol':'source-only 5-fold selection, target fixed evaluation'}
    for src,Xs,ys,tgt,Xt,yt in [('r1',X1,y1,'r2',X2,y2),('r2',X2,y2,'r1',X1,y1)]:
        sel,all_scores=select_source(Xs,ys); m=build_head(sel).fit(Xs,ys); out[f'{src}_selection']={'selected':sel,'top5':sorted(all_scores,key=lambda x:(-x['cv_balanced_accuracy'],-x['cv_accuracy']))[:5],f'{src}_to_{tgt}':met(yt,m.predict(Xt))}
    json.dump(out,open(a.output,'w'),indent=2); print(json.dumps(out,indent=2))
if __name__=='__main__':main()