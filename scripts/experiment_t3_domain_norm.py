#!/usr/bin/env python3
"""T3 训练侧域归一化对照；不读取 08 或 holdout。"""
import argparse, json, time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def metrics(y,p):
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),"n":int(len(y))}

def models():
    return {
      "cosine_knn5": make_pipeline(Normalizer(), KNeighborsClassifier(5,metric="cosine")),
      "standard_cosine_knn5": make_pipeline(StandardScaler(),Normalizer(),KNeighborsClassifier(5,metric="cosine")),
      "robust_cosine_knn5": make_pipeline(RobustScaler(quantile_range=(10,90)),Normalizer(),KNeighborsClassifier(5,metric="cosine")),
      "standard_logreg": make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0)),
    }

def cv_eval(X,y,seed=20260816):
    out={}; skf=StratifiedKFold(5,shuffle=True,random_state=seed)
    for name in models():
        ps=[]; ys=[]
        for tr,te in skf.split(X,y):
            m=models()[name]; m.fit(X[tr],y[tr]); ps.append(m.predict(X[te])); ys.append(y[te])
        out[name]=metrics(np.concatenate(ys),np.concatenate(ps))
    return out

def cross_eval(X1,y1,X2,y2):
    out={}
    for train_name,Xtr,ytr,eval_name,Xev,yev in [("r1",X1,y1,"r2",X2,y2),("r2",X2,y2,"r1",X1,y1)]:
        for name,m in models().items():
            m.fit(Xtr,ytr); out[f"{train_name}_to_{eval_name}_{name}"]=metrics(yev,m.predict(Xev))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--features",required=True); ap.add_argument("--output",required=True); a=ap.parse_args(); t=time.time()
    z=np.load(a.features); X1=z["X1t3"].astype(np.float64); y1=z["y1t3"].astype(int); X2=z["X2t3"].astype(np.float64); y2=z["y2t3"].astype(int)
    bothX=np.vstack([X1,X2]); bothy=np.concatenate([y1,y2])
    result={"experiment":"t3-domain-normalization-20260816","features":a.features,"shapes":{"r1":list(X1.shape),"r2":list(X2.shape)},"within_r1_cv":cv_eval(X1,y1),"within_r2_cv":cv_eval(X2,y2),"joint_cv":cv_eval(bothX,bothy),"cross_round":cross_eval(X1,y1,X2,y2),"runtime_s":time.time()-t}
    with open(a.output,"w",encoding="utf-8") as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()