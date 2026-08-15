import numpy as np
from scripts.experiment_t3_open_class import cosine_prototype_predict, apply_confidence_gate

def test_prototype_predict_returns_label_margin_and_support():
    centers=np.array([[1.,0.],[0.,1.],[-1.,0.]])
    x=np.array([[.9,.1],[.1,.9],[-.8,.1]])
    pred,margin,support=cosine_prototype_predict(centers,np.array([1,2,3]),x)
    assert pred.tolist()==[1,2,3]
    assert margin.shape==support.shape==(3,)

def test_gate_extremes_are_deterministic():
    base=np.array([1,1,2]); cand=np.array([1,2,3]); m=np.array([.1,.2,.3]); s=np.array([.4,.5,.6])
    out,cov=apply_confidence_gate(base,cand,m,s,float('inf'),float('inf'))
    assert np.array_equal(out,base) and cov==0
    out,cov=apply_confidence_gate(base,cand,m,s,-float('inf'),-float('inf'))
    assert np.array_equal(out,cand) and cov==1