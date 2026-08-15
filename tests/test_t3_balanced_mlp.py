import numpy as np
from scripts.experiment_t3_balanced_mlp import mixup_batch, TinyMLP

def test_mixup_targets_preserve_probability_mass():
    x=np.ones((4,3),np.float32); y=np.array([0,1,2,3])
    _,target=mixup_batch(x,y,5,0.4,np.random.default_rng(3))
    assert target.shape==(4,5)
    assert np.allclose(target.sum(axis=1),1.0)

def test_mlp_predicts_five_logits():
    import torch
    m=TinyMLP(8,5); out=m(torch.zeros(3,8))
    assert tuple(out.shape)==(3,5)