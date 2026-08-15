import numpy as np
from scripts.experiment_t3_feature_aug import augment_minority

def test_augmentation_only_adds_existing_minority_labels():
    x=np.arange(30,dtype=float).reshape(10,3); y=np.array([0,0,0,0,0,1,1,2,3,4])
    xa,ya=augment_minority(x,y,per_class=4,noise_scale=.01,rng=np.random.default_rng(1))
    assert len(xa)==len(ya)
    assert set(ya).issubset(set(y))
    assert np.sum(ya==3)>=4 and np.sum(ya==4)>=4