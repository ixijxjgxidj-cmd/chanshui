import numpy as np
from scripts.experiment_t3_fusion import fuse_predictions

def test_fusion_alpha_extremes():
    a=np.array([[.9,.1],[.2,.8]]); b=np.array([[.1,.9],[.8,.2]])
    assert np.array_equal(fuse_predictions(a,b,0).argmax(1),a.argmax(1))
    assert np.array_equal(fuse_predictions(a,b,1).argmax(1),b.argmax(1))

def test_fusion_probabilities_normalized():
    p=fuse_predictions(np.ones((2,5)),np.ones((2,5)),.4)
    assert np.allclose(p.sum(1),1)