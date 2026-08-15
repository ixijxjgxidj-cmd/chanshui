import numpy as np
from scripts.experiment_t3_coral import CORALAligner

def test_coral_fit_transform_has_zero_source_mean():
    rng=np.random.default_rng(2); x=rng.normal(size=(20,4))*np.array([1,2,3,4])+5
    a=CORALAligner(reg=1e-3).fit(x)
    z=a.transform(x)
    assert z.shape==x.shape
    assert np.allclose(z.mean(0),0,atol=1e-6)

def test_coral_transform_rejects_wrong_dimension():
    a=CORALAligner().fit(np.zeros((4,3)))
    try: a.transform(np.zeros((2,2)))
    except ValueError: return
    raise AssertionError('dimension mismatch must raise')