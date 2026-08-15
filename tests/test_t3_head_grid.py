from scripts.experiment_t3_head_grid import candidate_specs, build_head

def test_grid_is_fixed_and_unique():
    specs=candidate_specs()
    assert len(specs)>=12
    assert len({s['name'] for s in specs})==len(specs)

def test_each_head_fits_toy_data():
    import numpy as np
    x=np.array([[1,0],[.9,.1],[-1,0],[-.9,-.1],[0,1],[.1,.9]],float)
    y=np.array([1,1,2,2,3,3])
    for spec in candidate_specs():
        m=build_head(spec).fit(x,y)
        p=m.predict(x)
        assert p.shape==(6,)