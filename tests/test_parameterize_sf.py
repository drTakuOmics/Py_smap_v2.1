import numpy as np
from smap_tools_python.parameterize_sf import parameterize_sf


def test_parameterize_sf_carbon():
    a, b = parameterize_sf('C')
    np.testing.assert_allclose(a, [0.0893, 0.2563, 0.7570, 1.0487, 0.3575], rtol=1e-4)
    np.testing.assert_allclose(b, [0.2465, 1.7100, 6.4094, 18.6113, 50.2523], rtol=1e-4)


def test_parameterize_sf_unknown():
    a, b = parameterize_sf('Xx')
    assert np.all(a == 0) and np.all(b == 0)
