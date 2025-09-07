import numpy as np
from smap_tools_python import lb_bh_to_rrs


def test_lb_bh_to_rrs_identity():
    q_lb = np.eye(3)[None, :, :]
    q_bh = np.eye(3)[None, :, :]
    coeff = np.eye(3)
    mu = np.zeros(3)
    out = lb_bh_to_rrs(q_lb, q_bh, coeff, mu, coeff, mu)
    assert out.shape == (1, 3)
    assert np.allclose(out, 0.0)
