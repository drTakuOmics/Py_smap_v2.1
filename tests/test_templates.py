import numpy as np
from smap_tools_python import templates


def test_templates_identity_projection():
    vol = np.zeros((8, 8, 8), dtype=float)
    vol[3:5, 3:5, 3:5] = 1.0
    rot = np.eye(3)[..., None]
    tmpl = templates(vol, rot, dfs=None, pixel_size=1.0)
    assert tmpl.shape == (8, 8, 1)
    assert np.allclose(tmpl[:, :, 0], vol.sum(axis=2))
