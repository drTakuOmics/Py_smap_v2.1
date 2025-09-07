import numpy as np
from smap_tools_python.proj_view import proj_view
from smap_tools_python.crop_pad import extendj


def test_proj_view_basic():
    vol = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    out = proj_view(vol)
    edge = 4
    assert out.shape == (edge, edge, 3)
    vol_pad = extendj(vol, (edge, edge, edge), float(np.median(vol)))
    expected0 = vol_pad.sum(axis=0)
    expected1 = vol_pad.sum(axis=1)
    expected2 = vol_pad.sum(axis=2)
    assert np.allclose(out[:, :, 0], expected0)
    assert np.allclose(out[:, :, 1], expected1)
    assert np.allclose(out[:, :, 2], expected2)
