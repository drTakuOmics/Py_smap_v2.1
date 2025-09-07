import numpy as np

from smap_tools_python import psd_filter, psd_filter_3d


def test_psd_filter_shapes():
    im = np.random.rand(8, 8)
    filt, out, psbg = psd_filter(im)
    assert filt.shape == im.shape
    assert out.shape == im.shape
    assert psbg.shape == im.shape
    cp = im.shape[0] // 2
    assert filt[cp, cp] == 0


def test_psd_filter_3d_shapes():
    vol = np.random.rand(8, 8, 8)
    filt = psd_filter_3d(vol)
    assert filt.shape == vol.shape
    cp = vol.shape[0] // 2
    assert filt[cp, cp, cp] == 0

