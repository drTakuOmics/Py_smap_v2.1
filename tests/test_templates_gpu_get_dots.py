import numpy as np
from scipy.spatial.transform import Rotation as Rot

from smap_tools_python.templates import templates
from smap_tools_python.templates_gpu import templates_gpu, templates_half_gpu
from smap_tools_python.get_dots import get_dots


def test_templates_gpu_matches_cpu():
    vol = np.random.rand(8, 8, 8)
    rots = np.repeat(np.eye(3)[:, :, None], 2, axis=2)
    cpu = templates(vol, rots)
    gpu = templates_gpu(vol, rots)
    assert np.allclose(cpu, np.asarray(gpu))


def test_templates_half_gpu_downsamples():
    vol = np.random.rand(8, 8, 8)
    rots = np.eye(3)[:, :, None]
    full = templates_gpu(vol, rots)
    half = templates_half_gpu(vol, rots)
    assert half.shape[0] * 2 == full.shape[0]
    assert np.allclose(full[::2, ::2, 0], np.asarray(half)[:, :, 0])


def test_get_dots_reproducible():
    vol = np.zeros((16, 16, 16), float)
    x = np.arange(16) - 7.5
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    vol[(X ** 2 + Y ** 2 + Z ** 2) <= 4 ** 2] = 1.0
    rots = Rot.random(40, random_state=0).as_matrix().transpose(1, 2, 0)
    mean, std = get_dots(vol, rots, df=None, edge_size=16, n_samples=20)
    assert np.isclose(mean, 1.1897957200252536)
    assert np.isclose(std, 0.040022920317379196)
