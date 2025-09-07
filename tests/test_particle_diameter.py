import numpy as np

from smap_tools_python import particle_diameter


def test_particle_diameter_simple_sphere():
    # create binary sphere with radius 3
    r = 3
    size = 16
    x = np.arange(size) - size // 2
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    vol = ((X**2 + Y**2 + Z**2) <= r**2).astype(float)
    diameter = particle_diameter(vol, thresh=0.5)
    assert abs(diameter - 2 * r) <= 1
