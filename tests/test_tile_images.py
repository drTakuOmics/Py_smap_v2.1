import numpy as np
from smap_tools_python.tile_images import tile_images


def test_tile_images_basic():
    im1 = np.array([[1, 2], [3, 4]])
    im2 = np.array([[5, 6], [7, 8]])
    im3 = np.array([[9, 10], [11, 12]])
    out = tile_images([im1, im2, im3])
    expected = np.array([
        [1, 2, 5, 6],
        [3, 4, 7, 8],
        [9, 10, 6.5, 6.5],
        [11, 12, 6.5, 6.5],
    ])
    assert np.allclose(out, expected)
