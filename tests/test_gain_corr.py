import numpy as np
from smap_tools_python import gain_corr


def test_gain_corr_hot_pixel():
    rng = np.random.default_rng(0)
    movie = rng.normal(size=(8, 8, 5))
    gain = np.ones((8, 8))
    movie[2, 3, :] += 100  # introduce hot pixel
    corrected, hot = gain_corr(movie, gain, hot_threshold=5)
    assert corrected.shape == movie.shape
    assert [2, 3] in hot.tolist()
    assert corrected[2, 3, 0] < 50
