import numpy as np


def radial_mean(image: np.ndarray) -> np.ndarray:
    """Compute mean value for each integer radius from image center.

    Parameters
    ----------
    image : ndarray
        2-D array representing the image.

    Returns
    -------
    profile : ndarray
        1-D array where ``profile[r]`` is the mean of all pixels whose
        rounded radius equals ``r``.
    """
    y, x = np.indices(image.shape)
    cy = (image.shape[0] - 1) / 2.0
    cx = (image.shape[1] - 1) / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int64)

    r_flat = r.ravel()
    img_flat = image.ravel()

    sums = np.bincount(r_flat, weights=img_flat)
    counts = np.bincount(r_flat)
    counts[counts == 0] = 1
    return sums / counts


def radial_average(image: np.ndarray) -> np.ndarray:
    """Replace each pixel with the mean of its radius."""
    profile = radial_mean(image)
    y, x = np.indices(image.shape)
    cy = (image.shape[0] - 1) / 2.0
    cx = (image.shape[1] - 1) / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int64)
    return profile[r]


def radial_max(image: np.ndarray) -> np.ndarray:
    """Compute maximum value for each integer radius from image center."""
    y, x = np.indices(image.shape)
    cy = (image.shape[0] - 1) / 2.0
    cx = (image.shape[1] - 1) / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int64)

    r_flat = r.ravel()
    img_flat = image.ravel()
    max_r = r_flat.max() + 1
    out = np.full(max_r, -np.inf)
    for rad, val in zip(r_flat, img_flat):
        if val > out[rad]:
            out[rad] = val
    return out
