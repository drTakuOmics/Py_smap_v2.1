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
    r = np.round(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)).astype(np.int64)

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
    r = np.round(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)).astype(np.int64)
    return profile[r]


def radial_max(image: np.ndarray) -> np.ndarray:
    """Compute maximum value for each integer radius from image center."""
    y, x = np.indices(image.shape)
    cy = (image.shape[0] - 1) / 2.0
    cx = (image.shape[1] - 1) / 2.0
    r = np.round(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)).astype(np.int64)

    r_flat = r.ravel()
    img_flat = image.ravel()
    max_r = r_flat.max() + 1
    out = np.full(max_r, -np.inf)
    for rad, val in zip(r_flat, img_flat):
        if val > out[rad]:
            out[rad] = val
    return out


def radialmeanj(arr: np.ndarray, return_nd: bool = False):
    """Dimension-agnostic radial mean profile.

    Parameters
    ----------
    arr : ndarray
        Input array, 1-D, 2-D or 3-D.
    return_nd : bool, optional
        If ``True`` also return an array where each element is replaced by
        the mean of all pixels/voxels sharing its rounded radius.

    Returns
    -------
    profile : ndarray
        Radial mean values indexed by integer radius.
    arr_mean : ndarray, optional
        Only returned if ``return_nd`` is ``True``.
    """

    arr = np.asarray(arr, dtype=float)
    coords = np.meshgrid(*[np.arange(n) for n in arr.shape], indexing="ij")
    center = [(n - 1) / 2.0 for n in arr.shape]
    r = np.sqrt(sum((c - ctr) ** 2 for c, ctr in zip(coords, center)))
    r_int = np.round(r).astype(np.int64)

    sums = np.bincount(r_int.ravel(), weights=arr.ravel())
    counts = np.bincount(r_int.ravel())
    counts[counts == 0] = 1
    profile = sums / counts
    if return_nd:
        return profile, profile[r_int]
    return profile


def radialmean_im(image: np.ndarray) -> np.ndarray:
    """Wrapper mirroring MATLAB's ``radialmeanIm``."""

    return radialmeanj(image)


def radial_average_im(image: np.ndarray) -> np.ndarray:
    """Wrapper mirroring MATLAB's ``radialAverageIm``."""

    _, nd = radialmeanj(image, return_nd=True)
    return nd


def radialmaxj(arr: np.ndarray) -> np.ndarray:
    """Radial maximum profile for an N‑D array."""

    arr = np.asarray(arr, dtype=float)
    coords = np.meshgrid(*[np.arange(n) for n in arr.shape], indexing="ij")
    center = [(n - 1) / 2.0 for n in arr.shape]
    r = np.sqrt(sum((c - ctr) ** 2 for c, ctr in zip(coords, center)))
    r_int = np.round(r).astype(np.int64)
    out = np.full(r_int.max() + 1, -np.inf)
    np.maximum.at(out, r_int.ravel(), arr.ravel())
    return out


# MATLAB-style aliases
radialAverageIm = radial_average_im
radialmeanIm = radialmean_im


__all__ = [
    "radial_mean",
    "radial_average",
    "radial_max",
    "radialmeanj",
    "radialmean_im",
    "radial_average_im",
    "radialmaxj",
    "radialAverageIm",
    "radialmeanIm",
]
