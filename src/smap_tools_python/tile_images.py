import numpy as np


def tile_images(images):
    """Compose a tiled image from three projections.

    Parameters
    ----------
    images : sequence or array
        Either a list of three 2‑D arrays or a 3‑D array with the third
        dimension of size three.

    Returns
    -------
    numpy.ndarray
        A ``(2N, 2N)`` tiled image where ``N`` is the edge length of the input
        images. The bottom-right quadrant is filled with the median of the
        provided data.
    """
    if isinstance(images, np.ndarray):
        if images.ndim != 3 or images.shape[2] != 3:
            raise ValueError("expected array of shape (N, N, 3)")
        proj = [images[:, :, i] for i in range(3)]
    else:
        if len(images) != 3:
            raise ValueError("expected three images")
        proj = [np.asarray(im) for im in images]
    n = proj[0].shape[0]
    out = np.full((2 * n, 2 * n), np.nan, dtype=float)
    out[0:n, 0:n] = proj[0]
    out[0:n, n:2 * n] = proj[1]
    out[n:2 * n, 0:n] = proj[2]
    median = np.nanmedian(out)
    out[np.isnan(out)] = median
    return out
