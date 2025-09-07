import numpy as np


def rrj(inref):
    """Compute normalized radial distances for 2‑D or 3‑D grids.

    Parameters
    ----------
    inref : array_like or tuple of int
        Array whose shape defines the grid, or the shape itself.

    Returns
    -------
    numpy.ndarray
        Array of the same shape containing radii normalised by twice the
        maximum radius of the largest dimension.
    """

    if hasattr(inref, "shape"):
        dims = np.array(inref.shape, dtype=int)
    else:
        dims = np.array(inref, dtype=int)
    dim_max = dims.max()
    coords = [np.arange(n) - (n // 2) for n in dims]
    grids = np.meshgrid(*coords, indexing="ij")
    R = np.sqrt(sum(g.astype(float) ** 2 for g in grids))
    max_radius = dim_max // 2
    return R / (2 * max_radius)
