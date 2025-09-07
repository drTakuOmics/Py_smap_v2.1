import numpy as np



def particle_diameter(vol, thresh=0.005):
    """Estimate particle diameter from a 3-D volume.

    Parameters
    ----------
    vol : array_like
        Input 3-D volume.
    thresh : float, optional
        Fraction of the maximal radial profile used to determine the
        diameter. Defaults to 0.005, matching the MATLAB implementation.

    Returns
    -------
    float
        Estimated diameter in pixels.
    """
    vol = np.asarray(vol, float)
    coords = [np.arange(s) - (s // 2) for s in vol.shape]
    grids = np.meshgrid(*coords, indexing="ij")
    r = np.sqrt(sum(g ** 2 for g in grids))
    mask = vol > thresh
    if not np.any(mask):
        return 0.0
    max_r = r[mask].max()
    return float(2 * max_r)