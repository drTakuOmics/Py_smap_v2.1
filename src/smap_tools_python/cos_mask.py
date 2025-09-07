import numpy as np


def rrj(shape):


    """Compute normalized radial coordinates for an N-dimensional grid.

    Parameters
    ----------
    shape : tuple of int
        Desired output shape.

    Returns
    -------
    numpy.ndarray
        Array of radial distances normalized by twice the maximum radius.
    """
    coords = [np.arange(s) - (s // 2) for s in shape]
    grids = np.meshgrid(*coords, indexing="ij", sparse=True)
    sq = sum(g.astype(float) ** 2 for g in grids)
    R = np.sqrt(sq)
    max_radius = max(s // 2 for s in shape)
    return R / (2 * max_radius)


def variable_cos_mask(im_size, mask_edges, a_per_pix):
    """Replicates MATLAB's variableCosMask function."""
    mask_edge_in, mask_edge_out = mask_edges
    nn = rrj((im_size, im_size))
    R = nn / a_per_pix
    Rt = R <= mask_edge_in
    RR = np.abs(R - mask_edge_in) * (~Rt)
    Rtt = R > mask_edge_out
    RR[Rtt] = np.pi / 2
    T = (mask_edge_in - mask_edge_out) * 2
    RRR = 0.5 + 0.5 * np.cos(2 * np.pi * RR / T)
    RRR[Rtt] = 0
    return RRR
