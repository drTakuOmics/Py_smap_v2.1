import numpy as np
from .cos_mask import rrj


def make_filt(im_size, mask_edges, a_per_pix=1.0):
    """Generate a cosine-edged radial filter.

    Parameters
    ----------
    im_size : int or sequence of int
        Size of the output filter. If a scalar is given a square 2-D mask is
        produced.
    mask_edges : tuple of float
        Inner and outer edge in the same units as ``im_size * a_per_pix``.
    a_per_pix : float, optional
        Scaling factor converting pixel units to the mask domain.

    Returns
    -------
    numpy.ndarray
        Cosine-edged mask with values in ``[0, 1]``.
    """
    if np.isscalar(im_size):
        shape = (int(im_size), int(im_size))
    else:
        shape = tuple(int(s) for s in im_size)
    mask_edge_in, mask_edge_out = mask_edges

    nn = rrj(shape)
    R = nn / a_per_pix

    inner = R <= mask_edge_in
    RR = np.abs(R - mask_edge_in) * (~inner)
    outer = R > mask_edge_out
    RR[outer] = np.pi / 2
    T = (mask_edge_in - mask_edge_out) * 2
    mask = 0.5 + 0.5 * np.cos(2 * np.pi * RR / T)
    mask[outer] = 0
    return mask
