"""Cosine-edged radial masks."""

from __future__ import annotations

import numpy as np

from .rrj import rrj


def cos_mask(im_size: int, mask_edges, a_per_pix: float):
    """Create a cosine-edged circular mask.

    Parameters
    ----------
    im_size : int
        Size of the (square) output mask in pixels.
    mask_edges : tuple[float, float]
        Inner and outer radii of the cosine falloff in Å.
    a_per_pix : float
        Ångstroms per pixel for the grid.

    Returns
    -------
    numpy.ndarray
        ``im_size`` × ``im_size`` mask with values in ``[0, 1]``.
    """

    mask_edge_in, mask_edge_out = mask_edges
    nn = rrj((im_size, im_size))
    R = nn / a_per_pix
    inside = R <= mask_edge_in
    ramp = np.abs(R - mask_edge_in)
    outside = R > mask_edge_out
    ramp[outside] = np.pi / 2
    width = (mask_edge_in - mask_edge_out) * 2
    mask = 0.5 + 0.5 * np.cos(2 * np.pi * ramp / width)
    mask[outside] = 0
    mask[inside] = 1
    return mask


def variable_cos_mask(im_size: int, mask_edges, a_per_pix: float):
    """Backward-compatible wrapper for MATLAB's ``variableCosMask``."""

    return cos_mask(im_size, mask_edges, a_per_pix)


def cosMask(im_size: int, mask_edges, a_per_pix: float):
    """MATLAB-style alias of :func:`cos_mask`."""

    return cos_mask(im_size, mask_edges, a_per_pix)


__all__ = ["cos_mask", "variable_cos_mask", "cosMask"]