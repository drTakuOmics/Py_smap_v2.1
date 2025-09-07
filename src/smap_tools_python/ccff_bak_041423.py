import numpy as np
from .ccff import ccff


def ccff_bak_041423(image: np.ndarray, templates: np.ndarray, mode: str = "filt"):
    """Legacy wrapper around :func:`ccff`.

    Parameters
    ----------
    image : ndarray
        2-D image to correlate with templates.
    templates : ndarray
        Stack of templates matching the image size.
    mode : {"filt", "noFilt"}, optional
        Whitening mode passed through to :func:`ccff`.

    Returns
    -------
    tuple of (ndarray, ndarray)
        Cross-correlation volume and peak values, as produced by :func:`ccff`.
    """
    return ccff(image, templates, mode)


__all__ = ["ccff_bak_041423"]
