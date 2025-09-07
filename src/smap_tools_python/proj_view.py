import numpy as np
from .crop_pad import extendj


def proj_view(volume):
    """Generate orthogonal projections of a 3-D volume.

    The input volume is padded to a cube using its median value and then
    projected along each principal axis by summation.  The result is a stack of
    three 2-D projections matching the behaviour of the MATLAB ``projView``
    helper.

    Parameters
    ----------
    volume : array_like, shape (Z, Y, X)
        Input 3-D volume.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(edge, edge, 3)`` where ``edge`` is the maximum input
        dimension.  ``out[...,0]`` is summed over axis 0, ``out[...,1]`` over
        axis 1 and ``out[...,2]`` over axis 2.
    """

    vol = np.asarray(volume)
    edge = int(np.max(vol.shape))
    pad_shape = (edge, edge, edge)
    vol = extendj(vol, pad_shape, float(np.median(vol)))
    out = np.zeros((edge, edge, 3), dtype=vol.dtype)
    out[:, :, 0] = vol.sum(axis=0)
    out[:, :, 1] = vol.sum(axis=1)
    out[:, :, 2] = vol.sum(axis=2)
    return out
