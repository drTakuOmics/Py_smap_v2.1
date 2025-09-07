import numpy as np

from .crop_pad import extendj
from .rotate import rotate3d_matrix


def backproject(patches, rotations, pad_size=None):
    """Backproject 2-D patches into a 3-D volume.

    A simple real-space backprojection is performed by inserting each patch as
    a central slice of a cubic volume, rotating it according to ``rotations``
    and summing the results.

    Parameters
    ----------
    patches : array_like, shape (N, N, M)
        Stack of projection images.
    rotations : array_like, shape (M, 3, 3) or (3, 3, M)
        Rotation matrices associated with each patch.
    pad_size : int, optional
        Edge length of the output volume. Defaults to ``N``.

    Returns
    -------
    numpy.ndarray
        Reconstructed volume of shape ``(pad_size, pad_size, pad_size)``.
    """

    patches = np.asarray(patches, dtype=float)
    if patches.ndim != 3:
        raise ValueError("patches must have shape (N, N, M)")

    n, m, k = patches.shape
    if n != m:
        raise ValueError("patches must be square")

    pad_size = int(pad_size or n)
    vol = np.zeros((pad_size, pad_size, pad_size), dtype=float)

    rotations = np.asarray(rotations, dtype=float)
    if rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must contain 3x3 matrices")
    if rotations.shape[0] == 3 and rotations.shape[1] == 3 and rotations.ndim == 3:
        rotations = np.transpose(rotations, (2, 0, 1))
    if rotations.shape[0] != k:
        raise ValueError("Number of rotation matrices must match patches")

    center = pad_size // 2
    for i in range(k):
        patch = extendj(patches[:, :, i], (pad_size, pad_size), float(patches[:, :, i].mean()))
        slice_vol = np.zeros_like(vol)
        slice_vol[:, :, center] = patch
        vol += rotate3d_matrix(slice_vol, rotations[i])

    return vol / float(k)


__all__ = ["backproject"]
