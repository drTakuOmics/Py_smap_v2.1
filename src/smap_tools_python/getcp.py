import numpy as np


def get_center_pixel(shape_or_array):
    """Return the 0-based centre pixel index for each dimension.

    Parameters
    ----------
    shape_or_array : array_like or tuple of int
        Array or shape from which to determine the centre.

    Returns
    -------
    tuple of int
        Centre pixel coordinates using Python's 0-based indexing.
    """
    shape = getattr(shape_or_array, "shape", shape_or_array)
    shape = np.asarray(shape, dtype=int)
    return tuple(shape // 2)


# MATLAB compatibility alias
getcp = get_center_pixel
