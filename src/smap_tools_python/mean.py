import numpy as np


def mean(arr, axis=0):
    """NaN-aware mean along a given axis.

    This mirrors MATLAB's ``smap.mean`` helper, which averages while ignoring
    ``NaN`` values. The default axis follows MATLAB conventions, averaging
    along the first dimension (across rows).

    Parameters
    ----------
    arr : array_like
        Input array.
    axis : int, optional
        Axis along which to compute the mean. Defaults to ``0``.

    Returns
    -------
    numpy.ndarray or scalar
        The mean values with ``NaN`` entries skipped.
    """
    arr = np.asarray(arr)
    return np.nanmean(arr, axis=axis)
