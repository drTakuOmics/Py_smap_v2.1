import numpy as np


def nm(arr, axis=None):
    """Normalize an array to zero mean and unit variance.

    This utility mirrors MATLAB's ``smap.nm`` function. It computes the
    mean and standard deviation while ignoring ``NaN`` values and returns a
    z-scored array. By default the normalization is performed over the entire
    array, but an optional ``axis`` argument can be supplied to operate along a
    specific dimension.
    """
    arr = np.asarray(arr, dtype=float)
    mean = np.nanmean(arr, axis=axis, keepdims=True)
    std = np.nanstd(arr, axis=axis, ddof=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (arr - mean) / std
