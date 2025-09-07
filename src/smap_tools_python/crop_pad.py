import numpy as np

def crop_or_pad(arr, new_shape, pad_value=0):
    """Center crop or pad an array to ``new_shape``.

    Parameters
    ----------
    arr : array_like
        Input array to be cropped or padded.
    new_shape : tuple of int
        Desired output shape. Must have same length as ``arr.ndim``.
    pad_value : scalar, optional
        Value to use for padding when ``new_shape`` exceeds ``arr`` dimensions.

    Returns
    -------
    numpy.ndarray
        Array cropped or padded to ``new_shape``.
    """
    arr = np.asarray(arr)
    if len(new_shape) != arr.ndim:
        raise ValueError("new_shape must match number of dimensions of arr")

    slices = []
    pads = []
    for old, new in zip(arr.shape, new_shape):
        if new < old:
            start = (old - new) // 2
            slices.append(slice(start, start + new))
            pads.append((0, 0))
        else:
            slices.append(slice(0, old))
            before = (new - old) // 2
            after = new - old - before
            pads.append((before, after))
    cropped = arr[tuple(slices)]
    if any(new > old for old, new in zip(arr.shape, new_shape)):
        cropped = np.pad(cropped, pads, mode='constant', constant_values=pad_value)
    return cropped


def cutj(arr, new_shape):
    """Crop or pad ``arr`` to ``new_shape`` using the data mean for padding.

    This mirrors the behavior of SMAP's MATLAB ``cutj`` helper which crops an
    image to the requested dimensions and, when an expansion is requested,
    pads using the array mean rather than wrapping values.

    Parameters
    ----------
    arr : array_like
        Input array.
    new_shape : tuple of int
        Desired output shape.

    Returns
    -------
    numpy.ndarray
        Cropped or padded array.
    """
    arr = np.asarray(arr, dtype=float)
    if any(n > o for n, o in zip(new_shape, arr.shape)):
        pad_val = float(np.mean(arr))
    else:
        pad_val = 0
    return crop_or_pad(arr, new_shape, pad_val)


def extendj(arr, new_shape, pad_value=0):
    """Pad ``arr`` to ``new_shape`` using ``pad_value``.

    Parameters
    ----------
    arr : array_like
        Input array.
    new_shape : tuple of int
        Target shape after padding.
    pad_value : scalar, optional
        Value used to fill new elements when padding. Defaults to 0.

    Returns
    -------
    numpy.ndarray
        Padded array.
    """
    arr = np.asarray(arr)
    return crop_or_pad(arr, new_shape, pad_value)