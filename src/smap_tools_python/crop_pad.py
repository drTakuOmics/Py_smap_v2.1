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
