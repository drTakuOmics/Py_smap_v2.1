import numpy as np
try:
    from scipy.ndimage import distance_transform_edt
except Exception:  # pragma: no cover - SciPy may be unavailable
    distance_transform_edt = None


def mask_volume(mapref, mask_params, mode="mask"):
    """Mask a 3-D volume using raised cosine or shell modes.

    Parameters
    ----------
    mapref : array_like
        Input volume.
    mask_params : sequence
        Parameters controlling the mask. For ``mode='mask'`` these are
        ``(near_edge, cos_width)``. For ``mode='shell'`` they are
        ``(d_shell, t_shell)``.
    mode : {'mask', 'shell'}, optional
        Masking strategy. ``'mask'`` creates a raised cosine mask from the
        volume edges inward. ``'shell'`` creates a thin shell around the
        boundary of detected features.

    Returns
    -------
    tuple of numpy.ndarray
        ``(out, mask, D)`` where ``out`` is the masked volume, ``mask`` the
        mask itself, and ``D`` the distance transform used.
    """
    if distance_transform_edt is None:
        raise ImportError("scipy is required for mask_volume")

    mapref = np.asarray(mapref, dtype=float)
    mask_params = np.asarray(mask_params, dtype=float)

    bg_val = np.bincount(mapref.astype(int).ravel()).argmax()
    mm = mapref - bg_val
    thr = np.std(np.abs(mm))
    BW = np.abs(mm) > thr

    if mode == "mask":
        D = distance_transform_edt(~BW)
        near_edge, cos_width = mask_params
        far_edge = near_edge + cos_width
        D11 = D.copy()
        D11[D <= near_edge] = 0
        D11[D >= far_edge] = np.pi
        between = (D > near_edge) & (D < far_edge)
        temp = (D[between] - near_edge) * (np.pi / cos_width)
        D11[between] = temp
        mask = np.cos(D11) / 2 + 0.5
    elif mode == "shell":
        # preliminary cosine mask to dilate tight spaces
        D0 = distance_transform_edt(~BW)
        D11 = D0.copy()
        D11[D0 <= 0] = 0
        D11[D0 >= 1] = np.pi
        between = (D0 > 0) & (D0 < 1)
        D11[between] = D0[between] * np.pi
        BW = (np.cos(D11) / 2 + 0.5) > 0

        D = distance_transform_edt(~BW)
        d_shell, t_shell = mask_params
        D11 = np.abs(D - d_shell)
        D11[D == 0] = np.nan
        D11[D11 > t_shell] = np.nan
        mask = np.where(np.isnan(D11), 0, 1)
    else:
        raise ValueError("mode must be 'mask' or 'shell'")

    outref = mm * mask + bg_val
    return outref, mask, D
