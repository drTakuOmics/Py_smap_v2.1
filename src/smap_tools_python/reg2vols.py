import numpy as np
from .phase_shift import apply_phase_shifts


def _quad_peak(values, idx):
    if idx <= 0 or idx >= len(values) - 1:
        return 0.0
    a, b, c = values[idx - 1: idx + 2]
    denom = a - 2 * b + c
    if denom == 0:
        return 0.0
    return 0.5 * (a - c) / denom


def reg2vols(input_vol, ref_vol, rtu=100):
    """Register a volume against a reference using phase correlation.

    Parameters
    ----------
    input_vol, ref_vol : ndarray
        Volumes to register. ``input_vol`` will be shifted to match ``ref_vol``.
    rtu : int, optional
        Half-width of the cropped correlation cube used for subpixel fitting.

    Returns
    -------
    registered : ndarray
        ``input_vol`` shifted to align with ``ref_vol``.
    shifts : ndarray, shape (3,)
        Estimated subpixel shifts ``(dz, dy, dx)`` applied to ``input_vol``.
    peak : float
        Peak value of the correlation volume.
    """
    input_vol = np.asarray(input_vol, dtype=float)
    ref_vol = np.asarray(ref_vol, dtype=float)

    if input_vol.shape != ref_vol.shape:
        from .crop_pad import extendj
        input_vol = extendj(input_vol, ref_vol.shape, pad_value=input_vol.min())

    a = (input_vol - input_vol.mean()) / input_vol.std()
    b = (ref_vol - ref_vol.mean()) / ref_vol.std()

    outcc = np.fft.ifftn(np.fft.fftn(a) * np.conj(np.fft.fftn(b)))
    outcc = np.fft.fftshift(outcc.real)
    peak = float(outcc.max())
    z0, y0, x0 = np.unravel_index(np.argmax(outcc), outcc.shape)
    center = np.array(outcc.shape) // 2
    shifts_full = np.array([z0, y0, x0], dtype=float) - center

    dz = _quad_peak(outcc[:, y0, x0], z0)
    dy = _quad_peak(outcc[z0, :, x0], y0)
    dx = _quad_peak(outcc[z0, y0, :], x0)
    shifts = shifts_full + np.array([dz, dy, dx])

    registered = apply_phase_shifts(input_vol, -shifts)
    if np.isrealobj(input_vol):
        registered = registered.real
    return registered, shifts, peak
