import numpy as np
from numpy.fft import fftn, ifftn
from scipy.ndimage import fourier_shift


def _phase_correlation(ref, mov):
    """Estimate translation between ``ref`` and ``mov`` via phase correlation."""
    f_ref = fftn(ref)
    f_mov = fftn(mov)
    cross = f_ref * f_mov.conj()
    cross /= np.maximum(np.abs(cross), 1e-9)
    r = np.fft.fftshift(ifftn(cross))
    max_idx = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    max_idx = np.array(max_idx, dtype=float)
    mid = np.array(ref.shape) // 2
    shifts = max_idx - mid
    shifts[shifts > mid] -= np.array(ref.shape)[shifts > mid]
    return shifts


def motion_corr(frames, ref_index=None, axis=0):
    """Align movie frames by translational phase correlation.

    Parameters
    ----------
    frames : ndarray
        Stack of movie frames.
    ref_index : int, optional
        Index of the reference frame; defaults to the middle frame.
    axis : int, optional
        Axis corresponding to the frame index.

    Returns
    -------
    corrected : ndarray
        Motion-corrected stack with the same shape as ``frames``.
    shifts : ndarray
        Array of per-frame ``(dy, dx)`` shifts that were applied.
    """
    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError("expected a 3D stack of frames")
    frames = np.moveaxis(frames, axis, 0)
    n, h, w = frames.shape
    if ref_index is None:
        ref_index = n // 2
    ref = frames[ref_index]
    corrected = np.empty_like(frames, dtype=float)
    shifts = np.zeros((n, 2), dtype=float)
    for i, frame in enumerate(frames):
        shift = _phase_correlation(ref, frame)
        shifted = ifftn(fourier_shift(fftn(frame), shift)).real
        corrected[i] = shifted
        shifts[i] = shift
    corrected = np.moveaxis(corrected, 0, axis)
    return corrected, shifts


__all__ = ["motion_corr"]
