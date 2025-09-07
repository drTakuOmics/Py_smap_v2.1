"""Python FFT and GPU kernels for emClarity.

This module replaces legacy MATLAB/``mex`` implementations with pure
Python equivalents.  The functions provide both CPU (NumPy) and optional GPU
(CuPy) implementations that mimic MATLAB's normalization conventions used in
emClarity.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    import cupy as cp
    _cupy_available = True
except Exception:  # pragma: no cover - optional dependency
    cp = None
    _cupy_available = False


def _xp(use_gpu: bool):
    """Return the array module to use (NumPy or CuPy)."""
    return cp if use_gpu and _cupy_available else np


def has_gpu() -> bool:
    """Return ``True`` when the CuPy backend is available."""
    return _cupy_available


def fft(input_vol, use_gpu: bool = False):
    """Forward FFT with MATLAB-like normalization.

    Parameters
    ----------
    input_vol : array-like
        Volume or image to transform.
    use_gpu : bool, optional
        When ``True`` and CuPy is installed, the computation is performed on
        the GPU.
    """
    xp = _xp(use_gpu)
    arr = xp.asarray(input_vol)
    npix = int(np.prod(arr.shape))
    transformed = xp.fft.fftn(xp.fft.ifftshift(arr))
    return xp.fft.fftshift(transformed) / np.sqrt(npix)


def ifft(input_vol, use_gpu: bool = False):
    """Inverse FFT with MATLAB-like normalization."""
    xp = _xp(use_gpu)
    arr = xp.nan_to_num(xp.asarray(input_vol))
    npix = int(np.prod(arr.shape))
    transformed = xp.fft.ifftn(xp.fft.ifftshift(arr))
    result = xp.fft.fftshift(xp.real(transformed)) * np.sqrt(npix)
    return result
