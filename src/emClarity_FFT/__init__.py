"""Lightweight FFT helpers with optional GPU acceleration.

This module replaces the legacy CUDA/``mex`` implementation that previously
lived in :mod:`emClarity_FFT`.  The new implementation is pure Python and uses
``pyFFTW`` when available for high performance CPU transforms and ``CuPy`` for
GPU acceleration.  Both forward and inverse transforms match MATLAB's
normalisation so that existing pipelines relying on MATLAB parity continue to
work unchanged.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fftw

    fftw.interfaces.cache.enable()
    _fftn = fftw.fftn
    _ifftn = fftw.ifftn
except Exception:  # pragma: no cover - fallback
    _fftn = np.fft.fftn
    _ifftn = np.fft.ifftn

try:  # pragma: no cover - optional dependency
    import cupy as cp
    _cupy_available = True
except Exception:  # pragma: no cover - optional dependency
    cp = None
    _cupy_available = False

__all__ = ["fft", "ifft", "has_gpu"]


def has_gpu() -> bool:
    """Return ``True`` when a CuPy backend is available."""
    return _cupy_available


def _fft_cpu(arr: np.ndarray) -> np.ndarray:
    npix = int(np.prod(arr.shape))
    return np.fft.fftshift(_fftn(np.fft.ifftshift(arr))) / np.sqrt(npix)


def _ifft_cpu(arr: np.ndarray) -> np.ndarray:
    npix = int(np.prod(arr.shape))
    return (
        np.fft.fftshift(_ifftn(np.fft.ifftshift(arr))).real * np.sqrt(npix)
    )


def _fft_gpu(arr) -> "cp.ndarray":  # pragma: no cover - optional dependency
    xp = cp
    arr = xp.asarray(arr)
    npix = int(xp.prod(arr.shape))
    transformed = xp.fft.fftn(xp.fft.ifftshift(arr))
    return xp.fft.fftshift(transformed) / xp.sqrt(npix)


def _ifft_gpu(arr) -> "cp.ndarray":  # pragma: no cover - optional dependency
    xp = cp
    arr = xp.asarray(arr)
    npix = int(xp.prod(arr.shape))
    transformed = xp.fft.ifftn(xp.fft.ifftshift(arr))
    return xp.fft.fftshift(transformed).real * xp.sqrt(npix)


def fft(input_vol, use_gpu: bool = False):
    """Forward FFT with MATLAB-like normalisation.

    Parameters
    ----------
    input_vol : array-like
        Array to transform.
    use_gpu : bool, optional
        When ``True`` and a GPU backend is available the computation is
        executed with CuPy.
    """
    arr = np.asarray(input_vol)
    if use_gpu:
        if not has_gpu():  # pragma: no cover - optional dependency
            raise RuntimeError("CuPy is required for GPU FFT operations")
        return _fft_gpu(arr)
    return _fft_cpu(arr)


def ifft(input_vol, use_gpu: bool = False):
    """Inverse FFT with MATLAB-like normalisation."""
    arr = np.asarray(input_vol)
    if use_gpu:
        if not has_gpu():  # pragma: no cover - optional dependency
            raise RuntimeError("CuPy is required for GPU FFT operations")
        return _ifft_gpu(arr)
    return _ifft_cpu(arr)
