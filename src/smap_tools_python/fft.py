"""High level FFT wrappers.

The functions here provide a small, stable API that mirrors the original
MATLAB ``ftj``/``iftj`` routines.  They delegate the heavy lifting to the
:mod:`emClarity_FFT` backend which uses ``pyFFTW``/``NumPy`` on the CPU and
``CuPy`` on the GPU when available.
"""

from emClarity_FFT import fft as _fft_impl, ifft as _ifft_impl

__all__ = ["ftj", "iftj"]


def ftj(inref, use_gpu: bool = False):
    """Forward FFT with MATLAB-like normalization."""
    return _fft_impl(inref, use_gpu=use_gpu)


def iftj(inref, use_gpu: bool = False):
    """Inverse FFT matching MATLAB's ``iftj.m``."""
    return _ifft_impl(inref, use_gpu=use_gpu)
