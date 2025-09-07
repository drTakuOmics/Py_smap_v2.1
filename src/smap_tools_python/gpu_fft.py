"""GPU accelerated FFT helpers built on :mod:`emClarity_FFT`.

The functions here provide MATLAB compatible FFTs executed on a CUDA capable
GPU via CuPy.  They automatically fall back with a helpful error message when
no GPU backend is available.
"""

from emClarity_FFT import fft as _fft_impl, ifft as _ifft_impl, has_gpu

__all__ = ["gpu_ftj", "gpu_iftj", "ftjg", "iftjg"]


def gpu_ftj(inref):
    """Forward FFT executed on the GPU."""
    if not has_gpu():  # pragma: no cover - optional dependency
        raise RuntimeError("CuPy is required for GPU FFT operations")
    return _fft_impl(inref, use_gpu=True)


def gpu_iftj(inref):
    """Inverse FFT executed on the GPU."""
    if not has_gpu():  # pragma: no cover - optional dependency
        raise RuntimeError("CuPy is required for GPU FFT operations")
    return _ifft_impl(inref, use_gpu=True)


# MATLAB compatibility aliases
ftjg = gpu_ftj
iftjg = gpu_iftj
