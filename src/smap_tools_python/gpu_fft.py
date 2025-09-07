"""GPU accelerated FFT helpers."""

from emClarity_FFT import fft as _fft_impl, ifft as _ifft_impl, has_gpu


def gpu_ftj(inref):
    """Forward FFT executed on the GPU.

    Raises
    ------
    RuntimeError
        If a GPU backend (CuPy) is not available.
    """
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
