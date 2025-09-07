from emClarity_FFT import fft as _fft_impl, ifft as _ifft_impl


def ftj(inref, use_gpu=False):
    """Forward FFT with MATLAB-like normalization.

    Parameters
    ----------
    inref : array-like
        Input volume or image.
    use_gpu : bool, optional
        Perform the computation on the GPU when CuPy is available.
    """
    return _fft_impl(inref, use_gpu=use_gpu)


def iftj(inref, use_gpu=False):
    """Inverse FFT matching MATLAB's ``iftj.m``.

    Parameters
    ----------
    inref : array-like
        Input volume or image in Fourier space.
    use_gpu : bool, optional
        Perform the computation on the GPU when CuPy is available.
    """
    return _ifft_impl(inref, use_gpu=use_gpu)
