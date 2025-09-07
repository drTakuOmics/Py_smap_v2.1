import numpy as np


def ftj(inref):
    """Forward FFT with MATLAB-like normalization."""
    Npix = np.prod(inref.shape)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(inref))) / np.sqrt(Npix)


def iftj(inref):
    """Inverse FFT matching MATLAB's iftj.m."""
    Npix = np.prod(inref.shape)
    inref = np.nan_to_num(inref)
    return np.fft.fftshift(np.real(np.fft.ifftn(np.fft.ifftshift(inref)))) * np.sqrt(Npix)
