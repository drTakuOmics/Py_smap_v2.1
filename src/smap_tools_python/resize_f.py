import numpy as np
from .crop_pad import crop_or_pad


def resize_F(arr, sf, method='fixedSize'):
    """Resize an array by Fourier cropping/padding.

    Parameters
    ----------
    arr : ndarray
        Input array (2-D or 3-D).
    sf : float
        Scaling factor ``pitch_orig/pitch_target``. Values >1 crop (downsample)
        while values <1 pad (upsample).
    method : {'fixedSize', 'newSize'}, optional
        Whether to return an array with the original shape or the natural
        resized shape.
    """
    arr = np.asarray(arr, dtype=float)
    os = np.array(arr.shape)
    final_size = np.floor(os * sf).astype(int)
    f = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arr)))
    f_resized = crop_or_pad(f, final_size, pad_value=0)
    out = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f_resized))).real
    out *= sf ** arr.ndim
    if method == 'fixedSize':
        out = crop_or_pad(out, os)
    return out
