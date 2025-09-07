import numpy as np
from .crop_pad import crop_or_pad
from .radial import radial_average_im


def ccfn(image, templates):
    """Frequency-normalized cross-correlation.

    Parameters
    ----------
    image : ndarray, shape (M, N)
        Input 2D image.
    templates : ndarray, shape (M_t, N_t, K)
        Stack of ``K`` templates.

    Returns
    -------
    tuple of (ndarray, ndarray)
        ``cc`` is the cross-correlation volume of shape ``(M, N, K)`` and
        ``peaks`` holds the maximum value for each template.
    """
    image = np.asarray(image, dtype=float)
    templates = np.asarray(templates, dtype=float)
    if image.ndim != 2 or templates.ndim != 3:
        raise ValueError("image must be 2D and templates 3D")

    full_x, full_y = image.shape
    full_xy = full_x * full_y
    cp = full_x // 2

    # Zero-mean image and estimate power spectral density via radial averaging
    image = image - image.mean()
    f_amp = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))) / full_x
    f_amp_r = radial_average_im(f_amp)
    f_amp_r[cp, cp] = 1.0
    f_amp_r_inv = 1.0 / f_amp_r
    f_amp_r_inv[cp, cp] = np.nan
    f_amp_r_inv[cp, cp] = np.nanmean(
        f_amp_r_inv[cp - 1 : cp + 2, cp - 1 : cp + 2]
    )
    f_psd = (f_amp_r_inv / np.sum(np.abs(f_amp_r_inv.ravel()) ** 2)) * (full_x ** 2)
    f_psd = np.fft.ifftshift(f_psd)

    image_f = np.fft.fftn(np.fft.ifftshift(image)) * f_psd
    v = np.sum(np.abs(image_f.ravel()) ** 2) / full_xy
    denom = np.sqrt(v / full_xy)
    image_f /= denom

    n_templates = templates.shape[2]
    out = np.empty((full_x, full_y, n_templates), dtype=float)
    peaks = np.empty(n_templates, dtype=float)

    for i in range(n_templates):
        temp = templates[:, :, i]
        temp = temp - temp.mean()
        template = crop_or_pad(temp, image.shape, pad_value=0)
        template_f = np.fft.fftn(np.fft.ifftshift(template)) * f_psd
        cc_f = image_f * np.conj(template_f)
        temp_cc = np.real(np.fft.fftshift(np.fft.ifftn(cc_f)))
        out[:, :, i] = temp_cc
        peaks[i] = temp_cc.max()
    return out, peaks
