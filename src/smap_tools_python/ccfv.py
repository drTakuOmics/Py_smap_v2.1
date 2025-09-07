import numpy as np
from .crop_pad import crop_or_pad


def ccfv(image, templates):
    """Variance-normalized cross-correlation.

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

    image = image - image.mean()
    image_f = np.fft.fftn(np.fft.ifftshift(image))

    n_templates = templates.shape[2]
    out = np.empty((full_x, full_y, n_templates), dtype=float)
    peaks = np.empty(n_templates, dtype=float)

    for i in range(n_templates):
        temp = templates[:, :, i]
        temp = temp - temp.mean()
        template = crop_or_pad(temp, image.shape, pad_value=0)
        template_f = np.fft.fftn(np.fft.ifftshift(template))
        v = np.sum(np.abs(template_f.ravel()) ** 2) / full_xy
        denom = v / full_xy
        template_f /= denom
        cc_f = image_f * np.conj(template_f)
        temp_cc = np.real(np.fft.fftshift(np.fft.ifftn(cc_f))) / full_xy
        out[:, :, i] = temp_cc
        peaks[i] = temp_cc.max()
    return out, peaks
