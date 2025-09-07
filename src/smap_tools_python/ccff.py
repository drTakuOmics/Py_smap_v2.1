import numpy as np
from .psd_filter import psd_filter
from .crop_pad import crop_or_pad
from .nm import nm


def ccff(image: np.ndarray, templates: np.ndarray, mode: str = "filt"):
    """Whitened cross-correlation of an image with templates.

    Parameters
    ----------
    image : ndarray, shape (M, N)
        Input image.
    templates : ndarray, shape (M_t, N_t, K)
        Stack of ``K`` templates.
    mode : {"filt", "noFilt"}, optional
        If ``"filt"`` (default) the image is radially whitened prior to
        correlation using :func:`psd_filter`.  ``"noFilt"`` skips whitening.

    Returns
    -------
    tuple of (ndarray, ndarray)
        The cross-correlation volume of shape ``(M, N, K)`` and the peak value
        for each template.
    """
    image = np.asarray(image, dtype=float)
    templates = np.asarray(templates, dtype=float)
    if image.ndim != 2 or templates.ndim != 3:
        raise ValueError("image must be 2D and templates 3D")

    if mode not in {"filt", "noFilt"}:
        raise ValueError("mode must be 'filt' or 'noFilt'")

    if mode == "filt":
        f_psd, im_filt, _ = psd_filter(image, method="sqrt")
        im_filt = nm(im_filt)
    else:
        f_psd = np.ones_like(image)
        im_filt = nm(image)

    imref_F = np.fft.fftn(np.fft.ifftshift(im_filt)) / np.sqrt(image.size)
    f_psd = np.fft.ifftshift(f_psd)

    n_templates = templates.shape[2]
    out = np.empty((image.shape[0], image.shape[1], n_templates), dtype=float)
    peaks = np.empty(n_templates, dtype=float)

    for i in range(n_templates):
        temp = templates[:, :, i]
        temp = crop_or_pad(temp, image.shape, pad_value=np.median(temp))
        temp = nm(temp)
        template_F = np.fft.fftn(np.fft.ifftshift(temp)) / temp.size
        if mode == "filt":
            template_F *= f_psd
        template_F /= template_F.std()
        cc_F = imref_F * np.conj(template_F)
        cc = np.real(np.fft.fftshift(np.fft.ifftn(cc_F))) * np.sqrt(image.size)
        out[:, :, i] = cc
        peaks[i] = cc.max()

    return out, peaks
