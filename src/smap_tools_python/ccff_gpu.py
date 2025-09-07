import numpy as np
from .ccff import ccff
from .crop_pad import crop_or_pad
from .nm import nm

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None


def ccff_gpu(image, templates, mode="filt"):
    """GPU-accelerated :func:`ccff` using CuPy when available.

    When CuPy is not installed this function falls back to the CPU
    implementation. The API mirrors :func:`ccff`.
    """
    if cp is None:
        return ccff(np.asarray(image), np.asarray(templates), mode)

    image_gpu = cp.asarray(image, dtype=cp.float32)
    templates_gpu = cp.asarray(templates, dtype=cp.float32)

    if mode == "filt":
        from .psd_filter import psd_filter

        f_psd, im_filt, _ = psd_filter(cp.asnumpy(image_gpu), method="sqrt")
        im_filt = nm(im_filt)
        f_psd = cp.asarray(np.fft.ifftshift(f_psd))
        im_filt = cp.asarray(im_filt)
    else:
        f_psd = cp.ones_like(image_gpu)
        im_filt = cp.asarray(nm(cp.asnumpy(image_gpu)))

    imref_F = cp.fft.fftn(cp.fft.ifftshift(im_filt)) / cp.sqrt(image_gpu.size)

    n_templates = templates_gpu.shape[2]
    out = cp.empty((image_gpu.shape[0], image_gpu.shape[1], n_templates), dtype=cp.float32)
    peaks = cp.empty(n_templates, dtype=cp.float32)

    for i in range(n_templates):
        temp = cp.asnumpy(templates_gpu[:, :, i])
        temp = crop_or_pad(temp, image_gpu.shape[:2], pad_value=float(np.median(temp)))
        temp = nm(temp)
        temp_gpu = cp.asarray(temp)
        template_F = cp.fft.fftn(cp.fft.ifftshift(temp_gpu)) / temp_gpu.size
        if mode == "filt":
            template_F *= f_psd
        template_F /= cp.std(template_F)
        cc_F = imref_F * cp.conj(template_F)
        cc = cp.real(cp.fft.fftshift(cp.fft.ifftn(cc_F))) * cp.sqrt(image_gpu.size)
        out[:, :, i] = cc
        peaks[i] = cp.max(cc)

    return cp.asnumpy(out), cp.asnumpy(peaks)


__all__ = ["ccff_gpu"]
