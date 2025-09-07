import numpy as np
from .ccf import ccf
from .max_interp_f import max_interp_f
from .phase_shift import apply_phase_shifts
from .crop_pad import extendj, cutj


def make_template_stack(nf_im, templates=None, ref_template_stack=None):
    """Align templates to an image and return their stack and sum.

    Parameters
    ----------
    nf_im : ndarray
        The noise-filtered image used for alignment.
    templates : ndarray, optional
        Stack of templates with shape ``(M, N, K)``.
    ref_template_stack : ndarray, optional
        Reference stack to provide initial alignment coordinates.

    Returns
    -------
    tuple of (ndarray, ndarray)
        ``ti`` is the summed template image and ``template_im`` contains the
        individual aligned templates.
    """
    nf_im = np.asarray(nf_im, dtype=float)
    if templates is None:
        return np.zeros_like(nf_im), np.zeros(nf_im.shape + (0,), dtype=float)

    templates = np.asarray(templates, dtype=float)
    pad_val = np.nanmedian(templates)
    ti = np.ones_like(nf_im, dtype=float) * pad_val
    template_im = np.ones(nf_im.shape + (templates.shape[2],), dtype=float) * pad_val

    for j in range(templates.shape[2]):
        temp = templates[:, :, j]
        if ref_template_stack is None:
            cc, _ = ccf(nf_im, temp[:, :, None])
            cc = cc[:, :, 0]
        else:
            cc, _ = ccf(nf_im, ref_template_stack[:, :, j][:, :, None])
            cc = cc[:, :, 0]
        yt, xt = np.unravel_index(np.argmax(cc), cc.shape)
        shifts, _ = max_interp_f(cc, 10, 20, (yt, xt))
        padded = extendj(temp, (2048, 2048), pad_val)
        shifted = apply_phase_shifts(padded, shifts)
        template_im[:, :, j] = cutj(shifted, (nf_im.shape[0], nf_im.shape[1]))
        ti += template_im[:, :, j]

    return ti, template_im

__all__ = ["make_template_stack"]
