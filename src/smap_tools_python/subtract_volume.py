import numpy as np
from scipy.stats import mode
from .fft import ftj, iftj
from .nm import nm
from .rrj import rrj


def _radial_mean_map(vol):
    """Return a radial-mean map for a 3-D volume."""
    r = rrj(vol)
    r_ind = np.round(r * (vol.shape[0] / 2.0)).astype(int)
    flat_r = r_ind.ravel()
    flat_v = np.abs(vol).ravel()
    sums = np.bincount(flat_r, weights=flat_v)
    counts = np.bincount(flat_r)
    counts[counts == 0] = 1
    profile = sums / counts
    return profile[r_ind]


def subtract_volume(recref, templateref, subtraction_mode="normal"):
    """Subtract a scaled template from a volume.

    Parameters
    ----------
    recref : ndarray
        Input reconstruction volume.
    templateref : ndarray
        Template volume to subtract.
    subtraction_mode : {"normal", "twofold"}, optional
        Controls the scaling of the template before subtraction. ``"twofold"``
        subtracts twice the scaled template.

    Returns
    -------
    outref : ndarray
        Volume after template subtraction.
    scaled_template : ndarray
        Scaled template that was subtracted from ``recref``.
    """

    recref = np.asarray(recref, dtype=float)
    templateref = np.asarray(templateref, dtype=float)
    if recref.shape != templateref.shape:
        raise ValueError("volumes must have identical shapes")

    # remove background from template using mode (fall back to median)
    try:
        bg = mode(templateref.ravel(), keepdims=False).mode
        bg = float(bg)
    except Exception:
        bg = float(np.median(templateref))
    template = templateref - bg
    template_F = ftj(template)

    rec = recref - np.mean(recref)

    rec_norm = nm(rec)
    template_norm = nm(template)

    inds = np.abs(template_norm) > 0.5
    temp = np.zeros_like(rec)
    temp[inds] = rec[inds]
    rec_F = ftj(temp)

    q2 = _radial_mean_map(rec_F)
    q2_t = _radial_mean_map(template_F)
    ratio = np.divide(q2, q2_t, out=np.zeros_like(q2), where=q2_t != 0)
    template_mod_F = template_F * ratio
    scaled_template = iftj(template_mod_F)

    if subtraction_mode == "twofold":
        outref = rec - 2 * scaled_template
    else:
        outref = rec - scaled_template
    return outref, scaled_template
