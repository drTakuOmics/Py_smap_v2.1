import numpy as np


def pr_quick(table, prec_thrs=None, mode="standard"):
    """Estimate precision/recall curves and thresholds.

    Parameters
    ----------
    table : mapping
        Object with keys ``'peak_part'`` and ``'peak_part_ctrl'`` (or their
        ``'_opt'`` counterparts when ``mode='opt'``).
    prec_thrs : sequence of float, optional
        Precision levels for which thresholds should be returned.
    mode : {'standard', 'opt'}, optional
        Select which columns of ``table`` to use.

    Returns
    -------
    thr_ref : ndarray
        Thresholds achieving each requested precision level.
    prec : ndarray
        Estimated precision values for all histogram bins.
    recall : ndarray
        Estimated recall values for all histogram bins.
    thr_F1 : float
        Threshold giving maximum F1 score.
    F1_opt : float
        Maximum F1 score.
    """
    if prec_thrs is None:
        prec_thrs = [0.99, 0.95, 0.9, 0.85]
    vals_key = "peak_part_opt" if mode == "opt" else "peak_part"
    ctrl_key = vals_key + "_ctrl"
    vals = np.asarray(table[vals_key], dtype=float)
    vals_ctrl = np.asarray(table[ctrl_key], dtype=float)
    vmin = min(vals.min(), vals_ctrl.min())
    vmax = max(vals.max(), vals_ctrl.max())
    xx = np.linspace(vmin, vmax, 5000)
    peaks, _ = np.histogram(vals, bins=xx)
    peaks_ctrl, _ = np.histogram(vals_ctrl, bins=xx)
    cs = peaks.sum() - np.cumsum(peaks)
    cs_ctrl = peaks_ctrl.sum() - np.cumsum(peaks_ctrl)
    den = cs + cs_ctrl
    zero_mask = den == 0
    den[zero_mask] = 1
    prec = cs / den
    prec[zero_mask] = 0
    recall = cs / cs.max()
    thr_ref = []
    for pt in prec_thrs:
        inds = np.where(prec > pt)[0]
        if inds.size:
            thr_ref.append(xx[inds[0]])
    with np.errstate(divide="ignore", invalid="ignore"):
        F1 = 2.0 / (1.0 / recall + 1.0 / prec)
    F1 = np.nan_to_num(F1)
    idx = np.argmax(F1)
    thr_F1 = xx[idx]
    F1_opt = F1[idx]
    return np.array(thr_ref), prec, recall, thr_F1, F1_opt
