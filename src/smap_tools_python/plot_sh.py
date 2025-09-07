"""Compute survival histograms for peak values.

This is a minimalist translation of the MATLAB ``plotSH`` utility. It returns
sample-wise survival counts and theoretical Gaussian expectations. Plotting is
optional to keep the function lightweight.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc, erfcinv


def plot_sh(peak_vals, color="b", Nsamples=None, nfp=1, ax=None):
    """Return survival histogram data for ``peak_vals``.

    Parameters
    ----------
    peak_vals : array_like
        Peak scores to analyse.
    color : str, optional
        Line colour when plotting; ignored if ``ax`` is ``None``.
    Nsamples : int, optional
        Number of effective samples. Defaults to ``len(peak_vals)``.
    nfp : int, optional
        Number of false positives to expect when computing the threshold.
    ax : matplotlib axis, optional
        If provided, curves are plotted onto this axis using ``semilogy``.

    Returns
    -------
    tuple
        ``(xs, ys, YS, thr)`` where ``xs`` are bin centers, ``ys`` the empirical
        survival counts, ``YS`` the Gaussian expectation, and ``thr`` the
        detection threshold for ``nfp`` expected false positives.
    """

    v = np.ravel(peak_vals)
    if v.size == 0:
        raise ValueError("peak_vals must contain at least one value")
    if Nsamples is None:
        Nsamples = v.size
    xs = np.linspace(v.min(), v.max(), 10_000)
    hist, edges = np.histogram(v, bins=xs)
    ys = Nsamples - np.cumsum(hist)
    xs = edges[:-1]
    YS = erfc(xs / np.sqrt(2)) / 2 * Nsamples
    thr = np.sqrt(2) * erfcinv(nfp * 2 / Nsamples)
    if ax is not None:
        ax.semilogy(xs, ys, color)
        ax.semilogy(xs, YS, "k--")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(0.9, max(YS) * 2)
        ax.axis("square")
        ax.grid(True)
    return xs, ys, YS, thr
