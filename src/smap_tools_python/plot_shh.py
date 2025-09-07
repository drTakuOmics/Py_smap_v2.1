import numpy as np
from scipy.special import erfc, erfcinv


def plot_shh(bins, N, nfp=1.0, ax=None):
    """Plot survival histogram and Gaussian expectation.

    Parameters
    ----------
    bins : array_like
        Bin centres of the histogram.
    N : ndarray
        Counts per bin; summed over axis 1 to obtain total counts.
    nfp : float, optional
        Number of false positives expected. Default is 1.
    ax : matplotlib axis, optional
        Axis on which to plot. A new figure is created when omitted.

    Returns
    -------
    thr : float
        Threshold SNR corresponding to ``nfp`` false positives.
    """
    bins = np.asarray(bins, dtype=float)
    N = np.asarray(N, dtype=float)
    if N.ndim == 1:
        NN = N
    else:
        NN = N.sum(axis=1)
    Nsamples = NN.sum()
    YS = (erfc(bins / np.sqrt(2)) / 2.0) * Nsamples
    thr = np.sqrt(2) * erfcinv(nfp * 2.0 / Nsamples)

    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
    ax.semilogy(bins, YS, "k--")
    ax.semilogy(bins, Nsamples - np.cumsum(NN))
    ax.set_xlim(0, 13)
    ax.set_ylim(0.9, Nsamples * 2)
    ax.axvline(thr)
    ax.set_xlabel("SNR")
    ax.set_ylabel("survival count")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    return float(thr)
