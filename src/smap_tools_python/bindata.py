import numpy as np


def bindata(y, x, xrg, sf=None):
    """Bin data ``y`` according to coordinates ``x``.

    Parameters
    ----------
    y : array_like
        Values to be averaged within bins.
    x : array_like or complex array
        Coordinates for each value. Real and imaginary parts are used as
        separate dimensions when ``xrg`` is a sequence.
    xrg : array_like or sequence of arrays
        Bin edges. If a sequence of two arrays is provided, 2-D binning is
        performed.
    sf : int, optional
        Savitzky-Golay smoothing window size applied to 2-D outputs, matching
        MATLAB's optional smoothing. Ignored for 1-D data when not provided.

    Returns
    -------
    tuple
        ``(ym, yb, y_full)`` where ``ym`` are the mean values per bin,
        ``yb`` maps each input sample to its bin mean, and ``y_full`` are the
        bin centers.
    """

    y = np.asarray(y)
    x = np.asarray(x)

    if not (isinstance(xrg, (list, tuple)) and len(xrg) == 2):
        edges = np.asarray(xrg)
        bins = np.digitize(x, edges) - 1
        bins = np.clip(bins, 0, len(edges) - 2)
        counts = np.bincount(bins, minlength=len(edges) - 1)
        sums = np.bincount(bins, weights=y, minlength=len(edges) - 1)
        ym = np.full(len(edges) - 1, np.nan, dtype=float)
        mask = counts > 0
        ym[mask] = sums[mask] / counts[mask]
        yb = ym[bins]
        y_full = edges[:-1]
        return ym, yb, y_full

    # 2-D case: ``x`` contains complex numbers encoding x + 1j*y
    x1 = np.real(x)
    x2 = np.imag(x)
    x1rg, x2rg = map(np.asarray, xrg)
    bins1 = np.digitize(x1, x1rg) - 1
    bins1 = np.clip(bins1, 0, len(x1rg) - 2)
    bins2 = np.digitize(x2, x2rg) - 1
    bins2 = np.clip(bins2, 0, len(x2rg) - 2)
    bins = bins1 + bins2 * (len(x1rg) - 1)
    nbins = (len(x1rg) - 1) * (len(x2rg) - 1)
    counts = np.bincount(bins, minlength=nbins)
    sums = np.bincount(bins, weights=y, minlength=nbins)
    ym = np.full(nbins, np.nan, dtype=float)
    mask = counts > 0
    ym[mask] = sums[mask] / counts[mask]
    if sf is not None:
        from scipy.signal import savgol_filter

        ym[mask] = savgol_filter(ym[mask], sf, 1)
    y_full = ym[bins]
    yb = y_full
    ym = ym.reshape(len(x1rg) - 1, len(x2rg) - 1)
    return ym, yb, y_full
