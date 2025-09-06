import numpy as np


def g2(xyz, beta=(1.0, 0.5)):
    """Evaluate a Gaussian ``A*exp(-xyz**2/(2*sigma**2))``.

    Parameters
    ----------
    xyz : array_like
        Input coordinates.
    beta : tuple of float, optional
        Sequence ``(A, sigma)`` giving the amplitude and standard deviation.

    Returns
    -------
    numpy.ndarray
        Gaussian evaluated at ``xyz`` with the given parameters.
    """
    xyz = np.asarray(xyz, dtype=float)
    A, sigma = beta
    return A * np.exp(-(xyz ** 2) / (2.0 * sigma ** 2))
