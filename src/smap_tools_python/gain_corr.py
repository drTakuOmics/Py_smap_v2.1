import numpy as np


def gain_corr(movie: np.ndarray, gain: np.ndarray, hot_threshold: float = 7.0):
    """Apply gain correction and remove hot pixels from a movie stack.

    Parameters
    ----------
    movie : ndarray, shape (M, N, F)
        Stack of ``F`` frames.
    gain : ndarray, shape (M, N)
        Gain reference image.  Each frame is multiplied by this reference.
    hot_threshold : float, optional
        Pixels whose mean across the stack exceeds ``hot_threshold`` standard
        deviations are replaced by the mean of their ``3x3`` neighbourhood in
        each frame.

    Returns
    -------
    corrected : ndarray
        Gain-corrected movie.
    hot_pixels : ndarray, shape (K, 2)
        Coordinates of replaced hot pixels.
    """
    movie = np.asarray(movie, dtype=float)
    gain = np.asarray(gain, dtype=float)
    if movie.ndim != 3:
        raise ValueError("movie must be 3D (M, N, F)")
    if gain.shape != movie.shape[:2]:
        raise ValueError("gain shape must match movie xy dimensions")

    corrected = movie * gain[..., None]
    mean_im = corrected.mean(axis=2)
    z = (mean_im - mean_im.mean()) / mean_im.std()
    hot = np.argwhere(z > hot_threshold)

    for x, y in hot:
        xs = slice(max(x - 1, 0), min(x + 2, corrected.shape[0]))
        ys = slice(max(y - 1, 0), min(y + 2, corrected.shape[1]))
        neighbourhood = corrected[xs, ys, :]
        corrected[x, y, :] = neighbourhood.reshape(-1, neighbourhood.shape[-1]).mean(axis=0)

    return corrected, hot
