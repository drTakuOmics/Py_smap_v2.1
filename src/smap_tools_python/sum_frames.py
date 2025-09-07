import numpy as np

def sum_frames(frames, target_exposure, exposure_per_frame, axis=0):
    """Sum movie frames up to a desired total exposure.

    Parameters
    ----------
    frames : ndarray
        Stack of frames with shape (N, H, W) or (H, W, N).
    target_exposure : float
        Desired cumulative exposure in e/nm^2.
    exposure_per_frame : float
        Exposure of each frame.

    Returns
    -------
    summed : ndarray
        Sum of the first ``n`` frames where ``n`` is the number needed to reach
        ``target_exposure``.
    n_used : int
        Number of frames summed.
    """
    frames = np.asarray(frames)
    n_total = frames.shape[axis]
    n_to_use = int(np.ceil(target_exposure / exposure_per_frame))
    n_to_use = min(n_to_use, n_total)
    summed = np.take(frames, range(n_to_use), axis=axis).sum(axis=axis)
    return summed, n_to_use
