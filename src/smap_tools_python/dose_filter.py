import numpy as np
from .fft import ftj, iftj
from .ks import get_ks


def dose_filter(image_stack, total_dose, a_per_pix, norm_type="numerator_only", condition="LN"):
    """Apply dose-weighting to a stack of movie frames.

    Parameters
    ----------
    image_stack : ndarray
        Stack of frames with shape ``(nx, ny, n_frames)``.
    total_dose : float
        Total exposure in e/Å² over the stack.
    a_per_pix : float
        Pixel size in Å/pixel.
    norm_type : {"numerator_only", "noise_restored"}, optional
        If ``"noise_restored"`` the noise power is restored after filtering.
    condition : {"LN", "LHe"}, optional
        Dose model; helium mode doubles the critical dose constants.

    Returns
    -------
    ndarray
        Dose-filtered sum of frames.
    """

    imref = np.asarray(image_stack, dtype=np.float32)
    if imref.ndim != 3:
        raise ValueError("image_stack must be 3-D")
    edge_size, _, n_frames = imref.shape
    dose_per_frame = float(total_dose) / n_frames

    k, _ = get_ks(imref[:, :, 0], a_per_pix)
    a, b, c = 0.24499, -1.6649, 2.8141
    Nc = a * np.power(k, b) + c
    if condition == "LHe":
        Nc *= 2.0

    outref = np.zeros((edge_size, edge_size), dtype=np.float32)
    q2 = np.zeros_like(k, dtype=np.float32)

    for i in range(n_frames):
        N = dose_per_frame * (i + 1)
        q = np.exp(-N / (2.0 * Nc))
        frame = imref[:, :, i]
        dc = frame.mean()
        frame = frame - dc
        frame = iftj(ftj(frame) * q)
        outref += frame + dc
        q2 += q ** 2

    if norm_type == "noise_restored":
        outref = iftj(ftj(outref) / np.sqrt(q2))

    return outref


__all__ = ["dose_filter"]
