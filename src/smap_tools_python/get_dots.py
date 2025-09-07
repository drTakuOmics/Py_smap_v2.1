import numpy as np

from .templates import templates
from .rrj import rrj
from .particle_diameter import particle_diameter


def get_dots(volume, rotations, df, edge_size=None, n_samples=1000, pixel_size=1.0):
    """Estimate dot-product statistics from random template orientations.

    Parameters
    ----------
    volume : ndarray, shape (Z, Y, X)
        Scattering potential volume from which templates are generated.
    rotations : ndarray, shape (3, 3, N)
        Rotation matrices describing possible orientations.
    df : array_like, shape (3,), optional
        Defocus triplet used for all templates. ``None`` disables CTF
        modulation.
    edge_size : int, optional
        Output template size. Defaults to the volume size.
    n_samples : int, optional
        Number of random orientations to sample. Defaults to 1000.
    pixel_size : float, optional
        Pixel size in Angstroms passed to :func:`templates`.

    Returns
    -------
    tuple of float
        Mean and standard deviation of the masked template standard
        deviations, matching the MATLAB ``getDots`` heuristic.
    """
    vol = np.asarray(volume, float)
    rot = np.asarray(rotations, float)
    if rot.shape != (3, 3, rot.shape[2]):
        rot = rot.reshape(3, 3, -1)
    n_templates = rot.shape[2]

    rng = np.random.default_rng(1)
    if n_samples > n_templates:
        n_samples = n_templates
    rand_inds = rng.permutation(n_templates)[:n_samples]

    if edge_size is None:
        edge_size = vol.shape[0]

    tt = templates(vol, rot[:, :, rand_inds], df, pixel_size, edge_size)
    bg_val = np.median(tt)

    mask_size = min(particle_diameter(vol) * 1.25, edge_size)
    Rmask = rrj((edge_size, edge_size)) * (edge_size / 0.5)
    mask = np.ones_like(Rmask)
    mask[Rmask > (mask_size / 2)] = np.nan

    dots = []
    for i in range(tt.shape[2]):
        template = tt[:, :, i] - bg_val
        t_masked = template * mask
        dots.append(np.nanstd(t_masked))
    dots = np.asarray(dots)
    dot_mean = float(dots.mean() * 1.125)
    dot_std = float(dots.std())
    return dot_mean, dot_std


__all__ = ["get_dots"]
