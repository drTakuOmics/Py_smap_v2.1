import numpy as np
from scipy.ndimage import affine_transform
from .crop_pad import extendj
from .ctf import ctf
from .constants import def_consts


def _rotate_volume(vol, R):
    center = (np.array(vol.shape) - 1) / 2.0
    return affine_transform(
        vol,
        R.T,
        offset=center - R.T @ center,
        order=1,
        mode="constant",
        cval=float(np.median(vol)),
    )


def templates(volume, rotations, dfs=None, pixel_size=1.0, edge_size=None, params=None):
    """Generate projection templates from a scattering potential volume.

    Parameters
    ----------
    volume : ndarray, shape (Z, Y, X)
        Scattering potential volume.
    rotations : ndarray, shape (3, 3, N)
        Rotation matrices describing template orientations.
    dfs : array_like, shape (N, 3), optional
        Defocus triplets ``(df1, df2, ast)`` in nanometers. A single row can be
        provided and will be broadcast to all orientations. If ``None``, no CTF
        is applied.
    pixel_size : float, optional
        Pixel size in Angstroms used for CTF generation. Default is ``1.0``.
    edge_size : int, optional
        Output template size. Defaults to ``volume.shape[0]``.
    params : dict, optional
        Microscope parameters to pass to :func:`ctf`. Missing values default to
        those from :func:`def_consts`.

    Returns
    -------
    ndarray
        Stack of 2-D templates with shape ``(edge_size, edge_size, N)``.
    """
    vol = np.asarray(volume, dtype=float)
    rot = np.asarray(rotations, dtype=float)
    n_templates = rot.shape[2]

    if edge_size is None:
        edge_size = vol.shape[0]

    if dfs is None:
        dfs = np.zeros((n_templates, 3), dtype=float)
        use_ctf = False
    else:
        dfs = np.atleast_2d(np.asarray(dfs, dtype=float))
        if dfs.shape[0] == 1 and n_templates > 1:
            dfs = np.repeat(dfs, n_templates, axis=0)
        if dfs.shape[0] != n_templates:
            raise ValueError("dfs must have shape (N,3) or (1,3)")
        use_ctf = True

    if params is None:
        params = def_consts()
    params = dict(params)
    params.setdefault("aPerPix", pixel_size)

    out = np.empty((edge_size, edge_size, n_templates), dtype=float)
    for i in range(n_templates):
        vol_rot = _rotate_volume(vol, rot[:, :, i])
        proj = vol_rot.sum(axis=2)
        proj = extendj(proj, (edge_size, edge_size), np.median(proj))
        if use_ctf:
            ctf_img = np.real(ctf(dfs[i], edge_size, params))
            proj_F = np.fft.fftn(np.fft.ifftshift(proj))
            proj = np.real(np.fft.fftshift(np.fft.ifftn(proj_F * ctf_img)))
        out[:, :, i] = proj

    return out


__all__ = ["templates"]
