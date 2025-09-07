"""Utilities for working with icosahedral symmetry."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .icos import icos


def get_icos(
    q_best: np.ndarray,
    new_icos_flag: bool | int = True,
    a_per_vox: float = 0.97,
):
    """Return icosahedral symmetry operations and rotated reference points.

    Parameters
    ----------
    q_best : array_like, shape (4,)
        Quaternion representing the best orientation.  The quaternion should be
        in the ``[x, y, z, w]`` convention used by :mod:`scipy`.
    new_icos_flag : bool, optional
        Present for API compatibility with the MATLAB version.  It is ignored
        as the Python implementation always generates operations analytically.
    a_per_vox : float, optional
        Angstroms per voxel used to scale the returned coordinates.

    Returns
    -------
    tuple
        ``(q_out, xyz_sub, xyz_rnap)`` where ``q_out`` is an array of 60
        quaternions representing the icosahedral symmetry operations applied to
        ``q_best`` and the coordinate arrays hold the rotated reference points
        for each asymmetric unit.
    """

    base = Rotation.from_quat(np.asarray(q_best))
    ops = Rotation.create_group("I")
    q_out = (base * ops).as_quat()

    xyz_sub, xyz_rnap = icos(a_per_vox)
    xyz_sub = base.apply(xyz_sub)
    xyz_rnap = base.apply(xyz_rnap)

    return q_out, xyz_sub, xyz_rnap


__all__ = ["get_icos"]

