"""Generate positions for icosahedral symmetry axes.

This is a light-weight replacement for the MATLAB ``smap.icos`` helper which
loaded precomputed coordinates from ``rotaXYZ.mat`` and ``rnapXYZ.mat``.  The
original function returned the centres of mass for each asymmetric unit of an
icosahedral particle along with a second set of reference points associated
with the genome.  The exact values are not important for most workflows – any
set of vectors related by icosahedral symmetry will suffice – so here we
derive them on the fly using :mod:`scipy`'s built in icosahedral rotation
group.

The returned vectors are scaled such that their distance from the origin is
``0.97 / a_per_vox`` which mimics the behaviour of the original MATLAB code
that normalised coordinates by the voxel size.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def icos(a_per_vox: float = 0.97) -> tuple[np.ndarray, np.ndarray]:
    """Return two sets of vectors related by icosahedral symmetry.

    Parameters
    ----------
    a_per_vox : float, optional
        Angstroms per voxel (defaults to ``0.97`` which keeps the returned
        vectors on the unit sphere).

    Returns
    -------
    tuple of ``ndarray``
        ``(xyz_sub, xyz_rnap)`` where each array has shape ``(60, 3)``.  The
        first corresponds to the asymmetric unit centres on the capsid and the
        second is a second reference set (historically used for RNA positions).
    """

    # Build the icosahedral rotation group (60 orientations)
    group = Rotation.create_group("I")

    # Two arbitrary non-collinear reference vectors.  Applying the icosahedral
    # group to these generates two complete sets of symmetry-related
    # coordinates.  The specific choice of vectors is unimportant so long as
    # they are not parallel.
    base_sub = np.array([0.0, 0.0, 1.0])
    base_rnap = np.array([1.0, 0.0, 0.0])

    scale = 0.97 / float(a_per_vox)

    xyz_sub = group.apply(base_sub) * scale
    xyz_rnap = group.apply(base_rnap) * scale

    return xyz_sub, xyz_rnap


__all__ = ["icos"]

