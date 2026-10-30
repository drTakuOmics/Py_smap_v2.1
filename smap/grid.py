"""Orientation search grid (@smap/calculate_search_grid.m, @smap/q2R.m)."""

from __future__ import annotations

import math

import numpy as np

from .utils import matlab_colon, normalize_rm


def calculate_search_grid(symmetry: str, angular_step: float, psi_step: float,
                          dedupe: bool = False):
    """Port of calculate_search_grid.m -> (R (n,3,3), euler angles (3, n_unmirrored) deg).

    For C symmetry the C2x-mirrored set is appended (test_mirror), as in MATLAB.
    QUIRK: phi runs 0:phi_step:phi_max inclusive, so phi=0 and phi=360 both appear.
    """
    sym_type = symmetry[0].upper()
    sym_n = int(symmetry[1:]) if len(symmetry) > 1 else 1
    psi_max = 358.0
    table = {"C": (360.0 / sym_n, 90.0, 1), "D": (360.0 / sym_n, 90.0, 0),
             "T": (180.0, 54.7, 0), "O": (90.0, 54.7, 0), "I": (180.0, 31.7, 0)}
    phi_max, theta_max, test_mirror = table[sym_type]

    psi_vec = matlab_colon(0.0, psi_step, psi_max)
    theta_step = theta_max / math.floor(theta_max / angular_step + 0.5)
    cols = []
    for theta in matlab_colon(0.0, theta_step, theta_max + theta_step / 2.0):
        if theta == 0 or theta == 180:
            phi_step = phi_max
        else:
            phi_step = abs(angular_step / math.sin(theta * math.pi / 180))
            if phi_step > phi_max:
                phi_step = phi_max
            phi_step = phi_max / math.floor(phi_max / phi_step + 0.5)
        for phi in matlab_colon(0.0, phi_step, phi_max):
            for psi in psi_vec:
                cols.append((phi, theta, psi))
    ea = np.array(cols, dtype=np.float64).T  # (3, n)

    phi, theta, psi = np.deg2rad(ea)
    cph, cps, cth = np.cos(phi), np.cos(psi), np.cos(theta)
    sph, sps, sth = np.sin(phi), np.sin(psi), np.sin(theta)
    RM = np.empty((ea.shape[1], 3, 3))
    RM[:, 0, 0] = cph * cth * cps - sph * sps
    RM[:, 0, 1] = sph * cth * cps + cph * sps
    RM[:, 0, 2] = -sth * cps
    RM[:, 1, 0] = -cph * cth * sps - sph * cps
    RM[:, 1, 1] = -sph * cth * sps + cph * cps
    RM[:, 1, 2] = sth * sps
    RM[:, 2, 0] = sth * cph
    RM[:, 2, 1] = sth * sph
    RM[:, 2, 2] = cth

    if test_mirror:
        # quaternion.eulerangles('xyz',[pi 0 0]) -> 180 deg about x
        R_flip = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
        RM = normalize_rm(RM)
        RM_mirror = normalize_rm(RM @ R_flip)
        out = np.concatenate([RM, RM_mirror], axis=0)
    else:
        out = normalize_rm(RM)
    if dedupe:
        out = unique_rotations(out)
    return out, ea


def unique_rotations(R: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """OPTION ``dedupe_orientations``: drop repeats (quaternion distance < tol), keeping
    the first occurrence in grid order.

    v2.1 grids contain repeats (e.g. phi=0 and phi=phi_max at every theta). Repeats waste
    search time and make the rotation index of a peak ambiguous.
    """
    from scipy.spatial import cKDTree
    from .rot import rotm_to_quat_robust
    q = rotm_to_quat_robust(R)
    n = q.shape[0]
    pts = np.concatenate([q, -q])                       # q and -q are the same rotation
    pairs = cKDTree(pts).query_pairs(tol, output_type="ndarray")
    drop = np.zeros(n, bool)
    for i, j in pairs:
        a, b = i % n, j % n
        if a != b:
            drop[max(a, b)] = True
    return R[~drop]


def q2r(q) -> np.ndarray:
    """@smap/q2R.m: quaternion [w x y z] -> rotation matrix (orthonormalised)."""
    q = np.asarray(q, dtype=np.float64)
    d = math.sqrt(float(np.sum(q**2)))
    q = q / d if d > 0 else np.array([1.0, 0, 0, 0])
    a, b, c, e = q
    R = np.array([
        [a*a + b*b - c*c - e*e, 2*(b*c - a*e), 2*(b*e + a*c)],
        [2*(b*c + a*e), a*a - b*b + c*c - e*e, 2*(c*e - a*b)],
        [2*(b*e - a*c), 2*(c*e + a*b), a*a - b*b - c*c + e*e],
    ])
    return normalize_rm(R)
